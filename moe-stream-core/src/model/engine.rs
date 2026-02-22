//! MoE Streaming Engine: full forward pass and generation loop.
//!
//! Orchestrates the layer-by-layer forward through a GGUF model:
//! 1. Embedding lookup
//! 2. Layer loop (attention/DeltaNet + MoE for each layer)
//! 3. Final norm + LM head
//! 4. Greedy token generation with KV-cache
//!
//! Optimizations:
//! - Expert LRU cache: avoids re-dequantizing frequently used experts
//! - Attention weight residency: keeps attention weights in GPU memory
//! - Router gate residency: keeps router gates in GPU memory
//!
//! For hybrid models (Qwen3-Coder-Next 80B):
//! - DeltaNet state management (linear attention layers)
//! - Partial RoPE for full attention layers
//! - Shared expert support

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use rand::Rng;
use std::time::Instant;

use crate::chat_template::ChatTemplate;
use crate::config::{StreamingConfig, LayerTierConfig, InferenceMode, DevicePreference};
use crate::gguf::reader::GgufReader;
use crate::model::cache::{AttentionWeights, DeltaNetWeights, ExpertCache, ExpertWeights, GpuExpertProjection, GpuExpertWeights, NormWeights, SharedExpertWeights, ResidentWeights, LayerOutputCache, EntropyProfiler, EntropyLayerStats, RoutingStatsCollector, RoutingLayerStats, QuantizedAttentionWeights};
#[cfg(feature = "metal")]
use crate::model::cache::{PackedMxfp4Experts, PackedMxfp4Layer, Mxfp4Weight, Mxfp4AttentionWeights, QuantizedAttnWeight, QuantizedAttnMetalWeights};
use crate::gguf::reader::GgmlQuantType;
use crate::model::kv_cache::KvCache;
use crate::model::deltanet::DeltaNetState;
use crate::model::layer::{LayerForward, VqConfig, load_weight, load_expert, use_profile_layers, LayerTiming, StepTimingStats, ProfileStats};
use crate::ops;

/// Get total system RAM in bytes (macOS via sysctl).
fn get_system_ram() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        use std::mem;
        let mut size: u64 = 0;
        let mut len = mem::size_of::<u64>();
        let key = std::ffi::CString::new("hw.memsize").ok()?;
        let ret = unsafe {
            libc::sysctlbyname(
                key.as_ptr(),
                &mut size as *mut u64 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if ret == 0 { Some(size) } else { None }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<u64>() {
                            return Some(kb * 1024);
                        }
                    }
                }
            }
        }
        None
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

/// Get CUDA GPU total memory in bytes.
/// Uses candle's CudaDevice to query the device.
#[cfg(feature = "cuda")]
fn get_cuda_gpu_memory() -> Option<u64> {
    // Try to get CUDA device memory via nvidia-smi as a fallback.
    // candle doesn't expose a direct memory query API.
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits", "--id=0"])
        .output()
        .ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mib: u64 = stdout.trim().parse().ok()?;
    Some(mib * 1024 * 1024)
}

/// Sampling parameters for token generation.
#[derive(Clone, Debug)]
pub struct SamplingParams {
    /// Temperature for logit scaling (1.0 = no change, <1 = sharper, >1 = flatter).
    /// When set to 0.0, falls back to greedy argmax.
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold. Only tokens with cumulative probability
    /// <= top_p are considered. 1.0 = disabled.
    pub top_p: f32,
    /// Repetition penalty factor. Logits of previously generated tokens are divided
    /// by this value (if positive) or multiplied (if negative). 1.0 = disabled.
    pub repetition_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        }
    }
}

impl SamplingParams {
    /// Returns true if these params are equivalent to greedy argmax.
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }

    /// Sample a token from logits using temperature, repetition penalty, and top-p.
    ///
    /// When temperature is 0.0, falls back to greedy argmax.
    pub fn sample(&self, logits: &[f32], generated: &[u32], rng: &mut impl Rng) -> u32 {
        let mut logits = logits.to_vec();

        // Apply repetition penalty: divide logits of already-generated tokens
        if self.repetition_penalty != 1.0 {
            for &tok in generated {
                let idx = tok as usize;
                if idx < logits.len() {
                    if logits[idx] > 0.0 {
                        logits[idx] /= self.repetition_penalty;
                    } else {
                        logits[idx] *= self.repetition_penalty;
                    }
                }
            }
        }

        // Greedy fallback
        if self.temperature == 0.0 {
            return logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
        }

        // Temperature scaling
        if self.temperature != 1.0 {
            let inv_t = 1.0 / self.temperature;
            for v in logits.iter_mut() {
                *v *= inv_t;
            }
        }

        // Stable softmax
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|v| (v - max_val).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // Top-p (nucleus) filtering
        if self.top_p < 1.0 {
            let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut cumsum = 0.0f32;
            let mut cutoff = indexed.len();
            for (i, &(_, p)) in indexed.iter().enumerate() {
                cumsum += p;
                if cumsum > self.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            // Zero out tokens beyond the nucleus
            let kept: std::collections::HashSet<usize> =
                indexed[..cutoff].iter().map(|&(idx, _)| idx).collect();
            for (i, p) in probs.iter_mut().enumerate() {
                if !kept.contains(&i) {
                    *p = 0.0;
                }
            }
            // Re-normalize
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                for p in probs.iter_mut() {
                    *p /= sum;
                }
            }
        }

        // Weighted random sampling
        let r: f32 = rng.gen();
        let mut cumsum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i as u32;
            }
        }
        // Fallback (rounding edge case)
        (probs.len() - 1) as u32
    }
}

/// MoE Streaming Engine for autoregressive generation.
pub struct Engine {
    reader: GgufReader,
    config: StreamingConfig,
    device: Device,
    /// Pre-computed RoPE cos table [max_seq_len, head_dim/2] (on device)
    cos: Tensor,
    /// Pre-computed RoPE sin table [max_seq_len, head_dim/2] (on device)
    sin: Tensor,
    /// CPU copies of RoPE tables (avoid Metal→CPU transfer per layer)
    cos_cpu: Tensor,
    sin_cpu: Tensor,
    /// Embedding weight (kept resident: vocab_size × hidden_size)
    embed_weight: Tensor,
    /// LM head weight (kept resident: vocab_size × hidden_size)
    lm_head_weight: Tensor,
    /// LM head bias (optional, e.g. GPT-OSS)
    lm_head_bias: Option<Tensor>,
    /// Final norm weight (kept resident: hidden_size)
    final_norm_weight: Tensor,
    /// KV-cache for autoregressive generation (attention layers only)
    kv_cache: KvCache,
    /// Expert LRU cache (per-layer)
    expert_cache: ExpertCache,
    /// Resident weights (attention + router gates)
    resident: ResidentWeights,
    /// DeltaNet state for hybrid models (None for pure attention models)
    deltanet_state: Option<DeltaNetState>,
    /// Adaptive expert skip cache (router logits + MoE output per layer)
    layer_output_cache: LayerOutputCache,
    /// Chat template (auto-detected from model name/architecture)
    chat_template: ChatTemplate,
    /// Maximum number of layers to run (for debugging). 0 = all layers.
    max_layers: usize,
    /// Per-layer entropy profiler for routing analysis.
    entropy_profiler: EntropyProfiler,
    /// Per-layer routing statistics collector for calibration-based importance scoring.
    routing_stats: RoutingStatsCollector,
    /// Per-layer timing profiler (gated behind PROFILE_LAYERS=1).
    profile_stats: Option<ProfileStats>,
    /// Latest step timing from forward (used by generate to accumulate).
    last_step_timing: Option<StepTimingStats>,
    /// VQ codebooks: [layer][proj(0=gate,1=up,2=down)] → Vec<f32> (K * block_dim).
    /// Pre-loaded at startup for shared VQ models (~4.5 MB total). None for non-VQ or per-expert VQ.
    vq_codebooks: Option<Vec<[Vec<f32>; 3]>>,
    /// VQ configuration (block dimensions and codebook size). None for non-VQ models.
    vq_config: Option<VqConfig>,
    /// Whether VQ uses per-expert codebooks (read on-demand) vs shared (pre-loaded).
    vq_per_expert: bool,
}

impl Engine {
    /// Create a new engine from a GGUF file path.
    ///
    /// Loads embed/lm_head/norm as resident and auto-configures eviction policy.
    /// Call `preload_weights()` after for maximum speed (done automatically by generate.rs).
    pub fn open(gguf_path: &str, max_seq_len: usize) -> Result<Self> {
        Self::open_with_device(gguf_path, max_seq_len, DevicePreference::Auto)
    }

    /// Create a new engine from a GGUF file path with explicit device preference.
    ///
    /// Device preference controls inference mode selection:
    /// - Auto: picks GpuResident/RamResident/SsdStreaming based on model size and hardware
    /// - Gpu: forces GPU Resident mode (Metal)
    /// - Cpu: forces RamResident or SsdStreaming (no GPU)
    pub fn open_with_device(gguf_path: &str, max_seq_len: usize, device_preference: DevicePreference) -> Result<Self> {
        let device = {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).unwrap_or_else(|e| {
                    log::warn!("CUDA not available ({}), falling back to CPU", e);
                    Device::Cpu
                })
            }
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0).unwrap_or_else(|e| {
                    log::warn!("Metal not available ({}), falling back to CPU", e);
                    Device::Cpu
                })
            }
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            {
                log::info!("No GPU feature enabled, using CPU");
                Device::Cpu
            }
        };

        log::info!("Opening GGUF: {}", gguf_path);
        let reader = GgufReader::open(gguf_path)
            .map_err(|e| candle_core::Error::Msg(format!("GGUF open: {}", e)))?;
        log::debug!("GGUF loaded: {} tensors", reader.tensors.len());

        let mut config = StreamingConfig::from_gguf(&reader)
            .map_err(|e| candle_core::Error::Msg(format!("Config: {}", e)))?;
        config.device_preference = device_preference;

        let is_hybrid = config.is_deltanet_hybrid();
        log::info!(
            "Model: {} layers={}, heads={}, kv_heads={}, experts={}, hidden={}, hybrid={}, norm_topk_prob={}",
            config.architecture,
            config.num_layers,
            config.num_attention_heads,
            config.num_kv_heads,
            config.num_experts,
            config.hidden_size,
            is_hybrid,
            config.norm_topk_prob,
        );
        if is_hybrid {
            log::info!(
                "DeltaNet config: ssm_d_state={}, ssm_n_group={}, ssm_dt_rank={}, ssm_d_inner={}, full_attn_interval={}",
                config.ssm_d_state,
                config.ssm_n_group,
                config.ssm_dt_rank,
                config.ssm_d_inner,
                config.full_attention_interval,
            );
        }

        if config.sliding_window > 0 {
            log::info!(
                "Sliding window attention: window={}, pattern=alternating (even=SWA, odd=full)",
                config.sliding_window,
            );
        }

        // Pre-compute RoPE tables (on device and CPU)
        // Use rotary_dim (not head_dim) for frequency computation.
        // For partial RoPE (80B: rotary_dim=64, head_dim=256), frequencies must be
        // computed as 1/theta^(2i/rotary_dim), not 1/theta^(2i/head_dim).
        let (cos, sin) = ops::attention::precompute_rope_tables(
            config.rotary_dim,
            max_seq_len,
            config.rope_theta as f64,
            &device,
        )?;
        let cos_cpu = cos.to_device(&Device::Cpu)?;
        let sin_cpu = sin.to_device(&Device::Cpu)?;

        // Load resident weights (embed + lm_head + final_norm)
        log::info!("Loading resident weights...");
        let mut embed_weight = load_weight(&reader, "token_embd.weight", &device)?;
        let mut lm_head_weight = load_weight(&reader, "output.weight", &device)?;
        let mut lm_head_bias = if reader.tensors.contains_key("output.bias") {
            Some(load_weight(&reader, "output.bias", &device)?)
        } else {
            None
        };
        let final_norm_weight = load_weight(&reader, "output_norm.weight", &device)?;

        let embed_mb = (config.vocab_size * config.hidden_size * 4) as f64 / 1e6;
        let head_mb = (config.vocab_size * config.hidden_size * 4) as f64 / 1e6;
        let norm_mb = (config.hidden_size * 4) as f64 / 1e6;
        log::info!(
            "Resident: embed={:.1}MB, lm_head={:.1}MB, norm={:.1}MB, total={:.1}MB",
            embed_mb, head_mb, norm_mb, embed_mb + head_mb + norm_mb,
        );

        let kv_cache = KvCache::new(config.num_layers);
        let epl: Vec<usize> = (0..config.num_layers)
            .map(|i| config.experts_for_layer(i))
            .collect();
        let resident = ResidentWeights::empty(config.num_layers, &epl);

        // Initialize DeltaNet state for hybrid models
        let deltanet_state = if config.is_deltanet_hybrid() {
            let dn_state = DeltaNetState::new(&config);
            let state_mb = (config.num_layers - config.num_layers / config.full_attention_interval)
                * config.ssm_dt_rank * config.ssm_head_v_dim() * config.ssm_head_v_dim() * 4 / 1024 / 1024;
            log::info!("DeltaNet state: ~{}MB for {} linear attention layers", state_mb,
                config.num_layers - config.num_layers / config.full_attention_interval);
            Some(dn_state)
        } else {
            None
        };

        // Auto-detect inference mode based on model size, system RAM, and GPU availability.
        // Three modes: GpuResident > RamResident > SsdStreaming (priority order).
        let total_experts: u64 = (0..config.num_layers)
            .map(|i| config.experts_for_layer(i) as u64)
            .sum();
        let expert_f32_bytes = total_experts
            * 3 // gate + up + down projections
            * config.moe_intermediate_size as u64
            * config.hidden_size as u64
            * 4; // F32 bytes per element
        // Resident weights: embed + lm_head + attention (Q,K,V,O per layer) + gates + norms
        let resident_f32_bytes = {
            let embed_lm = config.vocab_size as u64 * config.hidden_size as u64 * 4 * 2;
            let attn_per_layer = (config.num_attention_heads as u64 * config.head_dim as u64
                + config.num_kv_heads as u64 * config.head_dim as u64 * 2
                + config.hidden_size as u64) // O projection
                * config.hidden_size as u64 * 4;
            let gates = total_experts * config.hidden_size as u64 * 4;
            embed_lm + attn_per_layer * config.num_layers as u64 + gates
        };
        let total_f32_estimate = expert_f32_bytes + resident_f32_bytes;

        let file_size = reader.file_size() as u64;
        let has_metal = matches!(device, Device::Metal(_));
        let has_cuda = matches!(device, Device::Cuda(_));
        let has_gpu = has_metal || has_cuda;
        let system_ram = get_system_ram();
        // For CUDA, use GPU VRAM for threshold (not system RAM).
        // For Metal (unified memory), system RAM = GPU memory.
        #[cfg(feature = "cuda")]
        let gpu_mem: Option<u64> = if has_cuda {
            get_cuda_gpu_memory()
        } else {
            system_ram
        };
        #[cfg(not(feature = "cuda"))]
        let gpu_mem: Option<u64> = system_ram;

        // Determine inference mode.
        //
        // 3-tier auto-detection based on GGUF file size vs system RAM:
        //   <80%  → GPU Resident: all weights on Metal GPU
        //   80-90% → GPU+SSD Hybrid: attention/gate/norm on GPU, experts from SSD
        //   >90%  → CPU+SSD Streaming: everything on CPU, experts from SSD
        //
        // RamResident is only available via --device cpu override (not auto-selected).
        let inference_mode = match config.device_preference {
            DevicePreference::Gpu => {
                log::info!("Inference mode: GPU Resident (user override --device gpu)");
                InferenceMode::GpuResident
            }
            DevicePreference::Cpu => {
                // CPU override: pick RamResident if it fits, else SsdStreaming
                if let Some(ram) = system_ram {
                    let available = ram as f64 * 0.75;
                    if (total_f32_estimate as f64) <= available {
                        log::info!("Inference mode: RAM Resident (user override --device cpu, F32 fits in RAM)");
                        InferenceMode::RamResident
                    } else {
                        log::info!("Inference mode: SSD Streaming (user override --device cpu)");
                        InferenceMode::SsdStreaming
                    }
                } else {
                    log::info!("Inference mode: SSD Streaming (user override --device cpu, RAM unknown)");
                    InferenceMode::SsdStreaming
                }
            }
            DevicePreference::Auto => {
                if let Some(mem) = gpu_mem {
                    let mem_gb = mem as f64 / 1e9;
                    let file_gb = file_size as f64 / 1e9;
                    let ratio_pct = file_size as f64 / mem as f64 * 100.0;
                    let gpu_label = if has_cuda { "CUDA VRAM" } else { "RAM" };

                    if has_gpu && file_size < (mem as f64 * 0.80) as u64 {
                        // Model GGUF fits in <80% of GPU memory → all weights on GPU
                        log::info!(
                            "Inference mode: GPU Resident (GGUF {:.1}GB = {:.0}% of {:.1}GB {} < 80%)",
                            file_gb, ratio_pct, mem_gb, gpu_label,
                        );
                        InferenceMode::GpuResident
                    } else if has_gpu && file_size < (mem as f64 * 0.90) as u64 {
                        // Model 80-90% of GPU memory → attention on GPU, experts from SSD/RAM
                        log::info!(
                            "Inference mode: GPU+SSD Hybrid (GGUF {:.1}GB = {:.0}% of {:.1}GB {}, 80-90%)",
                            file_gb, ratio_pct, mem_gb, gpu_label,
                        );
                        InferenceMode::GpuHybrid
                    } else {
                        // Model >90% of GPU memory → full CPU+SSD streaming
                        log::info!(
                            "Inference mode: CPU+SSD Streaming (GGUF {:.1}GB = {:.0}% of {:.1}GB {} > 90%)",
                            file_gb, ratio_pct, mem_gb, gpu_label,
                        );
                        InferenceMode::SsdStreaming
                    }
                } else {
                    log::info!("Inference mode: SSD Streaming (could not detect GPU/system memory)");
                    InferenceMode::SsdStreaming
                }
            }
        };

        // Apply inference mode to backward-compatible config fields
        config.inference_mode = Some(inference_mode);
        match inference_mode {
            InferenceMode::GpuResident => {
                config.gpu_compute = true;
                config.ram_resident = false; // weights on GPU, not CPU RAM
                config.evict_experts = false;
            }
            InferenceMode::GpuHybrid => {
                // Attention/gate/norm on GPU, experts stream from SSD via CPU matmul.
                config.gpu_compute = true;  // attention + routing on Metal GPU
                config.ram_resident = false; // experts NOT in RAM, stream from SSD
                config.evict_experts = false; // let OS page cache manage SSD reads
                log::info!(
                    "  GPU+SSD Hybrid: attention/gate/norm on GPU, experts stream from SSD",
                );
            }
            InferenceMode::RamResident => {
                config.gpu_compute = false;
                config.ram_resident = true;
                config.evict_experts = false;
                log::info!(
                    "  RAM Resident: experts={:.1}GB + resident={:.1}GB = {:.1}GB",
                    expert_f32_bytes as f64 / 1e9, resident_f32_bytes as f64 / 1e9,
                    total_f32_estimate as f64 / 1e9,
                );
            }
            InferenceMode::SsdStreaming => {
                config.gpu_compute = false;
                config.ram_resident = false;
                // Smart eviction: disable when enough RAM headroom exists for page cache
                if let Some(ram) = system_ram {
                    let preload_estimate = resident_f32_bytes as f64 * 2.0;
                    let headroom = ram as f64 - preload_estimate;
                    config.evict_experts = headroom < 4e9;
                    log::info!(
                        "  Expert eviction: {} (preload~{:.1}GB, headroom={:.1}GB)",
                        if config.evict_experts { "ON" } else { "OFF" },
                        preload_estimate / 1e9, headroom / 1e9,
                    );
                }
            }
        }

        // GPU modes (Resident + Hybrid): convert embed/lm_head from F32 to F16
        // to halve GPU memory. Forward pass casts F16→F32 at point of use
        // (negligible cost for single tokens).
        let is_gpu_mode = inference_mode == InferenceMode::GpuResident
            || inference_mode == InferenceMode::GpuHybrid;
        if is_gpu_mode {
            let f32_mb = (config.vocab_size * config.hidden_size * 4 * 2) as f64 / 1e6;
            embed_weight = embed_weight.to_dtype(DType::F16)?;
            lm_head_weight = lm_head_weight.to_dtype(DType::F16)?;
            if let Some(ref bias) = lm_head_bias {
                lm_head_bias = Some(bias.to_dtype(DType::F16)?);
            }
            let f16_mb = (config.vocab_size * config.hidden_size * 2 * 2) as f64 / 1e6;
            log::info!(
                "{}: embed/lm_head F32→F16 ({:.0}MB → {:.0}MB, saved {:.0}MB)",
                inference_mode, f32_mb, f16_mb, f32_mb - f16_mb,
            );
        }

        // Expert LRU cache: disabled for SSD streaming mode.
        // F32 cached experts are ~7× larger than their Q4 source pages, so caching
        // displaces OS page cache and makes cache misses extremely expensive (SSD reads).
        // Rely on OS page cache for Q4 data + fast CPU dequant instead.
        // Cache is only useful in RAM Resident mode (all experts loaded as F32).
        let expert_cache = ExpertCache::with_layer_capacities(
            config.layer_tier_config.layer_capacities(config.num_layers),
        );

        let layer_output_cache = LayerOutputCache::empty(config.num_layers, config.adaptive_skip_threshold);
        let entropy_profiler = EntropyProfiler::new(config.num_layers);
        let routing_stats = RoutingStatsCollector::new(config.num_layers, config.num_experts);

        let chat_template = ChatTemplate::detect(&config.model_name, &config.architecture);
        log::info!("Chat template: {} (model_name={:?})", chat_template.name(), config.model_name);

        Ok(Self {
            reader,
            config,
            device,
            cos,
            sin,
            cos_cpu,
            sin_cpu,
            embed_weight,
            lm_head_weight,
            lm_head_bias,
            final_norm_weight,
            kv_cache,
            expert_cache,
            resident,
            deltanet_state,
            layer_output_cache,
            chat_template,
            max_layers: 0,
            entropy_profiler,
            routing_stats,
            profile_stats: if use_profile_layers() { Some(ProfileStats::new()) } else { None },
            last_step_timing: None,
            vq_codebooks: None,
            vq_config: None,
            vq_per_expert: false,
        })
    }

    /// Pre-load router gate weights into GPU memory.
    ///
    /// Loads only the router gate weights (~50MB for 30B model) which are
    /// needed every step but are small. This eliminates 48 GGUF loads per step.
    /// Also detects dummy (padded) experts by checking for uniform gate weight rows.
    /// Loads router gate biases if present (e.g. GPT-OSS).
    pub fn preload_gates(&mut self) -> Result<()> {
        let t0 = Instant::now();

        let mut gate_mb = 0.0f64;
        let mut total_dummies = 0usize;
        let mut bias_count = 0usize;
        for layer_idx in 0..self.config.num_layers {
            let gate_name = format!("blk.{}.ffn_gate_inp.weight", layer_idx);
            // Load to Metal for actual MoE routing
            let gate_w = load_weight(&self.reader, &gate_name, &self.device)?;
            // CPU copy for MoE routing (avoids Metal→CPU sync barrier)
            let gate_w_cpu = gate_w.to_device(&Device::Cpu)?;
            gate_mb += gate_w.elem_count() as f64 * 4.0 / 1e6;

            // Load router gate bias if present (e.g. GPT-OSS ffn_gate_inp.bias)
            let bias_name = format!("blk.{}.ffn_gate_inp.bias", layer_idx);
            if self.reader.tensors.contains_key(&bias_name) {
                let bias = load_weight(&self.reader, &bias_name, &Device::Cpu)?;
                self.resident.router_gate_biases[layer_idx] = Some(bias);
                bias_count += 1;
            }

            // Detect dummy experts: rows where all elements have the same value
            // (zero-weight dummies from padding, or uniform negative fill from fix scripts).
            // Real expert gate rows have varied learned weights with non-zero variance.
            let (num_experts, hidden_size) = gate_w_cpu.dims2()?;
            let gate_data = gate_w_cpu.flatten_all()?.to_vec1::<f32>()?;
            let mut dummies = Vec::new();
            for expert_idx in 0..num_experts {
                let row_start = expert_idx * hidden_size;
                let row_end = row_start + hidden_size;
                let first = gate_data[row_start];
                let is_uniform = gate_data[row_start..row_end]
                    .iter()
                    .all(|&v| (v - first).abs() < 1e-6);
                if is_uniform {
                    dummies.push(expert_idx);
                }
            }
            if !dummies.is_empty() {
                log::info!(
                    "  Layer {}: {} dummy experts detected (indices: {}..{})",
                    layer_idx, dummies.len(), dummies[0], dummies[dummies.len() - 1]
                );
                total_dummies += dummies.len();
            }
            self.resident.dummy_experts[layer_idx] = dummies;

            self.resident.router_gates[layer_idx] = Some(gate_w);
            self.resident.router_gates_cpu[layer_idx] = Some(gate_w_cpu);
        }

        if bias_count > 0 {
            log::info!("  Router gate biases loaded for {} layers", bias_count);
        }

        log::info!(
            "Preloaded gates: {:.1}MB (+ CPU copies) in {:.2}s, {} dummy experts masked",
            gate_mb, t0.elapsed().as_secs_f64(), total_dummies
        );
        Ok(())
    }

    /// Pre-load attention weights into memory.
    ///
    /// For GPU Resident mode, loads to Metal GPU device.
    /// For other modes, loads to CPU memory.
    /// For hybrid models, only loads attention layers (skips DeltaNet layers).
    pub fn preload_attention(&mut self) -> Result<()> {
        let t0 = Instant::now();
        let is_gpu_mode = self.config.inference_mode == Some(InferenceMode::GpuResident)
            || self.config.inference_mode == Some(InferenceMode::GpuHybrid);
        let load_device = if is_gpu_mode { &self.device } else { &Device::Cpu };
        let device_name = if is_gpu_mode { "GPU" } else { "CPU" };

        let mut attn_mb = 0.0f64;
        let mut count = 0usize;
        for layer_idx in 0..self.config.num_layers {
            // Skip DeltaNet layers (they don't have attention weights)
            if self.config.is_deltanet_hybrid() && !self.config.is_attention_layer(layer_idx) {
                continue;
            }
            let prefix = format!("blk.{}", layer_idx);

            let q_weight = load_weight(&self.reader, &format!("{}.attn_q.weight", prefix), load_device)?;
            let k_weight = load_weight(&self.reader, &format!("{}.attn_k.weight", prefix), load_device)?;
            let v_weight = load_weight(&self.reader, &format!("{}.attn_v.weight", prefix), load_device)?;
            let o_weight = load_weight(&self.reader, &format!("{}.attn_output.weight", prefix), load_device)?;

            for w in [&q_weight, &k_weight, &v_weight, &o_weight] {
                attn_mb += w.elem_count() as f64 * 4.0 / 1e6;
            }

            let q_norm_name = format!("{}.attn_q_norm.weight", prefix);
            let (q_norm, k_norm) = if self.reader.tensors.contains_key(&q_norm_name) {
                let qn = load_weight(&self.reader, &q_norm_name, load_device)?;
                let kn = load_weight(&self.reader, &format!("{}.attn_k_norm.weight", prefix), load_device)?;
                (Some(qn), Some(kn))
            } else {
                (None, None)
            };

            // Load attention biases if present (e.g. Qwen1.5-MoE, GPT-OSS)
            let q_bias_name = format!("{}.attn_q.bias", prefix);
            let (q_bias, k_bias, v_bias) = if self.reader.tensors.contains_key(&q_bias_name) {
                let qb = load_weight(&self.reader, &q_bias_name, load_device)?;
                let kb = load_weight(&self.reader, &format!("{}.attn_k.bias", prefix), load_device)?;
                let vb = load_weight(&self.reader, &format!("{}.attn_v.bias", prefix), load_device)?;
                for b in [&qb, &kb, &vb] {
                    attn_mb += b.elem_count() as f64 * 4.0 / 1e6;
                }
                (Some(qb), Some(kb), Some(vb))
            } else {
                (None, None, None)
            };

            // Load output projection bias if present (e.g. GPT-OSS)
            let o_bias_name = format!("{}.attn_output.bias", prefix);
            let o_bias = if self.reader.tensors.contains_key(&o_bias_name) {
                let ob = load_weight(&self.reader, &o_bias_name, load_device)?;
                attn_mb += ob.elem_count() as f64 * 4.0 / 1e6;
                Some(ob)
            } else {
                None
            };

            // Load attention sinks if present (GPT-OSS: per-head learned sink logit)
            let sinks_name = format!("{}.attn_sinks.weight", prefix);
            let attn_sinks = if self.reader.tensors.contains_key(&sinks_name) {
                let sk = load_weight(&self.reader, &sinks_name, load_device)?;
                attn_mb += sk.elem_count() as f64 * 4.0 / 1e6;
                if layer_idx == 0 {
                    log::info!("  Attention sinks found: {} elements per layer", sk.elem_count());
                }
                Some(sk)
            } else {
                None
            };

            // Upload MXFP4 attention weight buffers for native-quantized matmul on Metal GPU.
            // This avoids the 4x bandwidth overhead of F32 dequantized matmul.
            #[cfg(feature = "metal")]
            let mxfp4 = if is_gpu_mode {
                let q_name = format!("{}.attn_q.weight", prefix);
                let is_mxfp4 = self.reader.tensors.get(&q_name)
                    .map(|t| t.quant_type == GgmlQuantType::MXFP4)
                    .unwrap_or(false);
                if is_mxfp4 {
                    let upload = |name: &str| -> Result<Mxfp4Weight> {
                        let info = self.reader.tensors.get(name).ok_or_else(|| {
                            candle_core::Error::Msg(format!("tensor not found: {}", name))
                        })?;
                        let raw_bytes = self.reader.tensor_data(name)
                            .map_err(|e| candle_core::Error::Msg(format!("MXFP4 load {}: {}", name, e)))?;
                        let buffer = crate::metal::upload_mxfp4_weights(&self.device, raw_bytes)?;
                        // GGUF dimensions: [cols, rows] = [in_features, out_features]
                        let in_features = info.dimensions[0] as usize;
                        let out_features = info.dimensions[1] as usize;
                        Ok(Mxfp4Weight { buffer, out_features, in_features })
                    };
                    let q = upload(&format!("{}.attn_q.weight", prefix))?;
                    let k = upload(&format!("{}.attn_k.weight", prefix))?;
                    let v = upload(&format!("{}.attn_v.weight", prefix))?;
                    let o = upload(&format!("{}.attn_output.weight", prefix))?;
                    let mxfp4_mb = [&q, &k, &v, &o].iter()
                        .map(|w| {
                            let blocks = w.in_features / 32;
                            (w.out_features * blocks * 17) as f64 / 1e6
                        })
                        .sum::<f64>();
                    if layer_idx == 0 {
                        log::info!("  MXFP4 attention: {:.1}MB/layer (4.25-bit, ~4x bandwidth reduction)", mxfp4_mb);
                    }
                    Some(Mxfp4AttentionWeights { q, k, v, o })
                } else {
                    None
                }
            } else {
                None
            };

            // Upload Q5_0/Q8_0 attention weights as raw Metal buffers for GPU Resident mode.
            // Uses our custom Metal kernels for native quantized matvec (no F32 dequant).
            #[cfg(feature = "metal")]
            let quantized_metal = if is_gpu_mode {
                let q_name = format!("{}.attn_q.weight", prefix);
                let q_info = self.reader.tensors.get(&q_name);
                let quant_type = q_info.and_then(|t| match t.quant_type {
                    GgmlQuantType::Q5_0 => Some(crate::metal::QuantizedAttnType::Q5_0),
                    GgmlQuantType::Q8_0 => Some(crate::metal::QuantizedAttnType::Q8_0),
                    _ => None,
                });
                if let Some(qt) = quant_type {
                    let upload = |name: &str| -> Result<Option<QuantizedAttnWeight>> {
                        let info = self.reader.tensors.get(name).ok_or_else(|| {
                            candle_core::Error::Msg(format!("tensor not found: {}", name))
                        })?;
                        // Only upload tensors that are actually Q5_0 or Q8_0.
                        // Other types (Q5_K, Q4_K_M, etc.) have different block formats
                        // and MUST NOT be dispatched through our Q5_0/Q8_0 Metal kernels.
                        let tensor_qt = match info.quant_type {
                            GgmlQuantType::Q5_0 => crate::metal::QuantizedAttnType::Q5_0,
                            GgmlQuantType::Q8_0 => crate::metal::QuantizedAttnType::Q8_0,
                            _ => return Ok(None), // unsupported type, skip Metal path
                        };
                        let raw_bytes = self.reader.tensor_data(name)
                            .map_err(|e| candle_core::Error::Msg(format!("Quantized load {}: {}", name, e)))?;
                        let buffer = crate::metal::upload_mxfp4_weights(&self.device, raw_bytes)?;
                        // GGUF dimensions: [cols, rows] = [in_features, out_features]
                        let in_features = info.dimensions[0] as usize;
                        let out_features = info.dimensions[1] as usize;
                        Ok(Some(QuantizedAttnWeight { buffer, out_features, in_features, quant_type: tensor_qt }))
                    };
                    let q = upload(&format!("{}.attn_q.weight", prefix))?;
                    let k = upload(&format!("{}.attn_k.weight", prefix))?;
                    let v = upload(&format!("{}.attn_v.weight", prefix))?;
                    let o = upload(&format!("{}.attn_output.weight", prefix))?;
                    // Per-weight optional Metal quantized path.
                    // Supported types (Q5_0/Q8_0) use native Metal kernel, others fall
                    // through to F32 matmul in run_attention.
                    let any_present = q.is_some() || k.is_some() || v.is_some() || o.is_some();
                    if any_present {
                        if layer_idx == 0 {
                            let qmb: f64 = [&q, &k, &v, &o].iter()
                                .filter_map(|w| w.as_ref())
                                .map(|w| {
                                    let blocks = w.in_features / 32;
                                    (w.out_features * blocks * w.quant_type.block_size()) as f64 / 1e6
                                })
                                .sum();
                            let count = [&q, &k, &v, &o].iter().filter(|w| w.is_some()).count();
                            log::info!("  Quantized attention ({:?}): Metal kernel on GPU ({}/4 weights, {:.1}MB/layer)", qt, count, qmb);
                        }
                        Some(QuantizedAttnMetalWeights { q, k, v, o })
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            // Load quantized attention weights (QMatMul) for CPU mode only.
            // QMatMul is CPU-based and panics if input is on Metal GPU.
            // For GPU mode with mixed quant types (e.g. Q5_0 + Q5_K), the attention
            // code falls through to F32 matmul with the already-loaded dequanted weights.
            let quantized = if !is_gpu_mode {
                let q_name = format!("{}.attn_q.weight", prefix);
                let q_info = self.reader.tensors.get(&q_name);
                let is_quantized = q_info.map(|t| {
                    t.quant_type != GgmlQuantType::F32
                        && t.quant_type != GgmlQuantType::F16
                        && t.quant_type != GgmlQuantType::MXFP4
                        && t.quant_type != GgmlQuantType::BF16
                }).unwrap_or(false);
                if is_quantized {
                    let load_qmatmul = |name: &str| -> Result<candle_core::quantized::QMatMul> {
                        let qt = self.reader.tensor_as_qtensor(name, &Device::Cpu)
                            .map_err(|e| candle_core::Error::Msg(format!("QTensor load {}: {}", name, e)))?;
                        candle_core::quantized::QMatMul::from_qtensor(qt)
                    };
                    match (|| -> Result<QuantizedAttentionWeights> {
                        Ok(QuantizedAttentionWeights {
                            q: load_qmatmul(&format!("{}.attn_q.weight", prefix))?,
                            k: load_qmatmul(&format!("{}.attn_k.weight", prefix))?,
                            v: load_qmatmul(&format!("{}.attn_v.weight", prefix))?,
                            o: load_qmatmul(&format!("{}.attn_output.weight", prefix))?,
                        })
                    })() {
                        Ok(qaw) => {
                            if layer_idx == 0 {
                                let qt = q_info.unwrap().quant_type;
                                log::info!("  Quantized attention ({:?}): QMatMul on CPU", qt);
                            }
                            Some(qaw)
                        }
                        Err(e) => {
                            if layer_idx == 0 {
                                log::warn!("  Quantized attention load failed, using F32 fallback: {}", e);
                            }
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

            self.resident.attention[layer_idx] = Some(AttentionWeights {
                q_weight,
                k_weight,
                v_weight,
                o_weight,
                q_norm,
                k_norm,
                q_bias,
                k_bias,
                v_bias,
                o_bias,
                attn_sinks,
                #[cfg(feature = "metal")]
                mxfp4,
                #[cfg(feature = "metal")]
                quantized_metal,
                quantized,
            });
            count += 1;
        }

        log::info!("Preloaded attention ({}): {:.1}MB ({} layers) in {:.2}s", device_name, attn_mb, count, t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Pre-load DeltaNet weights into CPU memory (for hybrid models).
    ///
    /// Supports both Unsloth GGUF format (split attn_qkv + attn_gate) and
    /// llama.cpp GGUF format (single ssm_in with per-group interleaved QKVZ).
    /// De-interleaving happens once at preload time for zero runtime cost.
    /// ~4.6GB for 80B (36 DeltaNet layers).
    pub fn preload_deltanet(&mut self) -> Result<()> {
        if !self.config.is_deltanet_hybrid() {
            log::info!("Not a hybrid model, skipping DeltaNet preload");
            return Ok(());
        }

        let t0 = Instant::now();
        let cpu = &Device::Cpu;

        // Detect GGUF format from first DeltaNet layer
        let first_dn_layer = (0..self.config.num_layers)
            .find(|&i| !self.config.is_attention_layer(i))
            .unwrap_or(0);
        let probe_name = format!("blk.{}.attn_qkv.weight", first_dn_layer);
        let is_unsloth = self.reader.tensors.contains_key(&probe_name);
        if is_unsloth {
            log::info!("DeltaNet: detected Unsloth GGUF format (split attn_qkv + attn_gate)");
        } else {
            log::info!("DeltaNet: detected llama.cpp GGUF format (fused ssm_in), will de-interleave");
        }

        let mut dn_mb = 0.0f64;
        let mut count = 0usize;
        for layer_idx in 0..self.config.num_layers {
            if self.config.is_attention_layer(layer_idx) {
                continue; // Skip attention layers
            }
            let prefix = format!("blk.{}", layer_idx);

            // Load input projections (handles both formats)
            use crate::model::deltanet::load_deltanet_projections;
            let (attn_qkv, attn_gate, ssm_ba, _) =
                load_deltanet_projections(&self.reader, &prefix, &self.config, cpu)?;

            let ssm_a = load_weight(&self.reader, &format!("{}.ssm_a", prefix), cpu)?;
            let ssm_dt_bias = load_weight(&self.reader, &format!("{}.ssm_dt.bias", prefix), cpu)?;
            let ssm_conv1d = load_weight(&self.reader, &format!("{}.ssm_conv1d.weight", prefix), cpu)?;
            let ssm_norm = load_weight(&self.reader, &format!("{}.ssm_norm.weight", prefix), cpu)?;
            let ssm_out = load_weight(&self.reader, &format!("{}.ssm_out.weight", prefix), cpu)?;

            for w in [&attn_qkv, &attn_gate, &ssm_ba, &ssm_a, &ssm_dt_bias, &ssm_conv1d, &ssm_norm, &ssm_out] {
                dn_mb += w.elem_count() as f64 * 4.0 / 1e6;
            }

            self.resident.deltanet[layer_idx] = Some(DeltaNetWeights {
                attn_qkv,
                attn_gate,
                ssm_ba,
                ssm_a,
                ssm_dt_bias,
                ssm_conv1d,
                ssm_norm,
                ssm_out,
            });
            count += 1;
        }

        log::info!("Preloaded DeltaNet: {:.1}MB ({} layers) in {:.2}s", dn_mb, count, t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Pre-load all expert weights into CPU memory (RAM Resident mode).
    ///
    /// Dequantizes all experts (gate/up/down) for every layer, plus shared experts.
    /// For Qwen1.5-MoE-A2.7B (24 layers × 60 experts): ~7GB F32.
    pub fn preload_all_experts(&mut self) -> Result<()> {
        let t0 = Instant::now();
        let cpu = &Device::Cpu;

        let mut total_mb = 0.0f64;
        let mut expert_count = 0usize;

        for layer_idx in 0..self.config.num_layers {
            let prefix = format!("blk.{}", layer_idx);
            let gate_exps_name = format!("{}.ffn_gate_exps.weight", prefix);
            let up_exps_name = format!("{}.ffn_up_exps.weight", prefix);
            let down_exps_name = format!("{}.ffn_down_exps.weight", prefix);

            let layer_num_experts = self.config.experts_for_layer(layer_idx);
            for expert_idx in 0..layer_num_experts {
                let gw = load_expert(&self.reader, &gate_exps_name, expert_idx, cpu)?;
                let uw = load_expert(&self.reader, &up_exps_name, expert_idx, cpu)?;
                let dw = load_expert(&self.reader, &down_exps_name, expert_idx, cpu)?;

                for w in [&gw, &uw, &dw] {
                    total_mb += w.elem_count() as f64 * 4.0 / 1e6;
                }

                self.resident.experts[layer_idx][expert_idx] = Some(ExpertWeights {
                    gate: gw,
                    up: uw,
                    down: dw,
                });
                expert_count += 1;
            }

            // Shared expert
            if self.config.has_shared_expert {
                let shexp_gate_name = format!("{}.ffn_gate_shexp.weight", prefix);
                if self.reader.tensors.contains_key(&shexp_gate_name) {
                    let shexp_gate = load_weight(&self.reader, &shexp_gate_name, cpu)?;
                    let shexp_up = load_weight(&self.reader, &format!("{}.ffn_up_shexp.weight", prefix), cpu)?;
                    let shexp_down = load_weight(&self.reader, &format!("{}.ffn_down_shexp.weight", prefix), cpu)?;

                    for w in [&shexp_gate, &shexp_up, &shexp_down] {
                        total_mb += w.elem_count() as f64 * 4.0 / 1e6;
                    }

                    let shexp_gate_inp_name = format!("{}.ffn_gate_inp_shexp.weight", prefix);
                    let gate_inp = if self.reader.tensors.contains_key(&shexp_gate_inp_name) {
                        let w = load_weight(&self.reader, &shexp_gate_inp_name, cpu)?;
                        total_mb += w.elem_count() as f64 * 4.0 / 1e6;
                        Some(w)
                    } else {
                        None
                    };

                    self.resident.shared_experts[layer_idx] = Some(SharedExpertWeights {
                        gate: shexp_gate,
                        up: shexp_up,
                        down: shexp_down,
                        gate_inp,
                    });
                }
            }

            if (layer_idx + 1) % 8 == 0 || layer_idx == self.config.num_layers - 1 {
                log::info!(
                    "Preloading experts: layer {}/{} ({:.0}MB so far)",
                    layer_idx + 1, self.config.num_layers, total_mb,
                );
            }
        }

        log::info!(
            "Preloaded experts: {:.1}MB ({} layers x {} experts = {}) in {:.2}s",
            total_mb, self.config.num_layers, self.config.num_experts,
            expert_count, t0.elapsed().as_secs_f64(),
        );
        Ok(())
    }

    /// Pre-load all expert weights onto Metal GPU (GPU Resident mode).
    ///
    /// For each expert's gate/up/down projection:
    /// - Q4_K and other candle-supported quantized types: Load as QMatMul on Metal (native quantized matmul)
    /// - MXFP4 (type 39): Upload raw 4-bit data to Metal buffer, dispatch via custom Metal kernel
    /// - F16/F32: Load directly to Metal
    ///
    /// Stores results in `self.resident.gpu_experts[layer][expert]`.
    pub fn preload_experts_gpu(&mut self) -> Result<()> {
        let t0 = Instant::now();
        // Use the engine's own Metal device to avoid device mismatch errors.
        // Creating a separate Device::new_metal(0) produces a different device instance
        // that candle treats as incompatible for tensor operations.
        let gpu_device = self.device.clone();

        let mut total_mb = 0.0f64;
        let mut expert_count = 0usize;
        let mut quantized_count = 0usize;
        let mut mxfp4_metal_count = 0usize;

        for layer_idx in 0..self.config.num_layers {
            let prefix = format!("blk.{}", layer_idx);
            let gate_exps_name = format!("{}.ffn_gate_exps.weight", prefix);
            let up_exps_name = format!("{}.ffn_up_exps.weight", prefix);
            let down_exps_name = format!("{}.ffn_down_exps.weight", prefix);

            // Detect quant type from the gate tensor
            let quant_type = self.reader.tensors.get(&gate_exps_name)
                .map(|info| info.quant_type)
                .unwrap_or(GgmlQuantType::F32);
            let is_mxfp4 = quant_type == GgmlQuantType::MXFP4;

            let layer_num_experts = self.config.experts_for_layer(layer_idx);

            // For MXFP4 layers, collect all expert data and create packed buffers.
            if is_mxfp4 {
                #[cfg(feature = "metal")]
                {
                    // Phase 1: Collect raw MXFP4 data for all experts
                    let mut gate_datas: Vec<(&[u8], [usize; 2])> = Vec::with_capacity(layer_num_experts);
                    let mut up_datas: Vec<(&[u8], [usize; 2])> = Vec::with_capacity(layer_num_experts);
                    let mut down_datas: Vec<(&[u8], [usize; 2])> = Vec::with_capacity(layer_num_experts);

                    for expert_idx in 0..layer_num_experts {
                        let (gd, gs) = self.reader.expert_slice_data(&gate_exps_name, expert_idx)
                            .map_err(|e| candle_core::Error::Msg(format!("MXFP4 L{} E{} gate: {}", layer_idx, expert_idx, e)))?;
                        let (ud, us) = self.reader.expert_slice_data(&up_exps_name, expert_idx)
                            .map_err(|e| candle_core::Error::Msg(format!("MXFP4 L{} E{} up: {}", layer_idx, expert_idx, e)))?;
                        let (dd, ds) = self.reader.expert_slice_data(&down_exps_name, expert_idx)
                            .map_err(|e| candle_core::Error::Msg(format!("MXFP4 L{} E{} down: {}", layer_idx, expert_idx, e)))?;
                        gate_datas.push((gd, [gs[0], gs[1]]));
                        up_datas.push((ud, [us[0], us[1]]));
                        down_datas.push((dd, [ds[0], ds[1]]));
                    }

                    // Phase 2: Pack into contiguous buffers with offset tracking
                    fn pack_expert_data(
                        datas: &[(&[u8], [usize; 2])],
                        gpu_device: &Device,
                    ) -> Result<(std::sync::Arc<metal::Buffer>, Vec<u64>, usize, usize)> {
                        let total_size: usize = datas.iter().map(|(d, _)| d.len()).sum();
                        let mut packed = Vec::with_capacity(total_size);
                        let mut offsets = Vec::with_capacity(datas.len());
                        let out_features = datas[0].1[0];
                        let in_features = datas[0].1[1];

                        for (data, _shape) in datas {
                            offsets.push(packed.len() as u64);
                            packed.extend_from_slice(data);
                        }

                        let buffer = crate::metal::upload_mxfp4_weights(gpu_device, &packed)?;
                        Ok((buffer, offsets, out_features, in_features))
                    }

                    let (gate_buf, gate_offsets, gate_out, gate_in) = pack_expert_data(&gate_datas, &gpu_device)?;
                    let (up_buf, up_offsets, _up_out, _up_in) = pack_expert_data(&up_datas, &gpu_device)?;
                    let (down_buf, down_offsets, down_out, down_in) = pack_expert_data(&down_datas, &gpu_device)?;

                    let layer_bytes: usize = gate_datas.iter().chain(up_datas.iter()).chain(down_datas.iter())
                        .map(|(d, _)| d.len()).sum();
                    total_mb += layer_bytes as f64 / 1e6;

                    // Phase 3: Store packed buffers for batched dispatch
                    self.resident.packed_mxfp4[layer_idx] = Some(PackedMxfp4Layer {
                        gate: PackedMxfp4Experts {
                            buffer: gate_buf.clone(), offsets: gate_offsets.clone(),
                            out_features: gate_out, in_features: gate_in,
                        },
                        up: PackedMxfp4Experts {
                            buffer: up_buf.clone(), offsets: up_offsets.clone(),
                            out_features: gate_out, in_features: gate_in,
                        },
                        down: PackedMxfp4Experts {
                            buffer: down_buf.clone(), offsets: down_offsets.clone(),
                            out_features: down_out, in_features: down_in,
                        },
                    });

                    // Phase 3b: Preload expert FFN biases to GPU (if present).
                    // Biases are small F32 tensors: [num_experts, intermediate_dim] for gate/up,
                    // [num_experts, hidden_dim] for down. Pack all experts contiguously.
                    let gate_bias_name = format!("{}.ffn_gate_exps.bias", prefix);
                    if self.reader.tensors.contains_key(&gate_bias_name) {
                        let up_bias_name = format!("{}.ffn_up_exps.bias", prefix);
                        let down_bias_name = format!("{}.ffn_down_exps.bias", prefix);

                        fn pack_bias_data(
                            reader: &crate::gguf::GgufReader,
                            tensor_name: &str,
                            num_experts: usize,
                            gpu_device: &Device,
                        ) -> Result<(std::sync::Arc<metal::Buffer>, usize, usize)> {
                            let mut all_data: Vec<f32> = Vec::new();
                            let mut dim = 0usize;
                            for expert_idx in 0..num_experts {
                                let (data, shape) = reader.dequantize_expert(tensor_name, expert_idx)
                                    .map_err(|e| candle_core::Error::Msg(
                                        format!("bias {}: {}", tensor_name, e)
                                    ))?;
                                dim = shape[shape.len() - 1]; // last dim is the feature dim
                                all_data.extend_from_slice(&data);
                            }
                            let metal_device = match gpu_device {
                                Device::Metal(m) => m,
                                _ => return Err(candle_core::Error::Msg("Expected Metal device".into())),
                            };
                            let buffer = metal_device.new_buffer_with_data(&all_data)?;
                            Ok((buffer, num_experts, dim))
                        }

                        let (gb_buf, gb_n, gb_dim) = pack_bias_data(&self.reader, &gate_bias_name, layer_num_experts, &gpu_device)?;
                        let (ub_buf, _ub_n, _ub_dim) = pack_bias_data(&self.reader, &up_bias_name, layer_num_experts, &gpu_device)?;
                        let (db_buf, db_n, db_dim) = pack_bias_data(&self.reader, &down_bias_name, layer_num_experts, &gpu_device)?;

                        self.resident.expert_bias_buffers[layer_idx] = Some(crate::metal::ExpertBiasBuffers {
                            gate: crate::metal::ExpertBiasBuffer { buffer: gb_buf, total_experts: gb_n, dim: gb_dim },
                            up: crate::metal::ExpertBiasBuffer { buffer: ub_buf, total_experts: gb_n, dim: gb_dim },
                            down: crate::metal::ExpertBiasBuffer { buffer: db_buf, total_experts: db_n, dim: db_dim },
                        });

                        log::debug!("Preloaded expert biases for layer {} ({} experts, gate_dim={}, down_dim={})",
                            layer_idx, layer_num_experts, gb_dim, db_dim);
                    }

                    // Phase 4: Also store per-expert entries (for fallback/multi-token path)
                    // These reference the packed buffer via Mxfp4Packed variant (zero extra memory).
                    for expert_idx in 0..layer_num_experts {
                        self.resident.gpu_experts[layer_idx][expert_idx] = Some(GpuExpertWeights {
                            gate: GpuExpertProjection::Mxfp4Packed(
                                gate_buf.clone(), gate_offsets[expert_idx],
                                gate_out, gate_in, gpu_device.clone(),
                            ),
                            up: GpuExpertProjection::Mxfp4Packed(
                                up_buf.clone(), up_offsets[expert_idx],
                                gate_out, gate_in, gpu_device.clone(),
                            ),
                            down: GpuExpertProjection::Mxfp4Packed(
                                down_buf.clone(), down_offsets[expert_idx],
                                down_out, down_in, gpu_device.clone(),
                            ),
                        });
                        expert_count += 1;
                        mxfp4_metal_count += 1;
                    }

                    // Phase 5: Create GPU routing offset tables for zero-sync MoE routing.
                    // Pre-uploads expert byte offsets to GPU so softmax_topk kernel can
                    // look up selected experts without CPU round-trip.
                    if let Some(packed) = &self.resident.packed_mxfp4[layer_idx] {
                        match crate::metal::create_gpu_routing_offsets(
                            &gpu_device,
                            packed,
                            &self.resident.dummy_experts[layer_idx],
                            layer_num_experts,
                        ) {
                            Ok(offsets) => {
                                self.resident.gpu_routing_offsets[layer_idx] = Some(offsets);
                                log::debug!("Created GPU routing offsets for layer {}", layer_idx);
                            }
                            Err(e) => {
                                log::warn!("Failed to create GPU routing offsets for layer {}: {}", layer_idx, e);
                            }
                        }
                    }
                }
                #[cfg(not(feature = "metal"))]
                {
                    for expert_idx in 0..layer_num_experts {
                        // Non-Metal fallback: dequantize MXFP4 to F32 -> F16 -> Dense tensor
                        let gate_f16 = self.dequant_expert_to_f16(&gate_exps_name, expert_idx, &gpu_device)?;
                        let up_f16 = self.dequant_expert_to_f16(&up_exps_name, expert_idx, &gpu_device)?;
                        let down_f16 = self.dequant_expert_to_f16(&down_exps_name, expert_idx, &gpu_device)?;

                        for w in [&gate_f16, &up_f16, &down_f16] {
                            total_mb += w.elem_count() as f64 * 2.0 / 1e6;
                        }
                        mxfp4_metal_count += 1;

                        self.resident.gpu_experts[layer_idx][expert_idx] = Some(GpuExpertWeights {
                            gate: GpuExpertProjection::Dense(gate_f16),
                            up: GpuExpertProjection::Dense(up_f16),
                            down: GpuExpertProjection::Dense(down_f16),
                        });
                        expert_count += 1;
                    }
                }
            } else {
                // Q4_K and other candle-supported types: load as QMatMul on Metal
                for expert_idx in 0..layer_num_experts {
                    let gate_q = self.reader.expert_slice_as_qtensor(&gate_exps_name, expert_idx, &gpu_device)
                        .map_err(|e| candle_core::Error::Msg(format!("GPU expert gate L{} E{}: {}", layer_idx, expert_idx, e)))?;
                    let up_q = self.reader.expert_slice_as_qtensor(&up_exps_name, expert_idx, &gpu_device)
                        .map_err(|e| candle_core::Error::Msg(format!("GPU expert up L{} E{}: {}", layer_idx, expert_idx, e)))?;
                    let down_q = self.reader.expert_slice_as_qtensor(&down_exps_name, expert_idx, &gpu_device)
                        .map_err(|e| candle_core::Error::Msg(format!("GPU expert down L{} E{}: {}", layer_idx, expert_idx, e)))?;

                    let gate_qmm = candle_core::quantized::QMatMul::from_qtensor(gate_q)?;
                    let up_qmm = candle_core::quantized::QMatMul::from_qtensor(up_q)?;
                    let down_qmm = candle_core::quantized::QMatMul::from_qtensor(down_q)?;

                    // Estimate size from GGUF raw bytes
                    if let Some(info) = self.reader.tensors.get(&gate_exps_name) {
                        let expert_elements = self.config.moe_intermediate_size * self.config.hidden_size;
                        let bytes_per_expert = info.quant_type.raw_size(expert_elements);
                        total_mb += bytes_per_expert as f64 * 3.0 / 1e6; // gate + up + down
                    }
                    quantized_count += 1;

                    let gpu_weights = GpuExpertWeights {
                        gate: GpuExpertProjection::Quantized(gate_qmm),
                        up: GpuExpertProjection::Quantized(up_qmm),
                        down: GpuExpertProjection::Quantized(down_qmm),
                    };

                    self.resident.gpu_experts[layer_idx][expert_idx] = Some(gpu_weights);
                    expert_count += 1;
                }
            }

            if (layer_idx + 1) % 4 == 0 || layer_idx == self.config.num_layers - 1 {
                log::info!(
                    "GPU preloading experts: layer {}/{} ({:.0}MB so far, {} experts)",
                    layer_idx + 1, self.config.num_layers, total_mb, expert_count,
                );
            }
        }

        // Allocate pre-allocated intermediate buffers for batched MoE dispatch.
        // Shared across all layers (dimensions must be uniform across layers for this to work).
        #[cfg(feature = "metal")]
        if mxfp4_metal_count > 0 {
            if let Some(packed) = &self.resident.packed_mxfp4[0] {
                let metal_device = match &gpu_device {
                    Device::Metal(m) => m,
                    _ => unreachable!(),
                };
                match crate::metal::BatchedMoeBuffers::new(
                    metal_device,
                    packed.gate.out_features,
                    packed.gate.in_features,
                ) {
                    Ok(bufs) => {
                        let buf_mb = (
                            crate::metal::MAX_BATCH_EXPERTS * packed.gate.out_features * 3  // gate + up + swiglu
                            + crate::metal::MAX_BATCH_EXPERTS * packed.down.out_features     // down
                            + packed.down.out_features                                       // final
                        ) as f64 * 4.0 / 1e6;
                        log::info!("Allocated batched MoE buffers: {:.1}MB (max {} experts)", buf_mb, crate::metal::MAX_BATCH_EXPERTS);
                        self.resident.batched_moe_buffers = Some(bufs);
                    }
                    Err(e) => {
                        log::warn!("Failed to allocate batched MoE buffers, using per-call allocation: {}", e);
                    }
                }
            }

            // Allocate reusable GPU routing output buffers (shared across all layers).
            // These hold softmax_topk kernel output: expert_indices, routing_weights, offsets.
            match crate::metal::create_gpu_routing_output_buffers(
                &gpu_device,
                self.config.num_experts_per_tok,
            ) {
                Ok(bufs) => {
                    log::info!("Allocated GPU routing output buffers (top_k={})", self.config.num_experts_per_tok);
                    self.resident.gpu_routing_out = Some(bufs);
                }
                Err(e) => {
                    log::warn!("Failed to allocate GPU routing output buffers: {}", e);
                }
            }
        }

        log::info!(
            "Preloaded GPU experts: {:.1}MB ({} experts, {} quantized, {} MXFP4 Metal) in {:.2}s",
            total_mb, expert_count, quantized_count, mxfp4_metal_count, t0.elapsed().as_secs_f64(),
        );
        Ok(())
    }

    /// Dequantize an expert slice to F32, convert to F16, and load to GPU device.
    /// Retained as fallback for non-Metal builds or debugging.
    #[allow(dead_code)]
    fn dequant_expert_to_f16(
        &self,
        tensor_name: &str,
        expert_idx: usize,
        gpu_device: &Device,
    ) -> Result<Tensor> {
        let (f32_data, shape) = self.reader.dequantize_expert(tensor_name, expert_idx)
            .map_err(|e| candle_core::Error::Msg(format!("dequant expert {}: {}", tensor_name, e)))?;

        // Convert F32 → F16
        let f16_data: Vec<half::f16> = f32_data.iter().map(|&v| half::f16::from_f32(v)).collect();

        // Create tensor on GPU device
        Tensor::from_vec(f16_data, shape.as_slice(), gpu_device)
    }

    /// Pre-load norm weights for all layers (~0.8MB total, saves 96 GGUF loads/token).
    ///
    /// Each layer has input_norm (attn_norm) and post_norm (post_attention_norm or ffn_norm).
    /// Tiny weights (~8KB each) but loaded 48×2=96 times per token from GGUF without this.
    /// Preloaded to self.device since rms_norm runs on the same device as hidden_states.
    pub fn preload_norms(&mut self) -> Result<()> {
        let t0 = Instant::now();
        let dev = &self.device;
        let mut norm_mb = 0.0f64;

        for layer_idx in 0..self.config.num_layers {
            let prefix = format!("blk.{}", layer_idx);
            let input_norm = load_weight(&self.reader, &format!("{}.attn_norm.weight", prefix), dev)?;
            let post_norm = load_weight(&self.reader, &format!("{}.post_attention_norm.weight", prefix), dev)
                .or_else(|_| load_weight(&self.reader, &format!("{}.ffn_norm.weight", prefix), dev))?;

            norm_mb += input_norm.elem_count() as f64 * 4.0 / 1e6;
            norm_mb += post_norm.elem_count() as f64 * 4.0 / 1e6;

            self.resident.norms[layer_idx] = Some(NormWeights { input_norm, post_norm });
        }

        log::info!("Preloaded norms: {:.1}MB ({} layers) in {:.2}s", norm_mb, self.config.num_layers, t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Pre-load shared expert weights for all layers (~900MB for 80B).
    ///
    /// Shared experts are computed every token at every layer. Without preloading,
    /// each token requires 3×48=144 GGUF dequantizations for shared experts alone.
    pub fn preload_shared_experts(&mut self) -> Result<()> {
        if !self.config.has_shared_expert {
            return Ok(());
        }

        let t0 = Instant::now();
        let cpu = &Device::Cpu;
        let mut total_mb = 0.0f64;
        let mut count = 0usize;

        for layer_idx in 0..self.config.num_layers {
            let prefix = format!("blk.{}", layer_idx);
            let shexp_gate_name = format!("{}.ffn_gate_shexp.weight", prefix);

            if !self.reader.tensors.contains_key(&shexp_gate_name) {
                continue;
            }

            let shexp_gate = load_weight(&self.reader, &shexp_gate_name, cpu)?;
            let shexp_up = load_weight(&self.reader, &format!("{}.ffn_up_shexp.weight", prefix), cpu)?;
            let shexp_down = load_weight(&self.reader, &format!("{}.ffn_down_shexp.weight", prefix), cpu)?;

            for w in [&shexp_gate, &shexp_up, &shexp_down] {
                total_mb += w.elem_count() as f64 * 4.0 / 1e6;
            }

            let shexp_gate_inp_name = format!("{}.ffn_gate_inp_shexp.weight", prefix);
            let gate_inp = if self.reader.tensors.contains_key(&shexp_gate_inp_name) {
                let w = load_weight(&self.reader, &shexp_gate_inp_name, cpu)?;
                total_mb += w.elem_count() as f64 * 4.0 / 1e6;
                Some(w)
            } else {
                None
            };

            self.resident.shared_experts[layer_idx] = Some(SharedExpertWeights {
                gate: shexp_gate,
                up: shexp_up,
                down: shexp_down,
                gate_inp,
            });
            count += 1;
        }

        log::info!("Preloaded shared experts: {:.1}MB ({} layers) in {:.2}s", total_mb, count, t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Calculate the optimal mlock budget based on system RAM.
    ///
    /// Formula: budget = system_ram × 15%
    ///
    /// Empirically determined on M4 Pro 24GB with Qwen3-80B v7 (27.7GB):
    /// - 4 GB (17%) → +15% speed, page cache ~10.9 GB (sufficient)
    /// - 5 GB (21%) → -14%, page cache starts to starve
    /// - 6 GB (25%) → -56%, severe page cache starvation
    ///
    /// 15% is conservative and accounts for:
    /// - Non-expert resident weights (~9 GB for 80B DeltaNet hybrid)
    /// - OS overhead (~2 GB)
    /// - Sufficient page cache for remaining SSD-streamed layers (~70% of RAM)
    pub fn compute_auto_budget(&self) -> Option<f32> {
        const MLOCK_RATIO: f64 = 0.15; // 15% of total system RAM
        const MIN_BUDGET: f32 = 1.0; // at least 1 GB

        let system_ram = get_system_ram()? as f64;
        let budget = (system_ram * MLOCK_RATIO / 1e9) as f32;
        Some(budget.max(MIN_BUDGET))
    }

    /// Pin (mlock) MoE expert tensor pages in Q4 format within the given RAM budget.
    /// Pages are pinned in the mmap — no dequantization, no extra memory allocation.
    /// The OS cannot evict pinned pages, guaranteeing zero page-fault I/O for these layers.
    /// Layers are pinned starting from layer 0 until the budget is exhausted.
    pub fn pin_experts_by_budget(&self, budget_gb: f32) -> Result<()> {
        let t0 = std::time::Instant::now();
        let budget_bytes = (budget_gb * 1e9) as u64;
        let mut used_bytes: u64 = 0;
        let mut pinned_layers = 0usize;

        for layer_idx in 0..self.config.num_layers {
            let prefix = format!("blk.{}", layer_idx);
            let gate_name = format!("{}.ffn_gate_exps.weight", prefix);
            let up_name = format!("{}.ffn_up_exps.weight", prefix);
            let down_name = format!("{}.ffn_down_exps.weight", prefix);

            // Calculate Q4 size for this layer's expert tensors
            let mut layer_bytes: u64 = 0;
            for tensor_name in [&gate_name, &up_name, &down_name] {
                if let Some(info) = self.reader.tensors.get(tensor_name.as_str()) {
                    layer_bytes += info.raw_size() as u64;
                }
            }

            if layer_bytes == 0 {
                continue; // non-MoE layer (e.g. dense FFN)
            }

            if used_bytes + layer_bytes > budget_bytes {
                break;
            }

            // mlock the Q4 pages for all 3 expert tensors
            for tensor_name in [&gate_name, &up_name, &down_name] {
                if self.reader.tensors.contains_key(tensor_name.as_str()) {
                    self.reader.mlock_tensor(tensor_name)
                        .map_err(|e| candle_core::Error::Msg(format!("mlock {}: {}", tensor_name, e)))?;
                }
            }

            used_bytes += layer_bytes;
            pinned_layers += 1;
        }

        let elapsed = t0.elapsed();
        eprintln!(
            "  mlock pinned {} / {} MoE layers ({:.2} GB Q4) in {:.1}ms",
            pinned_layers, self.config.num_layers,
            used_bytes as f64 / 1e9,
            elapsed.as_secs_f64() * 1000.0,
        );

        Ok(())
    }

    /// Pre-load all weights (gates + norms + shared experts + attention + DeltaNet + experts).
    ///
    /// Dispatches expert preloading based on inference mode:
    /// - GpuResident: preload_experts_gpu() (QMatMul/F16 on Metal)
    /// - RamResident: preload_all_experts() (F32 on CPU)
    /// - SsdStreaming: no expert preload (on-demand from GGUF)
    pub fn preload_weights(&mut self) -> Result<()> {
        let is_gpu_resident = self.config.inference_mode == Some(InferenceMode::GpuResident);

        self.preload_gates()?;
        self.preload_norms()?;
        self.preload_shared_experts()?;
        if self.config.is_deltanet_hybrid() {
            // Estimate DeltaNet F32 memory: ~470MB/layer for Qwen3.5 (d_inner=8192)
            let dn_layers = (0..self.config.num_layers)
                .filter(|&i| !self.config.is_attention_layer(i))
                .count();
            let dn_est_mb = dn_layers as f64
                * (self.config.ssm_conv_dim() * self.config.hidden_size  // attn_qkv
                 + self.config.ssm_d_inner * self.config.hidden_size     // attn_gate
                 + self.config.hidden_size * self.config.ssm_d_inner     // ssm_out
                 + 2 * self.config.ssm_dt_rank * self.config.hidden_size // ssm_ba
                ) as f64 * 4.0 / 1e6;
            let ram_mb = get_system_ram().unwrap_or(24_000_000_000) as f64 / 1e6;

            if dn_est_mb > ram_mb * 0.5 {
                log::info!(
                    "DeltaNet streaming mode: estimated {:.0}MB > 50% of {:.0}MB RAM, skipping preload",
                    dn_est_mb, ram_mb
                );
                // DeltaNet weights will be loaded from mmap on demand (SSD streaming)
            } else {
                self.preload_deltanet()?;
            }
            self.preload_attention()?;
        } else {
            self.preload_attention()?;
        }

        // Expert preloading:
        // - GpuResident: all experts to Metal GPU
        // - GpuHybrid: experts stream from SSD (no preload, same as SsdStreaming)
        // - RamResident: all experts dequantized to F32 in CPU RAM
        // - SsdStreaming: no expert preload, mlock budget for hot experts
        if is_gpu_resident {
            self.preload_experts_gpu()?;
        } else if self.config.ram_resident {
            self.preload_all_experts()?;
        } else if let Some(budget) = self.config.ram_budget_gb {
            if budget < 0.0 {
                // Auto mode: compute optimal budget from system RAM
                if let Some(auto_budget) = self.compute_auto_budget() {
                    eprintln!("  auto ram-budget: {:.1} GB", auto_budget);
                    self.pin_experts_by_budget(auto_budget)?;
                } else {
                    eprintln!("  auto ram-budget: could not determine system RAM, skipping mlock");
                }
            } else {
                self.pin_experts_by_budget(budget)?;
            }
        }

        // VQ codebook pre-loading (~4.5 MB total, instant).
        // Codebooks are loaded once and kept in memory for the entire session.
        if self.reader.is_vq_model() {
            self.preload_vq_codebooks()?;
        }

        Ok(())
    }

    /// Pre-load VQ codebooks from GGUF into memory.
    ///
    /// For shared VQ: reads all blk.{L}.ffn_{gate,up,down}_vq_cb.weight tensors (F16)
    /// and dequantizes to F32. Total ~4.5 MB for 48 layers × 3 projections × 32 KB.
    ///
    /// For per-expert VQ: codebooks are too large to pre-load (~1+ GB), so we only
    /// set the config and read codebooks on-demand during inference via expert_slice_data().
    fn preload_vq_codebooks(&mut self) -> Result<()> {
        let block_h = self.reader.vq_block_h();
        let block_w = self.reader.vq_block_w();
        let k = self.reader.vq_k();
        let block_dim = block_h * block_w;
        let per_expert = self.reader.is_vq_per_expert();

        log::info!("VQ model detected: block={}x{}, K={}, per_expert={}",
                   block_h, block_w, k, per_expert);

        let vq_cfg = VqConfig { block_h, block_w, k, block_dim };
        self.vq_per_expert = per_expert;

        if per_expert {
            // Per-expert mode: codebooks read on-demand in load_expert_vq
            log::info!("VQ per-expert mode: codebooks will be read on-demand");
            self.vq_codebooks = None;
        } else {
            // Shared mode: pre-load all codebooks
            let num_layers = self.config.num_layers;
            let mut codebooks = Vec::with_capacity(num_layers);

            for layer in 0..num_layers {
                let mut layer_cbs: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
                for (proj_idx, proj_name) in ["gate", "up", "down"].iter().enumerate() {
                    let tensor_name = format!("blk.{}.ffn_{}_vq_cb.weight", layer, proj_name);
                    if self.reader.tensors.contains_key(&tensor_name) {
                        let (data, _shape) = self.reader.dequantize_tensor(&tensor_name)
                            .map_err(|e| candle_core::Error::Msg(
                                format!("VQ codebook load {}: {}", tensor_name, e)))?;
                        layer_cbs[proj_idx] = data;
                    }
                }
                codebooks.push(layer_cbs);
            }

            let total_bytes: usize = codebooks.iter()
                .flat_map(|cbs| cbs.iter())
                .map(|cb| cb.len() * 4)
                .sum();
            log::info!("VQ codebooks loaded: {:.1} MB", total_bytes as f64 / 1e6);

            self.vq_codebooks = Some(codebooks);
        }

        self.vq_config = Some(vq_cfg);
        Ok(())
    }

    /// Run a GPU warmup pass to trigger Metal shader JIT compilation.
    ///
    /// Metal compiles compute shaders lazily on first use for each kernel type
    /// and tensor shape combination. This causes the first few inference steps
    /// to be 100-1000x slower than steady state.
    ///
    /// This method exercises the exact tensor shapes used in a real forward pass
    /// (attention projections, MoE expert matmuls, embedding, lm_head) to compile
    /// all Metal shaders upfront during initialization.
    ///
    /// Useful for GPU Resident and GPU+SSD Hybrid modes where Metal is the compute device.
    pub fn warmup_gpu(&mut self) -> Result<()> {
        let is_gpu_mode = self.config.inference_mode == Some(InferenceMode::GpuResident)
            || self.config.inference_mode == Some(InferenceMode::GpuHybrid);
        if !is_gpu_mode {
            return Ok(());
        }

        let t0 = Instant::now();
        log::info!("GPU warmup: compiling Metal shaders...");

        let dev = &self.device;
        let h = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let intermediate = self.config.moe_intermediate_size;

        // === Attention projection shapes ===
        // Exercise both prefill (seq=2) and decode (seq=1) matmul shapes using
        // the actual preloaded attention weights to match exact dtypes and dimensions.
        for seq_len in [1usize, 2] {
            if let Some(attn_w) = &self.resident.attention[0] {
                let dtype = attn_w.q_weight.dtype();
                let flat = Tensor::zeros((seq_len, h), dtype, dev)?;
                let _ = flat.matmul(&attn_w.q_weight.t()?)?;
                let _ = flat.matmul(&attn_w.k_weight.t()?)?;
                let _ = flat.matmul(&attn_w.v_weight.t()?)?;
                let attn_out = Tensor::zeros((seq_len, num_heads * head_dim), dtype, dev)?;
                let _ = attn_out.matmul(&attn_w.o_weight.t()?)?;
            }

            // QK^T: [batch, heads, seq_q, head_dim] x [batch, heads, head_dim, seq_k]
            for seq_k in [1usize, 2, 3] {
                let q = Tensor::zeros((1, num_heads, seq_len, head_dim), candle_core::DType::F32, dev)?;
                let kt = Tensor::zeros((1, num_heads, head_dim, seq_k), candle_core::DType::F32, dev)?;
                let scores = q.matmul(&kt)?;
                let attn_w = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
                let v = Tensor::zeros((1, num_heads, seq_k, head_dim), candle_core::DType::F32, dev)?;
                let _ = attn_w.matmul(&v)?;
            }
        }

        // === GQA expansion shapes ===
        if num_kv_heads < num_heads {
            let n_rep = num_heads / num_kv_heads;
            let kv = Tensor::zeros((1, num_kv_heads, 2, head_dim), candle_core::DType::F32, dev)?;
            let _ = kv.unsqueeze(2)?
                .expand((1, num_kv_heads, n_rep, 2, head_dim))?
                .reshape((1, num_heads, 2, head_dim))?
                .contiguous()?;
        }

        // === MoE expert shapes ===
        // Use actual GPU expert from layer 0 to compile the correct kernel type
        // (MXFP4 Metal, QMatMul, or Dense F16)
        if !self.resident.gpu_experts.is_empty() && !self.resident.gpu_experts[0].is_empty() {
            let token = Tensor::zeros((1, h), candle_core::DType::F32, dev)?;
            for expert_opt in self.resident.gpu_experts[0].iter() {
                if let Some(expert) = expert_opt {
                    let _ = expert.gate.forward(&token)?;
                    let _ = expert.up.forward(&token)?;
                    let mid = Tensor::zeros((1, intermediate), candle_core::DType::F32, dev)?;
                    let _ = expert.down.forward(&mid)?;
                    break; // One expert is enough to compile the shader
                }
            }
        }

        // === Embedding + LM head shapes ===
        let ids = Tensor::from_vec(vec![0u32, 1], (2,), dev)?;
        let emb = self.embed_weight.index_select(&ids, 0)?;
        // Cast to F32 if embed is F16 (GPU Resident), matching what forward() does
        let _emb_f32 = emb.to_dtype(candle_core::DType::F32)?;
        // lm_head: match actual dtype
        let lm_dtype = self.lm_head_weight.dtype();
        let lm_in = Tensor::zeros((1, h), lm_dtype, dev)?;
        let _ = lm_in.matmul(&self.lm_head_weight.t()?)?;

        // === Batched MXFP4 kernels (if packed experts available) ===
        // Compile all 5-8 Metal shader pipelines used by the batched dispatch path.
        // Without this, the first decode step incurs ~1.5s JIT compilation overhead.
        #[cfg(feature = "metal")]
        {
            if let Some(packed) = &self.resident.packed_mxfp4[0] {
                let bias_buffers = self.resident.expert_bias_buffers[0].as_ref();
                let dummy_input = Tensor::zeros((1, h), candle_core::DType::F32, dev)?;
                let dummy_experts = vec![
                    crate::metal::BatchedExpertInfo { expert_idx: 0, routing_weight: 1.0 },
                ];
                let _ = crate::metal::batched_moe_forward_metal(
                    dev, &dummy_input, packed, &dummy_experts,
                    self.config.architecture == "gpt-oss", 1.702, 7.0,
                    bias_buffers, None,
                );
                // Force GPU sync to complete the pipeline compilation
                let metal_device = match dev {
                    candle_core::Device::Metal(m) => Some(m),
                    _ => None,
                };
                if let Some(md) = metal_device {
                    let _ = md.wait_until_completed();
                }
            }
        }

        // === Element-wise ops (F32 path used throughout forward) ===
        let a = Tensor::zeros((1, h), candle_core::DType::F32, dev)?;
        let _ = a.exp()?;
        let _ = a.sin()?;
        let _ = a.cos()?;
        let _ = a.broadcast_add(&a)?;
        let _ = a.broadcast_mul(&a)?;
        let _ = a.sum_keepdim(candle_core::D::Minus1)?;

        // === Index add (MoE scatter) ===
        let idx = Tensor::from_vec(vec![0u32], (1,), dev)?;
        let val = Tensor::zeros((1, h), candle_core::DType::F32, dev)?;
        let base = Tensor::zeros((2, h), candle_core::DType::F32, dev)?;
        let _ = base.index_add(&idx, &val, 0)?;

        log::info!("GPU warmup: Metal shaders compiled in {:.2}s", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Issue madvise(WILLNEED) for a layer's weights to trigger async SSD readahead.
    /// Non-blocking: the OS starts fetching pages in the background.
    /// Skipped entirely in RAM Resident mode (all weights already in memory).
    #[allow(dead_code)]
    fn prefetch_layer_weights(&self, layer_idx: usize) {
        if layer_idx >= self.config.num_layers {
            return;
        }
        if self.config.ram_resident {
            return; // All weights resident, no SSD prefetch needed
        }
        let prefix = format!("blk.{}", layer_idx);
        let is_attn = self.config.is_attention_layer(layer_idx);

        // Norm weights (always needed, small — try both naming conventions)
        self.reader.prefetch_tensor(&format!("{}.attn_norm.weight", prefix));
        self.reader.prefetch_tensor(&format!("{}.post_attention_norm.weight", prefix));
        self.reader.prefetch_tensor(&format!("{}.ffn_norm.weight", prefix));

        if is_attn {
            // Attention weights (skip if resident)
            if self.resident.attention[layer_idx].is_none() {
                self.reader.prefetch_tensor(&format!("{}.attn_q.weight", prefix));
                self.reader.prefetch_tensor(&format!("{}.attn_k.weight", prefix));
                self.reader.prefetch_tensor(&format!("{}.attn_v.weight", prefix));
                self.reader.prefetch_tensor(&format!("{}.attn_output.weight", prefix));
                self.reader.prefetch_tensor(&format!("{}.attn_q_norm.weight", prefix));
                self.reader.prefetch_tensor(&format!("{}.attn_k_norm.weight", prefix));
                self.reader.prefetch_tensor(&format!("{}.attn_q.bias", prefix));
                self.reader.prefetch_tensor(&format!("{}.attn_k.bias", prefix));
                self.reader.prefetch_tensor(&format!("{}.attn_v.bias", prefix));
            }
        } else {
            // DeltaNet weights (skip if resident)
            if self.resident.deltanet[layer_idx].is_none() {
                // Support both Unsloth (attn_qkv + attn_gate) and llama.cpp (ssm_in) formats
                self.reader.prefetch_tensor(&format!("{}.attn_qkv.weight", prefix));
                self.reader.prefetch_tensor(&format!("{}.attn_gate.weight", prefix));
                self.reader.prefetch_tensor(&format!("{}.ssm_in.weight", prefix));
                self.reader.prefetch_tensor(&format!("{}.ssm_ba.weight", prefix));
                self.reader.prefetch_tensor(&format!("{}.ssm_alpha.weight", prefix));
                self.reader.prefetch_tensor(&format!("{}.ssm_beta.weight", prefix));
                self.reader.prefetch_tensor(&format!("{}.ssm_a", prefix));
                self.reader.prefetch_tensor(&format!("{}.ssm_dt.bias", prefix));
                self.reader.prefetch_tensor(&format!("{}.ssm_conv1d.weight", prefix));
                self.reader.prefetch_tensor(&format!("{}.ssm_norm.weight", prefix));
                self.reader.prefetch_tensor(&format!("{}.ssm_out.weight", prefix));
            }
        }

        // Shared expert weights (always needed for hybrid models)
        if self.config.has_shared_expert {
            self.reader.prefetch_tensor(&format!("{}.ffn_gate_shexp.weight", prefix));
            self.reader.prefetch_tensor(&format!("{}.ffn_up_shexp.weight", prefix));
            self.reader.prefetch_tensor(&format!("{}.ffn_down_shexp.weight", prefix));
            self.reader.prefetch_tensor(&format!("{}.ffn_gate_inp_shexp.weight", prefix));
        }
    }

    /// Full forward pass: input_ids → logits.
    ///
    /// input_ids shape: [batch, seq_len] (u32 token IDs)
    pub fn forward(&mut self, input_ids: &Tensor, use_cache: bool) -> Result<Tensor> {
        let (bsz, seq_len) = input_ids.dims2()?;
        let fwd_profile = std::env::var("FORWARD_PROFILE").is_ok();
        let t_fwd_start = Instant::now();

        // Embedding lookup: [batch, seq, hidden]
        // embed_weight may be F16 (GPU Resident) — cast to F32 for layer computation.
        let flat_ids = input_ids.flatten_all()?;
        let hidden_states = self.embed_weight.index_select(&flat_ids, 0)?;
        let hidden_states = if hidden_states.dtype() != DType::F32 {
            hidden_states.to_dtype(DType::F32)?
        } else {
            hidden_states
        };
        let mut hidden_states = hidden_states.reshape((bsz, seq_len, self.config.hidden_size))?;
        let embed_ms = if fwd_profile { t_fwd_start.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };

        // Layer loop (optionally limited for debugging)
        let num_layers = if self.max_layers > 0 {
            self.max_layers.min(self.config.num_layers)
        } else {
            self.config.num_layers
        };
        let profiling_enabled = self.entropy_profiler.enabled;
        let routing_stats_enabled = self.routing_stats.enabled;
        let layer_profiling = self.profile_stats.is_some();
        let top_k = self.config.num_experts_per_tok;
        let mut step_timing = if layer_profiling { Some(StepTimingStats::default()) } else { None };
        let mut layer_times_ms: Vec<f64> = if fwd_profile { Vec::with_capacity(num_layers) } else { Vec::new() };
        for layer_idx in 0..num_layers {
            let layer = LayerForward::new(
                &self.reader,
                &self.config,
                &self.device,
                &self.cos,
                &self.sin,
                &self.cos_cpu,
                &self.sin_cpu,
            ).with_vq(
                self.vq_codebooks.as_ref(),
                self.vq_config.as_ref(),
                self.vq_per_expert,
            );
            let mut entropy_out = if profiling_enabled { Some(0.0f32) } else { None };
            let mut routing_out: Option<(Vec<u32>, Vec<f32>)> = if routing_stats_enabled { Some((Vec::new(), Vec::new())) } else { None };
            let mut layer_timing: Option<LayerTiming> = if layer_profiling { Some(LayerTiming::default()) } else { None };
            let t_layer = Instant::now();
            hidden_states = layer.forward(
                &hidden_states,
                layer_idx,
                &mut self.kv_cache,
                use_cache,
                &mut self.expert_cache,
                &self.resident,
                self.deltanet_state.as_mut(),
                &mut self.layer_output_cache,
                &mut entropy_out,
                &mut routing_out,
                &mut layer_timing,
            )?;
            if fwd_profile {
                layer_times_ms.push(t_layer.elapsed().as_secs_f64() * 1000.0);
            }
            // Accumulate per-layer timing into step totals
            if let (Some(ref mut step), Some(lt)) = (&mut step_timing, layer_timing) {
                step.attention_ms += lt.attention_ms;
                step.norms_ms += lt.norms_ms;
                step.moe_routing_ms += lt.moe_routing_ms;
                step.moe_expert_io_ms += lt.moe_expert_io_ms;
                step.moe_expert_compute_ms += lt.moe_expert_compute_ms;
                step.moe_shared_expert_ms += lt.moe_shared_expert_ms;
            }
            if let Some(h) = entropy_out {
                self.entropy_profiler.record(layer_idx, h);
            }
            if let Some((indices, weights)) = routing_out {
                // indices/weights are flat: [num_tokens * K]
                // Record per-token chunks of size K
                let num_tokens = indices.len() / top_k;
                for t in 0..num_tokens {
                    let start = t * top_k;
                    let end = start + top_k;
                    self.routing_stats.record(
                        layer_idx,
                        &indices[start..end],
                        &weights[start..end],
                    );
                }
            }
        }

        let t_post = Instant::now();

        // Final norm
        hidden_states = ops::rms_norm(
            &hidden_states,
            &self.final_norm_weight,
            self.config.rms_norm_eps as f64,
        )?;

        // LM head: [batch, seq, hidden] → [batch, seq, vocab]
        // lm_head_weight may be F16 (GPU Resident) — match dtypes for matmul.
        let (bsz, seq_len, hidden_dim) = hidden_states.dims3()?;
        let flat = hidden_states.reshape((bsz * seq_len, hidden_dim))?;
        let flat = if flat.dtype() != self.lm_head_weight.dtype() {
            flat.to_dtype(self.lm_head_weight.dtype())?
        } else {
            flat
        };
        let mut logits = flat.matmul(&self.lm_head_weight.t()?)?;
        if let Some(ref bias) = self.lm_head_bias {
            logits = logits.broadcast_add(bias)?;
        }
        // Cast back to F32 for sampling/argmax
        let logits = if logits.dtype() != DType::F32 {
            logits.to_dtype(DType::F32)?
        } else {
            logits
        };
        let logits = logits.reshape((bsz, seq_len, self.config.vocab_size))?;

        // Forward pass profiling output
        if fwd_profile && seq_len == 1 {
            let post_ms = t_post.elapsed().as_secs_f64() * 1000.0;
            let total_ms = t_fwd_start.elapsed().as_secs_f64() * 1000.0;
            let layers_total: f64 = layer_times_ms.iter().sum();
            let layer_avg = layers_total / layer_times_ms.len() as f64;
            let layer_min = layer_times_ms.iter().copied().fold(f64::MAX, f64::min);
            let layer_max = layer_times_ms.iter().copied().fold(0.0f64, f64::max);
            eprintln!("[FWD_PROFILE] total={:.1}ms embed={:.1}ms layers={:.1}ms (avg={:.2}ms min={:.2}ms max={:.2}ms) post={:.1}ms (norm+lm_head)",
                total_ms, embed_ms, layers_total, layer_avg, layer_min, layer_max, post_ms);
        }

        // Store step timing for profiler (generate loop will collect it)
        self.last_step_timing = step_timing;

        Ok(logits)
    }

    /// Generate tokens autoregressively using greedy decoding.
    ///
    /// Returns generated token IDs (excluding the prompt).
    pub fn generate(&mut self, prompt_ids: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        self.kv_cache.clear();
        // Reset profile stats for this generation
        if let Some(ref mut ps) = self.profile_stats {
            *ps = ProfileStats::new();
        }

        let device = self.device.clone();
        let mut generated = Vec::new();

        // Prefill: process entire prompt at once
        let input = Tensor::from_vec(prompt_ids.to_vec(), (1, prompt_ids.len()), &device)?;
        let t0 = Instant::now();
        let logits = self.forward(&input, true)?;
        let prefill_time = t0.elapsed();

        // Record prefill time in profiler
        if let Some(ref mut ps) = self.profile_stats {
            ps.prefill_ms = prefill_time.as_secs_f64() * 1000.0;
        }

        // Get last token's logits → greedy argmax (pure greedy, no skipping)
        let last_logits_vec = logits.i((0, logits.dim(1)? - 1))?.to_vec1::<f32>()?;
        let mut next_token = {
            let mut indexed: Vec<(usize, f32)> = last_logits_vec.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            log::info!("Prefill top-5: {:?}", &indexed[..5]);
            indexed[0].0 as u32
        };
        generated.push(next_token);

        log::info!(
            "Prefill: {} tokens in {:.2}s ({:.1} tok/s), first generated: {}",
            prompt_ids.len(),
            prefill_time.as_secs_f64(),
            prompt_ids.len() as f64 / prefill_time.as_secs_f64(),
            next_token,
        );

        // Decode: one token at a time with KV-cache
        for step in 0..max_new_tokens - 1 {
            let input = Tensor::from_vec(vec![next_token], (1, 1), &device)?;
            let t0 = Instant::now();
            let logits = self.forward(&input, true)?;
            let step_time = t0.elapsed();

            // Collect decode step timing
            if let Some(mut timing) = self.last_step_timing.take() {
                timing.total_ms = step_time.as_secs_f64() * 1000.0;
                timing.other_ms = timing.total_ms - timing.attention_ms - timing.moe_routing_ms
                    - timing.moe_expert_io_ms - timing.moe_expert_compute_ms
                    - timing.moe_shared_expert_ms - timing.norms_ms;
                if let Some(ref mut ps) = self.profile_stats {
                    ps.steps.push(timing);
                }
            }

            // Pure greedy argmax
            let decode_logits_vec = logits.i((0, 0))?.to_vec1::<f32>()?;
            next_token = {
                let mut indexed: Vec<(usize, f32)> = decode_logits_vec.iter().cloned().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                if step < 5 {
                    eprintln!("[GEN] decode step {} top-5: {:?}", step + 2, &indexed[..5]);
                }
                indexed[0].0 as u32
            };

            generated.push(next_token);

            if step < 5 || (step + 2) % 10 == 0 {
                log::info!(
                    "Step {}: token={}, {:.2}s ({:.2} tok/s)",
                    step + 2,
                    next_token,
                    step_time.as_secs_f64(),
                    1.0 / step_time.as_secs_f64(),
                );
            }

            if self.chat_template.is_eos(next_token) {
                log::info!("EOS token generated at step {}", step + 2);
                break;
            }
        }

        let (hits, misses) = self.expert_cache.stats();
        log::info!(
            "Expert cache: hits={}, misses={}, hit_rate={:.1}%",
            hits, misses, self.expert_cache.hit_rate() * 100.0,
        );

        if self.config.adaptive_skip_enabled {
            log::info!(
                "Adaptive Skip: {:.1}% of layers skipped ({}/{})",
                self.layer_output_cache.skip_rate(),
                self.layer_output_cache.skip_count,
                self.layer_output_cache.total_count,
            );
        }

        // Print profiling summary
        if let Some(ref ps) = self.profile_stats {
            ps.print_summary();
        }

        Ok(generated)
    }

    /// Generate tokens with a per-token callback (for streaming output).
    ///
    /// The callback receives each generated token ID. Return `false` to stop early.
    pub fn generate_streaming<F>(&mut self, prompt_ids: &[u32], max_new_tokens: usize, mut on_token: F) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> bool,
    {
        self.kv_cache.clear();
        if let Some(ref mut ps) = self.profile_stats {
            *ps = ProfileStats::new();
        }

        let device = self.device.clone();
        let mut generated = Vec::new();

        // Prefill
        let input = Tensor::from_vec(prompt_ids.to_vec(), (1, prompt_ids.len()), &device)?;
        let t0 = Instant::now();
        let logits = self.forward(&input, true)?;
        let prefill_time = t0.elapsed();

        if let Some(ref mut ps) = self.profile_stats {
            ps.prefill_ms = prefill_time.as_secs_f64() * 1000.0;
        }

        let last_logits_vec = logits.i((0, logits.dim(1)? - 1))?.to_vec1::<f32>()?;
        let mut next_token = {
            let mut indexed: Vec<(usize, f32)> = last_logits_vec.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed[0].0 as u32
        };
        generated.push(next_token);
        if !on_token(next_token) {
            if let Some(ref ps) = self.profile_stats {
                ps.print_summary();
            }
            return Ok(generated);
        }

        log::info!(
            "Prefill: {} tokens in {:.2}s ({:.1} tok/s)",
            prompt_ids.len(),
            prefill_time.as_secs_f64(),
            prompt_ids.len() as f64 / prefill_time.as_secs_f64(),
        );

        // Decode
        for _step in 0..max_new_tokens - 1 {
            let input = Tensor::from_vec(vec![next_token], (1, 1), &device)?;
            let t0_step = Instant::now();
            let logits = self.forward(&input, true)?;
            let step_time = t0_step.elapsed();

            // Collect decode step timing
            if let Some(mut timing) = self.last_step_timing.take() {
                timing.total_ms = step_time.as_secs_f64() * 1000.0;
                timing.other_ms = timing.total_ms - timing.attention_ms - timing.moe_routing_ms
                    - timing.moe_expert_io_ms - timing.moe_expert_compute_ms
                    - timing.moe_shared_expert_ms - timing.norms_ms;
                if let Some(ref mut ps) = self.profile_stats {
                    ps.steps.push(timing);
                }
            }

            let decode_logits_vec = logits.i((0, 0))?.to_vec1::<f32>()?;
            next_token = {
                let mut indexed: Vec<(usize, f32)> = decode_logits_vec.iter().cloned().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed[0].0 as u32
            };

            generated.push(next_token);

            if self.chat_template.is_eos(next_token) {
                on_token(next_token);
                break;
            }

            if !on_token(next_token) {
                break;
            }
        }

        if self.config.adaptive_skip_enabled {
            log::info!(
                "Adaptive Skip: {:.1}% of layers skipped ({}/{})",
                self.layer_output_cache.skip_rate(),
                self.layer_output_cache.skip_count,
                self.layer_output_cache.total_count,
            );
        }

        if let Some(ref ps) = self.profile_stats {
            ps.print_summary();
        }

        Ok(generated)
    }

    /// Generate tokens with sampling (temperature, top-p, repetition penalty).
    ///
    /// Returns generated token IDs (excluding the prompt).
    pub fn generate_sampled(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
    ) -> Result<Vec<u32>> {
        self.kv_cache.clear();
        if let Some(ref mut ps) = self.profile_stats {
            *ps = ProfileStats::new();
        }
        let mut rng = rand::thread_rng();

        let device = self.device.clone();
        let mut generated = Vec::new();

        // Prefill
        let input = Tensor::from_vec(prompt_ids.to_vec(), (1, prompt_ids.len()), &device)?;
        let t0 = Instant::now();
        let logits = self.forward(&input, true)?;
        let prefill_time = t0.elapsed();

        if let Some(ref mut ps) = self.profile_stats {
            ps.prefill_ms = prefill_time.as_secs_f64() * 1000.0;
        }

        let last_logits_vec = logits.i((0, logits.dim(1)? - 1))?.to_vec1::<f32>()?;
        let mut next_token = params.sample(&last_logits_vec, &generated, &mut rng);
        generated.push(next_token);

        log::info!(
            "Prefill: {} tokens in {:.2}s ({:.1} tok/s), first generated: {}",
            prompt_ids.len(),
            prefill_time.as_secs_f64(),
            prompt_ids.len() as f64 / prefill_time.as_secs_f64(),
            next_token,
        );

        // Decode
        for step in 0..max_new_tokens - 1 {
            let input = Tensor::from_vec(vec![next_token], (1, 1), &device)?;
            let t0 = Instant::now();
            let logits = self.forward(&input, true)?;
            let step_time = t0.elapsed();

            // Collect decode step timing
            if let Some(mut timing) = self.last_step_timing.take() {
                timing.total_ms = step_time.as_secs_f64() * 1000.0;
                timing.other_ms = timing.total_ms - timing.attention_ms - timing.moe_routing_ms
                    - timing.moe_expert_io_ms - timing.moe_expert_compute_ms
                    - timing.moe_shared_expert_ms - timing.norms_ms;
                if let Some(ref mut ps) = self.profile_stats {
                    ps.steps.push(timing);
                }
            }

            let decode_logits_vec = logits.i((0, 0))?.to_vec1::<f32>()?;
            next_token = params.sample(&decode_logits_vec, &generated, &mut rng);
            generated.push(next_token);

            if step < 5 || (step + 2) % 10 == 0 {
                log::info!(
                    "Step {}: token={}, {:.2}s ({:.2} tok/s)",
                    step + 2, next_token,
                    step_time.as_secs_f64(),
                    1.0 / step_time.as_secs_f64(),
                );
            }

            if self.chat_template.is_eos(next_token) {
                log::info!("EOS token generated at step {}", step + 2);
                break;
            }
        }

        if self.config.adaptive_skip_enabled {
            log::info!(
                "Adaptive Skip: {:.1}% of layers skipped ({}/{})",
                self.layer_output_cache.skip_rate(),
                self.layer_output_cache.skip_count,
                self.layer_output_cache.total_count,
            );
        }

        if let Some(ref ps) = self.profile_stats {
            ps.print_summary();
        }

        Ok(generated)
    }

    /// Generate tokens with sampling and a per-token callback (for streaming).
    ///
    /// The callback receives each generated token ID. Return `false` to stop early.
    pub fn generate_streaming_sampled<F>(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        mut on_token: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> bool,
    {
        self.kv_cache.clear();
        if let Some(ref mut ps) = self.profile_stats {
            *ps = ProfileStats::new();
        }
        let mut rng = rand::thread_rng();

        let device = self.device.clone();
        let mut generated = Vec::new();

        // Prefill
        let input = Tensor::from_vec(prompt_ids.to_vec(), (1, prompt_ids.len()), &device)?;
        let t_prefill = Instant::now();
        let logits = self.forward(&input, true)?;
        let prefill_time = t_prefill.elapsed();

        if let Some(ref mut ps) = self.profile_stats {
            ps.prefill_ms = prefill_time.as_secs_f64() * 1000.0;
        }

        let last_logits_vec = logits.i((0, logits.dim(1)? - 1))?.to_vec1::<f32>()?;
        let mut next_token = params.sample(&last_logits_vec, &generated, &mut rng);
        generated.push(next_token);
        if !on_token(next_token) {
            if let Some(ref ps) = self.profile_stats {
                ps.print_summary();
            }
            return Ok(generated);
        }

        // Decode
        for _step in 0..max_new_tokens - 1 {
            let input = Tensor::from_vec(vec![next_token], (1, 1), &device)?;
            let t0 = Instant::now();
            let logits = self.forward(&input, true)?;
            let step_time = t0.elapsed();

            // Collect decode step timing
            if let Some(mut timing) = self.last_step_timing.take() {
                timing.total_ms = step_time.as_secs_f64() * 1000.0;
                timing.other_ms = timing.total_ms - timing.attention_ms - timing.moe_routing_ms
                    - timing.moe_expert_io_ms - timing.moe_expert_compute_ms
                    - timing.moe_shared_expert_ms - timing.norms_ms;
                if let Some(ref mut ps) = self.profile_stats {
                    ps.steps.push(timing);
                }
            }

            let decode_logits_vec = logits.i((0, 0))?.to_vec1::<f32>()?;
            next_token = params.sample(&decode_logits_vec, &generated, &mut rng);
            generated.push(next_token);

            if self.chat_template.is_eos(next_token) {
                on_token(next_token);
                break;
            }
            if !on_token(next_token) {
                break;
            }
        }

        if self.config.adaptive_skip_enabled {
            log::info!(
                "Adaptive Skip: {:.1}% of layers skipped ({}/{})",
                self.layer_output_cache.skip_rate(),
                self.layer_output_cache.skip_count,
                self.layer_output_cache.total_count,
            );
        }

        if let Some(ref ps) = self.profile_stats {
            ps.print_summary();
        }

        Ok(generated)
    }

    /// Clear KV-cache, DeltaNet state, and adaptive skip cache (for new conversation/prompt).
    pub fn clear_cache(&mut self) {
        self.kv_cache.clear();
        if let Some(dn_state) = &mut self.deltanet_state {
            dn_state.clear();
        }
        self.layer_output_cache.clear();
    }

    /// Check if this is a DeltaNet hybrid model.
    pub fn is_hybrid(&self) -> bool {
        self.config.is_deltanet_hybrid()
    }

    /// Set maximum layers for partial forward pass (0 = all layers).
    pub fn set_max_layers(&mut self, n: usize) {
        self.max_layers = n;
    }

    /// Override norm_topk_prob setting.
    ///
    /// When false (Qwen1.5-MoE default), routing weights are raw softmax
    /// probabilities (sum < 1). When true (Qwen3 default), weights are
    /// renormalized to sum to 1.
    pub fn set_norm_topk_prob(&mut self, normalize: bool) {
        self.config.norm_topk_prob = normalize;
        log::info!("norm_topk_prob = {}", normalize);
    }

    /// Enable or disable entropy-based dynamic K for MoE routing.
    ///
    /// When enabled, each layer computes router entropy and maps it to a K value
    /// in [k_min, k_max]. Low entropy (confident) uses fewer experts.
    /// k_max of 0 means use num_experts_per_tok from model config.
    pub fn set_dynamic_k(&mut self, enabled: bool, k_min: usize) {
        self.config.dynamic_k_enabled = enabled;
        self.config.dynamic_k_min = k_min;
        log::info!(
            "Dynamic K: enabled={}, k_min={}, k_max={}",
            enabled, k_min, self.config.effective_k_max(),
        );
    }

    /// Enable or disable adaptive expert skip.
    ///
    /// When enabled, layers with similar router logits between consecutive tokens
    /// reuse the previous MoE output, skipping SSD I/O entirely.
    pub fn set_adaptive_skip(&mut self, enabled: bool, threshold: f32) {
        self.config.adaptive_skip_enabled = enabled;
        self.config.adaptive_skip_threshold = threshold;
        self.layer_output_cache.similarity_threshold = threshold;
        log::info!(
            "Adaptive Skip: enabled={}, threshold={}",
            enabled, threshold,
        );
    }

    /// Set the maximum number of consecutive skips per layer for adaptive expert skip.
    pub fn set_adaptive_skip_max_consecutive(&mut self, max: u32) {
        self.layer_output_cache.max_consecutive_skips = max;
        log::info!("Adaptive Skip: max_consecutive_skips={}", max);
    }

    /// Override the maximum K for dynamic K routing.
    pub fn set_dynamic_k_max(&mut self, k_max: usize) {
        self.config.dynamic_k_max = k_max;
        log::info!("Dynamic K: k_max overridden to {}", k_max);
    }

    /// Enable layer-adaptive Dynamic K with tier-based K ranges.
    ///
    /// Automatically enables `dynamic_k_enabled`. Layers are classified into
    /// shallow/deep/final tiers, each with different (k_min, k_max) ranges.
    /// Optionally reinitializes the expert cache with per-layer capacities.
    pub fn set_layer_adaptive_k(&mut self, enabled: bool, tier_config: LayerTierConfig, adaptive_cache: bool) {
        self.config.layer_adaptive_k = enabled;
        self.config.layer_tier_config = tier_config;
        if enabled {
            self.config.dynamic_k_enabled = true;
        }
        if enabled && adaptive_cache {
            let capacities = self.config.layer_tier_config.layer_capacities(self.config.num_layers);
            self.expert_cache = ExpertCache::with_layer_capacities(capacities);
            log::info!("Layer-Adaptive Cache: per-layer capacities from tier config");
        }
        let tc = &self.config.layer_tier_config;
        log::info!(
            "Layer-Adaptive K: enabled={}, shallow={}-{}, deep={}-{}, final={}-{}",
            enabled,
            tc.shallow_k_min, tc.shallow_k_max,
            tc.deep_k_min, tc.deep_k_max,
            tc.final_k_min, tc.final_k_max,
        );
    }

    /// Enable or disable per-layer entropy profiling.
    ///
    /// When enabled, each MoE layer records per-token Shannon entropy of the
    /// full softmax routing distribution. Use `entropy_profile_summary()` to
    /// retrieve results after inference. Minimal overhead (~0.1ms/layer).
    pub fn set_entropy_profiling(&mut self, enabled: bool) {
        self.entropy_profiler.enabled = enabled;
        if enabled {
            self.entropy_profiler.clear();
        }
        log::info!("Entropy profiling: enabled={}", enabled);
    }

    /// Get per-layer entropy profiling results.
    pub fn entropy_profile_summary(&self) -> Vec<EntropyLayerStats> {
        self.entropy_profiler.summary()
    }

    /// Get total number of entropy samples collected.
    pub fn entropy_profile_samples(&self) -> usize {
        self.entropy_profiler.total_samples()
    }

    /// Enable or disable routing statistics collection for calibration-based
    /// expert importance scoring. When enabled, records top-K expert indices
    /// and gate weights for every token at every MoE layer.
    pub fn set_routing_stats(&mut self, enabled: bool) {
        self.routing_stats.enabled = enabled;
        if enabled {
            self.routing_stats.clear();
        }
        log::info!("Routing stats collection: enabled={}", enabled);
    }

    /// Get per-layer, per-expert routing statistics.
    pub fn routing_stats_summary(&self) -> Vec<RoutingLayerStats> {
        self.routing_stats.summary()
    }

    /// Get total tokens processed for routing stats.
    pub fn routing_stats_tokens(&self) -> u64 {
        self.routing_stats.total_tokens_any_layer()
    }

    /// Get model config.
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }

    /// Get the underlying GGUF reader (for training: expert weight dequantization).
    pub fn reader(&self) -> &crate::gguf::GgufReader {
        &self.reader
    }

    /// Get a reference to the pre-loaded resident weights (router gates, biases, etc.).
    pub fn resident_weights(&self) -> &ResidentWeights {
        &self.resident
    }

    /// Get adaptive skip statistics (skip_rate_percent, skip_count, total_count).
    pub fn adaptive_skip_stats(&self) -> (f64, u64, u64) {
        (
            self.layer_output_cache.skip_rate(),
            self.layer_output_cache.skip_count,
            self.layer_output_cache.total_count,
        )
    }

    /// Get the device used by this engine (Metal or CPU).
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the auto-detected chat template.
    pub fn chat_template(&self) -> &ChatTemplate {
        &self.chat_template
    }

    /// Check if running in RAM Resident mode (all weights in memory).
    pub fn is_ram_resident(&self) -> bool {
        self.config.ram_resident
    }

    /// Override RAM Resident mode (e.g. force SSD streaming even for small models).
    pub fn set_ram_resident(&mut self, resident: bool) {
        self.config.ram_resident = resident;
        log::info!("ram_resident = {}", resident);
    }

    pub fn set_gpu_compute(&mut self, enabled: bool) {
        self.config.gpu_compute = enabled;
        log::info!("gpu_compute = {} ({})", enabled, if enabled { "Metal" } else { "CPU" });
    }

    /// Set RAM budget for mlock-based hybrid RAM/SSD mode.
    /// Must be called before preload_weights().
    pub fn set_ram_budget(&mut self, budget_gb: Option<f32>) {
        self.config.ram_budget_gb = budget_gb;
        if let Some(gb) = budget_gb {
            log::info!("ram_budget = {:.1} GB (mlock Q4 pinning)", gb);
        }
    }
}
