//! StreamingConfig: model configuration extracted from GGUF metadata.

use crate::gguf::reader::{GgufReader, GgufError};


/// Inference mode selected by auto-detection or user override.
///
/// Determines how expert weights are loaded and computed:
/// - GpuResident: All weights on Metal GPU (fastest, requires model < 80% RAM)
/// - GpuHybrid: Attention/gate/norm on GPU, experts stream from SSD (80-90% RAM)
/// - RamResident: All weights dequantized to F32 in CPU RAM (manual override only)
/// - SsdStreaming: mmap + on-demand dequant from SSD (default for large models >90% RAM)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InferenceMode {
    /// All weights (experts + attention) resident on Metal GPU.
    /// Zero CPU↔GPU transfer. Requires model fits in ~80% of system RAM.
    GpuResident,
    /// Attention/gate/norm/embed on Metal GPU, experts stream from SSD.
    /// GPU handles attention + routing, CPU handles expert matmul from SSD.
    /// Best for models at 80-90% of system RAM.
    GpuHybrid,
    /// All expert weights dequantized to F32 and held in CPU RAM.
    /// Fast dequant-free inference for medium models. Manual override only.
    RamResident,
    /// Expert weights streamed from SSD via mmap + on-demand dequant.
    /// Default for models that don't fit in RAM (>90% of system RAM).
    SsdStreaming,
}

/// User's device preference for inference mode selection.
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Default)]
pub enum DevicePreference {
    /// Auto-detect best mode based on model size, RAM, and GPU availability.
    #[default]
    Auto,
    /// Force GPU Resident mode (Metal).
    Gpu,
    /// Force CPU/SSD Streaming mode (existing behavior).
    Cpu,
}


impl std::fmt::Display for InferenceMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceMode::GpuResident => write!(f, "GPU Resident"),
            InferenceMode::GpuHybrid => write!(f, "GPU+SSD Hybrid"),
            InferenceMode::RamResident => write!(f, "RAM Resident"),
            InferenceMode::SsdStreaming => write!(f, "SSD Streaming"),
        }
    }
}

/// Layer-adaptive tier configuration for Dynamic K and LRU cache.
///
/// Layers are classified into three tiers based on depth:
/// - Shallow (0 ~ shallow_pct): conservative, keep K high
/// - Deep (shallow_pct ~ final_pct): aggressive, reduce K for speed
/// - Final (final_pct ~ 1.0): recovery, moderate K for output quality
#[derive(Debug, Clone)]
pub struct LayerTierConfig {
    /// Fraction of layers considered "shallow" (default: 0.15).
    pub shallow_pct: f32,
    /// Fraction of layers where "final" tier begins (default: 0.85).
    pub final_pct: f32,
    /// Shallow tier K range (default: k_max, k_max — no reduction).
    pub shallow_k_min: usize,
    pub shallow_k_max: usize,
    /// Deep tier K range (default: 4, 7 — aggressive reduction).
    pub deep_k_min: usize,
    pub deep_k_max: usize,
    /// Final tier K range (default: 7, k_max — moderate).
    pub final_k_min: usize,
    pub final_k_max: usize,
    /// Per-tier LRU cache capacities.
    pub shallow_cache_cap: usize,
    pub deep_cache_cap: usize,
    pub final_cache_cap: usize,
}

impl LayerTierConfig {
    /// Create default tier config for a given model K_max.
    pub fn default_for_k(k_max: usize) -> Self {
        // F32 expert cache capacity per layer. Each cached expert uses ~19MB F32
        // (3 × intermediate × hidden × 4 bytes). Large cache saves dequant time but
        // competes with OS page cache for RAM. For models that fit in page cache,
        // cache=0 is optimal (let OS manage Q4 pages, 3.6× smaller than F32).
        // Non-zero cache is useful when eviction is ON (large models where page cache
        // can't hold the full file).
        Self {
            shallow_pct: 0.15,
            final_pct: 0.85,
            shallow_k_min: k_max,
            shallow_k_max: k_max,
            deep_k_min: 4.min(k_max),
            deep_k_max: (k_max * 2 / 3).max(4).min(k_max),
            final_k_min: (k_max * 2 / 3).max(4).min(k_max),
            final_k_max: k_max,
            shallow_cache_cap: 0,
            deep_cache_cap: 0,
            final_cache_cap: 0,
        }
    }

    /// Build per-layer cache capacities vector.
    pub fn layer_capacities(&self, num_layers: usize) -> Vec<usize> {
        (0..num_layers)
            .map(|i| {
                let depth = i as f32 / num_layers as f32;
                if depth < self.shallow_pct {
                    self.shallow_cache_cap
                } else if depth < self.final_pct {
                    self.deep_cache_cap
                } else {
                    self.final_cache_cap
                }
            })
            .collect()
    }
}

/// Model configuration extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub architecture: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// Whether this model has shared experts (Qwen3-Coder-Next)
    pub has_shared_expert: bool,
    /// Full attention interval (for DeltaNet hybrid models)
    pub full_attention_interval: usize,
    /// Partial rotary factor (fraction of head_dim that gets RoPE)
    pub partial_rotary_factor: f32,

    // === DeltaNet-specific (only used when is_deltanet_hybrid()) ===
    /// SSM state size = key head dim (128 for 80B)
    pub ssm_d_state: usize,
    /// SSM group count = num key heads (16 for 80B)
    pub ssm_n_group: usize,
    /// SSM dt rank = num value heads (32 for 80B)
    pub ssm_dt_rank: usize,
    /// SSM inner size = total value dim (4096 for 80B)
    pub ssm_d_inner: usize,
    /// SSM conv kernel size (4 for 80B)
    pub ssm_d_conv: usize,
    /// Rotary dim for attention layers (head_dim * partial_rotary_factor)
    pub rotary_dim: usize,

    // === Dense/MoE detection ===
    /// Whether this model uses MoE (Mixture of Experts) layers.
    /// When false, the model uses standard dense FFN layers.
    pub is_moe: bool,
    /// Hidden size for dense FFN layers (typically 4 * hidden_size for SwiGLU).
    pub ffn_hidden_size: usize,

    // === MoE routing ===
    /// Whether to renormalize top-k routing weights to sum to 1.0.
    /// When false (Qwen1.5-MoE, Qwen2MoE default), uses raw softmax probabilities.
    /// When true (Qwen3 default), renormalizes top-k weights.
    pub norm_topk_prob: bool,

    // === RAM Resident mode ===
    /// When true, all expert weights are preloaded into RAM (for models that fit).
    /// When false, experts are streamed from SSD via mmap (default for large models).
    pub ram_resident: bool,

    // === Dynamic K (entropy-based adaptive top-K routing) ===
    /// Enable entropy-based dynamic K selection for MoE routing.
    /// When false, uses fixed `num_experts_per_tok`.
    pub dynamic_k_enabled: bool,
    /// Minimum K value when dynamic K is enabled (default: 2).
    pub dynamic_k_min: usize,
    /// Maximum K value when dynamic K is enabled (0 = use num_experts_per_tok).
    pub dynamic_k_max: usize,

    // === Layer-Adaptive Dynamic K ===
    /// Enable layer-adaptive K ranges (different K per layer tier).
    /// When enabled, shallow/deep/final tiers use different k_min/k_max.
    pub layer_adaptive_k: bool,
    /// Tier configuration for layer-adaptive K and cache.
    pub layer_tier_config: LayerTierConfig,

    // === Adaptive Expert Skip ===
    /// Enable adaptive expert skip: reuse previous MoE output when router logits
    /// are similar between consecutive tokens (cosine similarity > threshold).
    pub adaptive_skip_enabled: bool,
    /// Cosine similarity threshold for adaptive expert skip (default: 0.95).
    pub adaptive_skip_threshold: f32,

    // === Model identity ===
    /// Model name from GGUF metadata (general.name), used for chat template detection.
    pub model_name: String,

    // === Variable expert count (pruned models) ===
    /// Per-layer expert counts for pruned models with variable expert counts.
    /// None = all layers have `num_experts` (uniform, default).
    /// Some(vec) = each layer has a potentially different expert count.
    pub experts_per_layer: Option<Vec<usize>>,

    // === Expert eviction ===
    /// Whether to evict (madvise MADV_FREE) expert mmap pages after use.
    /// Auto-enabled when GGUF file > 85% of system RAM (prevents page cache OOM).
    /// Disabled when file fits in RAM (page cache reuse is faster).
    /// Override with EXPERT_EVICT=0 or EXPERT_EVICT=1.
    pub evict_experts: bool,

    // === Compute device ===
    /// Use GPU (Metal) for expert/attention computation instead of CPU.
    /// Auto-enabled on Apple Silicon (unified memory = zero copy overhead).
    /// Override with --cpu-compute to force CPU.
    pub gpu_compute: bool,


    // === Sliding Window Attention ===
    /// Sliding window size for attention (0 = disabled, full attention on all layers).
    /// When > 0, alternating layers use sliding window attention (GPT-OSS pattern:
    /// even layers = full attention, odd layers = SWA with this window size).
    pub sliding_window: usize,

    // === Hybrid RAM/SSD mode (mlock-based) ===
    /// RAM budget in GB for mlock-pinning MoE expert pages in Q4 format.
    /// When set, the engine mlocks as many MoE layers' expert tensors as fit
    /// within the budget (starting from layer 0). Pinned pages stay in RAM
    /// without dequantization — the OS cannot evict them.
    /// Remaining layers stream from SSD via mmap as usual.
    /// None = no mlock pinning (default: pure SSD streaming or full RAM resident).
    pub ram_budget_gb: Option<f32>,

    // === Inference mode (auto-selected or user-overridden) ===
    /// User device preference: Auto (default), Gpu, or Cpu.
    /// Used by Engine::open() to determine inference_mode.
    pub device_preference: DevicePreference,
    /// Selected inference mode after auto-detection.
    /// Set by Engine::open() based on model size, RAM, and GPU availability.
    pub inference_mode: Option<InferenceMode>,
}

impl StreamingConfig {
    /// Build config from GGUF metadata.
    pub fn from_gguf(reader: &GgufReader) -> Result<Self, GgufError> {
        let arch = reader
            .get_metadata("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("qwen2moe")
            .to_string();

        let get_u32 = |key: &str| -> Option<u32> {
            reader.get_metadata(key).and_then(|v| v.as_u32())
        };

        let get_f32 = |key: &str| -> Option<f32> {
            reader.get_metadata(key).and_then(|v| v.as_f32())
        };

        let hidden_size = get_u32(&format!("{arch}.embedding_length"))
            .unwrap_or(2048) as usize;

        let num_layers = get_u32(&format!("{arch}.block_count"))
            .unwrap_or(48) as usize;

        let num_attention_heads = get_u32(&format!("{arch}.attention.head_count"))
            .unwrap_or(16) as usize;

        let num_kv_heads = get_u32(&format!("{arch}.attention.head_count_kv"))
            .unwrap_or(2) as usize;

        let head_dim = get_u32(&format!("{arch}.attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or(hidden_size / num_attention_heads);

        let num_experts = get_u32(&format!("{arch}.expert_count"))
            .unwrap_or(128) as usize;

        let num_experts_per_tok = get_u32(&format!("{arch}.expert_used_count"))
            .unwrap_or(8) as usize;

        let rope_theta = get_f32(&format!("{arch}.rope.freq_base"))
            .unwrap_or(1_000_000.0);

        let rms_norm_eps = get_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);

        // Infer vocab_size from embedding tensor shape if not in metadata
        let vocab_size = get_u32(&format!("{arch}.vocab_size"))
            .map(|v| v as usize)
            .or_else(|| {
                reader.tensors.get("token_embd.weight")
                    .map(|t| t.pt_shape()[0])
            })
            .unwrap_or(151936);

        // Infer moe_intermediate_size from expert tensor shape
        let moe_intermediate_size = get_u32(&format!("{arch}.expert_feed_forward_length"))
            .map(|v| v as usize)
            .or_else(|| {
                // Try to infer from ffn_gate_exps tensor shape
                reader.tensors.get("blk.0.ffn_gate_exps.weight")
                    .map(|t| {
                        let shape = t.pt_shape();
                        if shape.len() >= 3 { shape[1] } else { 768 }
                    })
            })
            .unwrap_or(768);

        // DeltaNet hybrid model detection (Qwen3-Coder-Next)
        // Detect from tensor presence (SSM tensors indicate hybrid model)
        let has_ssm_tensors = reader.tensors.contains_key("blk.0.ssm_a")
            || reader.tensors.contains_key("blk.0.ssm_conv1d.weight");

        // If SSM tensors exist, this is a hybrid model with interval=4 (every 4th layer is attention)
        let full_attention_interval = get_u32(&format!("{arch}.attention.full_attention_interval"))
            .map(|v| v as usize)
            .unwrap_or_else(|| if has_ssm_tensors { 4 } else { 0 });

        // Partial rotary: try partial_rotary_factor, then compute from dimension_count/head_dim
        let partial_rotary_factor = get_f32(&format!("{arch}.rope.partial_rotary_factor"))
            .or_else(|| {
                let dim_count = get_u32(&format!("{arch}.rope.dimension_count"))? as usize;
                Some(dim_count as f32 / head_dim as f32)
            })
            .unwrap_or(1.0);

        // Check for shared expert
        let has_shared_expert = reader.tensors.contains_key("blk.0.ffn_gate_shexp.weight")
            || reader.tensors.contains_key("blk.0.ffn_up_shexp.weight");

        // DeltaNet SSM parameters (from GGUF metadata)
        // Note: time_step_rank is the GGUF key name for dt_rank
        let ssm_d_state = get_u32(&format!("{arch}.ssm.state_size"))
            .unwrap_or(128) as usize;
        let ssm_n_group = get_u32(&format!("{arch}.ssm.group_count"))
            .unwrap_or(16) as usize;
        let ssm_dt_rank = get_u32(&format!("{arch}.ssm.time_step_rank"))
            .or_else(|| get_u32(&format!("{arch}.ssm.dt_rank")))
            .unwrap_or(32) as usize;
        let ssm_d_inner = get_u32(&format!("{arch}.ssm.inner_size"))
            .unwrap_or(4096) as usize;
        let ssm_d_conv = get_u32(&format!("{arch}.ssm.conv_kernel"))
            .unwrap_or(4) as usize;

        let rotary_dim = (head_dim as f32 * partial_rotary_factor) as usize;

        // norm_topk_prob: whether to renormalize top-k routing weights.
        // Qwen1.5-MoE and Qwen2MoE default to false; Qwen3 defaults to true.
        // Check GGUF metadata first, then infer from model name.
        let model_name = reader
            .get_metadata("general.name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let norm_topk_prob = !(model_name.contains("Qwen1.5") || model_name.contains("Qwen2"));

        // Sliding window attention (GPT-OSS: alternating SWA/full per layer)
        let sliding_window = get_u32(&format!("{arch}.attention.sliding_window"))
            .unwrap_or(0) as usize;

        let is_moe = num_experts > 1;
        let ffn_hidden_size = get_u32(&format!("{arch}.feed_forward_length"))
            .map(|v| v as usize)
            .unwrap_or(hidden_size * 4);

        // Variable expert count for pruned models
        let experts_per_layer = reader
            .get_metadata(&format!("{arch}.experts_per_layer"))
            .and_then(|v| v.as_u32_array())
            .map(|arr| arr.into_iter().map(|v| v as usize).collect::<Vec<_>>());

        if let Some(ref epl) = experts_per_layer {
            let min_e = epl.iter().min().copied().unwrap_or(0);
            let max_e = epl.iter().max().copied().unwrap_or(0);
            if min_e != max_e {
                log::info!(
                    "Variable expert count: {}-{} per layer (pruned model)",
                    min_e, max_e
                );
            }
        }

        Ok(Self {
            architecture: arch,
            hidden_size,
            num_layers,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            rope_theta,
            rms_norm_eps,
            has_shared_expert,
            full_attention_interval,
            partial_rotary_factor,
            ssm_d_state,
            ssm_n_group,
            ssm_dt_rank,
            ssm_d_inner,
            ssm_d_conv,
            rotary_dim,
            is_moe,
            ffn_hidden_size,
            norm_topk_prob,
            ram_resident: false, // auto-detected in Engine::open()
            dynamic_k_enabled: false,
            dynamic_k_min: 2,
            dynamic_k_max: 0,
            layer_adaptive_k: false,
            layer_tier_config: LayerTierConfig::default_for_k(num_experts_per_tok),
            adaptive_skip_enabled: false,
            adaptive_skip_threshold: 0.95,
            model_name,
            experts_per_layer,
            sliding_window,
            evict_experts: true, // auto-adjusted in Engine::open() based on file vs RAM size
            gpu_compute: false, // CPU faster for batch_size=1 (Metal kernel dispatch overhead)
            ram_budget_gb: None, // set via --ram-budget CLI flag
            device_preference: DevicePreference::Auto,
            inference_mode: None, // set by Engine::open() after auto-detection
        })
    }

    /// Get the number of experts for a specific layer.
    /// Returns per-layer count if available (pruned model), otherwise `num_experts`.
    pub fn experts_for_layer(&self, layer_idx: usize) -> usize {
        self.experts_per_layer
            .as_ref()
            .and_then(|epl| epl.get(layer_idx).copied())
            .unwrap_or(self.num_experts)
    }

    /// Check if this is a DeltaNet hybrid model (has DeltaNet + Attention layers).
    pub fn is_deltanet_hybrid(&self) -> bool {
        self.full_attention_interval > 0
    }

    /// Determine layer type for a given layer index.
    /// Returns true if the layer is a standard attention layer, false if DeltaNet.
    pub fn is_attention_layer(&self, layer_idx: usize) -> bool {
        if self.full_attention_interval == 0 {
            true // All layers are standard attention
        } else {
            (layer_idx + 1).is_multiple_of(self.full_attention_interval)
        }
    }

    /// Whether this layer uses sliding window attention (vs full attention).
    /// GPT-OSS pattern (dense_first=false): even layers = SWA, odd layers = full.
    /// Returns false if sliding_window is 0 (disabled).
    pub fn is_swa_layer(&self, layer_idx: usize) -> bool {
        self.sliding_window > 0 && layer_idx.is_multiple_of(2)
    }

    /// Effective k_max for dynamic K (0 means use num_experts_per_tok).
    pub fn effective_k_max(&self) -> usize {
        if self.dynamic_k_max > 0 { self.dynamic_k_max } else { self.num_experts_per_tok }
    }

    /// Get layer-specific (k_min, k_max) range based on tier classification.
    pub fn get_layer_k_range(&self, layer_idx: usize) -> (usize, usize) {
        let depth = layer_idx as f32 / self.num_layers as f32;
        let tier = &self.layer_tier_config;
        if depth < tier.shallow_pct {
            (tier.shallow_k_min, tier.shallow_k_max)
        } else if depth < tier.final_pct {
            (tier.deep_k_min, tier.deep_k_max)
        } else {
            (tier.final_k_min, tier.final_k_max)
        }
    }

    /// DeltaNet value head dim (ssm_d_inner / ssm_dt_rank).
    pub fn ssm_head_v_dim(&self) -> usize {
        if self.ssm_dt_rank == 0 { 128 } else { self.ssm_d_inner / self.ssm_dt_rank }
    }

    /// DeltaNet conv1d total dimension (Q + K + V channels).
    /// = 2 * head_k_dim * num_k_heads + head_v_dim * num_v_heads
    pub fn ssm_conv_dim(&self) -> usize {
        2 * self.ssm_d_state * self.ssm_n_group + self.ssm_d_inner
    }
}
