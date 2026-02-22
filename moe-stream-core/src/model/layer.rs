//! Single transformer layer forward pass with GGUF weight loading.
//!
//! Loads weights on-demand from memory-mapped GGUF, dequantizes to Metal,
//! runs forward, then drops weights. This is the core of SSD streaming.
//! Supports Expert LRU cache and resident weight pre-loading for speed.
//!
//! For hybrid models (Qwen3-Coder-Next), dispatches to DeltaNet for linear
//! attention layers and standard attention for full attention layers.

use candle_core::{Device, IndexOp, Module, Result, Tensor};

use crate::config::{StreamingConfig, InferenceMode};
use crate::gguf::dequant::{mxfp4_matvec_mul, mxfp4_matmul};
use crate::gguf::reader::{GgufReader, GgmlQuantType};
use crate::model::cache::{ExpertCache, ExpertWeights, LayerOutputCache, ResidentWeights};
use crate::model::kv_cache::KvCache;
use crate::model::deltanet::{DeltaNetState, deltanet_forward};
use crate::ops;

/// Load a GGUF tensor and move to device as F32.
pub fn load_weight(reader: &GgufReader, gguf_name: &str, device: &Device) -> Result<Tensor> {
    let (data, shape) = reader
        .dequantize_tensor(gguf_name)
        .map_err(|e| candle_core::Error::Msg(format!("GGUF load {}: {}", gguf_name, e)))?;
    Tensor::from_vec(data, shape.as_slice(), device)
}

/// Load a single expert slice from a stacked GGUF tensor (mmap path).
pub fn load_expert(
    reader: &GgufReader,
    gguf_name: &str,
    expert_idx: usize,
    device: &Device,
) -> Result<Tensor> {
    let (data, shape) = reader
        .dequantize_expert(gguf_name, expert_idx)
        .map_err(|e| candle_core::Error::Msg(format!("GGUF expert {}: {}", gguf_name, e)))?;
    Tensor::from_vec(data, shape.as_slice(), device)
}

/// Load a single expert slice as a QMatMul (no dequantization, quantized matmul).
///
/// Constructs a candle QTensor directly from the mmap'd quantized bytes,
/// skipping the dequant→F32→Tensor path. The QMatMul then performs matmul
/// natively in the quantized format.
pub fn load_expert_quantized(
    reader: &GgufReader,
    gguf_name: &str,
    expert_idx: usize,
    device: &candle_core::Device,
) -> Result<candle_core::quantized::QMatMul> {
    let qtensor = reader
        .expert_slice_as_qtensor(gguf_name, expert_idx, device)
        .map_err(|e| candle_core::Error::Msg(format!("GGUF expert Q {}: {}", gguf_name, e)))?;
    candle_core::quantized::QMatMul::from_qtensor(qtensor)
}

/// Load a single expert slice via F_NOCACHE pread (bypasses page cache).
#[allow(dead_code)]
fn load_expert_nocache(
    reader: &GgufReader,
    gguf_name: &str,
    expert_idx: usize,
    device: &Device,
    buf: &mut Vec<u8>,
) -> Result<Tensor> {
    let (data, shape) = reader
        .dequantize_expert_nocache(gguf_name, expert_idx, buf)
        .map_err(|e| candle_core::Error::Msg(format!("GGUF expert nocache {}: {}", gguf_name, e)))?;
    Tensor::from_vec(data, shape.as_slice(), device)
}

/// Reconstruct a single expert's weight matrix from VQ codebook + indices.
///
/// Reads the F16 indices from the GGUF stacked index tensor, then uses the
/// pre-loaded codebook to reconstruct the full F32 weight matrix via lookup.
///
/// The indices tensor is stored as [n_blocks, n_experts] in F16 format.
/// Each index value (0..K-1) maps to a codebook entry of block_dim floats.
/// Blocks are arranged in row-major order: n_bh blocks vertically × n_bw horizontally.
fn load_expert_vq(
    reader: &GgufReader,
    codebook: &[f32],   // [K * block_dim], pre-loaded
    vq_config: &VqConfig,
    idx_tensor_name: &str,
    expert_idx: usize,
    out_shape: (usize, usize),  // (H, W) e.g., (2048, 512) for the reconstructed weight
) -> Result<Tensor> {
    let (h, w) = out_shape;
    let block_h = vq_config.block_h;
    let block_w = vq_config.block_w;
    let block_dim = vq_config.block_dim;
    let n_bh = h / block_h;
    let n_bw = w / block_w;
    let n_blocks = n_bh * n_bw;

    // Read indices for this expert from the stacked F16 tensor.
    // The tensor shape is [n_blocks, n_experts] in GGUF order.
    // expert_slice_data gives us the raw bytes for one expert's slice.
    let (idx_data, _idx_shape) = reader
        .expert_slice_data(idx_tensor_name, expert_idx)
        .map_err(|e| candle_core::Error::Msg(format!("VQ idx {}: {}", idx_tensor_name, e)))?;

    // F16 bytes → f32 → u32 indices
    // Each index is 2 bytes (F16). n_blocks indices per expert.
    let idx_f16: &[half::f16] = unsafe {
        std::slice::from_raw_parts(idx_data.as_ptr() as *const half::f16, n_blocks)
    };

    // Reconstruct weight matrix via codebook lookup
    let mut weight = vec![0.0f32; h * w];

    for bi in 0..n_blocks {
        let code_idx = idx_f16[bi].to_f32() as usize;
        let code_offset = code_idx * block_dim;
        let brow = bi / n_bw;
        let bcol = bi % n_bw;
        for bh in 0..block_h {
            for bw_i in 0..block_w {
                let row = brow * block_h + bh;
                let col = bcol * block_w + bw_i;
                if row < h && col < w {
                    weight[row * w + col] = codebook[code_offset + bh * block_w + bw_i];
                }
            }
        }
    }

    Tensor::from_vec(weight, (h, w), &Device::Cpu)
}

/// Reconstruct a single expert's weight matrix from per-expert VQ codebook + indices.
///
/// Unlike `load_expert_vq`, this reads the codebook from the GGUF on-demand using
/// expert_slice_data (the codebook tensor is [block_dim, K, n_experts] in GGUF).
/// This avoids pre-loading all per-expert codebooks into memory (~1+ GB).
fn load_expert_vq_per_expert(
    reader: &GgufReader,
    vq_config: &VqConfig,
    cb_tensor_name: &str,
    idx_tensor_name: &str,
    expert_idx: usize,
    out_shape: (usize, usize),
) -> Result<Tensor> {
    let (h, w) = out_shape;
    let block_h = vq_config.block_h;
    let block_w = vq_config.block_w;
    let block_dim = vq_config.block_dim;
    let n_bh = h / block_h;
    let n_bw = w / block_w;
    let n_blocks = n_bh * n_bw;

    // Read per-expert codebook: tensor shape [block_dim, K, n_experts] in GGUF
    // expert_slice_data returns [block_dim, K] = [16, 2048] for one expert as raw F16 bytes.
    let (cb_data, _cb_shape) = reader
        .expert_slice_data(cb_tensor_name, expert_idx)
        .map_err(|e| candle_core::Error::Msg(format!("VQ cb {}: {}", cb_tensor_name, e)))?;

    let cb_elements = vq_config.k * block_dim;
    let cb_f16: &[half::f16] = unsafe {
        std::slice::from_raw_parts(cb_data.as_ptr() as *const half::f16, cb_elements)
    };

    // Convert codebook F16 → F32
    let codebook: Vec<f32> = cb_f16.iter().map(|v| v.to_f32()).collect();

    // Read indices (same as shared mode)
    let (idx_data, _idx_shape) = reader
        .expert_slice_data(idx_tensor_name, expert_idx)
        .map_err(|e| candle_core::Error::Msg(format!("VQ idx {}: {}", idx_tensor_name, e)))?;

    let idx_f16: &[half::f16] = unsafe {
        std::slice::from_raw_parts(idx_data.as_ptr() as *const half::f16, n_blocks)
    };

    // Reconstruct weight matrix via codebook lookup
    let mut weight = vec![0.0f32; h * w];

    for bi in 0..n_blocks {
        let code_idx = idx_f16[bi].to_f32() as usize;
        let code_offset = code_idx * block_dim;
        let brow = bi / n_bw;
        let bcol = bi % n_bw;
        for bh in 0..block_h {
            for bw_i in 0..block_w {
                let row = brow * block_h + bh;
                let col = bcol * block_w + bw_i;
                if row < h && col < w {
                    weight[row * w + col] = codebook[code_offset + bh * block_w + bw_i];
                }
            }
        }
    }

    Tensor::from_vec(weight, (h, w), &Device::Cpu)
}

/// Check if F_NOCACHE expert loading is enabled (checked once, cached).
fn use_nocache_experts() -> bool {
    static VAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VAL.get_or_init(|| {
        std::env::var("NOCACHE_EXPERTS").map(|v| v != "0").unwrap_or(false)
    })
}

/// Check if quantized matmul is enabled (QUANTIZED_MATMUL=1).
/// When enabled, expert weights stay in Q4_K format and matmul is done
/// directly on quantized data, skipping the dequant→F32 step.
fn use_quantized_matmul() -> bool {
    static VAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VAL.get_or_init(|| {
        let enabled = std::env::var("QUANTIZED_MATMUL").map(|v| v == "1").unwrap_or(false);
        if enabled {
            log::info!("Quantized matmul enabled: expert weights stay in quantized format (Q4_K/MXFP4)");
        }
        enabled
    })
}

/// Check if MADV_FREE expert eviction is enabled.
/// Reads from config.evict_experts (auto-set in Engine::open based on file size vs RAM).
/// Override: EXPERT_EVICT=0 to force disable, EXPERT_EVICT=1 to force enable.
fn use_expert_eviction(config_evict: bool) -> bool {
    static OVERRIDE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let ovr = *OVERRIDE.get_or_init(|| {
        std::env::var("EXPERT_EVICT").ok().map(|v| v != "0")
    });
    ovr.unwrap_or(config_evict)
}

/// Check if per-layer timing profiler is enabled (PROFILE_LAYERS=1).
/// When enabled, collects detailed per-component timing and prints a summary after generation.
pub fn use_profile_layers() -> bool {
    static VAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VAL.get_or_init(|| {
        let enabled = std::env::var("PROFILE_LAYERS").map(|v| v == "1").unwrap_or(false);
        if enabled {
            log::info!("Per-layer timing profiler enabled (PROFILE_LAYERS=1)");
        }
        enabled
    })
}

/// Accumulated timing statistics for a single forward step (all layers).
#[derive(Clone, Debug, Default)]
pub struct StepTimingStats {
    pub attention_ms: f64,
    pub moe_routing_ms: f64,
    pub moe_expert_io_ms: f64,
    pub moe_expert_compute_ms: f64,
    pub moe_shared_expert_ms: f64,
    pub norms_ms: f64,
    pub other_ms: f64,
    pub total_ms: f64,
}

/// Aggregated timing statistics across all decode steps.
#[derive(Clone, Debug)]
pub struct ProfileStats {
    pub steps: Vec<StepTimingStats>,
    pub prefill_ms: f64,
    pub embed_ms: f64,
    pub lm_head_ms: f64,
}

impl ProfileStats {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            prefill_ms: 0.0,
            embed_ms: 0.0,
            lm_head_ms: 0.0,
        }
    }

    pub fn print_summary(&self) {
        if self.steps.is_empty() {
            return;
        }
        let n = self.steps.len() as f64;
        let avg = |f: fn(&StepTimingStats) -> f64| -> f64 {
            self.steps.iter().map(f).sum::<f64>() / n
        };

        let avg_attn = avg(|s| s.attention_ms);
        let avg_routing = avg(|s| s.moe_routing_ms);
        let avg_expert_io = avg(|s| s.moe_expert_io_ms);
        let avg_expert_compute = avg(|s| s.moe_expert_compute_ms);
        let avg_shexp = avg(|s| s.moe_shared_expert_ms);
        let avg_norms = avg(|s| s.norms_ms);
        let avg_other = avg(|s| s.other_ms);
        let avg_total = avg(|s| s.total_ms);

        let avg_moe_total = avg_routing + avg_expert_io + avg_expert_compute + avg_shexp;
        let tok_per_sec = if avg_total > 0.0 { 1000.0 / avg_total } else { 0.0 };

        eprintln!();
        eprintln!("=== Layer Timing Profile (avg per token, {} decode steps) ===", self.steps.len());
        eprintln!("  Attention: {:.1}ms", avg_attn);
        eprintln!("  MoE: {:.1}ms (routing: {:.1}ms, expert_io: {:.1}ms, expert_compute: {:.1}ms, shared: {:.1}ms)",
            avg_moe_total, avg_routing, avg_expert_io, avg_expert_compute, avg_shexp);
        eprintln!("  Norms: {:.1}ms", avg_norms);
        eprintln!("  Other: {:.1}ms", avg_other);
        eprintln!("  Total: {:.1}ms ({:.1} tok/s)", avg_total, tok_per_sec);
        if self.prefill_ms > 0.0 {
            eprintln!("  Prefill: {:.1}ms", self.prefill_ms);
        }
        eprintln!();
    }
}

/// Per-layer timing collected during a single forward_layer call.
/// Returned to the engine for aggregation when profiling is enabled.
#[derive(Clone, Debug, Default)]
pub struct LayerTiming {
    pub attention_ms: f64,
    pub norms_ms: f64,
    pub moe_routing_ms: f64,
    pub moe_expert_io_ms: f64,
    pub moe_expert_compute_ms: f64,
    pub moe_shared_expert_ms: f64,
}

/// VQ compression configuration (passed from Engine).
#[derive(Clone, Debug)]
pub struct VqConfig {
    pub block_h: usize,
    pub block_w: usize,
    pub k: usize,
    pub block_dim: usize,
}

/// Handles the forward pass for a single transformer layer.
pub struct LayerForward<'a> {
    reader: &'a GgufReader,
    config: &'a StreamingConfig,
    device: &'a Device,
    cos_gpu: &'a Tensor,
    sin_gpu: &'a Tensor,
    /// CPU-side RoPE tables (avoid Metal→CPU transfer per layer)
    cos_cpu: &'a Tensor,
    sin_cpu: &'a Tensor,
    /// VQ codebooks for this model (None if not VQ or per-expert VQ). Shared ref from Engine.
    vq_codebooks: Option<&'a Vec<[Vec<f32>; 3]>>,
    /// VQ config (None if not VQ).
    vq_config: Option<&'a VqConfig>,
    /// Whether VQ uses per-expert codebooks (read on-demand from GGUF).
    vq_per_expert: bool,
}

impl<'a> LayerForward<'a> {
    pub fn new(
        reader: &'a GgufReader,
        config: &'a StreamingConfig,
        device: &'a Device,
        cos: &'a Tensor,
        sin: &'a Tensor,
        cos_cpu: &'a Tensor,
        sin_cpu: &'a Tensor,
    ) -> Self {
        Self {
            reader,
            config,
            device,
            cos_gpu: cos,
            sin_gpu: sin,
            cos_cpu,
            sin_cpu,
            vq_codebooks: None,
            vq_config: None,
            vq_per_expert: false,
        }
    }

    /// Set VQ codebooks and config (called from Engine for VQ models).
    pub fn with_vq(
        mut self,
        codebooks: Option<&'a Vec<[Vec<f32>; 3]>>,
        config: Option<&'a VqConfig>,
        per_expert: bool,
    ) -> Self {
        self.vq_codebooks = codebooks;
        self.vq_config = config;
        self.vq_per_expert = per_expert;
        self
    }

    /// Run a single transformer layer: pre-norm → attention/deltanet → residual → post-norm → MoE → residual.
    ///
    /// For hybrid models (Qwen3-Coder-Next), dispatches to DeltaNet for linear attention layers.
    /// MoE routing uses CPU gate weight copies to avoid Metal sync overhead (~10% speedup).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        layer_idx: usize,
        kv_cache: &mut KvCache,
        use_cache: bool,
        expert_cache: &mut ExpertCache,
        resident: &ResidentWeights,
        deltanet_state: Option<&mut DeltaNetState>,
        layer_output_cache: &mut LayerOutputCache,
        entropy_out: &mut Option<f32>,
        routing_stats_out: &mut Option<(Vec<u32>, Vec<f32>)>,
        timing_out: &mut Option<LayerTiming>,
    ) -> Result<Tensor> {
        let prefix = format!("blk.{}", layer_idx);
        let is_attention_layer = self.config.is_attention_layer(layer_idx);

        let t0 = std::time::Instant::now();

        // Use preloaded norm weights if available (already on self.device), else load from GGUF
        let (input_norm, post_norm) = if let Some(nw) = &resident.norms[layer_idx] {
            (nw.input_norm.clone(), nw.post_norm.clone())
        } else {
            let inp = load_weight(self.reader, &format!("{}.attn_norm.weight", prefix), self.device)?;
            let post = load_weight(self.reader, &format!("{}.post_attention_norm.weight", prefix), self.device)
                .or_else(|_| load_weight(self.reader, &format!("{}.ffn_norm.weight", prefix), self.device))?;
            (inp, post)
        };

        let norm_load_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // === DN/Attention compute (single-threaded) ===
        let residual = hidden_states.clone();
        let normed = ops::rms_norm(hidden_states, &input_norm, self.config.rms_norm_eps as f64)?;
        drop(input_norm);

        let t1 = std::time::Instant::now();
        let (_bsz, seq_len, _) = normed.dims3()?;

        let attn_out = if is_attention_layer {
            self.run_attention(&normed, layer_idx, kv_cache, use_cache, &prefix, resident)
        } else if let Some(dn_state) = deltanet_state {
            let dn_weights = resident.deltanet[layer_idx].as_ref();
            if seq_len == 1 {
                deltanet_forward(self.reader, self.config, &normed, layer_idx, dn_state, dn_weights)
            } else {
                // Prefill: sequential DeltaNet
                let mut outputs = Vec::with_capacity(seq_len);
                for t in 0..seq_len {
                    let token_hidden = normed.i((.., t..t+1, ..))?;
                    let token_out = deltanet_forward(self.reader, self.config, &token_hidden, layer_idx, dn_state, dn_weights)?;
                    outputs.push(token_out);
                }
                let refs: Vec<&Tensor> = outputs.iter().collect();
                Tensor::cat(&refs, 1)
            }
        } else {
            Err(candle_core::Error::Msg(format!(
                "Layer {} is DeltaNet but no DeltaNetState provided",
                layer_idx
            )))
        }?;

        let attn_ms = t1.elapsed().as_secs_f64() * 1000.0;

        drop(normed);
        let hidden_states = (residual + attn_out)?;

        // === MoE ===
        let residual = hidden_states.clone();
        let normed = ops::rms_norm(&hidden_states, &post_norm, self.config.rms_norm_eps as f64)?;
        drop(post_norm);

        let profiling = timing_out.is_some();
        let mut moe_timing: Option<(f64, f64, f64, f64)> = if profiling { Some((0.0, 0.0, 0.0, 0.0)) } else { None };

        let t2 = std::time::Instant::now();
        let moe_out = self.run_moe(&normed, layer_idx, &prefix, expert_cache, resident, layer_output_cache, entropy_out, routing_stats_out, &mut moe_timing)?;
        let moe_ms = t2.elapsed().as_secs_f64() * 1000.0;

        drop(normed);
        let hidden_states = (residual + moe_out)?;

        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Populate per-layer timing for profiler
        if let Some(ref mut timing) = timing_out {
            timing.attention_ms += attn_ms;
            timing.norms_ms += norm_load_ms;
            if let Some((routing, io, compute, shared)) = moe_timing {
                timing.moe_routing_ms += routing;
                timing.moe_expert_io_ms += io;
                timing.moe_expert_compute_ms += compute;
                timing.moe_shared_expert_ms += shared;
            }
        }

        // Log every layer at TRACE level, key layers at DEBUG
        let layer_type = if is_attention_layer { "attn" } else { "delta" };
        let log_msg = format!(
            "Layer {} ({}): total={:.1}ms (norm={:.1}, attn={:.1}, moe={:.1})",
            layer_idx, layer_type, total_ms, norm_load_ms, attn_ms, moe_ms,
        );
        if layer_idx == 0 || layer_idx == 3 || layer_idx == 23 || layer_idx == 47 {
            log::debug!("{}", log_msg);
        } else {
            log::trace!("{}", log_msg);
        }

        Ok(hidden_states)
    }

    /// Run self-attention with GQA and KV-cache.
    ///
    /// For decode (seq_len=1), uses CPU to avoid Metal launch overhead.
    /// For hybrid models (Qwen3-Coder-Next), supports:
    /// - Gated attention: Q projection outputs Q+gate, output *= sigmoid(gate)
    /// - Partial RoPE: only rotary_dim fraction of head_dim gets RoPE
    fn run_attention(
        &self,
        hidden_states: &Tensor,
        layer_idx: usize,
        kv_cache: &mut KvCache,
        use_cache: bool,
        prefix: &str,
        resident: &ResidentWeights,
    ) -> Result<Tensor> {
        let (bsz, seq_len, _) = hidden_states.dims3()?;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let hidden_dim = self.config.hidden_size;
        let is_hybrid = self.config.is_deltanet_hybrid();

        // Compute device: GPU (Metal) by default on Apple Silicon (unified memory),
        // or CPU with --cpu-compute flag.
        let compute_device = if self.config.gpu_compute { self.device } else { &Device::Cpu };
        let hidden_states = hidden_states.to_device(compute_device)?;

        let flat = hidden_states.reshape((bsz * seq_len, hidden_dim))?;

        // Load or use resident attention weights
        // For hybrid models, Q projection outputs [Q, gate] with double width
        let (q_raw, k, v, o_weight, q_norm_opt, k_norm_opt) =
            if let Some(attn_w) = &resident.attention[layer_idx] {
                // o_weight already on compute_device from preload_attention()
                let o_w = attn_w.o_weight.clone();

                // Per-weight projection: MXFP4 > Q5_0/Q8_0 Metal > QMatMul (CPU) > F32 fallback
                // Each weight checked individually to support mixed quant types (e.g. Q5_0 + Q5_K).
                #[cfg(feature = "metal")]
                let (mut q, mut k, mut v) = {
                    let mx = attn_w.mxfp4.as_ref();
                    let qm = attn_w.quantized_metal.as_ref();

                    macro_rules! proj {
                        ($mx_field:ident, $qm_field:ident, $dense_w:expr) => {
                            if let Some(m) = mx.map(|m| &m.$mx_field) {
                                crate::metal::mxfp4_matmul_metal_gpu_batched(&m.buffer, &flat, m.out_features, m.in_features)?
                            } else if let Some(q) = qm.and_then(|q| q.$qm_field.as_ref()) {
                                crate::metal::quantized_attn_matmul_metal_gpu_batched(&q.buffer, &flat, q.out_features, q.in_features, q.quant_type)?
                            } else {
                                // Weight already on compute_device from preload_attention()
                                flat.matmul(&$dense_w.t()?)?
                            }
                        };
                    }
                    (
                        proj!(q, q, attn_w.q_weight),
                        proj!(k, k, attn_w.k_weight),
                        proj!(v, v, attn_w.v_weight),
                    )
                };
                #[cfg(not(feature = "metal"))]
                let (mut q, mut k, mut v) = if let Some(ref qaw) = attn_w.quantized {
                    (qaw.q.forward(&flat)?, qaw.k.forward(&flat)?, qaw.v.forward(&flat)?)
                } else {
                    // Weights already on compute_device from preload_attention()
                    (flat.matmul(&attn_w.q_weight.t()?)?, flat.matmul(&attn_w.k_weight.t()?)?, flat.matmul(&attn_w.v_weight.t()?)?)
                };

                // Apply attention biases if present (e.g. Qwen1.5-MoE, GPT-OSS)
                // Biases/norms already on compute_device from preload_attention()
                if let Some(qb) = &attn_w.q_bias {
                    q = q.broadcast_add(qb)?;
                }
                if let Some(kb) = &attn_w.k_bias {
                    k = k.broadcast_add(kb)?;
                }
                if let Some(vb) = &attn_w.v_bias {
                    v = v.broadcast_add(vb)?;
                }
                let qn = attn_w.q_norm.clone();
                let kn = attn_w.k_norm.clone();
                (q, k, v, o_w, qn, kn)
            } else {
                let q_w = load_weight(self.reader, &format!("{}.attn_q.weight", prefix), compute_device)?;
                let k_w = load_weight(self.reader, &format!("{}.attn_k.weight", prefix), compute_device)?;
                let v_w = load_weight(self.reader, &format!("{}.attn_v.weight", prefix), compute_device)?;
                let o_w = load_weight(self.reader, &format!("{}.attn_output.weight", prefix), compute_device)?;

                let mut q = flat.matmul(&q_w.t()?)?;
                let mut k = flat.matmul(&k_w.t()?)?;
                let mut v = flat.matmul(&v_w.t()?)?;

                // Apply attention biases if present (e.g. Qwen1.5-MoE)
                let q_bias_name = format!("{}.attn_q.bias", prefix);
                if self.reader.tensors.contains_key(&q_bias_name) {
                    let qb = load_weight(self.reader, &q_bias_name, compute_device)?;
                    let kb = load_weight(self.reader, &format!("{}.attn_k.bias", prefix), compute_device)?;
                    let vb = load_weight(self.reader, &format!("{}.attn_v.bias", prefix), compute_device)?;
                    q = q.broadcast_add(&qb)?;
                    k = k.broadcast_add(&kb)?;
                    v = v.broadcast_add(&vb)?;
                }

                let q_norm_name = format!("{}.attn_q_norm.weight", prefix);
                let (qn, kn) = if self.reader.tensors.contains_key(&q_norm_name) {
                    let qn = load_weight(self.reader, &q_norm_name, compute_device)?;
                    let kn = load_weight(self.reader, &format!("{}.attn_k_norm.weight", prefix), compute_device)?;
                    (Some(qn), Some(kn))
                } else {
                    (None, None)
                };

                (q, k, v, o_w, qn, kn)
            };

        // For hybrid models: split Q into query + gate
        let (q, gate) = if is_hybrid {
            // Q projection output: [batch*seq, num_heads * head_dim * 2]
            // GGUF layout is interleaved per head:
            //   [Q_h0(head_dim), gate_h0(head_dim), Q_h1(head_dim), gate_h1(head_dim), ...]
            // Reshape to [batch*seq, num_heads, 2*head_dim] then split within each head
            let q_gate = q_raw.reshape((bsz * seq_len, num_heads, 2 * head_dim))?;
            let q = q_gate.narrow(2, 0, head_dim)?.contiguous()?
                .reshape((bsz * seq_len, num_heads * head_dim))?;
            let gate = q_gate.narrow(2, head_dim, head_dim)?.contiguous()?
                .reshape((bsz * seq_len, num_heads * head_dim))?;
            (q, Some(gate))
        } else {
            (q_raw, None)
        };

        // Reshape: [batch, seq, heads, head_dim]
        let q = q.reshape((bsz, seq_len, num_heads, head_dim))?;
        let k = k.reshape((bsz, seq_len, num_kv_heads, head_dim))?;
        let v = v.reshape((bsz, seq_len, num_kv_heads, head_dim))?;

        // QK norm (Qwen3-specific)
        let (q, k) = match (q_norm_opt, k_norm_opt) {
            (Some(qn), Some(kn)) => {
                let q = ops::rms_norm(&q, &qn, self.config.rms_norm_eps as f64)?;
                let k = ops::rms_norm(&k, &kn, self.config.rms_norm_eps as f64)?;
                (q, k)
            }
            _ => (q, k),
        };

        // RoPE — use pre-computed tables (GPU when gpu_compute, CPU otherwise)
        let position_offset = if use_cache { kv_cache.seq_len(layer_idx) } else { 0 };
        let (rope_cos, rope_sin) = if self.config.gpu_compute {
            (self.cos_gpu, self.sin_gpu)
        } else {
            (self.cos_cpu, self.sin_cpu)
        };
        // For hybrid models: use partial RoPE (only rotary_dim fraction)
        let (q, k) = if is_hybrid && self.config.rotary_dim < head_dim {
            ops::partial_rotary_embedding(&q, &k, rope_cos, rope_sin, position_offset, self.config.rotary_dim)?
        } else {
            ops::rotary_embedding(&q, &k, rope_cos, rope_sin, position_offset)?
        };

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // KV-cache update
        let (k, v) = if use_cache {
            kv_cache.update(layer_idx, &k, &v)?
        } else {
            (k, v)
        };

        // Scaled dot-product attention with GQA
        // Pass attention sinks if present (GPT-OSS: per-head virtual sink in softmax denominator)
        let scale = 1.0 / (head_dim as f64).sqrt();
        let causal = seq_len > 1;
        let sinks_ref = if let Some(attn_w) = &resident.attention[layer_idx] {
            attn_w.attn_sinks.as_ref()
        } else {
            None
        };
        let swa = if self.config.is_swa_layer(layer_idx) {
            Some(self.config.sliding_window)
        } else {
            None
        };
        let attn_out = ops::scaled_dot_product_attention(&q, &k, &v, scale, causal, sinks_ref, swa)?;

        // [batch, heads, seq, head_dim] → [batch*seq, heads * head_dim]
        let mut attn_out = attn_out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bsz * seq_len, num_heads * head_dim))?;

        // For hybrid models: apply gated attention (output *= sigmoid(gate))
        if let Some(gate) = gate {
            let gate_sigmoid = ops::sigmoid(&gate)?;
            attn_out = attn_out.mul(&gate_sigmoid)?;

        }

        // Output projection: MXFP4 > Q5_0/Q8_0 Metal > QMatMul > F32
        // O projection: MXFP4 > Q5_0/Q8_0 Metal > F32 fallback (per-weight)
        #[cfg(feature = "metal")]
        let mut output = {
            let attn_ref = resident.attention[layer_idx].as_ref();
            let mxfp4_o = attn_ref.and_then(|a| a.mxfp4.as_ref()).map(|m| &m.o);
            let qmetal_o = attn_ref.and_then(|a| a.quantized_metal.as_ref()).and_then(|q| q.o.as_ref());
            if let Some(mx) = mxfp4_o {
                crate::metal::mxfp4_matmul_metal_gpu_batched(&mx.buffer, &attn_out, mx.out_features, mx.in_features)?
            } else if let Some(qm) = qmetal_o {
                crate::metal::quantized_attn_matmul_metal_gpu_batched(&qm.buffer, &attn_out, qm.out_features, qm.in_features, qm.quant_type)?
            } else {
                attn_out.matmul(&o_weight.t()?)?
            }
        };
        #[cfg(not(feature = "metal"))]
        let mut output = {
            let attn_ref = resident.attention[layer_idx].as_ref();
            if let Some(qaw) = attn_ref.and_then(|a| a.quantized.as_ref()) {
                qaw.o.forward(&attn_out)?
            } else {
                attn_out.matmul(&o_weight.t()?)?
            }
        };

        // Apply output projection bias if present (e.g. GPT-OSS)
        if let Some(attn_w) = &resident.attention[layer_idx] {
            if let Some(ob) = &attn_w.o_bias {
                // o_bias already on compute_device from preload_attention()
                output = output.broadcast_add(ob)?;
            }
        } else {
            let o_bias_name = format!("{}.attn_output.bias", prefix);
            if self.reader.tensors.contains_key(&o_bias_name) {
                let ob = load_weight(self.reader, &o_bias_name, compute_device)?;
                output = output.broadcast_add(&ob)?;
            }
        }

        // Return on original device
        let output = output.reshape((bsz, seq_len, hidden_dim))?;
        output.to_device(self.device)
    }

    /// Run MoE layer with selective expert loading and LRU cache.
    ///
    /// Uses CPU gate weight copies for routing when available, avoiding Metal sync
    /// overhead (~1.5ms/layer saved). All expert computation happens on CPU.
    #[allow(clippy::too_many_arguments)]
    fn run_moe(
        &self,
        hidden_states: &Tensor,
        layer_idx: usize,
        prefix: &str,
        expert_cache: &mut ExpertCache,
        resident: &ResidentWeights,
        layer_output_cache: &mut LayerOutputCache,
        entropy_out: &mut Option<f32>,
        routing_stats_out: &mut Option<(Vec<u32>, Vec<f32>)>,
        moe_timing_out: &mut Option<(f64, f64, f64, f64)>,
    ) -> Result<Tensor> {
        let (bsz, seq_len, hidden_dim) = hidden_states.dims3()?;
        let hidden_flat = hidden_states.reshape((bsz * seq_len, hidden_dim))?;

        let t_routing_start = std::time::Instant::now();

        // GPT-OSS uses OAI SwiGLU activation (alpha=1.702, limit=7.0)
        let use_oai_swiglu = self.config.architecture == "gpt-oss";

        // Compute device for expert matmul: Metal (GPU) or CPU
        let compute_device = if self.config.gpu_compute { self.device } else { &Device::Cpu };

        // === GPU routing optimization (GPU Resident + GPU+SSD Hybrid) ===
        // In GPU modes, avoid transferring hidden_flat (2880 floats) to CPU for routing.
        // Instead, do routing matmul on GPU and only transfer the small router_logits
        // (32 floats per token) to CPU for top-k extraction.
        // This eliminates the dominant sync bottleneck (24 syncs/step → 0 syncs for routing).
        let is_gpu_resident = self.config.inference_mode == Some(InferenceMode::GpuResident);
        let is_gpu_hybrid = self.config.inference_mode == Some(InferenceMode::GpuHybrid);
        let use_gpu_routing = is_gpu_resident || is_gpu_hybrid;

        // Hidden states on CPU for expert matmul (needed for all non-GPU-Resident paths).
        // GpuHybrid needs this for expert Q4/MXFP4 matmul on CPU.
        let hidden_cpu = if !is_gpu_resident {
            hidden_flat.to_device(&Device::Cpu)?
        } else {
            // GPU Resident: create a lazy placeholder that won't be used.
            // hidden_flat stays on GPU. We use a zero-size CPU tensor as placeholder.
            Tensor::zeros((0,), candle_core::DType::F32, &Device::Cpu)?
        };
        // Hidden states on compute_device for expert matmul.
        let hidden_compute = if self.config.gpu_compute {
            hidden_flat.to_device(compute_device)?
        } else {
            hidden_cpu.clone()
        };

        // Router: GPU modes use Metal routing (no CPU transfer).
        // CPU modes use CPU gate weights to avoid Metal sync barrier.
        let gate_name = format!("{}.ffn_gate_inp.weight", prefix);
        let router_logits = if use_gpu_routing {
            // GPU mode: route on Metal GPU. The router_logits will be a small
            // tensor (1x32 for decode) transferred to CPU only for top-k extraction.
            if let Some(gate_w) = &resident.router_gates[layer_idx] {
                hidden_flat.matmul(&gate_w.t()?)?
            } else {
                let gate_weight = load_weight(self.reader, &gate_name, self.device)?;
                let logits = hidden_flat.matmul(&gate_weight.t()?)?;
                drop(gate_weight);
                logits
            }
        } else if let Some(gate_w_cpu) = &resident.router_gates_cpu[layer_idx] {
            // CPU routing: zero Metal sync (hidden_cpu and gate_w_cpu both on CPU)
            hidden_cpu.matmul(&gate_w_cpu.t()?)?
        } else if let Some(gate_w) = &resident.router_gates[layer_idx] {
            // Fallback: Metal routing
            hidden_flat.matmul(&gate_w.t()?)?
        } else {
            let gate_weight = load_weight(self.reader, &gate_name, self.device)?;
            let logits = hidden_flat.matmul(&gate_weight.t()?)?;
            drop(gate_weight);
            logits
        };

        // Apply router gate bias if present (e.g. GPT-OSS ffn_gate_inp.bias)
        let router_logits = if let Some(bias) = &resident.router_gate_biases[layer_idx] {
            let bias = bias.to_device(router_logits.device())?;
            router_logits.broadcast_add(&bias)?
        } else {
            router_logits
        };

        // === GPU-routed zero-sync fast path ===
        // When conditions are met, perform softmax+topk+MoE entirely on GPU
        // with zero CPU sync (eliminates 24 GPU syncs/token → 0).
        // Conditions: GPU Resident + single token + fixed K + MXFP4 packed.
        // Profiling (entropy_out/routing_stats_out) is handled post-hoc with a single
        // GPU→CPU sync — still far cheaper than the ~480 dispatch fallback path.
        #[cfg(feature = "metal")]
        if is_gpu_resident
            && bsz * seq_len == 1
            && !self.config.dynamic_k_enabled
            && !self.config.adaptive_skip_enabled
        {
            if let (Some(packed), Some(routing_offsets), Some(routing_out)) = (
                &resident.packed_mxfp4[layer_idx],
                &resident.gpu_routing_offsets[layer_idx],
                &resident.gpu_routing_out,
            ) {
                let n_experts_actual = self.config.experts_for_layer(layer_idx);
                let softmax_weight = self.config.architecture == "gpt-oss";
                let bias_buffers = resident.expert_bias_buffers[layer_idx].as_ref();
                let moe_buffers = resident.batched_moe_buffers.as_ref();

                let moe_output = crate::metal::gpu_routed_moe_forward_metal(
                    self.device,
                    &router_logits,
                    packed,
                    routing_offsets,
                    routing_out,
                    &hidden_flat,
                    self.config.num_experts_per_tok,
                    n_experts_actual,
                    use_oai_swiglu,
                    softmax_weight,
                    self.config.norm_topk_prob,
                    1.702,  // alpha for OAI SwiGLU
                    7.0,    // limit for OAI SwiGLU
                    bias_buffers,
                    moe_buffers,
                )?;

                // Post-hoc profiling: read routing data from GPU when requested.
                // This adds 1 GPU→CPU sync (router_logits readback) only when profiling
                // is active — normal inference remains zero-sync.
                if entropy_out.is_some() || routing_stats_out.is_some() {
                    let top_k = self.config.num_experts_per_tok;
                    // Single GPU→CPU sync: read the small router logits vector
                    let logits_vec: Vec<f32> = router_logits.flatten_all()?.to_vec1::<f32>()?;
                    let logits_cpu = Tensor::from_vec(
                        logits_vec, (1, n_experts_actual), &Device::Cpu,
                    )?;

                    if entropy_out.is_some() {
                        *entropy_out = Some(compute_router_entropy(&logits_cpu)?);
                    }

                    if routing_stats_out.is_some() {
                        // Compute top-k on CPU from logits (same as fallback path).
                        // The GPU already computed top-k for MoE dispatch, but reading
                        // Metal buffers directly requires unsafe code. CPU top-k on 32
                        // experts is negligible (~100ns).
                        let (topk_w, topk_i) = top_k_routing(
                            &logits_cpu, top_k, self.config.norm_topk_prob, use_oai_swiglu,
                        )?;
                        let indices = topk_i.flatten_all()?.to_vec1::<u32>()?;
                        let weights = topk_w.flatten_all()?.to_vec1::<f32>()?;
                        *routing_stats_out = Some((indices, weights));
                    }
                }

                // Handle shared expert (if any)
                let mut output = moe_output;
                if self.config.has_shared_expert {
                    let shexp_gate_name = format!("{}.ffn_gate_shexp.weight", prefix);
                    let has_shexp = resident.shared_experts[layer_idx].is_some()
                        || self.reader.tensors.contains_key(&shexp_gate_name);
                    if has_shexp {
                        let shexp_out = Self::run_shared_expert(
                            &hidden_flat, prefix, resident, self.reader, self.device,
                            layer_idx, self.config.has_shared_expert,
                        )?;
                        output = (output + shexp_out)?;
                    }
                }

                let result = output.reshape((bsz, seq_len, hidden_dim))?;

                if let Some(ref mut timing) = moe_timing_out {
                    let total_ms = t_routing_start.elapsed().as_secs_f64() * 1000.0;
                    *timing = (total_ms, 0.0, 0.0, 0.0);
                }

                return Ok(result);
            }
        }

        // === GPU routing: single sync via to_vec1 ===
        // Transfer small router_logits to CPU as Vec<f32> in one sync.
        // to_vec1 is faster than to_device(CPU) — avoids creating a CPU Tensor wrapper.
        // All downstream consumers (adaptive skip, entropy, top-k) work on CPU vecs.
        let router_logits_vec: Vec<f32> = router_logits.flatten_all()?.to_vec1::<f32>()?;
        let n_experts_actual = self.config.experts_for_layer(layer_idx);
        // Reconstruct as CPU tensor for downstream compatibility (lightweight — no GPU sync)
        let router_logits = Tensor::from_vec(
            router_logits_vec.clone(),
            (bsz * seq_len, n_experts_actual),
            &Device::Cpu,
        )?;

        // === Adaptive Expert Skip ===
        // Compare router logits between consecutive tokens. If the routing decision
        // is very similar, skip MoE computation and reuse the previous output.
        // Safety: max_consecutive_skips prevents unbounded error accumulation.
        // Never skip layer 0 (critical for representation) or last layer (output quality).
        let skip_eligible = self.config.adaptive_skip_enabled
            && layer_idx > 0
            && layer_idx < self.config.num_layers - 1;
        let skip_logits_vec = if skip_eligible {
            let logits_vec = router_logits.flatten_all()?.to_vec1::<f32>()?;
            layer_output_cache.total_count += 1;

            if layer_output_cache.should_skip(layer_idx, &logits_vec) {
                // Clone the cached output before mutating the cache
                let cached_output = layer_output_cache.get_cached_output(layer_idx).unwrap().clone();
                layer_output_cache.record_skip(layer_idx);
                log::trace!("Skip L{}: reusing cached MoE output", layer_idx);
                let output = Tensor::from_vec(
                    cached_output,
                    (bsz, seq_len, hidden_dim),
                    self.device,
                )?;
                // Update logits for next comparison (track drift)
                layer_output_cache.update_logits(layer_idx, logits_vec);
                return Ok(output);
            }
            layer_output_cache.record_compute(layer_idx);
            Some(logits_vec)
        } else {
            None
        };
        // === End Adaptive Expert Skip ===

        // Compute router entropy (always when profiling or Dynamic K is enabled)
        let should_compute_entropy = self.config.dynamic_k_enabled || entropy_out.is_some();
        let entropy_val = if should_compute_entropy {
            Some(compute_router_entropy(&router_logits)?)
        } else {
            None
        };

        // Write entropy to output for profiling
        if let Some(h) = entropy_val {
            *entropy_out = Some(h);
        }

        // Dynamic K: compute per-layer K from router entropy, or use fixed K
        let top_k = if self.config.dynamic_k_enabled {
            let entropy = entropy_val.unwrap();
            let (layer_k_min, layer_k_max) = if self.config.layer_adaptive_k {
                self.config.get_layer_k_range(layer_idx)
            } else {
                (self.config.dynamic_k_min, self.config.effective_k_max())
            };
            let k = entropy_to_k(
                entropy,
                self.config.experts_for_layer(layer_idx) as f32,
                layer_k_min,
                layer_k_max,
            );
            log::trace!("Dynamic K L{}: entropy={:.3}, K={} (range={}-{})", layer_idx, entropy, k, layer_k_min, layer_k_max);
            k
        } else {
            self.config.num_experts_per_tok
        };
        // Mask dummy (padded) experts to -∞ so they are never selected by top-k.
        // Dummy experts have zero/uniform gate weights and produce garbage output.
        // Post-matmul masking is the only reliable approach — constant gate weights
        // can't guarantee exclusion when sum(hidden_state) < 0 flips the sign.
        let router_logits = if !resident.dummy_experts[layer_idx].is_empty() {
            let (_n_tok, n_exp) = router_logits.dims2()?;
            let mut mask_data = vec![0.0f32; n_exp];
            for &dummy_idx in &resident.dummy_experts[layer_idx] {
                mask_data[dummy_idx] = f32::NEG_INFINITY;
            }
            let mask = Tensor::from_vec(mask_data, (1, n_exp), router_logits.device())?;
            (router_logits + mask)?
        } else {
            router_logits
        };
        let (topk_weights, topk_indices) = top_k_routing(&router_logits, top_k, self.config.norm_topk_prob, use_oai_swiglu)?;

        drop(router_logits);

        // Extract flat indices and weights once (avoids per-expert tensor ops).
        // top_k_routing always returns CPU tensors, so no GPU sync here.
        let indices_flat = topk_indices.flatten_all()?.to_vec1::<u32>()?;
        let weights_flat = topk_weights.flatten_all()?.to_vec1::<f32>()?;

        // === Activation dump for VQ KD ===
        // When DUMP_ACTIVATIONS=/path/to/dir is set, save pre-MoE hidden states
        // and routing decisions (expert indices + weights) for knowledge distillation.
        // Format: layer_LL_hidden.bin  = appended f32[hidden_dim] per token
        //         layer_LL_routing.bin = appended (u32[top_k] indices + f32[top_k] weights) per token
        {
            // Read env var every time (not OnceLock) so handle_train_step can
            // dynamically enable/disable activation dumping via set_var/remove_var.
            let dump_dir = std::env::var("DUMP_ACTIVATIONS").ok();
            if let Some(ref dir) = dump_dir {
                use std::io::Write;
                // Write metadata once (on first layer of first token)
                if layer_idx == 0 {
                    let meta_path = format!("{}/meta.json", dir);
                    if !std::path::Path::new(&meta_path).exists() {
                        let meta = format!(
                            "{{\"hidden_dim\":{},\"top_k\":{},\"n_experts\":{},\"n_layers\":{}}}",
                            hidden_dim, top_k, n_experts_actual, self.config.num_layers
                        );
                        std::fs::write(&meta_path, meta)?;
                    }
                }
                // Dump hidden states (pre-MoE, post-attention+layernorm)
                let hidden_data = hidden_cpu.flatten_all()?.to_vec1::<f32>()?;
                let hidden_bytes: Vec<u8> = hidden_data.iter()
                    .flat_map(|v| v.to_le_bytes()).collect();
                let hidden_path = format!("{}/layer_{:02}_hidden.bin", dir, layer_idx);
                let mut f = std::fs::OpenOptions::new()
                    .create(true).append(true).open(&hidden_path)?;
                f.write_all(&hidden_bytes)?;

                // Dump routing per-token: [top_k u32 indices][top_k f32 weights] per token
                let routing_path = format!("{}/layer_{:02}_routing.bin", dir, layer_idx);
                let mut f = std::fs::OpenOptions::new()
                    .create(true).append(true).open(&routing_path)?;
                let n_tok = bsz * seq_len;
                for t in 0..n_tok {
                    let start = t * top_k;
                    let end = start + top_k;
                    let idx_bytes: Vec<u8> = indices_flat[start..end].iter()
                        .flat_map(|v| v.to_le_bytes()).collect();
                    let wgt_bytes: Vec<u8> = weights_flat[start..end].iter()
                        .flat_map(|v| v.to_le_bytes()).collect();
                    f.write_all(&idx_bytes)?;
                    f.write_all(&wgt_bytes)?;
                }
            }
        }

        // Export routing stats for calibration profiling
        if routing_stats_out.is_some() {
            *routing_stats_out = Some((indices_flat.clone(), weights_flat.clone()));
        }

        let num_tokens = bsz * seq_len;

        // Pre-compute expert→(token_idx, weight) assignment map
        // Replaces per-expert get_expert_assignment() which does expensive tensor indexing
        let mut expert_assignment: std::collections::HashMap<usize, Vec<(usize, f32)>> =
            std::collections::HashMap::new();
        for t in 0..num_tokens {
            for k in 0..top_k {
                let flat_idx = t * top_k + k;
                let expert_idx = indices_flat[flat_idx] as usize;
                let weight = weights_flat[flat_idx];
                expert_assignment.entry(expert_idx).or_default().push((t, weight));
            }
        }

        let mut unique_experts: Vec<usize> = expert_assignment.keys().copied().collect();
        unique_experts.sort();

        let routing_ms = t_routing_start.elapsed().as_secs_f64() * 1000.0;

        // === GPU Resident path: all expert weights already on Metal GPU ===
        if is_gpu_resident {
            let t_gpu_compute = std::time::Instant::now();
            let result = self.run_moe_gpu_resident(
                &hidden_flat, layer_idx, prefix, resident,
                &expert_assignment, &unique_experts,
                num_tokens, hidden_dim, bsz, seq_len, use_oai_swiglu,
                skip_logits_vec,
                layer_output_cache,
            )?;
            if let Some(ref mut timing) = moe_timing_out {
                let gpu_compute_ms = t_gpu_compute.elapsed().as_secs_f64() * 1000.0;
                *timing = (routing_ms, 0.0, gpu_compute_ms, 0.0);
            }
            return Ok(result);
        }

        // Keep accumulation buffer on CPU to avoid repeated GPU↔CPU transfers
        let mut output_data = vec![0.0f32; num_tokens * hidden_dim];

        let gate_exps_name = format!("{}.ffn_gate_exps.weight", prefix);
        let up_exps_name = format!("{}.ffn_up_exps.weight", prefix);
        let down_exps_name = format!("{}.ffn_down_exps.weight", prefix);

        // Expert FFN biases (e.g. GPT-OSS: ffn_{gate,up,down}_exps.bias)
        let gate_bias_name = format!("{}.ffn_gate_exps.bias", prefix);
        let up_bias_name = format!("{}.ffn_up_exps.bias", prefix);
        let down_bias_name = format!("{}.ffn_down_exps.bias", prefix);
        let has_expert_bias = self.reader.tensors.contains_key(&gate_bias_name);

        let ram_resident = self.config.ram_resident;
        let nocache = use_nocache_experts();
        let quantized_mm = use_quantized_matmul();
        let do_evict = use_expert_eviction(self.config.evict_experts) && !nocache;

        // For mmap path: batch madvise(WILLNEED) for all selected experts
        if !ram_resident && !nocache {
            for &expert_idx in &unique_experts {
                self.reader.prefetch_expert_slice(&gate_exps_name, expert_idx);
                self.reader.prefetch_expert_slice(&up_exps_name, expert_idx);
                self.reader.prefetch_expert_slice(&down_exps_name, expert_idx);
            }
        }

        // === Phase 1: Load expert weights (parallel for GGUF, instant for resident/cached) ===
        let t_io = std::time::Instant::now();
        let is_single_token = num_tokens == 1;

        // Separate experts into already-available (resident/cached) and needs-load
        struct LoadedExpert {
            idx: usize,
            gate: Tensor,
            up: Tensor,
            down: Tensor,
        }
        let mut loaded_experts: Vec<LoadedExpert> = Vec::with_capacity(unique_experts.len());
        let mut needs_gguf_load: Vec<usize> = Vec::new();

        for &expert_idx in &unique_experts {
            if let Some(ew) = &resident.experts[layer_idx][expert_idx] {
                loaded_experts.push(LoadedExpert {
                    idx: expert_idx,
                    gate: ew.gate.clone(), up: ew.up.clone(), down: ew.down.clone(),
                });
            } else if let Some(cached) = expert_cache.get(layer_idx, expert_idx) {
                loaded_experts.push(LoadedExpert {
                    idx: expert_idx,
                    gate: cached.gate.clone(), up: cached.up.clone(), down: cached.down.clone(),
                });
            } else {
                needs_gguf_load.push(expert_idx);
            }
        }

        // === VQ path: codebook lookup → F32 weight → standard matmul ===
        // For VQ models, expert weights are stored as (codebook, indices) pairs.
        // We reconstruct full F32 weight matrices via codebook lookup, then do
        // standard matmul. Supports both shared (pre-loaded) and per-expert (on-demand)
        // codebook modes.
        let is_vq = self.vq_config.is_some() && (self.vq_codebooks.is_some() || self.vq_per_expert);

        if is_vq && !needs_gguf_load.is_empty() {
            let vq_cbs = self.vq_codebooks.as_ref(); // None for per-expert mode
            let vq_cfg = self.vq_config.as_ref().unwrap();
            let vq_pe = self.vq_per_expert;
            let reader = self.reader;
            let cpu = &Device::Cpu;

            let gate_idx_name = format!("{}.ffn_gate_vq_idx.weight", prefix);
            let up_idx_name = format!("{}.ffn_up_vq_idx.weight", prefix);
            let down_idx_name = format!("{}.ffn_down_vq_idx.weight", prefix);
            // Per-expert codebook tensor names (only used in per-expert mode)
            let gate_cb_name = format!("{}.ffn_gate_vq_cb.weight", prefix);
            let up_cb_name = format!("{}.ffn_up_vq_cb.weight", prefix);
            let down_cb_name = format!("{}.ffn_down_vq_cb.weight", prefix);

            // Get weight dimensions from config
            let intermediate_size = self.config.moe_intermediate_size;

            // Helper: load one expert's 3 weight matrices (gate, up, down)
            // Dispatches to shared or per-expert loader based on vq_pe flag.
            macro_rules! load_expert_weights {
                ($expert_idx:expr) => {{
                    if vq_pe {
                        let gate_w = load_expert_vq_per_expert(
                            reader, vq_cfg, &gate_cb_name, &gate_idx_name,
                            $expert_idx, (intermediate_size, hidden_dim),
                        )?;
                        let up_w = load_expert_vq_per_expert(
                            reader, vq_cfg, &up_cb_name, &up_idx_name,
                            $expert_idx, (intermediate_size, hidden_dim),
                        )?;
                        let down_w = load_expert_vq_per_expert(
                            reader, vq_cfg, &down_cb_name, &down_idx_name,
                            $expert_idx, (hidden_dim, intermediate_size),
                        )?;
                        (gate_w, up_w, down_w)
                    } else {
                        let cbs = vq_cbs.unwrap();
                        let gate_w = load_expert_vq(
                            reader, &cbs[layer_idx][0], vq_cfg,
                            &gate_idx_name, $expert_idx, (intermediate_size, hidden_dim),
                        )?;
                        let up_w = load_expert_vq(
                            reader, &cbs[layer_idx][1], vq_cfg,
                            &up_idx_name, $expert_idx, (intermediate_size, hidden_dim),
                        )?;
                        let down_w = load_expert_vq(
                            reader, &cbs[layer_idx][2], vq_cfg,
                            &down_idx_name, $expert_idx, (hidden_dim, intermediate_size),
                        )?;
                        (gate_w, up_w, down_w)
                    }
                }};
            }

            let hidden_ref = &hidden_cpu;

            if is_single_token {
                // Single-token VQ decode: parallel expert load+compute
                let expert_results: Vec<Result<(Vec<f32>, f32)>> = std::thread::scope(|s| {
                    let handles: Vec<_> = needs_gguf_load.iter().filter_map(|&expert_idx| {
                        let assignments = expert_assignment.get(&expert_idx)?;
                        if assignments.is_empty() { return None; }
                        let total_weight: f32 = assignments.iter().map(|(_, w)| w).sum();
                        let gi = &gate_idx_name;
                        let ui = &up_idx_name;
                        let di = &down_idx_name;
                        let gc = &gate_cb_name;
                        let uc = &up_cb_name;
                        let dc = &down_cb_name;
                        Some(s.spawn(move || -> Result<(Vec<f32>, f32)> {
                            let (gate_w, up_w, down_w) = if vq_pe {
                                let gw = load_expert_vq_per_expert(
                                    reader, vq_cfg, gc, gi,
                                    expert_idx, (intermediate_size, hidden_dim),
                                )?;
                                let uw = load_expert_vq_per_expert(
                                    reader, vq_cfg, uc, ui,
                                    expert_idx, (intermediate_size, hidden_dim),
                                )?;
                                let dw = load_expert_vq_per_expert(
                                    reader, vq_cfg, dc, di,
                                    expert_idx, (hidden_dim, intermediate_size),
                                )?;
                                (gw, uw, dw)
                            } else {
                                let cbs = vq_cbs.unwrap();
                                let gw = load_expert_vq(
                                    reader, &cbs[layer_idx][0], vq_cfg,
                                    gi, expert_idx, (intermediate_size, hidden_dim),
                                )?;
                                let uw = load_expert_vq(
                                    reader, &cbs[layer_idx][1], vq_cfg,
                                    ui, expert_idx, (intermediate_size, hidden_dim),
                                )?;
                                let dw = load_expert_vq(
                                    reader, &cbs[layer_idx][2], vq_cfg,
                                    di, expert_idx, (hidden_dim, intermediate_size),
                                )?;
                                (gw, uw, dw)
                            };

                            // Standard F32 matmul: hidden @ gate^T, hidden @ up^T
                            let gate_out = hidden_ref.matmul(&gate_w.t()?)?;
                            let up_out = hidden_ref.matmul(&up_w.t()?)?;
                            let expert_hidden = if use_oai_swiglu {
                                ops::swiglu_oai(&gate_out, &up_out, 1.702, 7.0)?
                            } else {
                                ops::silu_and_mul(&gate_out, &up_out)?
                            };
                            let expert_output = expert_hidden.matmul(&down_w.t()?)?;

                            let out_data = expert_output.flatten_all()?.to_vec1::<f32>()?;
                            Ok((out_data, total_weight))
                        }))
                    }).collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });

                for result in expert_results {
                    let (out_data, total_weight) = result?;
                    for j in 0..hidden_dim {
                        output_data[j] += out_data[j] * total_weight;
                    }
                }
            } else {
                // Multi-token VQ prefill: serial per-expert
                for &expert_idx in &needs_gguf_load {
                    let assignments = match expert_assignment.get(&expert_idx) {
                        Some(a) if !a.is_empty() => a,
                        _ => continue,
                    };
                    let active_tokens: Vec<u32> = assignments.iter().map(|(t, _)| *t as u32).collect();
                    let token_weights: Vec<f32> = assignments.iter().map(|(_, w)| *w).collect();

                    let indices_t = Tensor::from_vec(active_tokens.clone(), (active_tokens.len(),), cpu)?;
                    let expert_input = hidden_cpu.index_select(&indices_t, 0)?;

                    let (gate_w, up_w, down_w) = load_expert_weights!(expert_idx);

                    let gate_out = expert_input.matmul(&gate_w.t()?)?;
                    let up_out = expert_input.matmul(&up_w.t()?)?;
                    let expert_hidden = if use_oai_swiglu {
                        ops::swiglu_oai(&gate_out, &up_out, 1.702, 7.0)?
                    } else {
                        ops::silu_and_mul(&gate_out, &up_out)?
                    };
                    let expert_output = expert_hidden.matmul(&down_w.t()?)?;
                    let out_vec = expert_output.flatten_all()?.to_vec1::<f32>()?;

                    for (i, &tok_idx) in active_tokens.iter().enumerate() {
                        let w = token_weights[i];
                        let dst_start = tok_idx as usize * hidden_dim;
                        let src_start = i * hidden_dim;
                        for j in 0..hidden_dim {
                            output_data[dst_start + j] += out_vec[src_start + j] * w;
                        }
                    }
                }
            }
            // Clear needs_gguf_load since VQ handled everything
            needs_gguf_load.clear();
        }

        // Parallel dequantization from GGUF using scoped threads.
        // Each thread dequantizes one expert's 3 weight matrices (gate, up, down).
        // M4 Pro has 12 cores → 10 concurrent dequants ≈ 10× speedup.
        //
        // When quantized_mm is enabled, GGUF experts are loaded as QMatMul and
        // computed inline (Phase 1+2 merged), skipping dequantization entirely.
        if !needs_gguf_load.is_empty() {
            let reader = self.reader;
            let gate_name = &gate_exps_name;
            let up_name = &up_exps_name;
            let down_name = &down_exps_name;

            if quantized_mm {
                // Detect MXFP4 expert quantization (GPT-OSS uses MXFP4)
                let is_mxfp4 = reader.tensors.get(gate_name)
                    .map(|info| info.quant_type == GgmlQuantType::MXFP4)
                    .unwrap_or(false);

                if is_mxfp4 {
                    // === MXFP4 fused matmul path: raw bytes → dot product (no F32 intermediate) ===
                    // Each expert's MXFP4 data is read directly from mmap and the dequant
                    // is fused into the matmul, keeping experts in 4-bit format (~9.5 GB).
                    let cpu = &Device::Cpu;
                    let hidden_vec = hidden_cpu.flatten_all()?.to_vec1::<f32>()?;

                    for &expert_idx in &needs_gguf_load {
                        let assignments = match expert_assignment.get(&expert_idx) {
                            Some(a) if !a.is_empty() => a,
                            _ => continue,
                        };

                        // Get raw MXFP4 bytes for each weight matrix (no dequant, no copy)
                        let (gate_data, gate_shape) = reader.expert_slice_data(gate_name, expert_idx)
                            .map_err(|e| candle_core::Error::Msg(format!("MXFP4 gate: {}", e)))?;
                        let (up_data, _up_shape) = reader.expert_slice_data(up_name, expert_idx)
                            .map_err(|e| candle_core::Error::Msg(format!("MXFP4 up: {}", e)))?;
                        let (down_data, down_shape) = reader.expert_slice_data(down_name, expert_idx)
                            .map_err(|e| candle_core::Error::Msg(format!("MXFP4 down: {}", e)))?;

                        // Load expert biases if present (small F32 tensors)
                        let (gate_bias_vec, up_bias_vec, down_bias_vec) = if has_expert_bias {
                            let gb = load_expert(reader, &gate_bias_name, expert_idx, cpu)?
                                .flatten_all()?.to_vec1::<f32>()?;
                            let ub = load_expert(reader, &up_bias_name, expert_idx, cpu)?
                                .flatten_all()?.to_vec1::<f32>()?;
                            let db = load_expert(reader, &down_bias_name, expert_idx, cpu)?
                                .flatten_all()?.to_vec1::<f32>()?;
                            (Some(gb), Some(ub), Some(db))
                        } else {
                            (None, None, None)
                        };

                        // gate/up shape: [intermediate_dim, hidden_dim]
                        // down shape: [hidden_dim, intermediate_dim]
                        let gate_out_dim = gate_shape[0];
                        let gate_in_dim = gate_shape[1];
                        let down_out_dim = down_shape[0];
                        let down_in_dim = down_shape[1];

                        if is_single_token {
                            let total_weight: f32 = assignments.iter().map(|(_, w)| w).sum();

                            // Fused MXFP4 matmul: output = W @ input
                            let mut gate_out = mxfp4_matvec_mul(gate_data, &hidden_vec, gate_out_dim, gate_in_dim);
                            let mut up_out = mxfp4_matvec_mul(up_data, &hidden_vec, gate_out_dim, gate_in_dim);

                            // Add biases
                            if let Some(ref gb) = gate_bias_vec {
                                for (g, b) in gate_out.iter_mut().zip(gb.iter()) { *g += b; }
                            }
                            if let Some(ref ub) = up_bias_vec {
                                for (u, b) in up_out.iter_mut().zip(ub.iter()) { *u += b; }
                            }

                            // SwiGLU activation
                            let gate_t = Tensor::from_vec(gate_out, (1, gate_out_dim), cpu)?;
                            let up_t = Tensor::from_vec(up_out, (1, gate_out_dim), cpu)?;
                            let expert_hidden = if use_oai_swiglu {
                                ops::swiglu_oai(&gate_t, &up_t, 1.702, 7.0)?
                            } else {
                                ops::silu_and_mul(&gate_t, &up_t)?
                            };

                            // Down projection: MXFP4 matmul
                            let hidden_vec_mid = expert_hidden.flatten_all()?.to_vec1::<f32>()?;
                            let mut expert_out = mxfp4_matvec_mul(down_data, &hidden_vec_mid, down_out_dim, down_in_dim);

                            if let Some(ref db) = down_bias_vec {
                                for (o, b) in expert_out.iter_mut().zip(db.iter()) { *o += b; }
                            }

                            for j in 0..hidden_dim {
                                output_data[j] += expert_out[j] * total_weight;
                            }
                        } else {
                            // Multi-token (prefill) path
                            let active_tokens: Vec<u32> = assignments.iter().map(|(t, _)| *t as u32).collect();
                            let token_weights: Vec<f32> = assignments.iter().map(|(_, w)| *w).collect();
                            let n_active = active_tokens.len();

                            // Gather active token hidden states into flat buffer
                            let mut input_flat = vec![0.0f32; n_active * hidden_dim];
                            let full_hidden = hidden_cpu.flatten_all()?.to_vec1::<f32>()?;
                            for (i, &tok_idx) in active_tokens.iter().enumerate() {
                                let src = tok_idx as usize * hidden_dim;
                                input_flat[i * hidden_dim..(i + 1) * hidden_dim]
                                    .copy_from_slice(&full_hidden[src..src + hidden_dim]);
                            }

                            // Batched MXFP4 matmul
                            let mut gate_out = mxfp4_matmul(gate_data, &input_flat, n_active, gate_out_dim, gate_in_dim);
                            let mut up_out = mxfp4_matmul(up_data, &input_flat, n_active, gate_out_dim, gate_in_dim);

                            // Add biases (broadcast across tokens)
                            if let Some(ref gb) = gate_bias_vec {
                                for t in 0..n_active {
                                    for (j, b) in gb.iter().enumerate() {
                                        gate_out[t * gate_out_dim + j] += b;
                                    }
                                }
                            }
                            if let Some(ref ub) = up_bias_vec {
                                for t in 0..n_active {
                                    for (j, b) in ub.iter().enumerate() {
                                        up_out[t * gate_out_dim + j] += b;
                                    }
                                }
                            }

                            // SwiGLU activation (via Tensor ops)
                            let gate_t = Tensor::from_vec(gate_out, (n_active, gate_out_dim), cpu)?;
                            let up_t = Tensor::from_vec(up_out, (n_active, gate_out_dim), cpu)?;
                            let expert_hidden = if use_oai_swiglu {
                                ops::swiglu_oai(&gate_t, &up_t, 1.702, 7.0)?
                            } else {
                                ops::silu_and_mul(&gate_t, &up_t)?
                            };

                            // Down projection: batched MXFP4 matmul
                            let mid_vec = expert_hidden.flatten_all()?.to_vec1::<f32>()?;
                            let mut expert_out = mxfp4_matmul(down_data, &mid_vec, n_active, down_out_dim, down_in_dim);

                            if let Some(ref db) = down_bias_vec {
                                for t in 0..n_active {
                                    for (j, b) in db.iter().enumerate() {
                                        expert_out[t * down_out_dim + j] += b;
                                    }
                                }
                            }

                            // Scale by routing weights and accumulate
                            for (i, &tok_idx) in active_tokens.iter().enumerate() {
                                let w = token_weights[i];
                                let dst_start = tok_idx as usize * hidden_dim;
                                let src_start = i * hidden_dim;
                                for j in 0..hidden_dim {
                                    output_data[dst_start + j] += expert_out[src_start + j] * w;
                                }
                            }
                        }
                    }
                } else {
                    // === Q4 quantized matmul path: load Q4 → QMatMul → compute directly ===
                    // No dequantization, no F32 intermediate, no caching.
                    //
                    // SSD Streaming mode: parallel expert load+compute via thread::scope.
                    // Multiple experts trigger concurrent NVMe page faults, saturating
                    // SSD bandwidth (M4 Pro: 12 cores, NVMe supports high queue depth).
                    //
                    // RAM Resident mode: serial path (data already in memory, thread
                    // management overhead outweighs benefit).
                    let cpu = &Device::Cpu;
                    let use_parallel_q4 = !ram_resident && needs_gguf_load.len() >= 2;

                    if use_parallel_q4 && is_single_token {
                        // === SSD parallel single-token decode (hot path) ===
                        // Each thread loads one expert from SSD (page faults) and computes
                        // gate/up/down matmuls. Concurrent page faults across K experts
                        // saturate NVMe I/O bandwidth.
                        let hidden_ref = &hidden_cpu;
                        let gb_name = &gate_bias_name;
                        let ub_name = &up_bias_name;
                        let db_name = &down_bias_name;

                        let expert_results: Vec<Result<(Vec<f32>, f32)>> = std::thread::scope(|s| {
                            let handles: Vec<_> = needs_gguf_load.iter().filter_map(|&expert_idx| {
                                let assignments = expert_assignment.get(&expert_idx)?;
                                if assignments.is_empty() { return None; }
                                let total_weight: f32 = assignments.iter().map(|(_, w)| w).sum();
                                Some(s.spawn(move || -> Result<(Vec<f32>, f32)> {
                                    let cpu = &Device::Cpu;
                                    let gate_q = load_expert_quantized(reader, gate_name, expert_idx, cpu)?;
                                    let up_q = load_expert_quantized(reader, up_name, expert_idx, cpu)?;
                                    let down_q = load_expert_quantized(reader, down_name, expert_idx, cpu)?;

                                    let (gate_bias, up_bias, down_bias) = if has_expert_bias {
                                        let gb = load_expert(reader, gb_name, expert_idx, cpu)?;
                                        let ub = load_expert(reader, ub_name, expert_idx, cpu)?;
                                        let db = load_expert(reader, db_name, expert_idx, cpu)?;
                                        (Some(gb), Some(ub), Some(db))
                                    } else {
                                        (None, None, None)
                                    };

                                    let mut gate_out = gate_q.forward(hidden_ref)?;
                                    let mut up_out = up_q.forward(hidden_ref)?;
                                    if let Some(ref gb) = gate_bias {
                                        gate_out = gate_out.broadcast_add(gb)?;
                                    }
                                    if let Some(ref ub) = up_bias {
                                        up_out = up_out.broadcast_add(ub)?;
                                    }
                                    let expert_hidden = if use_oai_swiglu {
                                        ops::swiglu_oai(&gate_out, &up_out, 1.702, 7.0)?
                                    } else {
                                        ops::silu_and_mul(&gate_out, &up_out)?
                                    };
                                    let mut expert_output = down_q.forward(&expert_hidden)?;
                                    if let Some(ref db) = down_bias {
                                        expert_output = expert_output.broadcast_add(db)?;
                                    }
                                    let out_data = expert_output.flatten_all()?.to_vec1::<f32>()?;
                                    Ok((out_data, total_weight))
                                }))
                            }).collect();
                            handles.into_iter().map(|h| h.join().unwrap()).collect()
                        });
                        for result in expert_results {
                            let (out_data, total_weight) = result?;
                            for j in 0..hidden_dim {
                                output_data[j] += out_data[j] * total_weight;
                            }
                        }
                    } else if use_parallel_q4 {
                        // === SSD parallel multi-token (prefill) ===
                        let hidden_ref = &hidden_cpu;
                        let gb_name = &gate_bias_name;
                        let ub_name = &up_bias_name;
                        let db_name = &down_bias_name;
                        let hd = hidden_dim;

                        struct ExpertContrib {
                            token_offsets: Vec<usize>,
                            data: Vec<f32>,
                        }
                        let expert_results: Vec<Result<ExpertContrib>> = std::thread::scope(|s| {
                            let handles: Vec<_> = needs_gguf_load.iter().filter_map(|&expert_idx| {
                                let assignments = expert_assignment.get(&expert_idx)?;
                                if assignments.is_empty() { return None; }
                                let active_tokens: Vec<u32> = assignments.iter().map(|(t, _)| *t as u32).collect();
                                let token_weights: Vec<f32> = assignments.iter().map(|(_, w)| *w).collect();
                                Some(s.spawn(move || -> Result<ExpertContrib> {
                                    let cpu = &Device::Cpu;
                                    let gate_q = load_expert_quantized(reader, gate_name, expert_idx, cpu)?;
                                    let up_q = load_expert_quantized(reader, up_name, expert_idx, cpu)?;
                                    let down_q = load_expert_quantized(reader, down_name, expert_idx, cpu)?;
                                    let (gate_bias, up_bias, down_bias) = if has_expert_bias {
                                        let gb = load_expert(reader, gb_name, expert_idx, cpu)?;
                                        let ub = load_expert(reader, ub_name, expert_idx, cpu)?;
                                        let db = load_expert(reader, db_name, expert_idx, cpu)?;
                                        (Some(gb), Some(ub), Some(db))
                                    } else {
                                        (None, None, None)
                                    };
                                    let indices_tensor = Tensor::from_vec(
                                        active_tokens.clone(), (active_tokens.len(),), cpu)?;
                                    let expert_input = hidden_ref.index_select(&indices_tensor, 0)?;
                                    let mut gate_out = gate_q.forward(&expert_input)?;
                                    let mut up_out = up_q.forward(&expert_input)?;
                                    if let Some(ref gb) = gate_bias {
                                        gate_out = gate_out.broadcast_add(gb)?;
                                    }
                                    if let Some(ref ub) = up_bias {
                                        up_out = up_out.broadcast_add(ub)?;
                                    }
                                    let expert_hidden = if use_oai_swiglu {
                                        ops::swiglu_oai(&gate_out, &up_out, 1.702, 7.0)?
                                    } else {
                                        ops::silu_and_mul(&gate_out, &up_out)?
                                    };
                                    let mut expert_output = down_q.forward(&expert_hidden)?;
                                    if let Some(ref db) = down_bias {
                                        expert_output = expert_output.broadcast_add(db)?;
                                    }
                                    let weight_tensor = Tensor::from_vec(
                                        token_weights.clone(), (token_weights.len(), 1), cpu)?;
                                    let scaled = expert_output.broadcast_mul(&weight_tensor)?;
                                    let scaled_data = scaled.flatten_all()?.to_vec1::<f32>()?;
                                    let offsets: Vec<usize> = active_tokens.iter().map(|&t| t as usize).collect();
                                    Ok(ExpertContrib { token_offsets: offsets, data: scaled_data })
                                }))
                            }).collect();
                            handles.into_iter().map(|h| h.join().unwrap()).collect()
                        });
                        for result in expert_results {
                            let contrib = result?;
                            for (i, &tok_idx) in contrib.token_offsets.iter().enumerate() {
                                let dst_start = tok_idx * hd;
                                let src_start = i * hd;
                                for j in 0..hd {
                                    output_data[dst_start + j] += contrib.data[src_start + j];
                                }
                            }
                        }
                    } else {
                        // === Serial Q4 path: RAM resident or single expert ===
                        for &expert_idx in &needs_gguf_load {
                            let assignments = match expert_assignment.get(&expert_idx) {
                                Some(a) if !a.is_empty() => a,
                                _ => continue,
                            };

                            let gate_q = load_expert_quantized(reader, gate_name, expert_idx, cpu)?;
                            let up_q = load_expert_quantized(reader, up_name, expert_idx, cpu)?;
                            let down_q = load_expert_quantized(reader, down_name, expert_idx, cpu)?;

                            let (gate_bias, up_bias, down_bias) = if has_expert_bias {
                                let gb = load_expert(reader, &gate_bias_name, expert_idx, cpu)?;
                                let ub = load_expert(reader, &up_bias_name, expert_idx, cpu)?;
                                let db = load_expert(reader, &down_bias_name, expert_idx, cpu)?;
                                (Some(gb), Some(ub), Some(db))
                            } else {
                                (None, None, None)
                            };

                            if is_single_token {
                                let total_weight: f32 = assignments.iter().map(|(_, w)| w).sum();
                                let mut gate_out = gate_q.forward(&hidden_cpu)?;
                                let mut up_out = up_q.forward(&hidden_cpu)?;
                                if let Some(ref gb) = gate_bias {
                                    gate_out = gate_out.broadcast_add(gb)?;
                                }
                                if let Some(ref ub) = up_bias {
                                    up_out = up_out.broadcast_add(ub)?;
                                }
                                let expert_hidden = if use_oai_swiglu {
                                    ops::swiglu_oai(&gate_out, &up_out, 1.702, 7.0)?
                                } else {
                                    ops::silu_and_mul(&gate_out, &up_out)?
                                };
                                let mut expert_output = down_q.forward(&expert_hidden)?;
                                if let Some(ref db) = down_bias {
                                    expert_output = expert_output.broadcast_add(db)?;
                                }

                                let out_data = expert_output.flatten_all()?.to_vec1::<f32>()?;
                                for j in 0..hidden_dim {
                                    output_data[j] += out_data[j] * total_weight;
                                }
                            } else {
                                let active_tokens: Vec<u32> = assignments.iter().map(|(t, _)| *t as u32).collect();
                                let token_weights: Vec<f32> = assignments.iter().map(|(_, w)| *w).collect();

                                let indices_tensor =
                                    Tensor::from_vec(active_tokens.clone(), (active_tokens.len(),), cpu)?;
                                let expert_input = hidden_cpu.index_select(&indices_tensor, 0)?;

                                let mut gate_out = gate_q.forward(&expert_input)?;
                                let mut up_out = up_q.forward(&expert_input)?;
                                if let Some(ref gb) = gate_bias {
                                    gate_out = gate_out.broadcast_add(gb)?;
                                }
                                if let Some(ref ub) = up_bias {
                                    up_out = up_out.broadcast_add(ub)?;
                                }
                                let expert_hidden = if use_oai_swiglu {
                                    ops::swiglu_oai(&gate_out, &up_out, 1.702, 7.0)?
                                } else {
                                    ops::silu_and_mul(&gate_out, &up_out)?
                                };
                                let mut expert_output = down_q.forward(&expert_hidden)?;
                                if let Some(ref db) = down_bias {
                                    expert_output = expert_output.broadcast_add(db)?;
                                }

                                let weight_tensor = Tensor::from_vec(
                                    token_weights.clone(),
                                    (token_weights.len(), 1),
                                    cpu,
                                )?;
                                let scaled = expert_output.broadcast_mul(&weight_tensor)?;

                                let scaled_data = scaled.flatten_all()?.to_vec1::<f32>()?;
                                for (i, &tok_idx) in active_tokens.iter().enumerate() {
                                    let dst_start = tok_idx as usize * hidden_dim;
                                    let src_start = i * hidden_dim;
                                    for j in 0..hidden_dim {
                                        output_data[dst_start + j] += scaled_data[src_start + j];
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                // === Standard F32 path: dequantize → Tensor → cache ===
                let gguf_results: Vec<Result<LoadedExpert>> = std::thread::scope(|s| {
                    let handles: Vec<_> = needs_gguf_load.iter().map(|&expert_idx| {
                        s.spawn(move || {
                            let cpu = &Device::Cpu;
                            let gw = load_expert(reader, gate_name, expert_idx, cpu)?;
                            let uw = load_expert(reader, up_name, expert_idx, cpu)?;
                            let dw = load_expert(reader, down_name, expert_idx, cpu)?;
                            Ok(LoadedExpert { idx: expert_idx, gate: gw, up: uw, down: dw })
                        })
                    }).collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });

                for result in gguf_results {
                    let expert = result?;
                    // Cache on compute_device: CPU for SSD streaming, Metal for GPU compute.
                    // This avoids repeated CPU→GPU transfers on cache hits.
                    expert_cache.insert(layer_idx, expert.idx, ExpertWeights {
                        gate: expert.gate.to_device(compute_device)?,
                        up: expert.up.to_device(compute_device)?,
                        down: expert.down.to_device(compute_device)?,
                    });
                    loaded_experts.push(expert);
                }
            }

            // Batch eviction after parallel load (not per-expert during load)
            if do_evict {
                for &expert_idx in &needs_gguf_load {
                    self.reader.evict_expert_slice(&gate_exps_name, expert_idx);
                    self.reader.evict_expert_slice(&up_exps_name, expert_idx);
                    self.reader.evict_expert_slice(&down_exps_name, expert_idx);
                }
            }
        }

        let expert_io_ms = t_io.elapsed().as_secs_f64() * 1000.0;

        // === Phase 2: Compute F32 expert outputs (resident/cached only when quantized_mm) ===
        let t_compute = std::time::Instant::now();
        let expert_count = unique_experts.len();

        // Drop routing tensors early (no longer needed, already extracted flat arrays)
        drop(topk_indices);
        drop(topk_weights);

        for expert in &loaded_experts {
            let assignments = match expert_assignment.get(&expert.idx) {
                Some(a) if !a.is_empty() => a,
                _ => continue,
            };

            // Expert weights: move to compute_device if needed (e.g. CPU → Metal for GPU compute)
            let gate_w = expert.gate.to_device(compute_device)?;
            let up_w = expert.up.to_device(compute_device)?;
            let down_w = expert.down.to_device(compute_device)?;

            // Load expert FFN biases if present (e.g. GPT-OSS)
            let (gate_bias, up_bias, down_bias) = if has_expert_bias {
                let gb = load_expert(self.reader, &gate_bias_name, expert.idx, compute_device)?;
                let ub = load_expert(self.reader, &up_bias_name, expert.idx, compute_device)?;
                let db = load_expert(self.reader, &down_bias_name, expert.idx, compute_device)?;
                (Some(gb), Some(ub), Some(db))
            } else {
                (None, None, None)
            };

            if is_single_token {
                // Fast path: single-token decode (batch=1)
                // All experts have exactly 1 token. Skip index_select, broadcast_mul, etc.
                // Sum routing weights for this expert (handles duplicate assignments)
                let total_weight: f32 = assignments.iter().map(|(_, w)| w).sum();

                // SwiGLU: silu(gate_proj(x) + gate_bias) * (up_proj(x) + up_bias) → down_proj(...) + down_bias
                let mut gate_out = hidden_compute.matmul(&gate_w.t()?)?;
                let mut up_out = hidden_compute.matmul(&up_w.t()?)?;
                if let Some(ref gb) = gate_bias {
                    gate_out = gate_out.broadcast_add(gb)?;
                }
                if let Some(ref ub) = up_bias {
                    up_out = up_out.broadcast_add(ub)?;
                }
                let expert_hidden = if use_oai_swiglu {
                        ops::swiglu_oai(&gate_out, &up_out, 1.702, 7.0)?
                    } else {
                        ops::silu_and_mul(&gate_out, &up_out)?
                    };
                let mut expert_output = expert_hidden.matmul(&down_w.t()?)?;
                if let Some(ref db) = down_bias {
                    expert_output = expert_output.broadcast_add(db)?;
                }

                // Scale by routing weight and accumulate directly
                let out_data = expert_output.flatten_all()?.to_vec1::<f32>()?;
                for j in 0..hidden_dim {
                    output_data[j] += out_data[j] * total_weight;
                }
            } else {
                // General path: multi-token (prefill)
                let active_tokens: Vec<u32> = assignments.iter().map(|(t, _)| *t as u32).collect();
                let token_weights: Vec<f32> = assignments.iter().map(|(_, w)| *w).collect();

                let gate_w = gate_w.to_device(compute_device)?;
                let up_w = up_w.to_device(compute_device)?;
                let down_w = down_w.to_device(compute_device)?;

                let indices_tensor =
                    Tensor::from_vec(active_tokens.clone(), (active_tokens.len(),), compute_device)?;
                let expert_input = hidden_compute.index_select(&indices_tensor, 0)?;

                let mut gate_out = expert_input.matmul(&gate_w.t()?)?;
                let mut up_out = expert_input.matmul(&up_w.t()?)?;
                if let Some(ref gb) = gate_bias {
                    gate_out = gate_out.broadcast_add(&gb.to_device(compute_device)?)?;
                }
                if let Some(ref ub) = up_bias {
                    up_out = up_out.broadcast_add(&ub.to_device(compute_device)?)?;
                }
                let expert_hidden = if use_oai_swiglu {
                        ops::swiglu_oai(&gate_out, &up_out, 1.702, 7.0)?
                    } else {
                        ops::silu_and_mul(&gate_out, &up_out)?
                    };
                let mut expert_output = expert_hidden.matmul(&down_w.t()?)?;
                if let Some(ref db) = down_bias {
                    expert_output = expert_output.broadcast_add(&db.to_device(compute_device)?)?;
                }

                let weight_tensor = Tensor::from_vec(
                    token_weights.clone(),
                    (token_weights.len(), 1),
                    compute_device,
                )?;
                let scaled = expert_output.broadcast_mul(&weight_tensor)?;

                let scaled_data = scaled.flatten_all()?.to_vec1::<f32>()?;
                for (i, &tok_idx) in active_tokens.iter().enumerate() {
                    let dst_start = tok_idx as usize * hidden_dim;
                    let src_start = i * hidden_dim;
                    for j in 0..hidden_dim {
                        output_data[dst_start + j] += scaled_data[src_start + j];
                    }
                }
            }
        }
        let expert_compute_ms = t_compute.elapsed().as_secs_f64() * 1000.0;

        // === Shared Expert (Qwen3-Coder-Next) ===
        // Always-on MLP with sigmoid gating, added to routed output
        let t_shexp = std::time::Instant::now();
        if self.config.has_shared_expert {
            // Use resident shared expert weights if available, else load from GGUF
            let shexp_gate_name = format!("{}.ffn_gate_shexp.weight", prefix);

            let has_shexp = resident.shared_experts[layer_idx].is_some()
                || self.reader.tensors.contains_key(&shexp_gate_name);

            if has_shexp {
                let (shexp_gate_w, shexp_up_w, shexp_down_w, shexp_gate_inp_w) =
                    if let Some(se) = &resident.shared_experts[layer_idx] {
                        (se.gate.to_device(compute_device)?, se.up.to_device(compute_device)?,
                         se.down.to_device(compute_device)?, se.gate_inp.as_ref().map(|g| g.to_device(compute_device)).transpose()?)
                    } else {
                        let shexp_up_name = format!("{}.ffn_up_shexp.weight", prefix);
                        let shexp_down_name = format!("{}.ffn_down_shexp.weight", prefix);
                        let shexp_gate_inp_name = format!("{}.ffn_gate_inp_shexp.weight", prefix);
                        let gw = load_weight(self.reader, &shexp_gate_name, compute_device)?;
                        let uw = load_weight(self.reader, &shexp_up_name, compute_device)?;
                        let dw = load_weight(self.reader, &shexp_down_name, compute_device)?;
                        let giw = if self.reader.tensors.contains_key(&shexp_gate_inp_name) {
                            Some(load_weight(self.reader, &shexp_gate_inp_name, compute_device)?)
                        } else {
                            None
                        };
                        (gw, uw, dw, giw)
                    };

                // SwiGLU on compute_device: silu(gate_proj(x)) * up_proj(x) → down_proj(...)
                let shexp_gate_out = hidden_compute.matmul(&shexp_gate_w.t()?)?;
                let shexp_up_out = hidden_compute.matmul(&shexp_up_w.t()?)?;
                let shexp_hidden = ops::silu_and_mul(&shexp_gate_out, &shexp_up_out)?;
                let shexp_output = shexp_hidden.matmul(&shexp_down_w.t()?)?;

                // Apply shared expert gate (sigmoid scaling)
                let shexp_output = if let Some(gate_w) = &shexp_gate_inp_w {
                    let gate_logits = hidden_compute.broadcast_mul(&gate_w)?;
                    let gate_logits = gate_logits.sum_keepdim(1)?;
                    let gate_sigmoid = ops::sigmoid(&gate_logits)?;
                    shexp_output.broadcast_mul(&gate_sigmoid)?
                } else {
                    shexp_output
                };

                // Add shared expert output to routed expert output
                let shexp_data = shexp_output.flatten_all()?.to_vec1::<f32>()?;
                for t in 0..num_tokens {
                    let dst_start = t * hidden_dim;
                    for j in 0..hidden_dim {
                        output_data[dst_start + j] += shexp_data[t * hidden_dim + j];
                    }
                }
            }
        }

        let shexp_ms = t_shexp.elapsed().as_secs_f64() * 1000.0;

        // Populate MoE sub-timings for profiler
        if let Some(ref mut timing) = moe_timing_out {
            *timing = (routing_ms, expert_io_ms, expert_compute_ms, shexp_ms);
        }

        // Log MoE + shared expert breakdown
        if layer_idx == 0 || layer_idx == 3 || layer_idx == 47 {
            log::debug!(
                "  MoE+Shexp L{}: routed_io={:.1}ms routed_compute={:.1}ms shexp={:.1}ms ({} experts)",
                layer_idx, expert_io_ms, expert_compute_ms, shexp_ms, expert_count,
            );
        }

        // Single transfer: CPU → device at the end
        let final_output = Tensor::from_vec(output_data.clone(), (num_tokens, hidden_dim), self.device)?;
        let result = final_output.reshape((bsz, seq_len, hidden_dim))?;

        // Save router logits + MoE output for adaptive skip (next token comparison)
        if let Some(logits_vec) = skip_logits_vec {
            layer_output_cache.update(layer_idx, logits_vec, output_data);
        }

        Ok(result)
    }

    /// GPU Resident MoE path: all expert weights are preloaded on Metal GPU.
    ///
    /// No GGUF loading, no dequantization, no CPU↔GPU transfer for experts.
    /// Hidden states stay on GPU throughout. Uses GpuExpertProjection::forward()
    /// which dispatches to QMatMul (quantized) or tensor matmul (F16 dense).
    ///
    /// For MXFP4 layers with packed weights (e.g., GPT-OSS-20B), single-token decode
    /// uses the batched Metal dispatch path: all active experts are processed in just
    /// 5-8 Metal kernel dispatches instead of 12+ per-expert dispatches. This
    /// eliminates per-expert dispatch overhead (~10-50us each).
    #[allow(clippy::too_many_arguments)]
    fn run_moe_gpu_resident(
        &self,
        hidden_flat: &Tensor,
        layer_idx: usize,
        prefix: &str,
        resident: &ResidentWeights,
        expert_assignment: &std::collections::HashMap<usize, Vec<(usize, f32)>>,
        unique_experts: &[usize],
        num_tokens: usize,
        hidden_dim: usize,
        bsz: usize,
        seq_len: usize,
        use_oai_swiglu: bool,
        skip_logits_vec: Option<Vec<f32>>,
        layer_output_cache: &mut LayerOutputCache,
    ) -> Result<Tensor> {
        let gpu_device = self.device;
        let hidden_gpu = hidden_flat.to_device(gpu_device)?;
        let is_single_token = num_tokens == 1;

        // === Batched Metal dispatch path (MXFP4, single-token decode) ===
        // When packed MXFP4 weights are available, dispatch all active experts in a single
        // set of Metal kernel calls instead of per-expert loop. This reduces kernel dispatch
        // overhead from ~12+ dispatches (3 matmuls x 4 experts) to just 5-8 dispatches total.
        // Set MOE_NO_BATCH=1 to force per-expert fallback (for debugging/comparison).
        #[cfg(feature = "metal")]
        if is_single_token && std::env::var("MOE_NO_BATCH").is_err() {
            if let Some(packed) = &resident.packed_mxfp4[layer_idx] {
                if unique_experts.len() <= crate::metal::MAX_BATCH_EXPERTS {
                    // Build BatchedExpertInfo from routing assignments.
                    // For single-token, each expert has exactly one assignment with its weight.
                    let batched_experts: Vec<crate::metal::BatchedExpertInfo> = unique_experts.iter()
                        .filter_map(|&eidx| {
                            expert_assignment.get(&eidx).and_then(|assignments| {
                                if assignments.is_empty() { return None; }
                                let total_weight: f32 = assignments.iter().map(|(_, w)| w).sum();
                                Some(crate::metal::BatchedExpertInfo {
                                    expert_idx: eidx,
                                    routing_weight: total_weight,
                                })
                            })
                        })
                        .collect();

                    if !batched_experts.is_empty() {
                        let bias_buffers = resident.expert_bias_buffers[layer_idx].as_ref();
                        log::trace!(
                            "MoE L{}: batched Metal dispatch ({} experts, biases={})",
                            layer_idx, batched_experts.len(), bias_buffers.is_some()
                        );

                        let moe_buffers = resident.batched_moe_buffers.as_ref();
                        let batched_output = crate::metal::batched_moe_forward_metal(
                            gpu_device,
                            &hidden_gpu,
                            packed,
                            &batched_experts,
                            use_oai_swiglu,
                            1.702,  // alpha for OAI SwiGLU
                            7.0,    // limit for OAI SwiGLU
                            bias_buffers,
                            moe_buffers,
                        )?;

                        // Handle shared expert (if any)
                        let mut output = batched_output;
                        if self.config.has_shared_expert {
                            let shexp_gate_name = format!("{}.ffn_gate_shexp.weight", prefix);
                            let has_shexp = resident.shared_experts[layer_idx].is_some()
                                || self.reader.tensors.contains_key(&shexp_gate_name);

                            if has_shexp {
                                let shexp_out = Self::run_shared_expert(
                                    &hidden_gpu, prefix, resident, self.reader, gpu_device,
                                    layer_idx, self.config.has_shared_expert,
                                )?;
                                output = (output + shexp_out)?;
                            }
                        }

                        let result = output.reshape((bsz, seq_len, hidden_dim))?;

                        // Save for adaptive skip
                        if let Some(logits_vec) = skip_logits_vec {
                            let output_vec = output.reshape((num_tokens, hidden_dim))?
                                .flatten_all()?.to_vec1::<f32>()?;
                            layer_output_cache.update(layer_idx, logits_vec, output_vec);
                        }

                        return Ok(result);
                    }
                }
            }
        }

        // === Per-expert fallback path (QMatMul, Dense, or multi-token) ===

        // Expert FFN biases (e.g. GPT-OSS: ffn_{gate,up,down}_exps.bias)
        let gate_bias_name = format!("{}.ffn_gate_exps.bias", prefix);
        let up_bias_name = format!("{}.ffn_up_exps.bias", prefix);
        let down_bias_name = format!("{}.ffn_down_exps.bias", prefix);
        let has_expert_bias = self.reader.tensors.contains_key(&gate_bias_name);

        // Accumulate output on GPU
        let mut output_accum = Tensor::zeros((num_tokens, hidden_dim), candle_core::DType::F32, gpu_device)?;

        for &expert_idx in unique_experts {
            let assignments = match expert_assignment.get(&expert_idx) {
                Some(a) if !a.is_empty() => a,
                _ => continue,
            };

            let gpu_expert = match &resident.gpu_experts[layer_idx][expert_idx] {
                Some(e) => e,
                None => {
                    log::warn!("GPU Resident: missing expert L{} E{}, skipping", layer_idx, expert_idx);
                    continue;
                }
            };

            // Load expert biases if present (small F32 tensors)
            let (gate_bias, up_bias, down_bias) = if has_expert_bias {
                let gb = load_expert(self.reader, &gate_bias_name, expert_idx, gpu_device)?;
                let ub = load_expert(self.reader, &up_bias_name, expert_idx, gpu_device)?;
                let db = load_expert(self.reader, &down_bias_name, expert_idx, gpu_device)?;
                (Some(gb), Some(ub), Some(db))
            } else {
                (None, None, None)
            };

            if is_single_token {
                let total_weight: f32 = assignments.iter().map(|(_, w)| w).sum();

                // Forward through SwiGLU on GPU
                let mut gate_out = gpu_expert.gate.forward(&hidden_gpu)?;
                let mut up_out = gpu_expert.up.forward(&hidden_gpu)?;
                if let Some(ref gb) = gate_bias {
                    gate_out = gate_out.broadcast_add(gb)?;
                }
                if let Some(ref ub) = up_bias {
                    up_out = up_out.broadcast_add(ub)?;
                }
                let expert_hidden = if use_oai_swiglu {
                    ops::swiglu_oai(&gate_out, &up_out, 1.702, 7.0)?
                } else {
                    ops::silu_and_mul(&gate_out, &up_out)?
                };
                let mut expert_output = gpu_expert.down.forward(&expert_hidden)?;
                if let Some(ref db) = down_bias {
                    expert_output = expert_output.broadcast_add(db)?;
                }

                // Scale and accumulate on GPU
                let scaled = (expert_output * total_weight as f64)?;
                output_accum = (output_accum + scaled)?;
            } else {
                // Multi-token (prefill) path
                let active_tokens: Vec<u32> = assignments.iter().map(|(t, _)| *t as u32).collect();
                let token_weights: Vec<f32> = assignments.iter().map(|(_, w)| *w).collect();

                let indices_tensor = Tensor::from_vec(active_tokens.clone(), (active_tokens.len(),), gpu_device)?;
                let expert_input = hidden_gpu.index_select(&indices_tensor, 0)?;

                let mut gate_out = gpu_expert.gate.forward(&expert_input)?;
                let mut up_out = gpu_expert.up.forward(&expert_input)?;
                if let Some(ref gb) = gate_bias {
                    gate_out = gate_out.broadcast_add(gb)?;
                }
                if let Some(ref ub) = up_bias {
                    up_out = up_out.broadcast_add(ub)?;
                }
                let expert_hidden = if use_oai_swiglu {
                    ops::swiglu_oai(&gate_out, &up_out, 1.702, 7.0)?
                } else {
                    ops::silu_and_mul(&gate_out, &up_out)?
                };
                let mut expert_output = gpu_expert.down.forward(&expert_hidden)?;
                if let Some(ref db) = down_bias {
                    expert_output = expert_output.broadcast_add(db)?;
                }

                // Scale by routing weights
                let weight_tensor = Tensor::from_vec(
                    token_weights, (active_tokens.len(), 1), gpu_device,
                )?;
                let scaled = expert_output.broadcast_mul(&weight_tensor)?;

                // Scatter-add: accumulate scaled expert output to correct token positions.
                // index_add is the canonical scatter-add in candle.
                output_accum = output_accum.index_add(&indices_tensor, &scaled, 0)?;
            }
        }

        // === Shared Expert (Qwen3-Coder-Next) ===
        if self.config.has_shared_expert {
            let shexp_gate_name = format!("{}.ffn_gate_shexp.weight", prefix);
            let has_shexp = resident.shared_experts[layer_idx].is_some()
                || self.reader.tensors.contains_key(&shexp_gate_name);

            if has_shexp {
                let (shexp_gate_w, shexp_up_w, shexp_down_w, shexp_gate_inp_w) =
                    if let Some(se) = &resident.shared_experts[layer_idx] {
                        (se.gate.to_device(gpu_device)?, se.up.to_device(gpu_device)?,
                         se.down.to_device(gpu_device)?, se.gate_inp.as_ref().map(|g| g.to_device(gpu_device)).transpose()?)
                    } else {
                        let shexp_up_name = format!("{}.ffn_up_shexp.weight", prefix);
                        let shexp_down_name = format!("{}.ffn_down_shexp.weight", prefix);
                        let shexp_gate_inp_name = format!("{}.ffn_gate_inp_shexp.weight", prefix);
                        let gw = load_weight(self.reader, &shexp_gate_name, gpu_device)?;
                        let uw = load_weight(self.reader, &shexp_up_name, gpu_device)?;
                        let dw = load_weight(self.reader, &shexp_down_name, gpu_device)?;
                        let giw = if self.reader.tensors.contains_key(&shexp_gate_inp_name) {
                            Some(load_weight(self.reader, &shexp_gate_inp_name, gpu_device)?)
                        } else {
                            None
                        };
                        (gw, uw, dw, giw)
                    };

                let shexp_gate_out = hidden_gpu.matmul(&shexp_gate_w.t()?)?;
                let shexp_up_out = hidden_gpu.matmul(&shexp_up_w.t()?)?;
                let shexp_hidden = ops::silu_and_mul(&shexp_gate_out, &shexp_up_out)?;
                let shexp_output = shexp_hidden.matmul(&shexp_down_w.t()?)?;

                let shexp_output = if let Some(gate_w) = &shexp_gate_inp_w {
                    let gate_logits = hidden_gpu.broadcast_mul(gate_w)?;
                    let gate_logits = gate_logits.sum_keepdim(1)?;
                    let gate_sigmoid = ops::sigmoid(&gate_logits)?;
                    shexp_output.broadcast_mul(&gate_sigmoid)?
                } else {
                    shexp_output
                };

                output_accum = (output_accum + shexp_output)?;
            }
        }

        // Result stays on GPU device
        let result = output_accum.reshape((bsz, seq_len, hidden_dim))?;

        // Save for adaptive skip (need CPU vec for comparison)
        if let Some(logits_vec) = skip_logits_vec {
            let output_vec = output_accum.flatten_all()?.to_vec1::<f32>()?;
            layer_output_cache.update(layer_idx, logits_vec, output_vec);
        }

        Ok(result)
    }

    /// Compute shared expert output (Qwen3-Coder-Next style).
    ///
    /// Shared experts are dense FFN layers that process all tokens at every MoE layer,
    /// in addition to the routed experts. The output is added to the MoE output.
    ///
    /// This is extracted as a static method so it can be used by both the per-expert
    /// and batched dispatch paths.
    #[cfg(feature = "metal")]
    fn run_shared_expert(
        hidden_gpu: &Tensor,
        prefix: &str,
        resident: &ResidentWeights,
        reader: &crate::gguf::GgufReader,
        gpu_device: &Device,
        layer_idx: usize,
        _has_shared_expert: bool,
    ) -> Result<Tensor> {
        let shexp_gate_name = format!("{}.ffn_gate_shexp.weight", prefix);

        let (shexp_gate_w, shexp_up_w, shexp_down_w, shexp_gate_inp_w) =
            if let Some(se) = &resident.shared_experts[layer_idx] {
                (se.gate.to_device(gpu_device)?, se.up.to_device(gpu_device)?,
                 se.down.to_device(gpu_device)?, se.gate_inp.as_ref().map(|g| g.to_device(gpu_device)).transpose()?)
            } else {
                let shexp_up_name = format!("{}.ffn_up_shexp.weight", prefix);
                let shexp_down_name = format!("{}.ffn_down_shexp.weight", prefix);
                let shexp_gate_inp_name = format!("{}.ffn_gate_inp_shexp.weight", prefix);
                let gw = load_weight(reader, &shexp_gate_name, gpu_device)?;
                let uw = load_weight(reader, &shexp_up_name, gpu_device)?;
                let dw = load_weight(reader, &shexp_down_name, gpu_device)?;
                let giw = if reader.tensors.contains_key(&shexp_gate_inp_name) {
                    Some(load_weight(reader, &shexp_gate_inp_name, gpu_device)?)
                } else {
                    None
                };
                (gw, uw, dw, giw)
            };

        let shexp_gate_out = hidden_gpu.matmul(&shexp_gate_w.t()?)?;
        let shexp_up_out = hidden_gpu.matmul(&shexp_up_w.t()?)?;
        let shexp_hidden = ops::silu_and_mul(&shexp_gate_out, &shexp_up_out)?;
        let shexp_output = shexp_hidden.matmul(&shexp_down_w.t()?)?;

        let shexp_output = if let Some(gate_w) = &shexp_gate_inp_w {
            let gate_logits = hidden_gpu.broadcast_mul(gate_w)?;
            let gate_logits = gate_logits.sum_keepdim(1)?;
            let gate_sigmoid = ops::sigmoid(&gate_logits)?;
            shexp_output.broadcast_mul(&gate_sigmoid)?
        } else {
            shexp_output
        };

        Ok(shexp_output)
    }
}

/// Compute Shannon entropy of router logits (after softmax over all experts).
/// Returns a single f32 entropy value (averaged across tokens if batch > 1).
/// Low entropy = peaked distribution (confident), high entropy = flat (uncertain).
/// NOTE: Expects logits already on CPU (run_moe transfers them once via to_vec1).
fn compute_router_entropy(logits: &Tensor) -> Result<f32> {
    let num_tokens = logits.dim(0)?;
    let _num_experts = logits.dim(1)?;
    let mut total_entropy = 0.0f32;

    for t in 0..num_tokens {
        let row = logits.i(t)?.to_vec1::<f32>()?;
        // Stable softmax
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = row.iter().map(|v| (v - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        // Shannon entropy: -sum(p * ln(p)), skip p=0
        let mut h = 0.0f32;
        for &e in &exp_vals {
            let p = e / sum;
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        total_entropy += h;
    }

    Ok(total_entropy / num_tokens as f32)
}

/// Map entropy to K value using linear interpolation.
/// Low entropy (peaked) -> k_min, high entropy (flat) -> k_max.
fn entropy_to_k(entropy: f32, num_experts: f32, k_min: usize, k_max: usize) -> usize {
    let h_max = num_experts.ln(); // maximum entropy = ln(num_experts)
    let normalized = (entropy / h_max).clamp(0.0, 1.0);
    let k = k_min as f32 + normalized * (k_max - k_min) as f32;
    (k.round() as usize).clamp(k_min, k_max)
}

/// Top-k routing with configurable gating mode.
///
/// Standard mode (softmax_weight=false): softmax over all experts, then select top-k.
/// When `normalize` is true (norm_topk_prob=true, Qwen3 default), the top-k
/// weights are renormalized to sum to 1.0.
///
/// SOFTMAX_WEIGHT mode (softmax_weight=true, GPT-OSS): select top-k by raw logits,
/// then apply softmax ONLY on the selected top-k weights.
///
/// NOTE: Expects router_logits already on CPU (run_moe transfers them once via to_vec1).
fn top_k_routing(router_logits: &Tensor, k: usize, normalize: bool, softmax_weight: bool) -> Result<(Tensor, Tensor)> {
    let num_tokens = router_logits.dim(0)?;

    let mut all_weights = Vec::with_capacity(num_tokens * k);
    let mut all_indices = Vec::with_capacity(num_tokens * k);

    for t in 0..num_tokens {
        let row = router_logits.i(t)?.to_vec1::<f32>()?;

        if softmax_weight {
            // SOFTMAX_WEIGHT: select top-k by raw logits, then softmax on selected only
            let mut indexed: Vec<(usize, f32)> = row.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let topk_logits: Vec<f32> = indexed.iter().take(k).map(|(_, v)| *v).collect();
            let topk_idx: Vec<u32> = indexed.iter().take(k).map(|(i, _)| *i as u32).collect();

            // Softmax over top-k logits only
            let max_val = topk_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_vals: Vec<f32> = topk_logits.iter().map(|v| (v - max_val).exp()).collect();
            let sum: f32 = exp_vals.iter().sum();
            let weights: Vec<f32> = exp_vals.iter().map(|v| v / sum).collect();

            all_weights.extend(weights);
            all_indices.extend(topk_idx);
        } else {
            // Standard: softmax over ALL experts first (matching HF Qwen2MoE)
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_vals: Vec<f32> = row.iter().map(|v| (v - max_val).exp()).collect();
            let sum: f32 = exp_vals.iter().sum();
            let softmax_all: Vec<f32> = exp_vals.iter().map(|v| v / sum).collect();

            // Select top-k by softmax probability (monotonic, same as by logit)
            let mut indexed: Vec<(usize, f32)> = softmax_all.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let topk_probs: Vec<f32> = indexed.iter().take(k).map(|(_, v)| *v).collect();
            let topk_idx: Vec<u32> = indexed.iter().take(k).map(|(i, _)| *i as u32).collect();

            // Optionally renormalize to sum to 1.0 (norm_topk_prob=true)
            let weights = if normalize {
                let topk_sum: f32 = topk_probs.iter().sum();
                topk_probs.iter().map(|v| v / topk_sum).collect::<Vec<f32>>()
            } else {
                topk_probs
            };

            all_weights.extend(weights);
            all_indices.extend(topk_idx);
        }
    }

    // Always output on CPU: all callers extract to Vec immediately after.
    let weights = Tensor::from_vec(all_weights, (num_tokens, k), &Device::Cpu)?;
    let indices = Tensor::from_vec(all_indices, (num_tokens, k), &Device::Cpu)?;

    Ok((weights, indices))
}

/// Find which tokens are assigned to a given expert and their routing weights.
#[allow(dead_code)]
fn get_expert_assignment(
    topk_indices: &Tensor,
    topk_weights: &Tensor,
    expert_idx: usize,
    _device: &Device,
) -> Result<(Vec<u32>, Vec<f32>)> {
    let indices_cpu = topk_indices.to_device(&Device::Cpu)?;
    let weights_cpu = topk_weights.to_device(&Device::Cpu)?;

    let num_tokens = indices_cpu.dim(0)?;
    let k = indices_cpu.dim(1)?;

    let mut active_tokens = Vec::new();
    let mut token_weights = Vec::new();

    for t in 0..num_tokens {
        let mut weight_sum = 0.0f32;
        for j in 0..k {
            let idx = indices_cpu.i((t, j))?.to_scalar::<u32>()? as usize;
            if idx == expert_idx {
                weight_sum += weights_cpu.i((t, j))?.to_scalar::<f32>()?;
            }
        }
        if weight_sum > 0.0 {
            active_tokens.push(t as u32);
            token_weights.push(weight_sum);
        }
    }

    Ok((active_tokens, token_weights))
}
