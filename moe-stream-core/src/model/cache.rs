//! Expert LRU cache and resident weight management.
//!
//! Key optimizations for SSD streaming:
//! - ExpertCache: per-layer LRU avoids re-dequantizing frequently used experts
//! - ResidentWeights: attention + router weights kept in GPU memory

use candle_core::{Module, Tensor};
use candle_core::quantized::QMatMul;
use linked_hash_map::LinkedHashMap;

#[cfg(feature = "metal")]
use candle_core::Device;
#[cfg(feature = "metal")]
use std::sync::Arc;

/// Cached expert weights (gate, up, down projections).
pub struct ExpertWeights {
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
}

/// A single expert projection loaded on GPU, either as quantized (QMatMul), dense (Tensor),
/// or native MXFP4 Metal (raw 4-bit weights with custom Metal kernel).
///
/// QMatMul: For Q4_K and other candle-supported quantized types — native quantized matmul on Metal.
/// Dense: For F16/F32 weights — standard tensor matmul on Metal.
/// Mxfp4Metal: For MXFP4 weights — raw 4-bit data on Metal GPU, dispatched via custom Metal kernel.
/// Mxfp4Packed: References a region of a packed per-layer buffer (shared with batched dispatch).
pub enum GpuExpertProjection {
    /// Quantized matmul (Q4_K, Q5_K, Q6_K, etc.) — no dequant, native Metal quantized kernel.
    Quantized(QMatMul),
    /// Dense tensor (F16/F32) on Metal GPU.
    Dense(Tensor),
    /// MXFP4 raw weights on Metal GPU buffer, dispatched via custom Metal kernel.
    /// Fields: (buffer, out_features, in_features, device).
    #[cfg(feature = "metal")]
    Mxfp4Metal(Arc<metal::Buffer>, usize, usize, Device),
    /// MXFP4 weights referencing a region of a packed per-layer buffer.
    /// Uses the same buffer as PackedMxfp4Experts for zero-copy.
    /// Fields: (packed_buffer, byte_offset, out_features, in_features, device).
    #[cfg(feature = "metal")]
    Mxfp4Packed(Arc<metal::Buffer>, u64, usize, usize, Device),
}

/// Expert weights (gate, up, down) preloaded on GPU for GPU Resident mode.
pub struct GpuExpertWeights {
    pub gate: GpuExpertProjection,
    pub up: GpuExpertProjection,
    pub down: GpuExpertProjection,
}

impl GpuExpertProjection {
    /// Forward pass: input x weight^T -> output.
    /// For QMatMul: native quantized matmul. For Dense: standard tensor matmul.
    /// For Mxfp4Metal/Mxfp4Packed: custom Metal kernel dispatch via candle CustomOp1 (no dequant,
    /// 4-bit on GPU, zero sync — kernel is enqueued onto candle's shared command buffer).
    pub fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            GpuExpertProjection::Quantized(qmm) => qmm.forward(input),
            GpuExpertProjection::Dense(w) => {
                // Cast weight to match input dtype (e.g. F16 weight → F32) so output
                // stays in input's dtype. This avoids cascading dtype mismatches downstream.
                let w_compat = if w.dtype() != input.dtype() {
                    w.to_dtype(input.dtype())?
                } else {
                    w.clone()
                };
                input.matmul(&w_compat.t()?)
            }
            #[cfg(feature = "metal")]
            GpuExpertProjection::Mxfp4Metal(buffer, out_features, in_features, _device) => {
                crate::metal::mxfp4_matmul_metal_gpu_batched(buffer, input, *out_features, *in_features)
            }
            #[cfg(feature = "metal")]
            GpuExpertProjection::Mxfp4Packed(buffer, _offset, out_features, in_features, _device) => {
                crate::metal::mxfp4_matmul_metal_gpu_offset_batched(buffer, *_offset, input, *out_features, *in_features)
            }
        }
    }
}

/// Per-layer LRU cache for dequantized expert weights.
pub struct ExpertCache {
    layers: Vec<LinkedHashMap<usize, ExpertWeights>>,
    capacities: Vec<usize>,
    hits: u64,
    misses: u64,
}

impl ExpertCache {
    /// Create a new expert cache with uniform capacity per layer.
    pub fn new(num_layers: usize, capacity_per_layer: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| LinkedHashMap::with_capacity(capacity_per_layer))
            .collect();
        Self {
            layers,
            capacities: vec![capacity_per_layer; num_layers],
            hits: 0,
            misses: 0,
        }
    }

    /// Create a new expert cache with per-layer capacities.
    pub fn with_layer_capacities(capacities: Vec<usize>) -> Self {
        let layers = capacities
            .iter()
            .map(|&cap| LinkedHashMap::with_capacity(cap))
            .collect();
        Self {
            layers,
            capacities,
            hits: 0,
            misses: 0,
        }
    }

    /// Get cached expert weights if available (moves to front = most recent).
    pub fn get(&mut self, layer_idx: usize, expert_idx: usize) -> Option<&ExpertWeights> {
        if let Some(entry) = self.layers[layer_idx].get_refresh(&expert_idx) {
            self.hits += 1;
            Some(entry)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert expert weights into cache, evicting LRU if at capacity.
    /// Does nothing if capacity is 0 (cache disabled for this layer).
    pub fn insert(&mut self, layer_idx: usize, expert_idx: usize, weights: ExpertWeights) {
        let cap = self.capacities[layer_idx];
        if cap == 0 {
            return; // Cache disabled for this layer
        }
        let cache = &mut self.layers[layer_idx];
        if cache.len() >= cap {
            cache.pop_front();
        }
        cache.insert(expert_idx, weights);
    }

    /// Get hit rate as a fraction.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Get (hits, misses) counts.
    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}

/// A single MXFP4 weight matrix stored as a raw Metal buffer.
///
/// Unlike dequantized F32 tensors, this keeps the quantized data on GPU,
/// reducing memory usage by ~4x and memory bandwidth by the same factor.
/// Used with the custom `mxfp4_matvec_f32` Metal kernel.
#[cfg(feature = "metal")]
pub struct Mxfp4Weight {
    /// Raw MXFP4 data on GPU (out_features rows × weight_stride bytes each).
    pub buffer: Arc<metal::Buffer>,
    /// Number of output rows.
    pub out_features: usize,
    /// Number of input columns (must be a multiple of 32).
    pub in_features: usize,
}

/// MXFP4 attention weight matrices on Metal GPU.
///
/// Stores Q/K/V/O projection weights in native MXFP4 format, avoiding the
/// 4x memory overhead of F32 dequantization. The MXFP4 matvec kernel reads
/// these directly during attention computation.
#[cfg(feature = "metal")]
pub struct Mxfp4AttentionWeights {
    pub q: Mxfp4Weight,
    pub k: Mxfp4Weight,
    pub v: Mxfp4Weight,
    pub o: Mxfp4Weight,
}

/// A single quantized attention weight matrix stored as a raw Metal buffer.
///
/// For Q5_0/Q8_0 attention weights in GPU Resident mode. Uses our custom
/// Metal kernels (quantized_attn.metal) instead of dequanting to F32.
#[cfg(feature = "metal")]
pub struct QuantizedAttnWeight {
    pub buffer: Arc<metal::Buffer>,
    pub out_features: usize,
    pub in_features: usize,
    pub quant_type: crate::metal::QuantizedAttnType,
}

/// Quantized attention weight matrices on Metal GPU (Q5_0/Q8_0).
/// Fields are optional to support mixed quant types (e.g. Q5_0 Q/K/V + Q5_K O).
/// Unsupported types (Q5_K, Q4_K_M, etc.) fall through to F32 matmul.
#[cfg(feature = "metal")]
pub struct QuantizedAttnMetalWeights {
    pub q: Option<QuantizedAttnWeight>,
    pub k: Option<QuantizedAttnWeight>,
    pub v: Option<QuantizedAttnWeight>,
    pub o: Option<QuantizedAttnWeight>,
}

/// Pre-loaded attention weights for a single layer.
pub struct AttentionWeights {
    pub q_weight: Tensor,
    pub k_weight: Tensor,
    pub v_weight: Tensor,
    pub o_weight: Tensor,
    pub q_norm: Option<Tensor>,
    pub k_norm: Option<Tensor>,
    pub q_bias: Option<Tensor>,
    pub k_bias: Option<Tensor>,
    pub v_bias: Option<Tensor>,
    pub o_bias: Option<Tensor>,
    /// Attention sinks: per-head learned logit for virtual sink position (GPT-OSS).
    /// Shape: [num_heads]. Added to softmax denominator during attention.
    pub attn_sinks: Option<Tensor>,
    /// MXFP4 attention weights on Metal GPU (GPU Resident mode only).
    /// When present, attention matmuls use the MXFP4 kernel instead of F32 matmul,
    /// reducing memory bandwidth by ~4x (MXFP4 is 4.25 bits vs F32 32 bits).
    #[cfg(feature = "metal")]
    pub mxfp4: Option<Mxfp4AttentionWeights>,
    /// Quantized attention weights on Metal GPU (Q5_0/Q8_0).
    /// Uses our custom Metal kernels for native quantized matvec on GPU.
    /// Higher priority than QMatMul (CPU) and F32 fallback.
    #[cfg(feature = "metal")]
    pub quantized_metal: Option<QuantizedAttnMetalWeights>,
    /// Quantized attention weights (QMatMul) for Q4_K/Q5_0/Q8_0/etc.
    /// CPU-only path. On Metal GPU, use quantized_metal instead.
    pub quantized: Option<QuantizedAttentionWeights>,
}

/// Quantized attention projection weights using candle's QMatMul.
///
/// Stores Q/K/V/O as QMatMul which uses Metal quantized matmul kernels,
/// keeping weights in their native format (Q5_0, Q8_0, Q4_K, etc.).
pub struct QuantizedAttentionWeights {
    pub q: QMatMul,
    pub k: QMatMul,
    pub v: QMatMul,
    pub o: QMatMul,
}

/// Pre-loaded DeltaNet weights for a single layer (on CPU).
///
/// Official Unsloth GGUF splits the DeltaNet input projection into two tensors:
///   attn_qkv.weight = fused Q+K+V (8192 for 80B)
///   attn_gate.weight = output gate Z (4096 for 80B)
/// Earlier custom GGUF conversions used a single ssm_in.weight for all of ZQKV.
pub struct DeltaNetWeights {
    /// Fused Q+K+V projection: attn_qkv.weight [qkv_dim, hidden_size]
    pub attn_qkv: Tensor,
    /// Output gate Z projection: attn_gate.weight [d_inner, hidden_size]
    pub attn_gate: Tensor,
    pub ssm_ba: Tensor,
    pub ssm_a: Tensor,
    pub ssm_dt_bias: Tensor,
    pub ssm_conv1d: Tensor,
    pub ssm_norm: Tensor,
    pub ssm_out: Tensor,
}

/// Pre-loaded shared expert weights for a single layer (gate + up + down + optional sigmoid gate).
pub struct SharedExpertWeights {
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
    /// Sigmoid gate weight (for gated shared experts like Qwen3-Coder-Next).
    pub gate_inp: Option<Tensor>,
}

/// Per-layer packed MXFP4 expert weights for batched Metal dispatch.
///
/// All experts' weights for a given projection (gate, up, or down) are packed
/// into a single contiguous Metal buffer. The `offsets` array provides byte
/// offsets into this buffer for each expert.
///
/// This enables dispatching all active experts in a single Metal kernel call
/// using the Z grid dimension, eliminating per-expert dispatch overhead.
#[cfg(feature = "metal")]
pub struct PackedMxfp4Experts {
    /// Packed weight buffer: all experts' weights contiguous on GPU.
    pub buffer: Arc<metal::Buffer>,
    /// Byte offsets into `buffer` for each expert: offsets[expert_idx].
    pub offsets: Vec<u64>,
    /// Output features (rows) per expert weight matrix.
    pub out_features: usize,
    /// Input features (columns) per expert weight matrix.
    pub in_features: usize,
}

/// Per-layer packed MXFP4 projections (gate, up, down) for batched dispatch.
///
/// Bias data is stored separately in `ResidentWeights::expert_bias_buffers`.
#[cfg(feature = "metal")]
pub struct PackedMxfp4Layer {
    pub gate: PackedMxfp4Experts,
    pub up: PackedMxfp4Experts,
    pub down: PackedMxfp4Experts,
}

/// Pre-loaded norm weights for a single layer (input_norm + post_norm).
pub struct NormWeights {
    /// RMS norm before attention/DeltaNet (attn_norm.weight).
    pub input_norm: Tensor,
    /// RMS norm before MoE (post_attention_norm.weight or ffn_norm.weight).
    pub post_norm: Tensor,
}

/// Pre-loaded weights kept resident in memory.
pub struct ResidentWeights {
    /// Per-layer attention weights (None = load from GGUF on demand).
    pub attention: Vec<Option<AttentionWeights>>,
    /// Per-layer router gate weights on Metal (None = load from GGUF on demand).
    pub router_gates: Vec<Option<Tensor>>,
    /// Per-layer router gate weights on CPU.
    /// Avoids Metal→CPU sync barrier (~1.5ms/layer) during MoE routing.
    pub router_gates_cpu: Vec<Option<Tensor>>,
    /// Per-layer router gate bias on CPU (for models like GPT-OSS with ffn_gate_inp.bias).
    pub router_gate_biases: Vec<Option<Tensor>>,
    /// Per-layer DeltaNet weights on CPU (None = load from GGUF on demand).
    pub deltanet: Vec<Option<DeltaNetWeights>>,
    /// Per-layer per-expert weights for RAM Resident mode: [layer][expert].
    pub experts: Vec<Vec<Option<ExpertWeights>>>,
    /// Per-layer per-expert weights for GPU Resident mode: [layer][expert].
    /// Populated by preload_experts_gpu(). QMatMul for Q4 types, Dense(F16) for MXFP4 fallback.
    pub gpu_experts: Vec<Vec<Option<GpuExpertWeights>>>,
    /// Per-layer packed MXFP4 expert weights for batched Metal dispatch.
    /// Populated alongside gpu_experts during preload_experts_gpu() for MXFP4 layers.
    /// When present, run_moe_gpu_resident() uses the batched kernel path.
    #[cfg(feature = "metal")]
    pub packed_mxfp4: Vec<Option<PackedMxfp4Layer>>,
    /// Per-layer expert FFN bias data on Metal GPU for batched dispatch.
    /// Populated during preload_experts_gpu() if the model has expert biases
    /// (e.g. GPT-OSS has ffn_{gate,up,down}_exps.bias).
    #[cfg(feature = "metal")]
    pub expert_bias_buffers: Vec<Option<crate::metal::ExpertBiasBuffers>>,
    /// Pre-allocated intermediate buffers for batched MoE dispatch (shared across all layers).
    /// Avoids ~168 buffer allocations per token step. Allocated once after first packed layer.
    #[cfg(feature = "metal")]
    pub batched_moe_buffers: Option<crate::metal::BatchedMoeBuffers>,
    /// Per-layer shared expert weights.
    pub shared_experts: Vec<Option<SharedExpertWeights>>,
    /// Per-layer norm weights (tiny: ~16KB/layer, always preloaded).
    pub norms: Vec<Option<NormWeights>>,
    /// Per-layer indices of dummy (padded) experts that should never be routed to.
    /// Detected at gate preload time by checking for uniform gate weight rows.
    pub dummy_experts: Vec<Vec<usize>>,
    /// Per-layer GPU routing offset tables for GPU-side softmax+topk routing.
    /// Pre-uploaded at expert preload time when packed MXFP4 is available.
    /// Enables zero-sync MoE routing (24 GPU syncs/token → 0).
    #[cfg(feature = "metal")]
    pub gpu_routing_offsets: Vec<Option<crate::metal::GpuRoutingOffsets>>,
    /// Shared GPU routing output buffers (reused across layers).
    #[cfg(feature = "metal")]
    pub gpu_routing_out: Option<crate::metal::GpuRoutingOutputBuffers>,
}

impl ResidentWeights {
    /// Create empty (no pre-loaded weights).
    /// `experts_per_layer` gives per-layer expert counts (for pruned models with variable counts).
    pub fn empty(num_layers: usize, experts_per_layer: &[usize]) -> Self {
        Self {
            attention: (0..num_layers).map(|_| None).collect(),
            router_gates: (0..num_layers).map(|_| None).collect(),
            router_gates_cpu: (0..num_layers).map(|_| None).collect(),
            router_gate_biases: (0..num_layers).map(|_| None).collect(),
            deltanet: (0..num_layers).map(|_| None).collect(),
            experts: (0..num_layers)
                .map(|i| (0..experts_per_layer[i]).map(|_| None).collect())
                .collect(),
            gpu_experts: (0..num_layers)
                .map(|i| (0..experts_per_layer[i]).map(|_| None).collect())
                .collect(),
            #[cfg(feature = "metal")]
            packed_mxfp4: (0..num_layers).map(|_| None).collect(),
            #[cfg(feature = "metal")]
            expert_bias_buffers: (0..num_layers).map(|_| None).collect(),
            #[cfg(feature = "metal")]
            batched_moe_buffers: None,
            shared_experts: (0..num_layers).map(|_| None).collect(),
            norms: (0..num_layers).map(|_| None).collect(),
            dummy_experts: (0..num_layers).map(|_| Vec::new()).collect(),
            #[cfg(feature = "metal")]
            gpu_routing_offsets: (0..num_layers).map(|_| None).collect(),
            #[cfg(feature = "metal")]
            gpu_routing_out: None,
        }
    }
}

/// Per-layer entropy profiler for MoE routing analysis.
///
/// Collects per-token Shannon entropy of the full softmax routing distribution
/// at each layer. Used to determine whether layer-adaptive Dynamic K is viable.
pub struct EntropyProfiler {
    /// Accumulated entropy values per layer: entropy_log[layer_idx] = Vec<f32>
    entropy_log: Vec<Vec<f32>>,
    /// Whether profiling is active
    pub enabled: bool,
}

impl EntropyProfiler {
    /// Create a new profiler (disabled by default).
    pub fn new(num_layers: usize) -> Self {
        Self {
            entropy_log: (0..num_layers).map(|_| Vec::new()).collect(),
            enabled: false,
        }
    }

    /// Record an entropy value for a given layer.
    pub fn record(&mut self, layer_idx: usize, entropy: f32) {
        if self.enabled && layer_idx < self.entropy_log.len() {
            self.entropy_log[layer_idx].push(entropy);
        }
    }

    /// Get profiling results as a summary: per-layer (mean, std, min, max, count).
    pub fn summary(&self) -> Vec<EntropyLayerStats> {
        self.entropy_log
            .iter()
            .enumerate()
            .map(|(idx, vals)| {
                if vals.is_empty() {
                    return EntropyLayerStats {
                        layer_idx: idx,
                        count: 0,
                        mean: 0.0,
                        std: 0.0,
                        min: 0.0,
                        max: 0.0,
                    };
                }
                let n = vals.len() as f32;
                let mean = vals.iter().sum::<f32>() / n;
                let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
                let std = var.sqrt();
                let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                EntropyLayerStats {
                    layer_idx: idx,
                    count: vals.len(),
                    mean,
                    std,
                    min,
                    max,
                }
            })
            .collect()
    }

    /// Clear all collected data.
    pub fn clear(&mut self) {
        for v in &mut self.entropy_log {
            v.clear();
        }
    }

    /// Total number of entropy samples collected across all layers.
    pub fn total_samples(&self) -> usize {
        self.entropy_log.iter().map(|v| v.len()).sum()
    }
}

/// Per-layer entropy statistics.
pub struct EntropyLayerStats {
    pub layer_idx: usize,
    pub count: usize,
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
}

/// Per-expert activation statistics for calibration-based importance scoring.
///
/// Collects how often each expert is selected (top-K) and the mean gate weight
/// when selected. Used to compute `importance = frequency × mean_gate_weight`
/// for expert pruning decisions.
pub struct RoutingStatsCollector {
    /// Per-layer, per-expert: (activation_count, sum_of_gate_weights)
    expert_stats: Vec<Vec<(u64, f64)>>,
    /// Total tokens processed per layer
    total_tokens: Vec<u64>,
    /// Number of experts per layer
    num_experts: usize,
    /// Whether collection is active
    pub enabled: bool,
}

/// Per-layer routing statistics summary.
pub struct RoutingLayerStats {
    pub layer_idx: usize,
    pub total_tokens: u64,
    /// Per-expert: (expert_idx, count, frequency, mean_gate_weight, importance)
    pub experts: Vec<RoutingExpertStats>,
}

/// Per-expert routing statistics.
pub struct RoutingExpertStats {
    pub expert_idx: usize,
    pub count: u64,
    pub frequency: f64,
    pub mean_gate_weight: f64,
    pub importance: f64,
}

impl RoutingStatsCollector {
    /// Create a new collector (disabled by default).
    pub fn new(num_layers: usize, num_experts: usize) -> Self {
        Self {
            expert_stats: (0..num_layers)
                .map(|_| vec![(0u64, 0.0f64); num_experts])
                .collect(),
            total_tokens: vec![0u64; num_layers],
            num_experts,
            enabled: false,
        }
    }

    /// Record top-K routing decisions for one token at a given layer.
    ///
    /// `topk_indices` and `topk_weights` are parallel slices of length K.
    pub fn record(&mut self, layer_idx: usize, topk_indices: &[u32], topk_weights: &[f32]) {
        if !self.enabled || layer_idx >= self.expert_stats.len() {
            return;
        }
        self.total_tokens[layer_idx] += 1;
        for (&idx, &weight) in topk_indices.iter().zip(topk_weights.iter()) {
            let idx = idx as usize;
            if idx < self.num_experts {
                self.expert_stats[layer_idx][idx].0 += 1;
                self.expert_stats[layer_idx][idx].1 += weight as f64;
            }
        }
    }

    /// Get per-layer, per-expert routing statistics.
    pub fn summary(&self) -> Vec<RoutingLayerStats> {
        self.expert_stats
            .iter()
            .enumerate()
            .map(|(layer_idx, experts)| {
                let total = self.total_tokens[layer_idx];
                let expert_stats: Vec<RoutingExpertStats> = experts
                    .iter()
                    .enumerate()
                    .map(|(expert_idx, &(count, sum_weight))| {
                        let frequency = if total > 0 {
                            count as f64 / total as f64
                        } else {
                            0.0
                        };
                        let mean_gate_weight = if count > 0 {
                            sum_weight / count as f64
                        } else {
                            0.0
                        };
                        RoutingExpertStats {
                            expert_idx,
                            count,
                            frequency,
                            mean_gate_weight,
                            importance: frequency * mean_gate_weight,
                        }
                    })
                    .collect();
                RoutingLayerStats {
                    layer_idx,
                    total_tokens: total,
                    experts: expert_stats,
                }
            })
            .collect()
    }

    /// Clear all collected data.
    pub fn clear(&mut self) {
        for layer in &mut self.expert_stats {
            for stats in layer.iter_mut() {
                *stats = (0, 0.0);
            }
        }
        for t in &mut self.total_tokens {
            *t = 0;
        }
    }

    /// Total tokens processed across all layers.
    pub fn total_tokens_any_layer(&self) -> u64 {
        self.total_tokens.iter().max().copied().unwrap_or(0)
    }
}

/// Cache for adaptive expert skip: stores previous router logits and MoE outputs
/// per layer to detect when consecutive tokens have similar routing distributions.
/// When cosine similarity exceeds the threshold, the MoE computation is skipped
/// and the previous output is reused, eliminating SSD I/O for that layer.
///
/// Safety mechanisms to prevent quality degradation:
/// - Consecutive skip limit: forces recomputation after N consecutive skips
/// - Router logits comparison ensures the same experts would be selected
pub struct LayerOutputCache {
    /// Previous router logits per layer (CPU, f32).
    prev_router_logits: Vec<Option<Vec<f32>>>,
    /// Previous MoE block output per layer (CPU, f32, flattened).
    prev_moe_output: Vec<Option<Vec<f32>>>,
    /// Consecutive skip count per layer (reset on recompute).
    consecutive_skips: Vec<u32>,
    /// Maximum consecutive skips before forcing recompute (default: 3).
    pub max_consecutive_skips: u32,
    /// Cosine similarity threshold (default: 0.95).
    pub similarity_threshold: f32,
    /// Number of layers skipped.
    pub skip_count: u64,
    /// Total MoE layer invocations.
    pub total_count: u64,
}

impl LayerOutputCache {
    /// Create an empty cache for the given number of layers.
    pub fn empty(num_layers: usize, threshold: f32) -> Self {
        Self {
            prev_router_logits: (0..num_layers).map(|_| None).collect(),
            prev_moe_output: (0..num_layers).map(|_| None).collect(),
            consecutive_skips: vec![0; num_layers],
            max_consecutive_skips: 3,
            similarity_threshold: threshold,
            skip_count: 0,
            total_count: 0,
        }
    }

    /// Check if this layer should be skipped based on router logits similarity.
    /// Returns true if skip is safe (high similarity + cached output exists + not exceeded max skips).
    pub fn should_skip(&self, layer_idx: usize, current_logits: &[f32]) -> bool {
        // Don't skip too many times in a row (prevents error accumulation)
        if self.consecutive_skips[layer_idx] >= self.max_consecutive_skips {
            return false;
        }
        if let Some(prev) = &self.prev_router_logits[layer_idx] {
            if self.prev_moe_output[layer_idx].is_some() {
                return cosine_similarity(prev, current_logits) > self.similarity_threshold;
            }
        }
        false
    }

    /// Record that this layer was skipped (increments consecutive skip counter).
    pub fn record_skip(&mut self, layer_idx: usize) {
        self.consecutive_skips[layer_idx] += 1;
        self.skip_count += 1;
    }

    /// Record that this layer was recomputed (resets consecutive skip counter).
    pub fn record_compute(&mut self, layer_idx: usize) {
        self.consecutive_skips[layer_idx] = 0;
    }

    /// Get the cached MoE output for a layer (if available).
    pub fn get_cached_output(&self, layer_idx: usize) -> Option<&Vec<f32>> {
        self.prev_moe_output[layer_idx].as_ref()
    }

    /// Update the cache with new router logits and MoE output for a layer.
    pub fn update(&mut self, layer_idx: usize, router_logits: Vec<f32>, moe_output: Vec<f32>) {
        self.prev_router_logits[layer_idx] = Some(router_logits);
        self.prev_moe_output[layer_idx] = Some(moe_output);
    }

    /// Update only the router logits (used when skipping to track drift).
    pub fn update_logits(&mut self, layer_idx: usize, router_logits: Vec<f32>) {
        self.prev_router_logits[layer_idx] = Some(router_logits);
    }

    /// Skip rate as a percentage.
    pub fn skip_rate(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            self.skip_count as f64 / self.total_count as f64 * 100.0
        }
    }

    /// Clear all cached data (for new conversation/prompt).
    pub fn clear(&mut self) {
        for slot in &mut self.prev_router_logits {
            *slot = None;
        }
        for slot in &mut self.prev_moe_output {
            *slot = None;
        }
        for c in &mut self.consecutive_skips {
            *c = 0;
        }
        // Intentionally keep skip_count and total_count for stats
    }
}

/// Cosine similarity between two f32 slices.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}
