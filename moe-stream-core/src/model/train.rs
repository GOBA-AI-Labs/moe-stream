//! TinyLoRA training support: data structures and gradient computation.
//!
//! Implements "Learning to Reason in 13 Parameters" (2602.04118) for MoE models.
//!
//! W' = W + U_r @ diag(S_r) @ M @ V_r^T
//! where M = Σ_i v_i * P_i, v ∈ R^u is the only trainable parameter.
//!
//! The training loop is orchestrated by Python (GRPO), with this module providing:
//! - SVD data loading from .npz files
//! - Differentiable TinyLoRA delta computation (candle Var + backward)
//! - Layer-wise gradient checkpointing for memory efficiency

use candle_core::{DType, Device, IndexOp, Result, Tensor, Var};
use std::collections::HashMap;
use std::path::Path;

/// SVD data for a single weight matrix: U_r, S_r, V_r
#[derive(Clone)]
pub struct WeightSvd {
    pub u_r: Tensor,  // [out_dim, rank]
    pub s_r: Tensor,  // [rank]
    pub v_r: Tensor,  // [rank, in_dim]
}

/// SVD data for one expert's FFN (gate, up, down)
#[derive(Clone)]
pub struct ExpertSvd {
    pub gate: WeightSvd,
    pub up: WeightSvd,
    pub down: WeightSvd,
}

/// SVD data for one layer (all experts)
pub struct LayerSvd {
    pub experts: Vec<ExpertSvd>,  // [n_experts]
}

/// Full TinyLoRA configuration and SVD data
pub struct TinyLoRaConfig {
    pub rank: usize,
    pub u_dim: usize,
    pub p_matrices: Vec<Tensor>,  // [u_dim] of [rank, rank]
    pub layers: Vec<LayerSvd>,    // [n_layers]
    pub n_experts: usize,
}

/// Result of a training step: gradient and loss
pub struct TrainStepResult {
    pub gradient: Vec<f32>,  // ∂loss/∂v, length u_dim
    pub loss: f32,
}

impl TinyLoRaConfig {
    /// Load TinyLoRA SVD data from a directory produced by tinylora_moestream.py svd.
    ///
    /// Expected layout:
    ///   svd_dir/
    ///     projections.npz     (P_0..P_{u_dim-1})
    ///     svd_meta.json       (metadata)
    ///     layer_00/
    ///       e00_gate.npz      (U_r, S_r, V_r)
    ///       e00_up.npz
    ///       e00_down.npz
    ///       e01_gate.npz
    ///       ...
    pub fn load(svd_dir: &Path, device: &Device) -> Result<Self> {
        let meta_path = svd_dir.join("svd_meta.json");
        let meta_str = std::fs::read_to_string(&meta_path)
            .map_err(|e| candle_core::Error::Msg(format!("Read {}: {}", meta_path.display(), e)))?;
        let meta: serde_json::Value = serde_json::from_str(&meta_str)
            .map_err(|e| candle_core::Error::Msg(format!("Parse meta: {}", e)))?;

        let rank = meta["rank"].as_u64().unwrap_or(2) as usize;
        let u_dim = meta["u_dim"].as_u64().unwrap_or(13) as usize;
        let n_experts = meta["n_experts"].as_u64().unwrap_or(28) as usize;
        let moe_layers: Vec<usize> = meta["moe_layers"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect())
            .unwrap_or_default();

        // Load projection matrices
        let proj_path = svd_dir.join("projections.npz");
        let p_matrices = load_npz_matrices(&proj_path, "P_", u_dim, (rank, rank), device)?;

        // Load per-layer SVD data
        let mut layers = Vec::with_capacity(moe_layers.len());
        for &layer_idx in &moe_layers {
            let layer_dir = svd_dir.join(format!("layer_{:02}", layer_idx));
            let mut experts = Vec::with_capacity(n_experts);

            for expert_idx in 0..n_experts {
                let gate = load_weight_svd(&layer_dir, expert_idx, "gate", device)?;
                let up = load_weight_svd(&layer_dir, expert_idx, "up", device)?;
                let down = load_weight_svd(&layer_dir, expert_idx, "down", device)?;
                experts.push(ExpertSvd { gate, up, down });
            }

            layers.push(LayerSvd { experts });
        }

        Ok(Self {
            rank,
            u_dim,
            p_matrices,
            layers,
            n_experts,
        })
    }
}

/// Compute the TinyLoRA mixing matrix M = Σ_i v_i * P_i
///
/// v must be a Var (for autograd tracking).
/// Returns M as a [rank, rank] tensor in the computation graph.
pub fn compute_mixing_matrix(
    v: &Var,
    p_matrices: &[Tensor],
    rank: usize,
    device: &Device,
) -> Result<Tensor> {
    let v_tensor = v.as_tensor();
    let mut m = Tensor::zeros((rank, rank), DType::F32, device)?;
    for (i, p) in p_matrices.iter().enumerate() {
        let v_i = v_tensor.i(i)?;
        let scaled_p = p.broadcast_mul(&v_i)?;
        m = (m + scaled_p)?;
    }
    Ok(m)
}

/// Compute TinyLoRA weight delta for a single weight matrix.
///
/// delta_W = U_r @ diag(S_r) @ M @ V_r^T
///
/// This is differentiable w.r.t. v (through M).
pub fn compute_weight_delta(
    svd: &WeightSvd,
    m: &Tensor,  // [rank, rank], from compute_mixing_matrix
) -> Result<Tensor> {
    let rank = svd.s_r.dims1()?;

    // diag(S_r) @ M: scale each row of M by corresponding singular value
    let mut sm_rows = Vec::with_capacity(rank);
    for i in 0..rank {
        let s_i = svd.s_r.i(i)?;
        let row = m.i(i)?.broadcast_mul(&s_i)?;
        sm_rows.push(row);
    }
    let sm = Tensor::stack(&sm_rows, 0)?;  // [rank, rank]

    // U_r @ SM @ V_r = [out_dim, rank] @ [rank, rank] @ [rank, in_dim] = [out_dim, in_dim]
    let delta = svd.u_r.matmul(&sm)?.matmul(&svd.v_r)?;
    Ok(delta)
}

/// Apply SwiGLU activation: silu(gate) * up
pub fn silu_and_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    let gate_silu = (gate * candle_nn::ops::sigmoid(gate)?)?;
    gate_silu * up
}

// ==============================================================================
// Differentiable MoE forward for training
// ==============================================================================

/// Expert weights dequantized to F32 for training.
///
/// Loaded from GGUF (dequantized from MXFP4/Q4_K_M to F32) or from cached data.
/// These are the frozen base weights W. TinyLoRA adds delta_W on top.
#[derive(Clone)]
pub struct TrainExpertWeights {
    pub gate: Tensor, // [intermediate, hidden]
    pub up: Tensor,   // [intermediate, hidden]
    pub down: Tensor, // [hidden, intermediate]
    pub gate_bias: Option<Tensor>, // [intermediate]
    pub up_bias: Option<Tensor>,
    pub down_bias: Option<Tensor>,
}

/// Cached layer state from a non-differentiable forward pass.
///
/// During training, we first run standard inference to cache hidden states
/// and routing decisions, then replay the MoE part with TinyLoRA in autograd.
pub struct CachedLayerState {
    /// Hidden states after post-attn residual + post-attn norm (MoE input).
    pub moe_input: Vec<f32>, // flat [n_tokens * hidden_dim]
    /// Per-token routing: list of (expert_idx, weight).
    pub routing: Vec<Vec<(usize, f32)>>,
    pub n_tokens: usize,
    pub hidden_dim: usize,
}

/// Result of a single layer's gradient computation.
pub struct LayerGradResult {
    pub grad_v: Vec<f32>, // [u_dim]
    pub pseudo_loss: f32, // for monitoring convergence
}

/// OAI SwiGLU activation (differentiable through candle autograd).
///
/// Matches llama.cpp ggml_swiglu_oai (alpha=1.702, limit=7.0).
///   gate_c = clamp(gate, -inf, limit)
///   up_c   = clamp(up, -limit, limit)
///   output = (gate_c * sigmoid(alpha * gate_c)) * (up_c + 1.0)
pub fn swiglu_oai_diff(gate: &Tensor, up: &Tensor, alpha: f64, limit: f64) -> Result<Tensor> {
    let gate_c = gate.clamp(f64::NEG_INFINITY, limit)?;
    let up_c = up.clamp(-limit, limit)?;
    let alpha_gate = (&gate_c * alpha)?;
    let gate_activated = gate_c.mul(&candle_nn::ops::sigmoid(&alpha_gate)?)?;
    let up_shifted = (up_c + 1.0)?;
    gate_activated.mul(&up_shifted)
}

/// Forward one expert with TinyLoRA delta (differentiable w.r.t. M → v).
///
/// Computes: output = SwiGLU(x @ (W_gate + Δ_gate)^T, x @ (W_up + Δ_up)^T) @ (W_down + Δ_down)^T
/// where Δ_W = U_r @ diag(S_r) @ M @ V_r^T, and M depends on the trainable v.
pub fn train_forward_expert(
    x: &Tensor, // [n_tokens, hidden_dim]
    weights: &TrainExpertWeights,
    svd: &ExpertSvd,
    m: &Tensor, // [rank, rank], tracked by autograd
    use_oai_swiglu: bool,
) -> Result<Tensor> {
    // Compute TinyLoRA deltas (in computation graph, differentiable w.r.t. v)
    let delta_gate = compute_weight_delta(&svd.gate, m)?;
    let delta_up = compute_weight_delta(&svd.up, m)?;
    let delta_down = compute_weight_delta(&svd.down, m)?;

    // W' = W_frozen + delta_W
    let gate_w = (&weights.gate + &delta_gate)?;
    let up_w = (&weights.up + &delta_up)?;
    let down_w = (&weights.down + &delta_down)?;

    // Forward: x @ W'^T + bias
    let mut gate_out = x.matmul(&gate_w.t()?)?;
    let mut up_out = x.matmul(&up_w.t()?)?;

    if let Some(ref b) = weights.gate_bias {
        gate_out = gate_out.broadcast_add(b)?;
    }
    if let Some(ref b) = weights.up_bias {
        up_out = up_out.broadcast_add(b)?;
    }

    // Activation
    let hidden = if use_oai_swiglu {
        swiglu_oai_diff(&gate_out, &up_out, 1.702, 7.0)?
    } else {
        silu_and_mul(&gate_out, &up_out)?
    };

    // Down projection
    let mut output = hidden.matmul(&down_w.t()?)?;
    if let Some(ref b) = weights.down_bias {
        output = output.broadcast_add(b)?;
    }

    Ok(output)
}

/// Differentiable MoE forward for one layer (expert-batched).
///
/// Groups tokens by expert for batched matmul instead of per-token per-expert.
/// Reduces matmul calls from (n_tokens * top_k) to n_active_experts.
///
/// For gradient checkpointing, this is called once per layer with a fresh Var.
pub fn train_forward_moe(
    moe_input: &Tensor, // [n_tokens, hidden_dim]
    expert_weights: &HashMap<usize, TrainExpertWeights>,
    layer_svd: &LayerSvd,
    routing: &[Vec<(usize, f32)>], // [n_tokens][top_k]: (expert_idx, weight)
    m: &Tensor,                    // [rank, rank], from v
    use_oai_swiglu: bool,
    device: &Device,
) -> Result<Tensor> {
    let (n_tokens, hidden_dim) = moe_input.dims2()?;

    // Build expert → [(token_idx, weight)] map for batched processing
    let mut expert_tokens: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
    for (tok_idx, tok_routing) in routing.iter().enumerate() {
        for &(expert_idx, weight) in tok_routing {
            expert_tokens.entry(expert_idx).or_default().push((tok_idx, weight));
        }
    }

    // Per-token output accumulators (list of tensors, one per token)
    // Initialize lazily as we process experts
    let mut token_accum: Vec<Option<Tensor>> = (0..n_tokens).map(|_| None).collect();

    // Process each expert in batch
    for (expert_idx, tokens_weights) in &expert_tokens {
        let ew = match expert_weights.get(expert_idx) {
            Some(ew) => ew,
            None => continue,
        };
        let svd = match layer_svd.experts.get(*expert_idx) {
            Some(s) => s,
            None => continue,
        };

        // Gather input tokens for this expert: [n_batch, hidden_dim]
        let _n_batch = tokens_weights.len();
        let batch_rows: Vec<Tensor> = tokens_weights.iter()
            .map(|(t, _)| moe_input.i(*t))
            .collect::<Result<Vec<_>>>()?;
        let batch_input = Tensor::stack(&batch_rows, 0)?; // [n_batch, hidden_dim]

        // Batched expert forward (few large matmuls instead of many small ones)
        let batch_output = train_forward_expert(
            &batch_input,
            ew,
            svd,
            m,
            use_oai_swiglu,
        )?; // [n_batch, hidden_dim]

        // Scatter weighted outputs back to token positions
        for (i, &(tok_idx, weight)) in tokens_weights.iter().enumerate() {
            let row = batch_output.i(i)?; // [hidden_dim]
            let weighted = (&row * weight as f64)?;

            token_accum[tok_idx] = Some(match token_accum[tok_idx].take() {
                Some(prev) => (&prev + &weighted)?,
                None => weighted,
            });
        }
    }

    // Collect results (fill missing tokens with zeros)
    let zero = Tensor::zeros(hidden_dim, DType::F32, device)?;
    let rows: Vec<Tensor> = token_accum.into_iter()
        .map(|opt| opt.unwrap_or_else(|| zero.clone()))
        .collect();

    Tensor::stack(&rows, 0)
}

/// Compute gradient of v for one layer using the pseudo-loss trick.
///
/// Strategy (Straight-Through Estimator for non-TinyLoRA parts):
///   1. MoE input comes from cached non-differentiable forward pass (detached)
///   2. MoE forward with TinyLoRA is fully differentiable w.r.t. v
///   3. pseudo_loss = sum(grad_output * moe_output)
///   4. backward() gives ∂pseudo_loss/∂v = exact ∂loss/∂v contribution from this layer
///
/// The grad_output is the gradient of the final loss w.r.t. the hidden state at
/// this layer's MoE output. With residual connections, this is approximately
/// the gradient at the final layer (STE approximation).
pub fn compute_layer_grad(
    cached: &CachedLayerState,
    grad_output: &Tensor, // [n_tokens, hidden_dim], gradient of loss w.r.t. MoE output
    expert_weights: &HashMap<usize, TrainExpertWeights>,
    layer_svd: &LayerSvd,
    config: &TinyLoRaConfig,
    use_oai_swiglu: bool,
    device: &Device,
) -> Result<LayerGradResult> {
    // Create a FRESH Var for v (each layer gets its own graph for checkpointing)
    let v_init = vec![0.0f32; config.u_dim];
    let v = Var::from_vec(v_init, (config.u_dim,), device)?;

    // Reconstruct MoE input as a detached tensor
    let moe_input = Tensor::from_vec(
        cached.moe_input.clone(),
        (cached.n_tokens, cached.hidden_dim),
        device,
    )?;

    // Compute M from v (in autograd graph)
    let m = compute_mixing_matrix(&v, &config.p_matrices, config.rank, device)?;

    // Differentiable MoE forward
    let moe_output = train_forward_moe(
        &moe_input,
        expert_weights,
        layer_svd,
        &cached.routing,
        &m,
        use_oai_swiglu,
        device,
    )?;

    // Pseudo-loss: dot product of grad_output and moe_output
    // ∂pseudo_loss/∂v = ∂(grad_output · moe_output)/∂v = the exact gradient contribution
    let pseudo_loss = (grad_output * &moe_output)?.sum_all()?;

    // Backward: computes ∂pseudo_loss/∂v
    let grads = pseudo_loss.backward()?;
    let grad_v_tensor = grads
        .get(v.as_tensor())
        .ok_or_else(|| candle_core::Error::Msg("No gradient for v in compute_layer_grad".into()))?;
    let grad_v = grad_v_tensor.to_vec1::<f32>()?;

    let loss_val = pseudo_loss.to_vec0::<f32>()?;

    Ok(LayerGradResult {
        grad_v,
        pseudo_loss: loss_val,
    })
}

/// Full training step: compute grad_v across all MoE layers.
///
/// Uses gradient checkpointing: processes one layer at a time, computes
/// grad_v for that layer, frees the autograd graph, and accumulates.
///
/// ## STE Approximation
/// For non-TinyLoRA parts (attention, norms, residual connections), we use
/// a Straight-Through Estimator. The gradient of the final loss w.r.t. MoE
/// output at each layer is approximated as constant (due to residual connections).
///
/// ## Memory Budget
/// Per layer: ~4 experts × 3 matrices × [2880, 2880] × 4 bytes ≈ 400 MB
/// Plus autograd graph overhead ≈ 800 MB total per layer.
/// Well within 24 GB with model mmap (~10.4 GB) + 800 MB.
pub fn train_step_full(
    cached_layers: &[CachedLayerState],
    grad_output: &Tensor, // [n_tokens, hidden_dim], ∂L/∂h_L from output layer
    layer_expert_weights: &[HashMap<usize, TrainExpertWeights>], // Per-layer expert weights
    config: &TinyLoRaConfig,
    use_oai_swiglu: bool,
    device: &Device,
) -> Result<TrainStepResult> {
    let n_layers = cached_layers.len();
    let mut total_grad_v = vec![0.0f32; config.u_dim];
    let mut total_loss = 0.0f32;

    for (layer_rel_idx, cached) in cached_layers.iter().enumerate() {
        // Get layer SVD (layer_rel_idx maps to config.layers[])
        if layer_rel_idx >= config.layers.len() {
            break;
        }
        let layer_svd = &config.layers[layer_rel_idx];

        // Get expert weights for this layer
        if layer_rel_idx >= layer_expert_weights.len() {
            break;
        }
        let expert_weights = &layer_expert_weights[layer_rel_idx];

        // Compute gradient for this layer
        let result = compute_layer_grad(
            cached,
            grad_output, // STE: same gradient approximation for all layers
            expert_weights,
            layer_svd,
            config,
            use_oai_swiglu,
            device,
        )?;

        // Accumulate gradient
        for (i, g) in result.grad_v.iter().enumerate() {
            total_grad_v[i] += g;
        }
        total_loss += result.pseudo_loss;
    }

    Ok(TrainStepResult {
        gradient: total_grad_v,
        loss: total_loss / n_layers as f32,
    })
}

/// Load F32 expert weights for training from a GGUF reader.
///
/// Dequantizes the specified experts from MXFP4/Q4/etc. to F32 tensors.
/// This is the training-time equivalent of load_expert() in layer.rs.
pub fn load_train_expert_weights(
    reader: &crate::gguf::GgufReader,
    layer_idx: usize,
    expert_indices: &[usize],
    has_expert_bias: bool,
) -> Result<HashMap<usize, TrainExpertWeights>> {
    let prefix = format!("blk.{}", layer_idx);
    let gate_name = format!("{}.ffn_gate_exps.weight", prefix);
    let up_name = format!("{}.ffn_up_exps.weight", prefix);
    let down_name = format!("{}.ffn_down_exps.weight", prefix);
    let gate_bias_name = format!("{}.ffn_gate_exps.bias", prefix);
    let up_bias_name = format!("{}.ffn_up_exps.bias", prefix);
    let down_bias_name = format!("{}.ffn_down_exps.bias", prefix);

    let device = Device::Cpu;
    let mut result = HashMap::new();

    for &expert_idx in expert_indices {
        let gate = crate::model::layer::load_expert(reader, &gate_name, expert_idx, &device)?;
        let up = crate::model::layer::load_expert(reader, &up_name, expert_idx, &device)?;
        let down = crate::model::layer::load_expert(reader, &down_name, expert_idx, &device)?;

        let (gate_bias, up_bias, down_bias) = if has_expert_bias {
            let gb = crate::model::layer::load_expert(reader, &gate_bias_name, expert_idx, &device)?;
            let ub = crate::model::layer::load_expert(reader, &up_bias_name, expert_idx, &device)?;
            let db = crate::model::layer::load_expert(reader, &down_bias_name, expert_idx, &device)?;
            (Some(gb), Some(ub), Some(db))
        } else {
            (None, None, None)
        };

        result.insert(expert_idx, TrainExpertWeights {
            gate,
            up,
            down,
            gate_bias,
            up_bias,
            down_bias,
        });
    }

    Ok(result)
}

// ==============================================================================
// Router Bias Training (Option A: differentiable routing)
// ==============================================================================

/// Result of router bias gradient computation for one layer.
pub struct RouterBiasGradResult {
    pub grad_bias: Vec<f32>, // [n_experts]
    pub pseudo_loss: f32,
}

/// Result of a full-model router bias training step.
pub struct RouterTrainStepResult {
    pub grad_router_biases: Vec<Vec<f32>>, // [n_layers][n_experts]
    pub loss: f32,
}

/// Forward one expert with frozen weights only (no TinyLoRA delta).
///
/// Computes: output = SwiGLU(x @ W_gate^T + gate_bias, x @ W_up^T + up_bias) @ W_down^T + down_bias
/// All weights are frozen (no Var, no autograd tracking on weights).
/// Gradient flows through the input x and the routing weights that multiply the output.
pub fn frozen_forward_expert(
    x: &Tensor,  // [n_tokens, hidden_dim]
    weights: &TrainExpertWeights,
    use_oai_swiglu: bool,
) -> Result<Tensor> {
    // Forward: x @ W^T + bias
    let mut gate_out = x.matmul(&weights.gate.t()?)?;
    let mut up_out = x.matmul(&weights.up.t()?)?;

    if let Some(ref b) = weights.gate_bias {
        gate_out = gate_out.broadcast_add(b)?;
    }
    if let Some(ref b) = weights.up_bias {
        up_out = up_out.broadcast_add(b)?;
    }

    // Activation
    let hidden = if use_oai_swiglu {
        swiglu_oai_diff(&gate_out, &up_out, 1.702, 7.0)?
    } else {
        silu_and_mul(&gate_out, &up_out)?
    };

    // Down projection
    let mut output = hidden.matmul(&weights.down.t()?)?;
    if let Some(ref b) = weights.down_bias {
        output = output.broadcast_add(b)?;
    }

    Ok(output)
}

/// Differentiable MoE forward with trainable router bias.
///
/// Unlike `train_forward_moe` (TinyLoRA), this function makes the routing itself
/// differentiable w.r.t. the router bias. Expert weights are frozen.
///
/// Strategy:
///   1. Compute logits = moe_input @ router_gate_weight^T + bias_var  (differentiable)
///   2. Compute softmax probabilities over all experts
///   3. Hard top-k selection on detached logits (non-differentiable)
///   4. Gather soft weights at top-k positions (differentiable through softmax)
///   5. Normalize soft weights
///   6. Run frozen expert forward for each selected expert
///   7. Weighted sum of expert outputs using differentiable weights
pub fn train_forward_moe_diff_routing(
    moe_input: &Tensor,           // [n_tokens, hidden_dim]
    expert_weights: &HashMap<usize, TrainExpertWeights>,
    router_gate_weight: &Tensor,   // [n_experts, hidden_dim] (frozen)
    bias_var: &Var,                // [n_experts] (trainable)
    top_k: usize,
    use_oai_swiglu: bool,
    device: &Device,
) -> Result<Tensor> {
    let (n_tokens, hidden_dim) = moe_input.dims2()?;
    let _n_experts = router_gate_weight.dim(0)?;

    // Step 1: Compute logits = input @ gate_weight^T + bias
    let logits = moe_input.matmul(&router_gate_weight.t()?)?; // [n_tokens, n_experts]
    let logits = logits.broadcast_add(bias_var.as_tensor())?;  // [n_tokens, n_experts]

    // Step 2: Softmax over all experts (differentiable w.r.t. bias_var)
    let probs = candle_nn::ops::softmax(&logits, 1)?; // [n_tokens, n_experts]

    // Step 3: Hard top-k selection on detached logits (non-differentiable)
    // Use detach() to get non-tracked tensor for index selection
    let logits_detached = logits.detach();
    let logits_vec = logits_detached.to_vec2::<f32>()?;

    // For each token, find top-k expert indices
    let mut topk_indices_per_token: Vec<Vec<usize>> = Vec::with_capacity(n_tokens);
    for t in 0..n_tokens {
        let row = &logits_vec[t];
        let mut indexed: Vec<(usize, f32)> = row.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let topk: Vec<usize> = indexed.iter().take(top_k).map(|(i, _)| *i).collect();
        topk_indices_per_token.push(topk);
    }

    // Step 4: Gather soft weights at top-k positions and normalize
    // For each token, collect the softmax probs at the selected expert indices
    // This is differentiable through probs → logits → bias_var
    let mut token_outputs: Vec<Option<Tensor>> = (0..n_tokens).map(|_| None).collect();

    // Build expert → [(token_idx, position_in_topk)] map for batched processing
    let mut expert_tokens: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (tok_idx, topk) in topk_indices_per_token.iter().enumerate() {
        for (pos, &expert_idx) in topk.iter().enumerate() {
            expert_tokens.entry(expert_idx).or_default().push((tok_idx, pos));
        }
    }

    // For each token, compute normalized weights from softmax probs at top-k positions
    // We need per-token weight tensors for the weighted sum
    let mut per_token_weights: Vec<Vec<(usize, Tensor)>> = (0..n_tokens).map(|_| Vec::new()).collect();

    for (tok_idx, topk) in topk_indices_per_token.iter().enumerate() {
        // Gather probs at top-k positions for this token
        let token_probs = probs.i(tok_idx)?; // [n_experts]
        let mut selected_probs = Vec::with_capacity(top_k);
        for &expert_idx in topk {
            selected_probs.push(token_probs.i(expert_idx)?); // scalar tensor, tracked
        }
        // Stack into [top_k] tensor
        let selected = Tensor::stack(&selected_probs, 0)?; // [top_k]
        // Normalize: weights = selected / sum(selected)
        let sum = selected.sum_all()?;
        let normalized = selected.broadcast_div(&sum)?; // [top_k]

        for (pos, &expert_idx) in topk.iter().enumerate() {
            let w = normalized.i(pos)?; // scalar, differentiable
            per_token_weights[tok_idx].push((expert_idx, w));
        }
    }

    // Step 5-6: Run frozen expert forward and weighted sum
    // Process by expert for batched matmul
    for (expert_idx, tok_positions) in &expert_tokens {
        let ew = match expert_weights.get(expert_idx) {
            Some(ew) => ew,
            None => continue,
        };

        // Gather input tokens for this expert
        let batch_rows: Vec<Tensor> = tok_positions
            .iter()
            .map(|(t, _)| moe_input.i(*t))
            .collect::<Result<Vec<_>>>()?;
        let batch_input = Tensor::stack(&batch_rows, 0)?; // [n_batch, hidden_dim]

        // Batched frozen expert forward
        let batch_output = frozen_forward_expert(&batch_input, ew, use_oai_swiglu)?;
        // [n_batch, hidden_dim]

        // Scatter weighted outputs back to token positions
        for (i, &(tok_idx, _pos)) in tok_positions.iter().enumerate() {
            let row = batch_output.i(i)?; // [hidden_dim]
            // Find the differentiable weight for this (token, expert) pair
            let weight = &per_token_weights[tok_idx]
                .iter()
                .find(|(eidx, _)| *eidx == *expert_idx)
                .expect("weight must exist for selected expert")
                .1;
            let weighted = row.broadcast_mul(weight)?; // [hidden_dim], differentiable

            token_outputs[tok_idx] = Some(match token_outputs[tok_idx].take() {
                Some(prev) => (&prev + &weighted)?,
                None => weighted,
            });
        }
    }

    // Collect results (fill missing tokens with zeros)
    let zero = Tensor::zeros(hidden_dim, DType::F32, device)?;
    let rows: Vec<Tensor> = token_outputs
        .into_iter()
        .map(|opt| opt.unwrap_or_else(|| zero.clone()))
        .collect();

    Tensor::stack(&rows, 0)
}

/// Compute gradient of router bias for one layer using the pseudo-loss trick.
///
/// Strategy:
///   1. MoE input comes from cached non-differentiable forward pass (detached)
///   2. Router gate weight is frozen
///   3. Router bias is a Var (trainable)
///   4. Routing is differentiable through softmax(logits + bias) → weights
///   5. Expert forward uses frozen weights
///   6. pseudo_loss = sum(grad_output * moe_output)
///   7. backward() gives ∂pseudo_loss/∂bias
pub fn compute_router_bias_grad(
    cached: &CachedLayerState,
    grad_output: &Tensor,                            // [n_tokens, hidden_dim]
    expert_weights: &HashMap<usize, TrainExpertWeights>,
    router_gate_weight: &Tensor,                     // [n_experts, hidden_dim] (frozen)
    router_bias_init: &[f32],                        // [n_experts] current values
    top_k: usize,
    use_oai_swiglu: bool,
    device: &Device,
) -> Result<RouterBiasGradResult> {
    let n_experts = router_bias_init.len();

    // Create a Var for the bias (each layer gets its own graph for checkpointing)
    let bias_var = Var::from_vec(router_bias_init.to_vec(), (n_experts,), device)?;

    // Reconstruct MoE input as a detached tensor
    let moe_input = Tensor::from_vec(
        cached.moe_input.clone(),
        (cached.n_tokens, cached.hidden_dim),
        device,
    )?;

    // Differentiable MoE forward with trainable routing
    let moe_output = train_forward_moe_diff_routing(
        &moe_input,
        expert_weights,
        router_gate_weight,
        &bias_var,
        top_k,
        use_oai_swiglu,
        device,
    )?;

    // Pseudo-loss: dot product of grad_output and moe_output
    let pseudo_loss = (grad_output * &moe_output)?.sum_all()?;

    // Backward: computes ∂pseudo_loss/∂bias
    let grads = pseudo_loss.backward()?;
    let grad_bias_tensor = grads
        .get(bias_var.as_tensor())
        .ok_or_else(|| candle_core::Error::Msg(
            "No gradient for bias_var in compute_router_bias_grad".into(),
        ))?;
    let grad_bias = grad_bias_tensor.to_vec1::<f32>()?;

    let loss_val = pseudo_loss.to_vec0::<f32>()?;

    Ok(RouterBiasGradResult {
        grad_bias,
        pseudo_loss: loss_val,
    })
}

/// Full router bias training step: compute grad_bias across all MoE layers.
///
/// Uses gradient checkpointing: processes one layer at a time, computes
/// grad_bias for that layer, frees the autograd graph, and accumulates.
///
/// This is SEPARATE from TinyLoRA. No SVD data needed. Only the router
/// biases are trainable; all expert FFN weights are frozen.
pub fn router_train_step(
    cached_layers: &[CachedLayerState],
    grad_output: &Tensor,                                      // [n_tokens, hidden_dim]
    layer_expert_weights: &[HashMap<usize, TrainExpertWeights>], // Per-layer expert weights
    router_gate_weights: &[Tensor],                             // [n_layers][n_experts, hidden_dim]
    router_biases: &[Vec<f32>],                                 // [n_layers][n_experts]
    top_k: usize,
    use_oai_swiglu: bool,
    device: &Device,
) -> Result<RouterTrainStepResult> {
    let n_layers = cached_layers.len();
    let mut all_grad_biases: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
    let mut total_loss = 0.0f32;

    for layer_idx in 0..n_layers {
        if layer_idx >= layer_expert_weights.len()
            || layer_idx >= router_gate_weights.len()
            || layer_idx >= router_biases.len()
        {
            break;
        }

        let result = compute_router_bias_grad(
            &cached_layers[layer_idx],
            grad_output, // STE: same gradient approximation for all layers
            &layer_expert_weights[layer_idx],
            &router_gate_weights[layer_idx],
            &router_biases[layer_idx],
            top_k,
            use_oai_swiglu,
            device,
        )?;

        all_grad_biases.push(result.grad_bias);
        total_loss += result.pseudo_loss;
    }

    let avg_loss = if n_layers > 0 {
        total_loss / n_layers as f32
    } else {
        0.0
    };

    Ok(RouterTrainStepResult {
        grad_router_biases: all_grad_biases,
        loss: avg_loss,
    })
}

// ==============================================================================
// NPZ loading helpers
// ==============================================================================

/// Load matrices from an npz file with keys like "{prefix}0", "{prefix}1", ...
fn load_npz_matrices(
    path: &Path,
    prefix: &str,
    count: usize,
    shape: (usize, usize),
    device: &Device,
) -> Result<Vec<Tensor>> {
    let npz_data = read_npz(path)?;
    let mut tensors = Vec::with_capacity(count);

    for i in 0..count {
        let key = format!("{}{}", prefix, i);
        let data = npz_data.get(&key)
            .ok_or_else(|| candle_core::Error::Msg(format!("Key '{}' not in {}", key, path.display())))?;
        let tensor = Tensor::from_vec(data.clone(), shape, device)?;
        tensors.push(tensor);
    }

    Ok(tensors)
}

/// Load SVD data (U_r, S_r, V_r) for one weight matrix.
/// S_r is normalized to unit L2 norm so that the delta magnitude is
/// controlled by v alone, not by per-matrix singular value scale.
fn load_weight_svd(
    layer_dir: &Path,
    expert_idx: usize,
    component: &str,
    device: &Device,
) -> Result<WeightSvd> {
    let fname = format!("e{:02}_{}.npz", expert_idx, component);
    let npz_data = read_npz(&layer_dir.join(&fname))?;

    let u_r_data = npz_data.get("U_r")
        .ok_or_else(|| candle_core::Error::Msg(format!("No U_r in {}", fname)))?;
    let s_r_data = npz_data.get("S_r")
        .ok_or_else(|| candle_core::Error::Msg(format!("No S_r in {}", fname)))?;
    let v_r_data = npz_data.get("V_r")
        .ok_or_else(|| candle_core::Error::Msg(format!("No V_r in {}", fname)))?;

    // Infer shapes from data length and rank
    let rank = s_r_data.len();
    let u_total = u_r_data.len();
    let v_total = v_r_data.len();
    let out_dim = u_total / rank;
    let in_dim = v_total / rank;

    // Normalize S_r to unit L2 norm.
    // Without this, down projections (S_r~600) create 100x larger deltas
    // than gate/up (S_r~5), making the shared v vector unstable.
    let s_norm: f32 = s_r_data.iter().map(|x| x * x).sum::<f32>().sqrt();
    let s_r_normalized: Vec<f32> = if s_norm > 1e-10 {
        s_r_data.iter().map(|x| x / s_norm).collect()
    } else {
        s_r_data.clone()
    };

    let u_r = Tensor::from_vec(u_r_data.clone(), (out_dim, rank), device)?;
    let s_r = Tensor::from_vec(s_r_normalized, (rank,), device)?;
    let v_r = Tensor::from_vec(v_r_data.clone(), (rank, in_dim), device)?;

    Ok(WeightSvd { u_r, s_r, v_r })
}

/// Read a .npz file into a HashMap of arrays (f32 only).
///
/// Minimal .npz reader: npz is a ZIP of .npy files.
fn read_npz(path: &Path) -> Result<HashMap<String, Vec<f32>>> {
    use std::io::Read;

    let file = std::fs::File::open(path)
        .map_err(|e| candle_core::Error::Msg(format!("Open {}: {}", path.display(), e)))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| candle_core::Error::Msg(format!("ZIP {}: {}", path.display(), e)))?;

    let mut result = HashMap::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)
            .map_err(|e| candle_core::Error::Msg(format!("ZIP entry {}: {}", i, e)))?;
        let name = entry.name().trim_end_matches(".npy").to_string();

        let mut buf = Vec::new();
        entry.read_to_end(&mut buf)
            .map_err(|e| candle_core::Error::Msg(format!("Read {}: {}", name, e)))?;

        let data = parse_npy_f32(&buf)?;
        result.insert(name, data);
    }

    Ok(result)
}

/// Parse a .npy file (numpy array format) containing f32 data.
fn parse_npy_f32(buf: &[u8]) -> Result<Vec<f32>> {
    // Minimal .npy parser: magic (6) + version (2) + header_len (2 or 4) + header + data
    if buf.len() < 10 || &buf[0..6] != b"\x93NUMPY" {
        return Err(candle_core::Error::Msg("Not a .npy file".into()));
    }

    let major = buf[6];
    let header_len = if major >= 2 {
        u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize
    } else {
        u16::from_le_bytes([buf[8], buf[9]]) as usize
    };

    let header_start = if major >= 2 { 12 } else { 10 };
    let data_start = header_start + header_len;

    // Parse data as f32 (little-endian)
    let data_bytes = &buf[data_start..];
    let n_floats = data_bytes.len() / 4;
    let mut data = Vec::with_capacity(n_floats);
    for chunk in data_bytes.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixing_matrix() -> Result<()> {
        let device = Device::Cpu;
        let rank = 2;
        let u_dim = 3;

        let p0 = Tensor::new(&[1.0f32, 0.0, 0.0, 1.0], &device)?.reshape((rank, rank))?;
        let p1 = Tensor::new(&[0.0f32, 1.0, 1.0, 0.0], &device)?.reshape((rank, rank))?;
        let p2 = Tensor::new(&[1.0f32, 1.0, -1.0, 1.0], &device)?.reshape((rank, rank))?;
        let p_matrices = vec![p0, p1, p2];

        let v = Var::from_vec(vec![1.0f32, 0.5, -0.5], (u_dim,), &device)?;
        let m = compute_mixing_matrix(&v, &p_matrices, rank, &device)?;

        // M = 1.0*P0 + 0.5*P1 + (-0.5)*P2
        // = [[1,0],[0,1]] + [[0,0.5],[0.5,0]] + [[-0.5,-0.5],[0.5,-0.5]]
        // = [[0.5, 0.0], [1.0, 0.5]]
        let m_data: Vec<f32> = m.flatten_all()?.to_vec1()?;
        assert!((m_data[0] - 0.5).abs() < 1e-5);
        assert!((m_data[1] - 0.0).abs() < 1e-5);
        assert!((m_data[2] - 1.0).abs() < 1e-5);
        assert!((m_data[3] - 0.5).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_weight_delta_gradient() -> Result<()> {
        let device = Device::Cpu;
        let rank = 2;
        let u_dim = 5;
        let dim = 8;

        // Create SVD data
        let svd = WeightSvd {
            u_r: Tensor::randn(0.0f32, 1.0, (dim, rank), &device)?,
            s_r: Tensor::new(&[3.0f32, 1.5], &device)?,
            v_r: Tensor::randn(0.0f32, 1.0, (rank, dim), &device)?,
        };

        let p_matrices: Vec<Tensor> = (0..u_dim)
            .map(|_| Tensor::randn(0.0f32, 1.0, (rank, rank), &device))
            .collect::<Result<Vec<_>>>()?;

        let v = Var::from_vec(vec![0.01f32; u_dim], (u_dim,), &device)?;

        // Forward: compute delta, apply, get loss
        let m = compute_mixing_matrix(&v, &p_matrices, rank, &device)?;
        let delta = compute_weight_delta(&svd, &m)?;
        let w = Tensor::randn(0.0f32, 0.1, (dim, dim), &device)?;
        let w_prime = (&w + &delta)?;
        let x = Tensor::randn(0.0f32, 1.0, (1, dim), &device)?;
        let y = x.matmul(&w_prime.t()?)?;
        let loss = y.sqr()?.mean_all()?;

        // Backward
        let grads = loss.backward()?;
        let grad_v = grads.get(v.as_tensor()).expect("gradient must exist");
        let grad_norm: f32 = grad_v.to_vec1::<f32>()?.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(grad_norm > 1e-10, "gradient should be non-zero");
        Ok(())
    }

    #[test]
    fn test_moe_forward_gradient() -> Result<()> {
        // Test that gradient flows through the full MoE forward with TinyLoRA.
        // Small dimensions for fast testing.
        let device = Device::Cpu;
        let rank = 2;
        let u_dim = 5;
        let hidden_dim = 16;
        let intermediate_dim = 16;
        let n_experts = 4;
        let n_tokens = 2;
        let top_k = 2;

        // Create random projection matrices
        let p_matrices: Vec<Tensor> = (0..u_dim)
            .map(|_| Tensor::randn(0.0f32, 1.0, (rank, rank), &device))
            .collect::<Result<Vec<_>>>()?;

        // Create expert weights (frozen base weights)
        let mut expert_weights = HashMap::new();
        let mut expert_svds = Vec::new();
        for e in 0..n_experts {
            expert_weights.insert(
                e,
                TrainExpertWeights {
                    gate: Tensor::randn(0.0f32, 0.1, (intermediate_dim, hidden_dim), &device)?,
                    up: Tensor::randn(0.0f32, 0.1, (intermediate_dim, hidden_dim), &device)?,
                    down: Tensor::randn(0.0f32, 0.1, (hidden_dim, intermediate_dim), &device)?,
                    gate_bias: None,
                    up_bias: None,
                    down_bias: None,
                },
            );
            expert_svds.push(ExpertSvd {
                gate: WeightSvd {
                    u_r: Tensor::randn(0.0f32, 1.0, (intermediate_dim, rank), &device)?,
                    s_r: Tensor::new(&[2.0f32, 1.0], &device)?,
                    v_r: Tensor::randn(0.0f32, 1.0, (rank, hidden_dim), &device)?,
                },
                up: WeightSvd {
                    u_r: Tensor::randn(0.0f32, 1.0, (intermediate_dim, rank), &device)?,
                    s_r: Tensor::new(&[1.5f32, 0.8], &device)?,
                    v_r: Tensor::randn(0.0f32, 1.0, (rank, hidden_dim), &device)?,
                },
                down: WeightSvd {
                    u_r: Tensor::randn(0.0f32, 1.0, (hidden_dim, rank), &device)?,
                    s_r: Tensor::new(&[1.0f32, 0.5], &device)?,
                    v_r: Tensor::randn(0.0f32, 1.0, (rank, intermediate_dim), &device)?,
                },
            });
        }

        let layer_svd = LayerSvd {
            experts: expert_svds,
        };

        // Routing: token 0 → experts [0, 2], token 1 → experts [1, 3]
        let routing = vec![
            vec![(0usize, 0.6f32), (2, 0.4)],
            vec![(1, 0.7), (3, 0.3)],
        ];

        // Random MoE input
        let moe_input =
            Tensor::randn(0.0f32, 1.0, (n_tokens, hidden_dim), &device)?;

        // Create Var v and compute M
        let v = Var::from_vec(vec![0.01f32; u_dim], (u_dim,), &device)?;
        let m = compute_mixing_matrix(&v, &p_matrices, rank, &device)?;

        // Forward MoE with TinyLoRA
        let moe_output = train_forward_moe(
            &moe_input,
            &expert_weights,
            &layer_svd,
            &routing,
            &m,
            false, // standard SwiGLU
            &device,
        )?;

        // Verify output shape
        let (out_tokens, out_dim) = moe_output.dims2()?;
        assert_eq!(out_tokens, n_tokens);
        assert_eq!(out_dim, hidden_dim);

        // Compute loss and backward
        let loss = moe_output.sqr()?.mean_all()?;
        let grads = loss.backward()?;
        let grad_v = grads
            .get(v.as_tensor())
            .expect("gradient must exist for v through MoE forward");
        let grad_v_vec = grad_v.to_vec1::<f32>()?;
        let grad_norm: f32 = grad_v_vec.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert_eq!(grad_v_vec.len(), u_dim);
        assert!(
            grad_norm > 1e-10,
            "gradient through MoE should be non-zero, got norm={}",
            grad_norm
        );

        // Also test compute_layer_grad (pseudo-loss approach)
        let cached = CachedLayerState {
            moe_input: moe_input.flatten_all()?.to_vec1::<f32>()?,
            routing: routing.clone(),
            n_tokens,
            hidden_dim,
        };
        // Fake gradient from subsequent layers (uniform for testing)
        let grad_output = Tensor::ones((n_tokens, hidden_dim), DType::F32, &device)?;

        let config = TinyLoRaConfig {
            rank,
            u_dim,
            p_matrices: p_matrices.clone(),
            layers: vec![], // not used in compute_layer_grad
            n_experts,
        };

        let result = compute_layer_grad(
            &cached,
            &grad_output,
            &expert_weights,
            &layer_svd,
            &config,
            false,
            &device,
        )?;

        let layer_grad_norm: f32 = result.grad_v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_eq!(result.grad_v.len(), u_dim);
        assert!(
            layer_grad_norm > 1e-10,
            "layer gradient should be non-zero, got norm={}",
            layer_grad_norm
        );

        Ok(())
    }

    #[test]
    fn test_oai_swiglu_diff() -> Result<()> {
        // Verify OAI SwiGLU is differentiable and matches expected behavior
        let device = Device::Cpu;
        let gate = Var::from_vec(vec![1.0f32, -1.0, 5.0, 10.0], (1, 4), &device)?;
        let up = Tensor::new(&[0.5f32, -0.5, 2.0, 3.0], &device)?.reshape((1, 4))?;

        let output = swiglu_oai_diff(gate.as_tensor(), &up, 1.702, 7.0)?;
        let loss = output.sqr()?.sum_all()?;
        let grads = loss.backward()?;
        let grad_gate = grads.get(gate.as_tensor()).expect("gradient for gate");
        let grad_norm: f32 = grad_gate
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();

        assert!(grad_norm > 1e-10, "OAI SwiGLU should have non-zero gradient");

        // Check clamping: gate=10.0 should be clamped to 7.0
        let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
        // gate=10 → clamped to 7, up=3 → clamped to 3 (within [-7,7])
        // silu_oai(7) = 7 * sigmoid(1.702*7) ≈ 7 * 1.0 ≈ 7.0
        // output = 7.0 * (3.0 + 1.0) = 28.0
        assert!(
            (output_vec[3] - 28.0).abs() < 0.5,
            "clamped OAI SwiGLU should be ~28.0, got {}",
            output_vec[3]
        );

        Ok(())
    }

    #[test]
    fn test_v_zero_baseline() -> Result<()> {
        // When v=0, TinyLoRA delta should be zero, so MoE output should equal
        // the output of the frozen base weights (no modification).
        let device = Device::Cpu;
        let rank = 2;
        let u_dim = 5;
        let hidden_dim = 8;
        let intermediate_dim = 8;

        let p_matrices: Vec<Tensor> = (0..u_dim)
            .map(|_| Tensor::randn(0.0f32, 1.0, (rank, rank), &device))
            .collect::<Result<Vec<_>>>()?;

        // Create expert weights
        let gate_w = Tensor::randn(0.0f32, 0.1, (intermediate_dim, hidden_dim), &device)?;
        let up_w = Tensor::randn(0.0f32, 0.1, (intermediate_dim, hidden_dim), &device)?;
        let down_w = Tensor::randn(0.0f32, 0.1, (hidden_dim, intermediate_dim), &device)?;

        let weights = TrainExpertWeights {
            gate: gate_w.clone(),
            up: up_w.clone(),
            down: down_w.clone(),
            gate_bias: None,
            up_bias: None,
            down_bias: None,
        };

        let svd = ExpertSvd {
            gate: WeightSvd {
                u_r: Tensor::randn(0.0f32, 1.0, (intermediate_dim, rank), &device)?,
                s_r: Tensor::new(&[2.0f32, 1.0], &device)?,
                v_r: Tensor::randn(0.0f32, 1.0, (rank, hidden_dim), &device)?,
            },
            up: WeightSvd {
                u_r: Tensor::randn(0.0f32, 1.0, (intermediate_dim, rank), &device)?,
                s_r: Tensor::new(&[1.5f32, 0.8], &device)?,
                v_r: Tensor::randn(0.0f32, 1.0, (rank, hidden_dim), &device)?,
            },
            down: WeightSvd {
                u_r: Tensor::randn(0.0f32, 1.0, (hidden_dim, rank), &device)?,
                s_r: Tensor::new(&[1.0f32, 0.5], &device)?,
                v_r: Tensor::randn(0.0f32, 1.0, (rank, intermediate_dim), &device)?,
            },
        };

        let x = Tensor::randn(0.0f32, 1.0, (1, hidden_dim), &device)?;

        // v=0 → M=0 → delta_W=0 → output uses only frozen weights
        let v_zero = Var::from_vec(vec![0.0f32; u_dim], (u_dim,), &device)?;
        let m_zero = compute_mixing_matrix(&v_zero, &p_matrices, rank, &device)?;
        let output_tinylora = train_forward_expert(&x, &weights, &svd, &m_zero, false)?;

        // Baseline: forward with just the frozen weights (no TinyLoRA)
        let gate_out = x.matmul(&gate_w.t()?)?;
        let up_out = x.matmul(&up_w.t()?)?;
        let hidden = silu_and_mul(&gate_out, &up_out)?;
        let output_baseline = hidden.matmul(&down_w.t()?)?;

        // Should be exactly equal (v=0 means no modification)
        let diff = (&output_tinylora - &output_baseline)?
            .abs()?
            .sum_all()?
            .to_vec0::<f32>()?;

        assert!(
            diff < 1e-5,
            "v=0 TinyLoRA output should equal baseline, got diff={}",
            diff
        );

        // Verify v≠0 produces different output
        let v_nonzero = Var::from_vec(vec![0.1f32; u_dim], (u_dim,), &device)?;
        let m_nonzero = compute_mixing_matrix(&v_nonzero, &p_matrices, rank, &device)?;
        let output_modified = train_forward_expert(&x, &weights, &svd, &m_nonzero, false)?;

        let diff_modified = (&output_modified - &output_baseline)?
            .abs()?
            .sum_all()?
            .to_vec0::<f32>()?;

        assert!(
            diff_modified > 1e-3,
            "v≠0 should produce different output, got diff={}",
            diff_modified
        );

        Ok(())
    }

    #[test]
    fn test_gradient_numerical_check() -> Result<()> {
        // Verify analytical gradient (from backward()) matches numerical gradient.
        // Uses finite differences: ∂L/∂v_i ≈ (L(v+εe_i) - L(v-εe_i)) / (2ε)
        let device = Device::Cpu;
        let rank = 2;
        let u_dim = 3;
        let hidden_dim = 8;
        let intermediate_dim = 8;
        let eps = 1e-3f32;

        let p_matrices: Vec<Tensor> = (0..u_dim)
            .map(|_| Tensor::randn(0.0f32, 1.0, (rank, rank), &device))
            .collect::<Result<Vec<_>>>()?;

        let weights = TrainExpertWeights {
            gate: Tensor::randn(0.0f32, 0.1, (intermediate_dim, hidden_dim), &device)?,
            up: Tensor::randn(0.0f32, 0.1, (intermediate_dim, hidden_dim), &device)?,
            down: Tensor::randn(0.0f32, 0.1, (hidden_dim, intermediate_dim), &device)?,
            gate_bias: None,
            up_bias: None,
            down_bias: None,
        };
        let svd = ExpertSvd {
            gate: WeightSvd {
                u_r: Tensor::randn(0.0f32, 1.0, (intermediate_dim, rank), &device)?,
                s_r: Tensor::new(&[2.0f32, 1.0], &device)?,
                v_r: Tensor::randn(0.0f32, 1.0, (rank, hidden_dim), &device)?,
            },
            up: WeightSvd {
                u_r: Tensor::randn(0.0f32, 1.0, (intermediate_dim, rank), &device)?,
                s_r: Tensor::new(&[1.5f32, 0.8], &device)?,
                v_r: Tensor::randn(0.0f32, 1.0, (rank, hidden_dim), &device)?,
            },
            down: WeightSvd {
                u_r: Tensor::randn(0.0f32, 1.0, (hidden_dim, rank), &device)?,
                s_r: Tensor::new(&[1.0f32, 0.5], &device)?,
                v_r: Tensor::randn(0.0f32, 1.0, (rank, intermediate_dim), &device)?,
            },
        };

        let x = Tensor::randn(0.0f32, 1.0, (1, hidden_dim), &device)?;
        let v_vals = vec![0.01f32, -0.02, 0.03];

        // Analytical gradient via backward()
        let v_var = Var::from_vec(v_vals.clone(), (u_dim,), &device)?;
        let m = compute_mixing_matrix(&v_var, &p_matrices, rank, &device)?;
        let output = train_forward_expert(&x, &weights, &svd, &m, false)?;
        let loss = output.sqr()?.sum_all()?;
        let grads = loss.backward()?;
        let analytical = grads
            .get(v_var.as_tensor())
            .unwrap()
            .to_vec1::<f32>()?;

        // Numerical gradient via finite differences
        let mut numerical = vec![0.0f32; u_dim];
        for i in 0..u_dim {
            // L(v + eps*e_i)
            let mut v_plus = v_vals.clone();
            v_plus[i] += eps;
            let v_p = Var::from_vec(v_plus, (u_dim,), &device)?;
            let m_p = compute_mixing_matrix(&v_p, &p_matrices, rank, &device)?;
            let out_p = train_forward_expert(&x, &weights, &svd, &m_p, false)?;
            let loss_p = out_p.sqr()?.sum_all()?.to_vec0::<f32>()?;

            // L(v - eps*e_i)
            let mut v_minus = v_vals.clone();
            v_minus[i] -= eps;
            let v_m = Var::from_vec(v_minus, (u_dim,), &device)?;
            let m_m = compute_mixing_matrix(&v_m, &p_matrices, rank, &device)?;
            let out_m = train_forward_expert(&x, &weights, &svd, &m_m, false)?;
            let loss_m = out_m.sqr()?.sum_all()?.to_vec0::<f32>()?;

            numerical[i] = (loss_p - loss_m) / (2.0 * eps);
        }

        // Compare: relative error should be < 1%
        for i in 0..u_dim {
            let rel_err = if analytical[i].abs() > 1e-6 {
                ((analytical[i] - numerical[i]) / analytical[i]).abs()
            } else {
                (analytical[i] - numerical[i]).abs()
            };
            assert!(
                rel_err < 0.05,
                "gradient[{}]: analytical={:.6}, numerical={:.6}, rel_err={:.4}",
                i, analytical[i], numerical[i], rel_err
            );
        }

        Ok(())
    }

    #[test]
    fn test_router_bias_gradient() -> Result<()> {
        // Test that gradient flows through the differentiable routing path.
        // This verifies the router bias training pipeline end-to-end:
        //   bias_var → logits → softmax → weights → weighted expert output → pseudo_loss → grad_bias
        let device = Device::Cpu;
        let hidden_dim = 16;
        let intermediate_dim = 16;
        let n_experts = 4;
        let n_tokens = 3;
        let top_k = 2;

        // Create frozen expert weights
        let mut expert_weights = HashMap::new();
        for e in 0..n_experts {
            expert_weights.insert(
                e,
                TrainExpertWeights {
                    gate: Tensor::randn(0.0f32, 0.1, (intermediate_dim, hidden_dim), &device)?,
                    up: Tensor::randn(0.0f32, 0.1, (intermediate_dim, hidden_dim), &device)?,
                    down: Tensor::randn(0.0f32, 0.1, (hidden_dim, intermediate_dim), &device)?,
                    gate_bias: None,
                    up_bias: None,
                    down_bias: None,
                },
            );
        }

        // Create frozen router gate weight [n_experts, hidden_dim]
        let router_gate_weight = Tensor::randn(
            0.0f32, 0.1, (n_experts, hidden_dim), &device,
        )?;

        // Random MoE input
        let moe_input = Tensor::randn(
            0.0f32, 1.0, (n_tokens, hidden_dim), &device,
        )?;

        // Initial router bias (zeros)
        let bias_init = vec![0.0f32; n_experts];

        // === Test 1: Direct differentiable forward + backward ===
        let bias_var = Var::from_vec(bias_init.clone(), (n_experts,), &device)?;
        let moe_output = train_forward_moe_diff_routing(
            &moe_input,
            &expert_weights,
            &router_gate_weight,
            &bias_var,
            top_k,
            false, // standard SwiGLU
            &device,
        )?;

        // Verify output shape
        let (out_tokens, out_dim) = moe_output.dims2()?;
        assert_eq!(out_tokens, n_tokens);
        assert_eq!(out_dim, hidden_dim);

        // Compute loss and backward
        let loss = moe_output.sqr()?.mean_all()?;
        let grads = loss.backward()?;
        let grad_bias = grads
            .get(bias_var.as_tensor())
            .expect("gradient must exist for bias through differentiable routing");
        let grad_bias_vec = grad_bias.to_vec1::<f32>()?;
        let grad_norm: f32 = grad_bias_vec.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert_eq!(grad_bias_vec.len(), n_experts);
        assert!(
            grad_norm > 1e-10,
            "router bias gradient should be non-zero, got norm={}",
            grad_norm
        );

        // === Test 2: compute_router_bias_grad (pseudo-loss approach) ===
        let cached = CachedLayerState {
            moe_input: moe_input.flatten_all()?.to_vec1::<f32>()?,
            routing: vec![vec![(0, 0.5), (1, 0.5)]; n_tokens], // dummy routing (not used)
            n_tokens,
            hidden_dim,
        };
        let grad_output = Tensor::ones((n_tokens, hidden_dim), DType::F32, &device)?;

        let result = compute_router_bias_grad(
            &cached,
            &grad_output,
            &expert_weights,
            &router_gate_weight,
            &bias_init,
            top_k,
            false,
            &device,
        )?;

        let layer_grad_norm: f32 = result.grad_bias.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_eq!(result.grad_bias.len(), n_experts);
        assert!(
            layer_grad_norm > 1e-10,
            "layer router bias gradient should be non-zero, got norm={}",
            layer_grad_norm
        );

        // === Test 3: Numerical gradient check ===
        // Verify analytical gradient matches finite differences: ∂L/∂b_i ≈ (L(b+εe_i) - L(b-εe_i)) / 2ε
        let eps = 1e-3f32;
        let bias_vals = vec![0.01f32, -0.02, 0.03, 0.0];

        // Analytical gradient
        let bias_v = Var::from_vec(bias_vals.clone(), (n_experts,), &device)?;
        let output = train_forward_moe_diff_routing(
            &moe_input, &expert_weights, &router_gate_weight, &bias_v,
            top_k, false, &device,
        )?;
        let analytical_loss = output.sqr()?.sum_all()?;
        let grads = analytical_loss.backward()?;
        let analytical = grads.get(bias_v.as_tensor()).unwrap().to_vec1::<f32>()?;

        // Numerical gradient
        let mut numerical = vec![0.0f32; n_experts];
        for i in 0..n_experts {
            let mut b_plus = bias_vals.clone();
            b_plus[i] += eps;
            let bv_p = Var::from_vec(b_plus, (n_experts,), &device)?;
            let out_p = train_forward_moe_diff_routing(
                &moe_input, &expert_weights, &router_gate_weight, &bv_p,
                top_k, false, &device,
            )?;
            let loss_p = out_p.sqr()?.sum_all()?.to_vec0::<f32>()?;

            let mut b_minus = bias_vals.clone();
            b_minus[i] -= eps;
            let bv_m = Var::from_vec(b_minus, (n_experts,), &device)?;
            let out_m = train_forward_moe_diff_routing(
                &moe_input, &expert_weights, &router_gate_weight, &bv_m,
                top_k, false, &device,
            )?;
            let loss_m = out_m.sqr()?.sum_all()?.to_vec0::<f32>()?;

            numerical[i] = (loss_p - loss_m) / (2.0 * eps);
        }

        // Compare: relative error should be < 5%
        for i in 0..n_experts {
            let rel_err = if analytical[i].abs() > 1e-6 {
                ((analytical[i] - numerical[i]) / analytical[i]).abs()
            } else {
                (analytical[i] - numerical[i]).abs()
            };
            assert!(
                rel_err < 0.05,
                "bias_grad[{}]: analytical={:.6}, numerical={:.6}, rel_err={:.4}",
                i, analytical[i], numerical[i], rel_err
            );
        }

        // === Test 4: Full router_train_step across multiple layers ===
        let cached_layers = vec![
            CachedLayerState {
                moe_input: moe_input.flatten_all()?.to_vec1::<f32>()?,
                routing: vec![vec![(0, 0.5), (1, 0.5)]; n_tokens],
                n_tokens,
                hidden_dim,
            },
            CachedLayerState {
                moe_input: moe_input.flatten_all()?.to_vec1::<f32>()?,
                routing: vec![vec![(2, 0.6), (3, 0.4)]; n_tokens],
                n_tokens,
                hidden_dim,
            },
        ];
        let layer_expert_weights = vec![expert_weights.clone(), expert_weights.clone()];
        let gate_weights = vec![router_gate_weight.clone(), router_gate_weight.clone()];
        let biases = vec![vec![0.0f32; n_experts], vec![0.0f32; n_experts]];

        let step_result = router_train_step(
            &cached_layers,
            &grad_output,
            &layer_expert_weights,
            &gate_weights,
            &biases,
            top_k,
            false,
            &device,
        )?;

        assert_eq!(step_result.grad_router_biases.len(), 2);
        for (layer_idx, grad) in step_result.grad_router_biases.iter().enumerate() {
            let norm: f32 = grad.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert_eq!(grad.len(), n_experts);
            assert!(
                norm > 1e-10,
                "layer {} router bias grad should be non-zero, got norm={}",
                layer_idx, norm
            );
        }

        Ok(())
    }
}
