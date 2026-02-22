//! Gated DeltaNet: linear attention with recurrent state for SSM hybrid models.
//!
//! Implements the Gated Delta Rule (ICLR 2025) for Qwen3-Coder-Next 80B:
//!   S_t = alpha_t * S_{t-1} + k_t (x) delta_t
//!   where delta_t = (v_t - S_{t-1} * k_t) * beta_t
//!
//! Key advantages over standard attention:
//! - O(1) decode per token (vs O(n) for KV-cache attention)
//! - Fixed memory: ~72MB for all 36 DeltaNet layers (80B)
//! - No KV-cache needed for DeltaNet layers (75% KV memory reduction)

use candle_core::{Device, Result, Tensor};

use crate::config::StreamingConfig;
use crate::gguf::reader::GgufReader;
use crate::model::cache::DeltaNetWeights;
use crate::model::layer::load_weight;

/// De-interleave ssm_in.weight (llama.cpp per-group interleaved QKVZ) into flat attn_qkv + attn_gate.
///
/// llama.cpp stores ssm_in as [qkvz_dim, hidden_size] with per-group interleaving:
///   Group g: [Q_g(head_k_dim), K_g(head_k_dim), V_g(v_per_k * head_v_dim), Z_g(v_per_k * head_v_dim)]
///   repeated for each of num_k_heads groups.
///
/// Returns: (attn_qkv [q+k+v_total, hidden], attn_gate [z_total, hidden])
pub fn deinterleave_ssm_in(ssm_in: &Tensor, config: &StreamingConfig) -> Result<(Tensor, Tensor)> {
    let num_k_heads = config.ssm_n_group;      // 16
    let head_k_dim = config.ssm_d_state;        // 128
    let num_v_heads = config.ssm_dt_rank;       // 32
    let head_v_dim = config.ssm_head_v_dim();    // 128
    let v_per_k = num_v_heads / num_k_heads;    // 2

    let q_per_group = head_k_dim;               // 128
    let k_per_group = head_k_dim;               // 128
    let v_per_group = v_per_k * head_v_dim;     // 256
    let z_per_group = v_per_k * head_v_dim;     // 256
    let group_size = q_per_group + k_per_group + v_per_group + z_per_group; // 768

    // Collect Q, K, V, Z slices from each group, then cat into flat layout
    let mut q_parts = Vec::with_capacity(num_k_heads);
    let mut k_parts = Vec::with_capacity(num_k_heads);
    let mut v_parts = Vec::with_capacity(num_k_heads);
    let mut z_parts = Vec::with_capacity(num_k_heads);

    for g in 0..num_k_heads {
        let base = g * group_size;
        q_parts.push(ssm_in.narrow(0, base, q_per_group)?);
        k_parts.push(ssm_in.narrow(0, base + q_per_group, k_per_group)?);
        v_parts.push(ssm_in.narrow(0, base + q_per_group + k_per_group, v_per_group)?);
        z_parts.push(ssm_in.narrow(0, base + q_per_group + k_per_group + v_per_group, z_per_group)?);
    }

    let q_refs: Vec<&Tensor> = q_parts.iter().collect();
    let k_refs: Vec<&Tensor> = k_parts.iter().collect();
    let v_refs: Vec<&Tensor> = v_parts.iter().collect();
    let z_refs: Vec<&Tensor> = z_parts.iter().collect();

    let flat_q = Tensor::cat(&q_refs, 0)?;
    let flat_k = Tensor::cat(&k_refs, 0)?;
    let flat_v = Tensor::cat(&v_refs, 0)?;
    let flat_z = Tensor::cat(&z_refs, 0)?;

    // attn_qkv = [all_Q, all_K, all_V] in flat layout
    let attn_qkv = Tensor::cat(&[&flat_q, &flat_k, &flat_v], 0)?;
    // attn_gate = all_Z in flat layout
    let attn_gate = flat_z;

    Ok((attn_qkv, attn_gate))
}

/// De-interleave ssm_ba.weight from per-head interleaved [β₀,α₀,β₁,α₁,...] to flat [all_β, all_α].
///
/// llama.cpp preserves HF layout: [β₀, α₀, β₁, α₁, ..., β_{n-1}, α_{n-1}]
/// moe-stream expects: [β₀, β₁, ..., β_{n-1}, α₀, α₁, ..., α_{n-1}]
pub fn deinterleave_ssm_ba(ssm_ba: &Tensor, config: &StreamingConfig) -> Result<Tensor> {
    let num_v_heads = config.ssm_dt_rank; // 32

    let mut beta_parts = Vec::with_capacity(num_v_heads);
    let mut alpha_parts = Vec::with_capacity(num_v_heads);

    for h in 0..num_v_heads {
        beta_parts.push(ssm_ba.narrow(0, h * 2, 1)?);
        alpha_parts.push(ssm_ba.narrow(0, h * 2 + 1, 1)?);
    }

    let beta_refs: Vec<&Tensor> = beta_parts.iter().collect();
    let alpha_refs: Vec<&Tensor> = alpha_parts.iter().collect();

    let flat_beta = Tensor::cat(&beta_refs, 0)?;
    let flat_alpha = Tensor::cat(&alpha_refs, 0)?;

    Tensor::cat(&[&flat_beta, &flat_alpha], 0)
}

/// Load DeltaNet input projections, supporting both Unsloth format (split attn_qkv + attn_gate)
/// and llama.cpp format (single ssm_in with per-group interleaved QKVZ).
///
/// Returns: (attn_qkv, attn_gate, ssm_ba, is_llamacpp_format)
pub fn load_deltanet_projections(
    reader: &GgufReader,
    prefix: &str,
    config: &StreamingConfig,
    cpu: &Device,
) -> Result<(Tensor, Tensor, Tensor, bool)> {
    let qkv_name = format!("{}.attn_qkv.weight", prefix);
    if reader.tensors.contains_key(&qkv_name) {
        // Unsloth format: already de-interleaved
        let attn_qkv = load_weight(reader, &qkv_name, cpu)?;
        let attn_gate = load_weight(reader, &format!("{}.attn_gate.weight", prefix), cpu)?;
        let ssm_ba = load_weight(reader, &format!("{}.ssm_ba.weight", prefix), cpu)?;
        Ok((attn_qkv, attn_gate, ssm_ba, false))
    } else {
        // llama.cpp format: single ssm_in with per-group interleaved QKVZ
        let ssm_in = load_weight(reader, &format!("{}.ssm_in.weight", prefix), cpu)?;
        let (attn_qkv, attn_gate) = deinterleave_ssm_in(&ssm_in, config)?;
        drop(ssm_in);
        let ssm_ba_raw = load_weight(reader, &format!("{}.ssm_ba.weight", prefix), cpu)?;
        let ssm_ba = deinterleave_ssm_ba(&ssm_ba_raw, config)?;
        Ok((attn_qkv, attn_gate, ssm_ba, true))
    }
}

/// Per-layer DeltaNet state (conv1d buffer + recurrent matrix).
pub struct DeltaNetLayerState {
    /// Conv1d state buffer: [conv_dim, kernel_size - 1] (flattened, row-major)
    pub conv_state: Vec<f32>,
    /// Recurrent state: [num_v_heads, head_k_dim, head_v_dim] (flattened, row-major)
    pub recurrent_state: Vec<f32>,
}

/// DeltaNet state manager for all layers.
pub struct DeltaNetState {
    pub layers: Vec<Option<DeltaNetLayerState>>,
    pub conv_dim: usize,
    pub kernel_size: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
}

impl DeltaNetState {
    /// Create empty DeltaNet state (all zeros, initialized on first use).
    pub fn new(config: &StreamingConfig) -> Self {
        let num_layers = config.num_layers;
        let conv_dim = config.ssm_conv_dim();
        let kernel_size = config.ssm_d_conv;
        let num_v_heads = config.ssm_dt_rank;
        let head_k_dim = config.ssm_d_state;
        let head_v_dim = config.ssm_head_v_dim();

        Self {
            layers: (0..num_layers).map(|_| None).collect(),
            conv_dim,
            kernel_size,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        }
    }

    /// Get or create state for a layer (zero-initialized on first access).
    pub fn get_or_init(&mut self, layer_idx: usize) -> &mut DeltaNetLayerState {
        if self.layers[layer_idx].is_none() {
            let conv_size = self.conv_dim * (self.kernel_size - 1);
            let recurrent_size = self.num_v_heads * self.head_k_dim * self.head_v_dim;
            self.layers[layer_idx] = Some(DeltaNetLayerState {
                conv_state: vec![0.0f32; conv_size],
                recurrent_state: vec![0.0f32; recurrent_size],
            });
        }
        self.layers[layer_idx].as_mut().unwrap()
    }

    /// Clear all states (for new generation).
    pub fn clear(&mut self) {
        for entry in &mut self.layers {
            *entry = None;
        }
    }
}

/// Run DeltaNet forward for a single token (autoregressive decode).
///
/// Processes one token through: projection → conv1d → delta rule → gated norm → output.
///
/// hidden_states: [batch=1, seq=1, hidden_size]
/// Returns: [batch=1, seq=1, hidden_size]
pub fn deltanet_forward(
    reader: &GgufReader,
    config: &StreamingConfig,
    hidden_states: &Tensor,
    layer_idx: usize,
    state: &mut DeltaNetState,
    resident: Option<&DeltaNetWeights>,
) -> Result<Tensor> {
    let prefix = format!("blk.{}", layer_idx);
    let (bsz, seq_len, hidden_dim) = hidden_states.dims3()?;
    let orig_device = hidden_states.device().clone();

    // All computation on CPU for single-token decode
    let cpu = &Device::Cpu;
    let hidden_cpu = hidden_states.to_device(cpu)?.reshape((bsz * seq_len, hidden_dim))?;

    let num_k_heads = config.ssm_n_group;    // 16 for 80B
    let head_k_dim = config.ssm_d_state;     // 128 for 80B
    let num_v_heads = config.ssm_dt_rank;    // 32 for 80B
    let head_v_dim = config.ssm_head_v_dim(); // 128 for 80B
    let v_per_k = num_v_heads / num_k_heads; // 2 for 80B

    let t_proj = std::time::Instant::now();

    // === 1. Input projections ===
    // Supports both Unsloth format (split attn_qkv + attn_gate) and
    // llama.cpp format (single ssm_in with per-group interleaved QKVZ).
    // When resident weights exist, they're already de-interleaved at preload time.
    let (attn_qkv_w, attn_gate_w, ssm_ba_w) = match resident {
        Some(r) => (r.attn_qkv.clone(), r.attn_gate.clone(), r.ssm_ba.clone()),
        None => {
            let (qkv, gate, ba, is_llcpp) = load_deltanet_projections(reader, &prefix, config, cpu)?;
            if is_llcpp {
                log::debug!("L{}: using llama.cpp ssm_in format (de-interleaved on-the-fly)", layer_idx);
            }
            (qkv, gate, ba)
        }
    };
    let mixed_qkv = hidden_cpu.matmul(&attn_qkv_w.t()?)?;
    drop(attn_qkv_w);

    let gate_proj_tensor = hidden_cpu.matmul(&attn_gate_w.t()?)?;
    drop(attn_gate_w);

    // ssm_ba: projects to [beta, alpha_log] per value head (64 = 2*32 for 80B)
    let mixed_ba = hidden_cpu.matmul(&ssm_ba_w.t()?)?;
    drop(ssm_ba_w);

    // === 2. Split Q, K, V from attn_qkv and get Z from attn_gate ===
    // Unsloth GGUF stores attn_qkv in flat (concatenated) layout:
    //   attn_qkv output: [all_Q(q_total), all_K(k_total), all_V(v_total)]
    //   attn_gate output: Z gate [d_inner]
    // Note: HF's in_proj_qkvz uses per-group interleaving, but the Unsloth converter
    // de-interleaves the fused QKVZ weight into separate flat Q, K, V and Z tensors.
    let q_total = head_k_dim * num_k_heads;  // 2048
    let k_total = head_k_dim * num_k_heads;  // 2048
    let v_total = head_v_dim * num_v_heads;  // 4096

    let mixed_qkv_vec = mixed_qkv.flatten_all()?.to_vec1::<f32>()?;

    // Flat layout:
    //   [all_Q(q_total), all_K(k_total), all_V(v_total)]
    // This differs from HF's in_proj_qkvz which uses per-group interleaving.
    let mut q_data = vec![0.0f32; q_total];
    let mut k_data = vec![0.0f32; k_total];
    let mut v_data = vec![0.0f32; v_total];

    q_data.copy_from_slice(&mixed_qkv_vec[0..q_total]);
    k_data.copy_from_slice(&mixed_qkv_vec[q_total..q_total + k_total]);
    v_data.copy_from_slice(&mixed_qkv_vec[q_total + k_total..q_total + k_total + v_total]);

    // Z (output gate) comes directly from attn_gate projection
    let gate_proj_vec = gate_proj_tensor.flatten_all()?.to_vec1::<f32>()?;

    // === 3. Split beta/alpha and compute gates ===
    // Unsloth GGUF stores ssm_ba in flat (concatenated) layout:
    //   [all_beta(num_v_heads), all_alpha(num_v_heads)]
    // Note: HF's in_proj_ba uses per-group interleaving, but the Unsloth converter
    // de-interleaves into flat beta and alpha sections.
    let mixed_ba_vec = mixed_ba.flatten_all()?.to_vec1::<f32>()?;

    // Flat layout:
    //   [all_beta(num_v_heads), all_alpha(num_v_heads)]
    let mut beta_data = vec![0.0f32; num_v_heads];
    let mut alpha_log_data = vec![0.0f32; num_v_heads];

    beta_data.copy_from_slice(&mixed_ba_vec[0..num_v_heads]);
    alpha_log_data.copy_from_slice(&mixed_ba_vec[num_v_heads..2 * num_v_heads]);

    // Load A: GGUF stores -exp(A_log), already negated and exponentiated
    let ssm_a = match resident {
        Some(r) => r.ssm_a.clone(),
        None => load_weight(reader, &format!("{}.ssm_a", prefix), cpu)?,
    };
    let ssm_a_vec = ssm_a.flatten_all()?.to_vec1::<f32>()?;

    // Load dt_bias
    let dt_bias = match resident {
        Some(r) => r.ssm_dt_bias.clone(),
        None => load_weight(reader, &format!("{}.ssm_dt.bias", prefix), cpu)?,
    };
    let dt_bias_vec = dt_bias.flatten_all()?.to_vec1::<f32>()?;

    // Compute gate (decay log): gate = ssm_a * softplus(alpha + dt_bias)
    // ssm_a = -exp(A_log), so gate = -exp(A_log) * softplus(alpha + dt_bias) < 0
    let mut alpha_data = vec![0.0f32; num_v_heads];
    for h in 0..num_v_heads {
        let alpha_biased = alpha_log_data[h] + dt_bias_vec[h];
        let softplus_val = (1.0 + alpha_biased.exp()).ln();
        alpha_data[h] = ssm_a_vec[h] * softplus_val;
    }

    // beta = sigmoid(beta)
    for val in beta_data.iter_mut().take(num_v_heads) {
        *val = 1.0 / (1.0 + (-*val).exp());
    }

    let proj_ms = t_proj.elapsed().as_secs_f64() * 1000.0;
    let t_conv = std::time::Instant::now();

    // === 4. Conv1d update ===
    let conv_dim = config.ssm_conv_dim();
    let kernel_size = config.ssm_d_conv;

    // Build conv input: concat(Q_flat, K_flat, V_flat)
    let mut conv_input = Vec::with_capacity(conv_dim);
    conv_input.extend_from_slice(&q_data);
    conv_input.extend_from_slice(&k_data);
    conv_input.extend_from_slice(&v_data);
    assert_eq!(conv_input.len(), conv_dim, "Conv input dim mismatch: {} vs {}", conv_input.len(), conv_dim);

    // Load conv1d weights (no bias in this architecture)
    let conv_w = match resident {
        Some(r) => r.ssm_conv1d.clone(),
        None => load_weight(reader, &format!("{}.ssm_conv1d.weight", prefix), cpu)?,
    };
    let conv_w_vec = conv_w.flatten_all()?.to_vec1::<f32>()?;

    // Get/init layer state
    let layer_state = state.get_or_init(layer_idx);

    // Conv1d: depthwise causal convolution
    // State holds last kernel_size-1 inputs per channel
    let ks_minus_1 = kernel_size - 1;
    let mut conv_output = vec![0.0f32; conv_dim];

    for c in 0..conv_dim {
        let state_offset = c * ks_minus_1;
        let weight_offset = c * kernel_size;

        // Build full window: [state[c], new_input[c]]
        let mut window = vec![0.0f32; kernel_size];
        window[..ks_minus_1].copy_from_slice(&layer_state.conv_state[state_offset..(state_offset + ks_minus_1)]);
        window[ks_minus_1] = conv_input[c];

        // Apply convolution: sum(window * weight)
        let mut sum = 0.0f32;
        for j in 0..kernel_size {
            sum += window[j] * conv_w_vec[weight_offset + j];
        }
        conv_output[c] = sum;

        // Update state: shift left, append new input
        for j in 0..ks_minus_1 - 1 {
            layer_state.conv_state[state_offset + j] = layer_state.conv_state[state_offset + j + 1];
        }
        if ks_minus_1 > 0 {
            layer_state.conv_state[state_offset + ks_minus_1 - 1] = conv_input[c];
        }
    }

    // Apply SiLU activation to conv output
    for val in conv_output.iter_mut().take(conv_dim) {
        let x = *val;
        *val = x / (1.0 + (-x).exp()); // silu(x) = x * sigmoid(x)
    }

    // Split conv output back into Q, K, V
    let q_conv: Vec<f32> = conv_output[0..q_total].to_vec();
    let k_conv: Vec<f32> = conv_output[q_total..q_total + k_total].to_vec();
    let v_conv: Vec<f32> = conv_output[q_total + k_total..].to_vec();

    // === 5. Expand Q, K from num_k_heads to num_v_heads (repeat-interleave) ===
    let mut q_expanded = vec![0.0f32; head_k_dim * num_v_heads];
    let mut k_expanded = vec![0.0f32; head_k_dim * num_v_heads];
    for g in 0..num_k_heads {
        for r in 0..v_per_k {
            let dst_head = g * v_per_k + r;
            q_expanded[dst_head * head_k_dim..(dst_head + 1) * head_k_dim]
                .copy_from_slice(&q_conv[g * head_k_dim..(g + 1) * head_k_dim]);
            k_expanded[dst_head * head_k_dim..(dst_head + 1) * head_k_dim]
                .copy_from_slice(&k_conv[g * head_k_dim..(g + 1) * head_k_dim]);
        }
    }

    // === 6. L2 normalize Q, K per head; scale Q ===
    let eps = config.rms_norm_eps as f64;
    let scale = 1.0 / (head_v_dim as f32).sqrt();

    for h in 0..num_v_heads {
        let start = h * head_k_dim;
        let end = start + head_k_dim;

        // L2 norm Q
        let q_norm: f32 = q_expanded[start..end].iter().map(|x| x * x).sum::<f32>().sqrt().max(eps as f32);
        for val in q_expanded[start..end].iter_mut() {
            *val = *val / q_norm * scale;
        }

        // L2 norm K
        let k_norm: f32 = k_expanded[start..end].iter().map(|x| x * x).sum::<f32>().sqrt().max(eps as f32);
        for val in k_expanded[start..end].iter_mut() {
            *val /= k_norm;
        }
    }

    let conv_ms = t_conv.elapsed().as_secs_f64() * 1000.0;
    let t_state_update = std::time::Instant::now();

    // === 7. Delta rule state update ===
    // state shape: [num_v_heads, head_k_dim, head_v_dim] (row-major)
    // For each head h:
    //   state[h] *= exp(alpha[h])    (decay)
    //   kv_mem = state[h] @ k[h]     (read: [head_v_dim])
    //   delta = (v[h] - kv_mem) * beta[h]  (error correction)
    //   state[h] += outer(k[h], delta)      (write)
    //   output[h] = state[h] @ q[h]  (query)

    let hkd = head_k_dim;
    let hvd = head_v_dim;
    let mut output_data = vec![0.0f32; num_v_heads * hvd];

    let rec_state = &mut layer_state.recurrent_state;
    for h in 0..num_v_heads {
        let state_offset = h * hkd * hvd;
        let q_offset = h * hkd;
        let k_offset = h * hkd;
        let v_offset = h * hvd;

        // Decay: state *= exp(alpha)
        let decay = alpha_data[h].exp();
        for i in 0..hkd * hvd {
            rec_state[state_offset + i] *= decay;
        }

        // Read: kv_mem[j] = sum_i(k[i] * state[i][j])
        // state is [hkd, hvd], k is [hkd], result is [hvd]
        let mut kv_mem = vec![0.0f32; hvd];
        for i in 0..hkd {
            let row_start = state_offset + i * hvd;
            let k_i = k_expanded[k_offset + i];
            for j in 0..hvd {
                kv_mem[j] += k_i * rec_state[row_start + j];
            }
        }

        // Error correction: delta = (v - kv_mem) * beta
        let beta = beta_data[h];
        let mut delta = vec![0.0f32; hvd];
        for j in 0..hvd {
            delta[j] = (v_conv[v_offset + j] - kv_mem[j]) * beta;
        }

        // Write: state += outer(k, delta) - state[i][j] += k[i] * delta[j]
        for i in 0..hkd {
            let row_start = state_offset + i * hvd;
            let k_i = k_expanded[k_offset + i];
            for j in 0..hvd {
                rec_state[row_start + j] += k_i * delta[j];
            }
        }

        // Query: output[h][j] = sum_i(q[i] * state[i][j])
        for i in 0..hkd {
            let row_start = state_offset + i * hvd;
            let q_i = q_expanded[q_offset + i];
            for j in 0..hvd {
                output_data[h * hvd + j] += q_i * rec_state[row_start + j];
            }
        }
    }

    let state_ms = t_state_update.elapsed().as_secs_f64() * 1000.0;
    let t_norm_gate = std::time::Instant::now();

    // === 8. Gated RMS norm: output * (rms_norm(output) * weight) ===
    // Note: ssm_norm.weight is [head_v_dim] (128), applied per head
    let ssm_norm_w = match resident {
        Some(r) => r.ssm_norm.clone(),
        None => load_weight(reader, &format!("{}.ssm_norm.weight", prefix), cpu)?,
    };
    let ssm_norm_vec = ssm_norm_w.flatten_all()?.to_vec1::<f32>()?;

    // RMS norm per head
    for h in 0..num_v_heads {
        let start = h * hvd;
        let end = start + hvd;

        let sq_sum: f32 = output_data[start..end].iter().map(|x| x * x).sum();
        let rms = (sq_sum / hvd as f32 + eps as f32).sqrt();

        for (local_i, val) in output_data[start..end].iter_mut().enumerate() {
            *val = *val / rms * ssm_norm_vec[local_i];
        }
    }

    // === 9. Apply output gate: output *= silu(Z) ===
    // gate_proj_vec (Z) was extracted from ssm_in in step 2
    for i in 0..v_total {
        let g = gate_proj_vec[i];
        let silu_g = g / (1.0 + (-g).exp());
        output_data[i] *= silu_g;
    }

    let norm_gate_ms = t_norm_gate.elapsed().as_secs_f64() * 1000.0;
    let t_output = std::time::Instant::now();

    // === 10. Output projection ===
    let ssm_out_w = match resident {
        Some(r) => r.ssm_out.clone(),
        None => load_weight(reader, &format!("{}.ssm_out.weight", prefix), cpu)?,
    };
    let output_tensor = Tensor::from_vec(output_data, (1, v_total), cpu)?;
    let projected = output_tensor.matmul(&ssm_out_w.t()?)?;

    let output_ms = t_output.elapsed().as_secs_f64() * 1000.0;

    // Log detailed timing for DeltaNet layers
    if layer_idx == 0 || layer_idx == 1 || layer_idx == 2 || layer_idx == 47 {
        log::debug!(
            "  DeltaNet L{}: proj={:.1}ms conv={:.1}ms state={:.1}ms norm_gate={:.1}ms out={:.1}ms total={:.1}ms",
            layer_idx, proj_ms, conv_ms, state_ms, norm_gate_ms, output_ms,
            proj_ms + conv_ms + state_ms + norm_gate_ms + output_ms,
        );
    }

    // Return as [batch, seq, hidden], move back to original device
    let result = projected.reshape((bsz, seq_len, hidden_dim))?;
    result.to_device(&orig_device)
}
