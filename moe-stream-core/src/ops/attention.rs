//! Attention operations: RoPE and Scaled Dot-Product Attention.

use candle_core::{Device, IndexOp, Result, Tensor, D};
use candle_nn::ops::softmax;

/// Pre-compute cosine/sine tables for Rotary Position Embedding.
///
/// Returns (cos, sin) tensors of shape [max_seq_len, head_dim].
pub fn precompute_rope_tables(
    head_dim: usize,
    max_seq_len: usize,
    theta: f64,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;

    // inv_freq[i] = 1.0 / theta^(2i / head_dim)
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / (theta.powf(2.0 * i as f64 / head_dim as f64) as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half_dim), device)?;

    // positions: [max_seq_len, 1]
    let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
    let positions = Tensor::from_vec(positions, (max_seq_len, 1), device)?;

    // freqs = positions @ inv_freq: [max_seq_len, half_dim]
    let freqs = positions.matmul(&inv_freq)?;

    let cos = freqs.cos()?;
    let sin = freqs.sin()?;

    Ok((cos, sin))
}

/// Apply rotary embedding to query and key tensors.
///
/// q, k shape: [batch, seq_len, num_heads, head_dim]
/// cos, sin shape: [max_seq_len, head_dim/2]  (sliced to seq_len internally)
/// position_offset: starting position for this generation step (for KV-cache)
///
/// Uses the "half-rotate" convention:
///   x_rot = x[..., :dim//2], x_pass = x[..., dim//2:]
///   x_rot_out = x_rot * cos - x_complement * sin
///   where x_complement = rotate_half(x_rot)
///
/// On Metal GPU with seq_len=1 (decode), uses a fused kernel (1 dispatch per tensor
/// replaces ~12 candle ops: narrow×4, broadcast_mul×4, broadcast_sub/add×2, cat×2).
pub fn rotary_embedding(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    position_offset: usize,
) -> Result<(Tensor, Tensor)> {
    let seq_len = q.dim(1)?;
    let head_dim = q.dim(3)?;
    let half_dim = head_dim / 2;

    // Fused Metal kernel path for single-token decode (the hot path).
    // Reduces ~12 candle ops per tensor to 1 Metal dispatch per tensor.
    #[cfg(feature = "metal")]
    {
        if seq_len == 1 {
            if let candle_core::Device::Metal(_) = q.device() {
                let num_q_heads = q.dim(2)?;
                let num_k_heads = k.dim(2)?;

                // Slice cos/sin for current position: [1, half_dim] → flatten to [half_dim]
                let cos_slice = cos.i(position_offset..position_offset + 1)?.contiguous()?;
                let sin_slice = sin.i(position_offset..position_offset + 1)?.contiguous()?;
                let cos_flat = cos_slice.flatten_all()?;
                let sin_flat = sin_slice.flatten_all()?;

                // Flatten Q and K for the kernel: [1, 1, heads, head_dim] → [heads * head_dim]
                let q_flat = q.contiguous()?.flatten_all()?;
                let k_flat = k.contiguous()?.flatten_all()?;

                let q_rot_flat = crate::metal::fused_rope_metal(
                    q.device(), &q_flat, &cos_flat, &sin_flat, num_q_heads, head_dim,
                )?;
                let k_rot_flat = crate::metal::fused_rope_metal(
                    k.device(), &k_flat, &cos_flat, &sin_flat, num_k_heads, head_dim,
                )?;

                // Reshape back to [batch, seq_len, heads, head_dim]
                let q_rot = q_rot_flat.reshape(q.shape())?;
                let k_rot = k_rot_flat.reshape(k.shape())?;

                return Ok((q_rot, k_rot));
            }
        }
    }

    // CPU / multi-token fallback: standard candle ops.
    // Slice cos/sin for current positions: [seq_len, half_dim]
    let cos = cos.i(position_offset..position_offset + seq_len)?;
    let sin = sin.i(position_offset..position_offset + seq_len)?;

    // Reshape for broadcasting: [1, seq_len, 1, half_dim]
    let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

    let apply_rope = |x: &Tensor| -> Result<Tensor> {
        let x1 = x.narrow(3, 0, half_dim)?;
        let x2 = x.narrow(3, half_dim, half_dim)?;

        // x1 * cos - x2 * sin
        let rotated_x1 = x1.broadcast_mul(&cos)?.broadcast_sub(&x2.broadcast_mul(&sin)?)?;
        // x2 * cos + x1 * sin
        let rotated_x2 = x2.broadcast_mul(&cos)?.broadcast_add(&x1.broadcast_mul(&sin)?)?;

        Tensor::cat(&[&rotated_x1, &rotated_x2], 3)
    };

    let q_rot = apply_rope(q)?;
    let k_rot = apply_rope(k)?;

    Ok((q_rot, k_rot))
}

/// Apply partial rotary embedding (only applies RoPE to a fraction of head_dim).
///
/// Used by Qwen3-Coder-Next (partial_rotary_factor = 0.25).
/// cos/sin tables may be pre-computed for the full rotary_dim; they will be
/// narrowed to rotary_dim/2 columns automatically.
pub fn partial_rotary_embedding(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    position_offset: usize,
    rotary_dim: usize,
) -> Result<(Tensor, Tensor)> {
    let head_dim = q.dim(3)?;

    if rotary_dim >= head_dim {
        return rotary_embedding(q, k, cos, sin, position_offset);
    }

    let pass_dim = head_dim - rotary_dim;
    let half_rotary = rotary_dim / 2;

    // Narrow cos/sin tables to match rotary_dim/2 columns (they may be wider)
    let cos_table_dim = cos.dim(1)?;
    let (cos_narrow, sin_narrow) = if cos_table_dim > half_rotary {
        (cos.narrow(1, 0, half_rotary)?, sin.narrow(1, 0, half_rotary)?)
    } else {
        (cos.clone(), sin.clone())
    };

    let q_rot_part = q.narrow(3, 0, rotary_dim)?;
    let q_pass = q.narrow(3, rotary_dim, pass_dim)?;
    let k_rot_part = k.narrow(3, 0, rotary_dim)?;
    let k_pass = k.narrow(3, rotary_dim, pass_dim)?;

    let (q_rotated, k_rotated) =
        rotary_embedding(&q_rot_part, &k_rot_part, &cos_narrow, &sin_narrow, position_offset)?;

    let q_out = Tensor::cat(&[&q_rotated, &q_pass], 3)?;
    let k_out = Tensor::cat(&[&k_rotated, &k_pass], 3)?;

    Ok((q_out, k_out))
}

/// Scaled dot-product attention with optional attention sinks and sliding window.
///
/// query shape:  [batch, num_heads, seq_q, head_dim]
/// key shape:    [batch, num_kv_heads, seq_k, head_dim]
/// value shape:  [batch, num_kv_heads, seq_k, head_dim]
/// attn_sinks:   Optional [num_heads] — per-head learned sink logit (GPT-OSS)
/// sliding_window: Optional window size — positions further than window are masked
///
/// Handles GQA (Grouped Query Attention): num_heads / num_kv_heads > 1
/// by repeating KV heads to match query heads.
///
/// causal_mask: if true, applies causal attention mask.
pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    scale: f64,
    causal: bool,
    attn_sinks: Option<&Tensor>,
    sliding_window: Option<usize>,
) -> Result<Tensor> {
    let num_heads = query.dim(1)?;
    let num_kv_heads = key.dim(1)?;
    let seq_q = query.dim(2)?;
    let seq_k = key.dim(2)?;

    // GQA: expand KV heads (must be contiguous for Metal matmul)
    let (key, value) = if num_kv_heads < num_heads {
        let n_rep = num_heads / num_kv_heads;
        let key = key
            .unsqueeze(2)?
            .expand((key.dim(0)?, num_kv_heads, n_rep, seq_k, key.dim(3)?))?
            .reshape((key.dim(0)?, num_heads, seq_k, key.dim(3)?))?
            .contiguous()?;
        let value = value
            .unsqueeze(2)?
            .expand((value.dim(0)?, num_kv_heads, n_rep, seq_k, value.dim(3)?))?
            .reshape((value.dim(0)?, num_heads, seq_k, value.dim(3)?))?
            .contiguous()?;
        (key, value)
    } else {
        (key.clone(), value.clone())
    };

    // Attention scores: Q @ K^T → [batch, heads, seq_q, seq_k]
    let query = query.contiguous()?;
    let key_t = key.transpose(2, 3)?.contiguous()?;
    let scores = query.matmul(&key_t)?;
    let scores = (scores * scale)?;

    // Causal mask + sliding window mask
    let sw = sliding_window.unwrap_or(0);
    let needs_mask = (causal && seq_q > 1) || sw > 0;
    let scores = if needs_mask {
        let device = scores.device();
        let k_start = seq_k.saturating_sub(seq_q);
        let mask_data: Vec<f32> = (0..seq_q)
            .flat_map(|q_pos| {
                let q_abs = k_start + q_pos;
                (0..seq_k).map(move |k_pos| {
                    // Causal: future positions masked (k_pos after q_abs)
                    if causal && k_pos > q_abs {
                        return f32::NEG_INFINITY;
                    }
                    // SWA: positions too far back masked
                    if sw > 0 && q_abs.saturating_sub(k_pos) >= sw {
                        return f32::NEG_INFINITY;
                    }
                    0.0
                })
            })
            .collect();
        let mask = Tensor::from_vec(mask_data, (seq_q, seq_k), device)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        scores.broadcast_add(&mask)?
    } else {
        scores
    };

    // Softmax with optional attention sinks (GPU-side, no CPU round-trip)
    let attn_weights = if let Some(sinks) = attn_sinks {
        // Attention sinks (GPT-OSS): add a virtual sink position to the softmax denominator.
        //   max' = max(max_over_keys(scores), sink[head])
        //   attn[key] = exp(scores[key] - max') / (sum(exp(scores - max')) + exp(sink - max'))
        //
        // All operations stay on GPU via tensor ops (no to_vec1 / CPU loop).
        let sinks_4d = sinks
            .to_device(scores.device())?
            .reshape((1, num_heads, 1, 1))?;

        // max of scores along key dim: [batch, heads, seq_q, 1]
        let max_scores = scores.max_keepdim(D::Minus1)?;

        // Numerically stable max including sink: max(a, b) = (a + b + |a - b|) / 2
        // Use broadcast_add/sub (candle's + doesn't auto-broadcast)
        let sum_ab = max_scores.broadcast_add(&sinks_4d)?;
        let diff_ab = max_scores.broadcast_sub(&sinks_4d)?.abs()?;
        let max_val = (sum_ab.broadcast_add(&diff_ab)? * 0.5)?;

        // Shifted exponentials (stays on GPU)
        let exp_scores = scores.broadcast_sub(&max_val)?.exp()?;
        let exp_sink = sinks_4d.broadcast_sub(&max_val)?.exp()?;

        // Denominator: sum(exp_scores) + exp_sink
        let denom = exp_scores.sum_keepdim(D::Minus1)?.broadcast_add(&exp_sink)?;

        exp_scores.broadcast_div(&denom)?
    } else {
        softmax(&scores, D::Minus1)?
    };

    // Output: attn @ V → [batch, heads, seq_q, head_dim]
    attn_weights.matmul(&value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_rope_tables() {
        let device = Device::Cpu;
        let (cos, sin) = precompute_rope_tables(128, 32, 10000.0, &device).unwrap();
        assert_eq!(cos.dims(), &[32, 64]);
        assert_eq!(sin.dims(), &[32, 64]);

        // cos(0) should be 1.0, sin(0) should be 0.0
        let cos_vals = cos.i(0).unwrap().to_vec1::<f32>().unwrap();
        assert!((cos_vals[0] - 1.0).abs() < 1e-5);
        let sin_vals = sin.i(0).unwrap().to_vec1::<f32>().unwrap();
        assert!(sin_vals[0].abs() < 1e-5);
    }

    #[test]
    fn test_sdpa_single_token() {
        let device = Device::Cpu;
        // [batch=1, heads=2, seq=1, dim=4]
        let q = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            (1, 2, 1, 4),
            &device,
        )
        .unwrap();
        let k = q.clone();
        let v = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            (1, 2, 1, 4),
            &device,
        )
        .unwrap();

        let out =
            scaled_dot_product_attention(&q, &k, &v, 1.0 / (4.0f64).sqrt(), false, None, None).unwrap();
        assert_eq!(out.dims(), &[1, 2, 1, 4]);

        // Single token: output = V (softmax over single element = 1.0)
        let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-5);
        assert!((vals[4] - 5.0).abs() < 1e-5);
    }
}
