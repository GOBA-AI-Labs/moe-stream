//! Activation functions for MoE and DeltaNet.

use candle_core::{Result, Tensor, D};
use candle_nn::ops::silu;

/// SwiGLU activation: SiLU(gate) * up.
///
/// Used in Qwen3 MoE expert MLPs:
///   output = silu(gate_proj(x)) * up_proj(x)
///
/// gate shape: [batch, intermediate]
/// up shape:   [batch, intermediate]
pub fn silu_and_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    silu(gate)?.mul(up)
}

/// OAI SwiGLU activation used by GPT-OSS.
///
/// Different from standard SwiGLU: uses clamping and a learnable-like alpha.
///   gate_c = min(gate, limit)
///   up_c   = clamp(up, -limit, limit)
///   output = (gate_c * sigmoid(alpha * gate_c)) * (up_c + 1.0)
///
/// Reference: llama.cpp ggml_swiglu_oai (alpha=1.702, limit=7.0).
pub fn swiglu_oai(gate: &Tensor, up: &Tensor, alpha: f32, limit: f32) -> Result<Tensor> {
    // gate_clamped = clamp(gate, -inf, limit) = min(gate, limit)
    let gate_c = gate.clamp(f32::NEG_INFINITY, limit)?;
    // up_clamped = clamp(up, -limit, limit)
    let up_c = up.clamp(-limit, limit)?;

    // silu_oai(gate_c) = gate_c * sigmoid(alpha * gate_c)
    let alpha_gate = (&gate_c * alpha as f64)?;
    let gate_activated = gate_c.mul(&sigmoid(&alpha_gate)?)?;

    // output = gate_activated * (up_c + 1.0)
    let up_shifted = (up_c + 1.0f64)?;
    gate_activated.mul(&up_shifted)
}

/// Sigmoid activation: 1 / (1 + exp(-x)).
pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    let neg_x = x.neg()?;
    let exp_neg = neg_x.exp()?;
    (exp_neg + 1.0f64)?.recip()
}

/// Softplus activation: log(1 + exp(x)).
pub fn softplus(x: &Tensor) -> Result<Tensor> {
    let exp_x = x.exp()?;
    (exp_x + 1.0f64)?.log()
}

/// L2 normalization along the last dimension.
///
/// l2_norm(x) = x / sqrt(sum(x^2, dim=-1) + eps)
pub fn l2_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let x_sq = x.sqr()?;
    let norm = (x_sq.sum_keepdim(D::Minus1)? + eps)?.sqrt()?;
    x.broadcast_div(&norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_silu_and_mul() {
        let device = Device::Cpu;
        let gate = Tensor::from_vec(vec![1.0f32, -1.0, 2.0, 0.0], (1, 4), &device).unwrap();
        let up = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], (1, 4), &device).unwrap();
        let out = silu_and_mul(&gate, &up).unwrap();
        let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // silu(1.0) = 1.0 * sigmoid(1.0) ≈ 0.7311
        // silu(-1.0) = -1.0 * sigmoid(-1.0) ≈ -0.2689
        // silu(0.0) = 0.0
        assert!((vals[0] - 0.7311).abs() < 0.01);
        assert!((vals[1] - (-0.2689)).abs() < 0.01);
        assert!(vals[3].abs() < 0.001);
    }
}
