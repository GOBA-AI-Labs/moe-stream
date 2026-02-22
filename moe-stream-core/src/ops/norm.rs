//! RMS Normalization.
//!
//! RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
//!
//! On Metal GPU, dispatches to a fused kernel (1 dispatch replaces 7 candle ops).

use candle_core::{Result, Tensor, D};

/// Apply RMS normalization: x / sqrt(mean(x^2) + eps) * weight.
///
/// Input shape: [..., hidden_size]
/// Weight shape: [hidden_size]
///
/// When both x and weight are on Metal GPU, uses a fused Metal kernel
/// (1 dispatch vs 7 candle ops: to_dtype, sqr, mean_keepdim, add, sqrt, recip, broadcast_mul×2).
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    // Fused Metal kernel path: 1 dispatch replaces 7 candle ops.
    #[cfg(feature = "metal")]
    {
        if let candle_core::Device::Metal(_) = x.device() {
            let x_f32 = x.to_dtype(candle_core::DType::F32)?;
            return crate::metal::fused_rms_norm_metal(
                x.device(),
                &x_f32,
                weight,
                eps as f32,
            );
        }
    }

    // CPU fallback: standard candle ops.
    let x_f32 = x.to_dtype(candle_core::DType::F32)?;
    let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
    let inv_rms = (variance + eps)?.sqrt()?.recip()?;
    let normed = x_f32.broadcast_mul(&inv_rms)?;
    normed.broadcast_mul(weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_rms_norm_basic() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &device).unwrap();
        let w = Tensor::ones((4,), candle_core::DType::F32, &device).unwrap();
        let out = rms_norm(&x, &w, 1e-6).unwrap();
        let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        // normed = [0.3651, 0.7303, 1.0954, 1.4606]
        assert!((vals[0] - 0.3651).abs() < 0.01);
        assert!((vals[3] - 1.4606).abs() < 0.01);
    }
}
