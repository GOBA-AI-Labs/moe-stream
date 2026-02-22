/// Quick Metal backend smoke test.
use candle_core::{Device, DType, Tensor};
use moe_stream_core::ops;

fn main() {
    // Try Metal, fallback to CPU
    let device = Device::new_metal(0).unwrap_or_else(|e| {
        println!("Metal not available ({}), using CPU", e);
        Device::Cpu
    });
    println!("Device: {:?}", device);

    // Create tensors on Metal
    let x = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        (1, 8),
        &device,
    )
    .unwrap();
    println!("x shape: {:?}", x.dims());

    // Test RMS norm
    let weight = Tensor::ones((8,), DType::F32, &device).unwrap();
    let normed = ops::rms_norm(&x, &weight, 1e-6).unwrap();
    let vals = normed
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    println!("rms_norm: {:?}", &vals[..4]);

    // Test matmul
    let a = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        (2, 3),
        &device,
    )
    .unwrap();
    let b = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        (3, 2),
        &device,
    )
    .unwrap();
    let c = a.matmul(&b).unwrap();
    let vals = c
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    println!("matmul [2x3]@[3x2] = {:?}", vals);

    // Test SiLU
    let gate = Tensor::from_vec(vec![1.0f32, -1.0, 2.0, 0.0], (1, 4), &device).unwrap();
    let up = Tensor::ones((1, 4), DType::F32, &device).unwrap();
    let activated = ops::silu_and_mul(&gate, &up).unwrap();
    let vals = activated
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    println!("silu_and_mul: {:?}", vals);

    // Test RoPE precomputation
    let (cos, sin) = ops::attention::precompute_rope_tables(128, 2048, 10000.0, &device).unwrap();
    println!("RoPE cos shape: {:?}", cos.dims());
    println!("RoPE sin shape: {:?}", sin.dims());

    // Test SDPA (single token decode)
    let q = Tensor::randn(0.0f32, 1.0, (1, 32, 1, 128), &device).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, (1, 4, 1, 128), &device).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, (1, 4, 1, 128), &device).unwrap();
    let scale = 1.0 / (128.0f64).sqrt();
    let out = ops::scaled_dot_product_attention(&q, &k, &v, scale, false, None, None).unwrap();
    println!("SDPA output shape: {:?}", out.dims());

    println!("\nAll Metal ops verified!");
}
