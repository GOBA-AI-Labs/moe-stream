//! Minimal test to verify candle Var + backward() for TinyLoRA gradient computation.
//!
//! Tests:
//! 1. Create a Var (simulating v, 13 params)
//! 2. Use it in a TinyLoRA-style computation: delta_W = U @ diag(v * S) @ V^T
//! 3. Apply delta: y = (W + delta_W) @ x
//! 4. Compute scalar loss
//! 5. Call backward() and verify gradient is non-zero
//!
//! Run: cargo run -p moe-stream-core --example test_autograd

use candle_core::{DType, Device, IndexOp, Result, Tensor, Var};

fn main() -> Result<()> {
    let device = Device::Cpu;
    println!("=== Candle Autograd Test for TinyLoRA ===\n");

    // Simulate TinyLoRA parameters
    let rank = 2;
    let u_dim = 13; // number of mixing coefficients
    let hidden = 64; // small hidden dim for test

    // Frozen SVD components (U, S, V)
    let u_matrix = Tensor::randn(0.0f32, 1.0, (hidden, rank), &device)?; // [hidden, rank]
    let s_vector = Tensor::new(&[3.0f32, 1.5f32], &device)?; // [rank] singular values
    let v_matrix = Tensor::randn(0.0f32, 1.0, (rank, hidden), &device)?; // [rank, hidden]

    // Random projection matrices P_i (frozen)
    let p_matrices: Vec<Tensor> = (0..u_dim)
        .map(|_| Tensor::randn(0.0f32, 1.0, (rank, rank), &device))
        .collect::<Result<Vec<_>>>()?;

    // Trainable: v (the only parameter!)
    let v = Var::from_vec(vec![0.01f32; u_dim], (u_dim,), &device)?;

    // Frozen weight matrix W
    let w = Tensor::randn(0.0f32, 0.1, (hidden, hidden), &device)?;

    // Input
    let x = Tensor::randn(0.0f32, 1.0, (1, hidden), &device)?;

    // === Forward pass ===

    // Step 1: Compute mixing matrix M = sum_i v_i * P_i
    let v_tensor = v.as_tensor();
    let mut m = Tensor::zeros((rank, rank), DType::F32, &device)?;
    for i in 0..u_dim {
        let v_i = v_tensor.i(i)?; // scalar
        let scaled_p = p_matrices[i].broadcast_mul(&v_i)?;
        m = (m + scaled_p)?;
    }

    // Step 2: delta_W = U @ diag(S) @ M @ V^T
    // diag(S) @ M: scale rows of M by S
    let s0 = s_vector.i(0)?;
    let s1 = s_vector.i(1)?;
    let sm_row0 = m.i(0)?.broadcast_mul(&s0)?;
    let sm_row1 = m.i(1)?.broadcast_mul(&s1)?;
    let sm = Tensor::stack(&[sm_row0, sm_row1], 0)?; // [rank, rank]

    let delta_w = u_matrix.matmul(&sm)?.matmul(&v_matrix)?; // [hidden, hidden]

    // Step 3: W' = W + delta_W
    let w_prime = (&w + &delta_w)?;

    // Step 4: y = x @ W'^T
    let y = x.matmul(&w_prime.t()?)?; // [1, hidden]

    // Step 5: loss = mean(y^2) (simple MSE-like loss)
    let loss = y.sqr()?.mean_all()?;

    println!("Loss: {:?}", loss.to_scalar::<f32>()?);

    // === Backward pass ===
    let grad_store = loss.backward()?;

    // Retrieve gradient for v
    match grad_store.get(v.as_tensor()) {
        Some(grad_v) => {
            let grad_vec: Vec<f32> = grad_v.to_vec1()?;
            let grad_norm: f32 = grad_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            println!("\nGradient w.r.t. v (13 params):");
            for (i, g) in grad_vec.iter().enumerate() {
                println!("  v[{i}]: grad = {g:.6}");
            }
            println!("\nGradient L2 norm: {grad_norm:.6}");

            if grad_norm > 1e-10 {
                println!("\nSUCCESS: Gradient is non-zero! Autograd works for TinyLoRA.");
            } else {
                println!("\nFAIL: Gradient is zero. Autograd may not be tracking v correctly.");
                std::process::exit(1);
            }
        }
        None => {
            println!("\nFAIL: No gradient found for v. backward() did not track the variable.");
            std::process::exit(1);
        }
    }

    // === Gradient descent sanity check ===
    println!("\n--- Gradient Descent Sanity Check (10 steps) ---");
    let lr = 0.001f64;
    for step in 0..10 {
        // Forward pass (rebuild graph each step)
        let vt = v.as_tensor();
        let mut m_step = Tensor::zeros((rank, rank), DType::F32, &device)?;
        for i in 0..u_dim {
            let v_i = vt.i(i)?;
            let scaled_p = p_matrices[i].broadcast_mul(&v_i)?;
            m_step = (m_step + scaled_p)?;
        }
        let s0 = s_vector.i(0)?;
        let s1 = s_vector.i(1)?;
        let sm_row0 = m_step.i(0)?.broadcast_mul(&s0)?;
        let sm_row1 = m_step.i(1)?.broadcast_mul(&s1)?;
        let sm_step = Tensor::stack(&[sm_row0, sm_row1], 0)?;
        let delta_w_step = u_matrix.matmul(&sm_step)?.matmul(&v_matrix)?;
        let w_prime_step = (&w + &delta_w_step)?;
        let y_step = x.matmul(&w_prime_step.t()?)?;
        let loss_step = y_step.sqr()?.mean_all()?;
        let loss_val = loss_step.to_scalar::<f32>()?;

        // Backward
        let grads = loss_step.backward()?;
        let grad_v = grads.get(v.as_tensor()).expect("grad must exist");

        // SGD update: v = v - lr * grad
        let updated = (vt - (grad_v * lr)?)?;
        v.set(&updated)?;

        if step % 3 == 0 || step == 9 {
            let gn: f32 = grad_v
                .to_vec1::<f32>()?
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            println!("  step {step}: loss = {loss_val:.6}, grad_norm = {gn:.6}");
        }
    }

    println!("\n=== All tests passed! ===");
    Ok(())
}
