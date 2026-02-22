/// Full numeric comparison: saves dequantized tensors as binary for diffing.
///
/// Usage: cargo run -p moe-stream-core --release --example full_compare -- <gguf>
use moe_stream_core::gguf::reader::GgufReader;
use std::fs;
use std::io::Write;

fn out_dir() -> String {
    std::env::temp_dir().join("gguf_verify").to_string_lossy().to_string()
}

fn save_f32(data: &[f32], path: &str) {
    let mut file = fs::File::create(path).expect("Failed to create file");
    for &v in data {
        file.write_all(&v.to_le_bytes()).expect("write failed");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("Usage: full_compare <gguf>");

    let dir = out_dir();
    fs::create_dir_all(&dir).ok();

    let reader = GgufReader::open(path).expect("Failed to open GGUF");

    // Test tensors: (gguf_name, expert_idx_or_none, label)
    let tests: Vec<(&str, Option<usize>, &str)> = vec![
        ("output_norm.weight", None, "output_norm"),
        ("blk.0.ffn_gate_exps.weight", Some(0), "expert0_gate"),
        ("blk.0.ffn_gate_exps.weight", Some(42), "expert42_gate"),
        ("blk.0.ffn_down_exps.weight", Some(0), "expert0_down"),
        ("blk.0.attn_q.weight", None, "attn_q"),
        ("blk.0.attn_v.weight", None, "attn_v"),
    ];

    for (name, expert_idx, label) in &tests {
        let result = match expert_idx {
            Some(idx) => reader.dequantize_expert(name, *idx),
            None => reader.dequantize_tensor(name),
        };

        match result {
            Ok((data, shape)) => {
                let out_path = format!("{}/rust_{}.bin", dir, label);
                save_f32(&data, &out_path);
                println!(
                    "[Rust] {}: shape={:?}, elements={}, saved to {}",
                    label,
                    shape,
                    data.len(),
                    out_path
                );
            }
            Err(e) => {
                println!("[Rust] {}: ERROR: {}", label, e);
            }
        }
    }

    println!(
        "Done. Compare with: python3 -c 'import numpy as np; \
         r=np.fromfile(\"{dir}/rust_X.bin\",np.float32); \
         p=np.fromfile(\"{dir}/python_X.bin\",np.float32); \
         print(f\"max_diff={{np.max(np.abs(r-p))}}\")'",
        dir = dir
    );
}
