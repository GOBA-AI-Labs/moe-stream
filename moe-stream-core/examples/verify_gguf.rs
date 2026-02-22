/// Verify Rust GGUF reader against real GGUF file.
/// Outputs dequantized tensor values for comparison with Python.
///
/// Usage: cargo run -p moe-stream-core --example verify_gguf -- <path_to_gguf>
use moe_stream_core::gguf::reader::GgufReader;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("Usage: verify_gguf <path_to_gguf>");

    println!("Opening GGUF: {}", path);
    let reader = GgufReader::open(path).expect("Failed to open GGUF");

    // Print basic info
    println!("Tensors: {}", reader.tensors.len());
    println!("Metadata keys: {}", reader.metadata.len());
    println!("---");

    // List first 20 tensors (sorted by name for reproducibility)
    let mut tensor_names: Vec<&String> = reader.tensors.keys().collect();
    tensor_names.sort();
    println!("First 20 tensors:");
    for (i, name) in tensor_names.iter().take(20).enumerate() {
        let info = &reader.tensors[*name];
        println!(
            "  [{}] {} {:?} {:?} ({})",
            i,
            info.name,
            info.pt_shape(),
            info.quant_type,
            info.n_elements
        );
    }
    println!("---");

    // Test 1: Dequantize a small tensor (output_norm.weight = hidden_size = 2048 elements)
    let norm_name = "output_norm.weight";
    println!("Dequantizing '{}'...", norm_name);
    match reader.dequantize_tensor(norm_name) {
        Ok((data, shape)) => {
            println!("  Shape: {:?}, len: {}", shape, data.len());
            println!("  First 20 values:");
            for (i, v) in data.iter().take(20).enumerate() {
                println!("  [{}] {:.8}", i, v);
            }
            println!("  Last 5 values:");
            let n = data.len();
            for i in (n.saturating_sub(5))..n {
                println!("  [{}] {:.8}", i, data[i]);
            }
        }
        Err(e) => println!("  ERROR: {}", e),
    }
    println!("---");

    // Test 2: Dequantize an expert slice (layer 0, expert 0, gate_proj)
    let expert_tensor = "blk.0.ffn_gate_exps.weight";
    let expert_idx = 0;
    println!(
        "Dequantizing expert slice: {} [expert {}]...",
        expert_tensor, expert_idx
    );
    match reader.dequantize_expert(expert_tensor, expert_idx) {
        Ok((data, shape)) => {
            println!("  Shape: {:?}, len: {}", shape, data.len());
            println!("  First 20 values:");
            for (i, v) in data.iter().take(20).enumerate() {
                println!("  [{}] {:.8}", i, v);
            }
        }
        Err(e) => println!("  ERROR: {}", e),
    }
    println!("---");

    // Test 3: Dequantize another expert to verify slicing is correct
    let expert_idx2 = 42;
    println!(
        "Dequantizing expert slice: {} [expert {}]...",
        expert_tensor, expert_idx2
    );
    match reader.dequantize_expert(expert_tensor, expert_idx2) {
        Ok((data, shape)) => {
            println!("  Shape: {:?}, len: {}", shape, data.len());
            println!("  First 20 values:");
            for (i, v) in data.iter().take(20).enumerate() {
                println!("  [{}] {:.8}", i, v);
            }
        }
        Err(e) => println!("  ERROR: {}", e),
    }
    println!("---");

    // Test 4: Config extraction
    println!("Config from GGUF metadata:");
    match moe_stream_core::config::StreamingConfig::from_gguf(&reader) {
        Ok(cfg) => {
            println!("  architecture: {}", cfg.architecture);
            println!("  hidden_size: {}", cfg.hidden_size);
            println!("  num_layers: {}", cfg.num_layers);
            println!("  num_attention_heads: {}", cfg.num_attention_heads);
            println!("  num_kv_heads: {}", cfg.num_kv_heads);
            println!("  vocab_size: {}", cfg.vocab_size);
            println!("  num_experts: {}", cfg.num_experts);
            println!("  num_experts_per_tok: {}", cfg.num_experts_per_tok);
            println!("  moe_intermediate_size: {}", cfg.moe_intermediate_size);
            println!("  rope_theta: {}", cfg.rope_theta);
            println!("  rms_norm_eps: {}", cfg.rms_norm_eps);
            println!("  has_shared_expert: {}", cfg.has_shared_expert);
        }
        Err(e) => println!("  ERROR: {}", e),
    }

    println!("\nDone!");
}
