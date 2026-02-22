/// Generate text from GGUF model using SSD streaming.
///
/// Usage:
///   # One-shot generation:
///   generate <gguf_path> [max_tokens] [--prompt "text"] [--stream] [--preload-gates] [--preload-attn] [--dynamic-k] [--k-min N] [--k-max N]
///
///   # Persistent server (JSONL over stdin/stdout):
///   generate <gguf_path> --server [--preload-gates] [--preload-attn] [--dynamic-k] [--k-min N] [--k-max N]
///
/// Server mode protocol:
///   stdin:  {"prompt": "text", "max_tokens": 100}
///   stdin:  {"prompt": "text + more", "max_tokens": 1, "continue": true}  (reuse KV-cache)
///   stdout: {"token": "decoded text"}     (per token)
///   stdout: {"done": true, "tokens": N, "elapsed": 1.23, "cached_tokens": M}
use candle_core::{IndexOp, Tensor};
use moe_stream_core::config::{LayerTierConfig, DevicePreference};
use moe_stream_core::model::{Engine, SamplingParams};
use moe_stream_core::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};
use std::time::Instant;

#[derive(Deserialize)]
struct ServerRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// When true, reuse KV-cache from the previous request. Only the new
    /// tokens beyond the cached prefix are processed. When false (default),
    /// the cache is cleared and the full prompt is re-processed.
    #[serde(default)]
    #[serde(rename = "continue")]
    continue_: bool,
}

fn default_max_tokens() -> usize { 512 }

#[derive(Serialize)]
#[serde(untagged)]
enum ServerResponse {
    Token { token: String },
    Done { done: bool, tokens: usize, elapsed: f64, cached_tokens: usize },
    Error { error: String },
}

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let gguf_path = args.get(1).expect(
        "Usage: generate <gguf_path> [max_tokens] [--prompt \"text\"] [--stream] [--server] [--device {auto|gpu|cpu}] [--dynamic-k] [--k-min N] [--k-max N] [--layer-adaptive-k] [--deep-k-min N] [--deep-k-max N] [--layer-adaptive-cache] [--adaptive-skip] [--adaptive-skip-threshold F] [--entropy-profile]",
    );
    let max_tokens: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let no_preload = args.iter().any(|a| a == "--no-preload");
    let preload_all = !no_preload; // Auto-preload by default; --no-preload to disable
    let preload_gates = args.iter().any(|a| a == "--preload-gates") || preload_all;
    let preload_attn = args.iter().any(|a| a == "--preload-attn") || preload_all;
    let preload_dn = args.iter().any(|a| a == "--preload-dn") || preload_all;
    let cpu_compute = args.iter().any(|a| a == "--cpu-compute");
    let gpu_compute = args.iter().any(|a| a == "--gpu-compute");
    // --device {auto|gpu|cpu}: controls inference mode selection
    // --gpu-compute is an alias for --device gpu, --cpu-compute for --device cpu
    let device_arg: Option<String> = args
        .iter()
        .position(|a| a == "--device")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let device_preference = if gpu_compute || device_arg.as_deref() == Some("gpu") {
        DevicePreference::Gpu
    } else if cpu_compute || device_arg.as_deref() == Some("cpu") {
        DevicePreference::Cpu
    } else {
        DevicePreference::Auto
    };
    let server_mode = args.iter().any(|a| a == "--server");
    let max_layers: usize = args
        .iter()
        .position(|a| a == "--max-layers")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let custom_tokens: Option<Vec<u32>> = args
        .iter()
        .position(|a| a == "--tokens")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect());
    let prompt_text: Option<String> = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let tokenizer_path: Option<String> = args
        .iter()
        .position(|a| a == "--tokenizer")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let streaming = args.iter().any(|a| a == "--stream");
    let dynamic_k = args.iter().any(|a| a == "--dynamic-k");
    let k_min: usize = args
        .iter()
        .position(|a| a == "--k-min")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let k_max: Option<usize> = args
        .iter()
        .position(|a| a == "--k-max")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());

    // Adaptive expert skip
    let entropy_profile = args.iter().any(|a| a == "--entropy-profile");
    let routing_stats = args.iter().any(|a| a == "--routing-stats");
    let routing_stats_json: Option<String> = args
        .iter()
        .position(|a| a == "--routing-stats-json")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let adaptive_skip = args.iter().any(|a| a == "--adaptive-skip");
    let adaptive_skip_threshold: f32 = args
        .iter()
        .position(|a| a == "--adaptive-skip-threshold")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.95);
    let adaptive_skip_max_consecutive: u32 = args
        .iter()
        .position(|a| a == "--adaptive-skip-max-consecutive")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    // Layer-adaptive K
    let layer_adaptive_k = args.iter().any(|a| a == "--layer-adaptive-k");
    let layer_adaptive_cache = args.iter().any(|a| a == "--layer-adaptive-cache");
    let deep_k_min: Option<usize> = args
        .iter()
        .position(|a| a == "--deep-k-min")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());
    let deep_k_max: Option<usize> = args
        .iter()
        .position(|a| a == "--deep-k-max")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());

    // Sampling parameters
    let temperature: Option<f32> = args
        .iter()
        .position(|a| a == "--temperature")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());
    let top_p: f32 = args
        .iter()
        .position(|a| a == "--top-p")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.9);
    let repetition_penalty: f32 = args
        .iter()
        .position(|a| a == "--repetition-penalty")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);

    // Hybrid RAM/SSD: mlock Q4 expert pages within budget
    // --ram-budget N    : pin N GB of expert pages
    // --ram-budget auto : automatically compute optimal budget (30% of free RAM)
    let ram_budget: Option<f32> = args
        .iter()
        .position(|a| a == "--ram-budget")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| {
            if s == "auto" {
                Some(-1.0f32) // sentinel for auto mode
            } else {
                s.parse().ok()
            }
        });

    // Build sampling params: greedy by default (backward compat), sampled when --temperature is set
    let sampling = match temperature {
        Some(t) => SamplingParams { temperature: t, top_p, repetition_penalty },
        None => SamplingParams { temperature: 0.0, top_p: 1.0, repetition_penalty: 1.0 },
    };

    // Auto-detect tokenizer.json in model directory
    let tokenizer_path = tokenizer_path.or_else(|| {
        let model_dir = std::path::Path::new(gguf_path).parent()?;
        let candidate = model_dir.join("tokenizer.json");
        if candidate.exists() {
            Some(candidate.to_string_lossy().to_string())
        } else {
            None
        }
    });

    eprintln!("=== MoE Streaming Engine (Rust) ===");
    eprintln!("Model: {}", gguf_path);
    if !server_mode {
        eprintln!("Max tokens: {}", max_tokens);
    }
    eprintln!("Device preference: {:?}", device_preference);
    eprintln!("Preload gates: {}, attention: {}", preload_gates, preload_attn);
    if sampling.temperature > 0.0 {
        eprintln!(
            "Sampling: temperature={}, top_p={}, repetition_penalty={}",
            sampling.temperature, sampling.top_p, sampling.repetition_penalty,
        );
    } else {
        eprintln!("Sampling: greedy (use --temperature to enable sampling)");
    }
    if layer_adaptive_k {
        eprintln!("Layer-Adaptive K: enabled (cache={})", layer_adaptive_cache);
    }
    if adaptive_skip {
        eprintln!("Adaptive Skip: enabled (threshold={})", adaptive_skip_threshold);
    }
    if let Some(budget) = ram_budget {
        if budget < 0.0 {
            eprintln!("RAM Budget: auto (15% of system RAM)");
        } else {
            eprintln!("RAM Budget: {:.1} GB (mlock Q4 expert pages)", budget);
        }
    }
    if server_mode {
        eprintln!("Mode: JSONL server (stdin/stdout)");
    }
    eprintln!();

    // Load tokenizer
    let tokenizer = tokenizer_path.as_ref().map(|p| {
        let tok = Tokenizer::from_file(p).expect("Failed to load tokenizer");
        eprintln!("Tokenizer loaded: {}", p);
        tok
    });

    // Load engine
    let t0 = Instant::now();
    let mut engine = Engine::open_with_device(gguf_path, 4096, device_preference)
        .expect("Failed to open model");
    if max_layers > 0 {
        engine.set_max_layers(max_layers);
    }
    if dynamic_k {
        engine.set_dynamic_k(true, k_min);
        if let Some(k_max_val) = k_max {
            engine.set_dynamic_k_max(k_max_val);
        }
    }
    if layer_adaptive_k {
        let k_max_val = engine.config().num_experts_per_tok;
        let mut tier = LayerTierConfig::default_for_k(k_max_val);
        if let Some(v) = deep_k_min {
            tier.deep_k_min = v;
        }
        if let Some(v) = deep_k_max {
            tier.deep_k_max = v;
        }
        engine.set_layer_adaptive_k(true, tier, layer_adaptive_cache);
    }
    if adaptive_skip {
        engine.set_adaptive_skip(true, adaptive_skip_threshold);
        engine.set_adaptive_skip_max_consecutive(adaptive_skip_max_consecutive);
    }
    if entropy_profile {
        engine.set_entropy_profiling(true);
    }
    if routing_stats || routing_stats_json.is_some() {
        engine.set_routing_stats(true);
    }
    // Note: --gpu-compute and --cpu-compute are handled via device_preference
    // passed to open_with_device() above. No need for post-hoc set_gpu_compute().
    if let Some(budget) = ram_budget {
        engine.set_ram_budget(Some(budget));
    }
    let mode_str = engine.config().inference_mode
        .map(|m| format!("{}", m))
        .unwrap_or_else(|| "Unknown".to_string());
    eprintln!(
        "Engine loaded in {:.2}s (config: {} layers, {} experts, mode={})",
        t0.elapsed().as_secs_f64(),
        engine.config().num_layers,
        engine.config().num_experts,
        mode_str,
    );

    // Preload weights (auto: gates + norms + shared experts + DeltaNet + attention)
    if preload_all {
        let t0 = Instant::now();
        engine.preload_weights().expect("Failed to preload weights");
        eprintln!("All weights preloaded in {:.2}s", t0.elapsed().as_secs_f64());
    } else {
        if preload_gates {
            engine.preload_gates().expect("Failed to preload gates");
        }
        if preload_dn {
            engine.preload_deltanet().expect("Failed to preload DeltaNet");
        }
        if preload_attn {
            engine.preload_attention().expect("Failed to preload attention");
        }
    }

    // GPU warmup: compile Metal shaders before real generation
    engine.warmup_gpu().expect("GPU warmup failed");

    if server_mode {
        run_server(&mut engine, tokenizer.as_ref().expect("Server mode requires tokenizer"), &sampling);
    } else {
        run_oneshot(&mut engine, tokenizer.as_ref(), &args, &prompt_text, &custom_tokens, &prompt_ids_from_args(&args, tokenizer.as_ref(), &prompt_text, &custom_tokens), max_tokens, streaming, &sampling);
    }

    // Output entropy profile results
    if entropy_profile {
        let summary = engine.entropy_profile_summary();
        let samples = engine.entropy_profile_samples();
        let num_experts = engine.config().num_experts;
        let h_max = (num_experts as f32).ln();
        eprintln!("\n=== Entropy Profile ({} samples across {} layers) ===", samples, summary.len());
        eprintln!("{:<8} {:>10} {:>10} {:>8} {:>8} {:>8} {:>8}",
            "Layer", "Mean(nats)", "% of Max", "Std", "Min", "Max", "Count");
        for s in &summary {
            if s.count > 0 {
                eprintln!("{:<8} {:>10.3} {:>9.1}% {:>8.3} {:>8.3} {:>8.3} {:>8}",
                    s.layer_idx, s.mean, s.mean / h_max * 100.0, s.std, s.min, s.max, s.count);
            }
        }
        // JSON output for programmatic use
        eprintln!("\n--- JSON ---");
        eprint!("{{\"h_max\":{:.4},\"num_experts\":{},\"layers\":[", h_max, num_experts);
        for (i, s) in summary.iter().enumerate() {
            if i > 0 { eprint!(","); }
            eprint!("{{\"layer\":{},\"mean\":{:.4},\"std\":{:.4},\"min\":{:.4},\"max\":{:.4},\"count\":{}}}",
                s.layer_idx, s.mean, s.std, s.min, s.max, s.count);
        }
        eprintln!("]}}");
    }

    // Output routing statistics for calibration-based importance scoring
    if routing_stats || routing_stats_json.is_some() {
        let summary = engine.routing_stats_summary();
        let total_tokens = engine.routing_stats_tokens();
        let num_experts = engine.config().num_experts;

        if routing_stats {
            eprintln!("\n=== Routing Stats ({} tokens, {} layers, {} experts) ===",
                total_tokens, summary.len(), num_experts);
            for ls in &summary {
                if ls.total_tokens == 0 { continue; }
                // Show top-5 and bottom-5 experts by importance
                let mut sorted: Vec<_> = ls.experts.iter().collect();
                sorted.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());
                let top5: Vec<_> = sorted.iter().take(5)
                    .map(|e| format!("E{}({:.4})", e.expert_idx, e.importance))
                    .collect();
                let bot5: Vec<_> = sorted.iter().rev().take(5)
                    .map(|e| format!("E{}({:.4})", e.expert_idx, e.importance))
                    .collect();
                eprintln!("L{:>2}: tokens={}, top=[{}], bot=[{}]",
                    ls.layer_idx, ls.total_tokens,
                    top5.join(", "), bot5.join(", "));
            }
        }

        // JSON output
        if let Some(json_path) = &routing_stats_json {
            let mut json = String::from("{\n");
            json.push_str(&format!("  \"model\": \"Qwen3-Coder-Next-80B\",\n"));
            json.push_str(&format!("  \"num_experts\": {},\n", num_experts));
            json.push_str(&format!("  \"total_tokens\": {},\n", total_tokens));
            json.push_str("  \"layers\": [\n");
            for (li, ls) in summary.iter().enumerate() {
                json.push_str("    {\n");
                json.push_str(&format!("      \"layer\": {},\n", ls.layer_idx));
                json.push_str(&format!("      \"total_tokens\": {},\n", ls.total_tokens));
                json.push_str("      \"experts\": {\n");
                for (ei, e) in ls.experts.iter().enumerate() {
                    let comma = if ei < ls.experts.len() - 1 { "," } else { "" };
                    json.push_str(&format!(
                        "        \"{}\": {{\"count\": {}, \"frequency\": {:.6}, \"mean_gate_weight\": {:.6}, \"importance\": {:.8}}}{}\n",
                        e.expert_idx, e.count, e.frequency, e.mean_gate_weight, e.importance, comma
                    ));
                }
                json.push_str("      }\n");
                let comma = if li < summary.len() - 1 { "," } else { "" };
                json.push_str(&format!("    }}{}\n", comma));
            }
            json.push_str("  ]\n}\n");
            std::fs::write(json_path, &json).expect("Failed to write routing stats JSON");
            eprintln!("Routing stats saved to: {}", json_path);
        }
    }
}

fn prompt_ids_from_args(
    args: &[String],
    tokenizer: Option<&Tokenizer>,
    prompt_text: &Option<String>,
    custom_tokens: &Option<Vec<u32>>,
) -> Vec<u32> {
    if let Some(ref text) = prompt_text {
        let tok = tokenizer.expect("--prompt requires tokenizer");
        tok.encode(text).expect("Failed to encode prompt")
    } else if let Some(ref tokens) = custom_tokens {
        tokens.clone()
    } else if args.iter().any(|a| a == "--hello") {
        vec![9707]
    } else if args.iter().any(|a| a == "--chat") {
        vec![151644, 872, 198, 7985, 264, 75698, 729, 304, 13027, 151645, 198, 151644, 77091, 198]
    } else if args.iter().any(|a| a == "--think") {
        vec![151644, 872, 198, 7985, 264, 75698, 729, 304, 13027, 151645, 198, 151644, 77091, 198, 151667, 198]
    } else if args.iter().any(|a| a == "--comment") {
        vec![2, 9645, 264, 75698, 729, 304, 13027, 198, 750, 75698, 1445, 982]
    } else {
        vec![750, 75698, 1445, 982]
    }
}

/// Persistent server mode: read JSONL from stdin, write JSONL to stdout.
/// Engine + weights are loaded once; each request only pays generation cost.
///
/// KV-cache persistence: when a request includes `"continue": true`, the
/// KV-cache from the previous request is reused. Only tokens beyond the
/// cached prefix are processed, dramatically reducing latency for
/// sequential verification (Mode C).
fn run_server(engine: &mut Engine, tokenizer: &Tokenizer, sampling: &SamplingParams) {
    eprintln!("Server ready. Waiting for JSONL requests on stdin...");

    // Signal readiness to parent process
    let ready = serde_json::json!({"ready": true});
    println!("{}", serde_json::to_string(&ready).unwrap());
    io::stdout().flush().ok();

    // Track cached token IDs across requests for KV-cache persistence
    let mut cached_ids: Vec<u32> = Vec::new();
    let mut rng = rand::thread_rng();

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let request: ServerRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = ServerResponse::Error { error: format!("Invalid JSON: {}", e) };
                println!("{}", serde_json::to_string(&resp).unwrap());
                io::stdout().flush().ok();
                continue;
            }
        };

        // Encode prompt
        let prompt_ids = match tokenizer.encode(&request.prompt) {
            Ok(ids) => ids,
            Err(e) => {
                let resp = ServerResponse::Error { error: format!("Encode failed: {}", e) };
                println!("{}", serde_json::to_string(&resp).unwrap());
                io::stdout().flush().ok();
                continue;
            }
        };

        // Determine how many tokens are already cached
        let prefix_len = if request.continue_ {
            // Find the common prefix between cached_ids and prompt_ids
            cached_ids.iter().zip(prompt_ids.iter())
                .take_while(|(a, b)| a == b)
                .count()
        } else {
            0
        };

        if !request.continue_ || prefix_len == 0 {
            // Fresh request: clear cache and process full prompt
            engine.clear_cache();
            cached_ids.clear();
        } else if prefix_len < cached_ids.len() {
            // Prompt diverged from cache: must clear and re-process from scratch.
            // KV-cache cannot be partially truncated.
            engine.clear_cache();
            cached_ids.clear();
            eprintln!(
                "Cache miss: prompt diverged at token {} (cached {}), full re-process",
                prefix_len, cached_ids.len(),
            );
        }
        // else: prefix_len == cached_ids.len() -> perfect prefix match, reuse cache

        let new_ids = &prompt_ids[cached_ids.len()..];
        eprintln!(
            "Request: prompt={} tokens, cached={}, new={}, continue={}, max_tokens={}",
            prompt_ids.len(), cached_ids.len(), new_ids.len(),
            request.continue_, request.max_tokens,
        );

        let t0 = Instant::now();

        // Prefill the uncached portion and get logits for the last position.
        // If all tokens are cached (new_ids is empty), we still need to get
        // logits from the last cached position for the first decode step.
        let prefill_logits = if !new_ids.is_empty() {
            let input = match Tensor::from_vec(
                new_ids.to_vec(), (1, new_ids.len()), engine.device(),
            ) {
                Ok(t) => t,
                Err(e) => {
                    let resp = ServerResponse::Error { error: format!("Tensor error: {}", e) };
                    println!("{}", serde_json::to_string(&resp).unwrap());
                    io::stdout().flush().ok();
                    continue;
                }
            };
            match engine.forward(&input, true) {
                Ok(logits) => {
                    cached_ids.extend_from_slice(new_ids);
                    // Get logits from last position
                    let seq_len = new_ids.len();
                    match logits.i((0, seq_len - 1)) {
                        Ok(t) => t.to_vec1::<f32>().ok(),
                        Err(_) => None,
                    }
                }
                Err(e) => {
                    let resp = ServerResponse::Error { error: format!("Prefill failed: {}", e) };
                    println!("{}", serde_json::to_string(&resp).unwrap());
                    io::stdout().flush().ok();
                    engine.clear_cache();
                    cached_ids.clear();
                    continue;
                }
            }
        } else {
            // All tokens already cached, nothing new to process.
            // With continue=true and identical prompt, there's no new context.
            // The client should extend the prompt with new tokens.
            let resp = ServerResponse::Error {
                error: "No new tokens to process (prompt identical to cache). \
                        Extend the prompt or use continue=false.".to_string(),
            };
            println!("{}", serde_json::to_string(&resp).unwrap());
            io::stdout().flush().ok();
            continue;
        };

        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Generate tokens one at a time with streaming output
        let mut all_ids = prompt_ids.clone();
        let mut generated = Vec::new();
        let mut gen_ok = true;

        // Use prefill logits for the first token
        let first_logits = prefill_logits;
        let mut next_logits: Option<Vec<f32>> = first_logits;

        for _step in 0..request.max_tokens {
            // Get logits: either from prefill (first step) or from decode
            let logits_vec = if let Some(lv) = next_logits.take() {
                lv
            } else {
                // Should not happen; break as safety
                break;
            };

            // Sample (greedy when temperature==0.0, otherwise stochastic)
            let next_token = sampling.sample(&logits_vec, &generated, &mut rng);

            cached_ids.push(next_token);
            generated.push(next_token);
            all_ids.push(next_token);

            // Incremental decode with full context
            let full_text = tokenizer.decode(&all_ids).unwrap_or_default();
            let prev_text = tokenizer.decode(&all_ids[..all_ids.len() - 1]).unwrap_or_default();
            let start = prev_text.len();
            if start < full_text.len() && full_text.is_char_boundary(start) {
                let new_text = &full_text[start..];
                let resp = ServerResponse::Token { token: new_text.to_string() };
                println!("{}", serde_json::to_string(&resp).unwrap());
                io::stdout().flush().ok();
            }

            // EOS check
            if engine.chat_template().is_eos(next_token) {
                break;
            }

            // Stop if we've generated enough
            if generated.len() >= request.max_tokens {
                break;
            }

            // Decode next token
            let input = match Tensor::from_vec(
                vec![next_token], (1, 1), engine.device(),
            ) {
                Ok(t) => t,
                Err(e) => {
                    let resp = ServerResponse::Error { error: format!("Tensor error: {}", e) };
                    println!("{}", serde_json::to_string(&resp).unwrap());
                    io::stdout().flush().ok();
                    gen_ok = false;
                    break;
                }
            };
            match engine.forward(&input, true) {
                Ok(logits) => {
                    next_logits = logits.i((0, 0)).ok().and_then(|t| t.to_vec1::<f32>().ok());
                }
                Err(e) => {
                    let resp = ServerResponse::Error { error: format!("Decode failed: {}", e) };
                    println!("{}", serde_json::to_string(&resp).unwrap());
                    io::stdout().flush().ok();
                    gen_ok = false;
                    break;
                }
            }
        }

        let elapsed = t0.elapsed().as_secs_f64();
        if gen_ok {
            let resp = ServerResponse::Done {
                done: true,
                tokens: generated.len(),
                elapsed,
                cached_tokens: cached_ids.len(),
            };
            println!("{}", serde_json::to_string(&resp).unwrap());
            io::stdout().flush().ok();
            eprintln!(
                "Generated {} tokens in {:.2}s ({:.3} tok/s, prefill={:.1}ms, cached={})",
                generated.len(), elapsed,
                generated.len() as f64 / elapsed,
                prefill_ms, cached_ids.len(),
            );
        }
    }

    eprintln!("Server: stdin closed, shutting down.");
}

/// One-shot generation mode (original behavior).
fn run_oneshot(
    engine: &mut Engine,
    tokenizer: Option<&Tokenizer>,
    _args: &[String],
    _prompt_text: &Option<String>,
    _custom_tokens: &Option<Vec<u32>>,
    prompt_ids: &[u32],
    max_tokens: usize,
    streaming: bool,
    sampling: &SamplingParams,
) {
    eprintln!("Prompt tokens: {} tokens", prompt_ids.len());
    if let Some(ref tok) = tokenizer {
        if let Ok(decoded) = tok.decode(prompt_ids) {
            eprintln!("Prompt text: {:?}", decoded);
        }
    }
    eprintln!("Generating {} tokens...\n", max_tokens);

    let t0 = Instant::now();

    if streaming && tokenizer.is_some() {
        let tok = tokenizer.unwrap();
        let mut all_ids = prompt_ids.to_vec();

        let generated = if sampling.is_greedy() {
            engine.generate_streaming(prompt_ids, max_tokens, |token_id| {
                all_ids.push(token_id);
                let full_text = tok.decode(&all_ids).unwrap_or_default();
                let prev_text = tok.decode(&all_ids[..all_ids.len() - 1]).unwrap_or_default();
                let start = prev_text.len();
                if start < full_text.len() && full_text.is_char_boundary(start) {
                    let new_text = &full_text[start..];
                    print!("{}", new_text);
                    io::stdout().flush().ok();
                }
                true
            })
        } else {
            engine.generate_streaming_sampled(prompt_ids, max_tokens, sampling, |token_id| {
                all_ids.push(token_id);
                let full_text = tok.decode(&all_ids).unwrap_or_default();
                let prev_text = tok.decode(&all_ids[..all_ids.len() - 1]).unwrap_or_default();
                let start = prev_text.len();
                if start < full_text.len() && full_text.is_char_boundary(start) {
                    let new_text = &full_text[start..];
                    print!("{}", new_text);
                    io::stdout().flush().ok();
                }
                true
            })
        }.expect("Generation failed");

        let total_time = t0.elapsed();
        println!();
        eprintln!(
            "\nGenerated {} tokens in {:.2}s ({:.3} tok/s)",
            generated.len(),
            total_time.as_secs_f64(),
            generated.len() as f64 / total_time.as_secs_f64(),
        );
        let (skip_rate, skip_count, total_count) = engine.adaptive_skip_stats();
        if total_count > 0 {
            eprintln!(
                "Adaptive Skip: {:.1}% of layers skipped ({}/{})",
                skip_rate, skip_count, total_count,
            );
        }
    } else {
        let generated = if sampling.is_greedy() {
            engine.generate(prompt_ids, max_tokens)
        } else {
            engine.generate_sampled(prompt_ids, max_tokens, sampling)
        }.expect("Generation failed");
        let total_time = t0.elapsed();

        // Decode with full context then strip prompt text
        if let Some(ref tok) = tokenizer {
            let mut all_ids = prompt_ids.to_vec();
            all_ids.extend_from_slice(&generated);
            let full_text = tok.decode(&all_ids).unwrap_or_default();
            let prompt_decoded = tok.decode(prompt_ids).unwrap_or_default();
            let generated_text = if full_text.len() >= prompt_decoded.len() {
                &full_text[prompt_decoded.len()..]
            } else {
                &full_text
            };
            println!("{}", generated_text);
        }

        eprintln!(
            "\nGenerated {} tokens in {:.2}s ({:.3} tok/s)",
            generated.len(),
            total_time.as_secs_f64(),
            generated.len() as f64 / total_time.as_secs_f64(),
        );
        let (skip_rate, skip_count, total_count) = engine.adaptive_skip_stats();
        if total_count > 0 {
            eprintln!(
                "Adaptive Skip: {:.1}% of layers skipped ({}/{})",
                skip_rate, skip_count, total_count,
            );
        }
        eprintln!("Token IDs: {:?}", generated);
    }
}
