use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use candle_core::{IndexOp, Tensor};
use rand::Rng;
use moe_stream_core::chat_template::ChatMessage;
use moe_stream_core::config::DevicePreference;
use moe_stream_core::model::{Engine, SamplingParams};
use moe_stream_core::tokenizer::Tokenizer;
use tokio::sync::oneshot;

/// Information about the loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_name: String,
    pub architecture: String,
    pub num_layers: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub inference_mode: String,
    pub chat_template: String,
}

/// A generated token event sent from the engine thread.
#[derive(Debug)]
pub enum TokenEvent {
    /// A decoded text fragment.
    Token(String),
    /// Generation finished.
    Done {
        generated_tokens: usize,
        prompt_tokens: usize,
        elapsed_s: f64,
    },
    /// An error occurred.
    Error(String),
}

/// Commands sent to the engine thread.
pub enum EngineCommand {
    ChatCompletion {
        messages: Vec<ChatMessage>,
        max_tokens: usize,
        sampling: SamplingParams,
        token_tx: tokio::sync::mpsc::UnboundedSender<TokenEvent>,
    },
    GetModelInfo {
        reply: oneshot::Sender<ModelInfo>,
    },
    Tokenize {
        text: String,
        reply: oneshot::Sender<Result<Vec<u32>, String>>,
    },
}

/// Handle to the engine running on a dedicated OS thread.
/// Cloneable — all clones share the same engine thread.
#[derive(Clone)]
pub struct EngineHandle {
    cmd_tx: mpsc::Sender<EngineCommand>,
    model_info: ModelInfo,
}

impl std::fmt::Debug for EngineHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineHandle")
            .field("model", &self.model_info.model_name)
            .finish()
    }
}

impl EngineHandle {
    /// Spawn the engine on a dedicated OS thread.
    ///
    /// This blocks the calling thread until the engine is loaded and ready.
    pub fn spawn(
        model_path: &str,
        device_preference: DevicePreference,
        max_seq_len: usize,
    ) -> Result<Self, String> {
        let model_path = model_path.to_string();
        let (cmd_tx, cmd_rx) = mpsc::channel::<EngineCommand>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<ModelInfo, String>>();

        thread::Builder::new()
            .name("moe-engine".into())
            .spawn(move || {
                engine_thread_main(&model_path, device_preference, max_seq_len, ready_tx, cmd_rx);
            })
            .map_err(|e| format!("Failed to spawn engine thread: {}", e))?;

        let model_info = ready_rx
            .recv()
            .map_err(|_| "Engine thread died before ready".to_string())?
            .map_err(|e| format!("Engine init failed: {}", e))?;

        Ok(EngineHandle { cmd_tx, model_info })
    }

    /// Send a chat completion request.
    /// Returns a tokio mpsc receiver that streams TokenEvents.
    pub fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: usize,
        sampling: SamplingParams,
    ) -> tokio::sync::mpsc::UnboundedReceiver<TokenEvent> {
        let (token_tx, token_rx) = tokio::sync::mpsc::unbounded_channel();
        let cmd = EngineCommand::ChatCompletion {
            messages,
            max_tokens,
            sampling,
            token_tx,
        };
        if self.cmd_tx.send(cmd).is_err() {
            eprintln!("[EngineHandle] Engine thread is gone");
        }
        token_rx
    }

    /// Get cached model info (no round-trip to engine thread).
    pub fn model_info(&self) -> &ModelInfo {
        &self.model_info
    }

    /// Tokenize text (round-trip to engine thread).
    pub async fn tokenize(&self, text: String) -> Result<Vec<u32>, String> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let cmd = EngineCommand::Tokenize {
            text,
            reply: reply_tx,
        };
        self.cmd_tx
            .send(cmd)
            .map_err(|_| "Engine thread is gone".to_string())?;
        reply_rx
            .await
            .map_err(|_| "Engine thread dropped reply".to_string())?
    }
}

fn engine_thread_main(
    model_path: &str,
    device_preference: DevicePreference,
    max_seq_len: usize,
    ready_tx: mpsc::Sender<Result<ModelInfo, String>>,
    cmd_rx: mpsc::Receiver<EngineCommand>,
) {
    // Load engine
    eprintln!("[Engine] Loading model: {}", model_path);
    let t0 = Instant::now();
    let mut engine = match Engine::open_with_device(model_path, max_seq_len, device_preference) {
        Ok(e) => e,
        Err(e) => {
            let _ = ready_tx.send(Err(format!("{}", e)));
            return;
        }
    };

    // Preload all weights
    if let Err(e) = engine.preload_weights() {
        let _ = ready_tx.send(Err(format!("preload_weights: {}", e)));
        return;
    }

    // GPU warmup
    if let Err(e) = engine.warmup_gpu() {
        let _ = ready_tx.send(Err(format!("warmup_gpu: {}", e)));
        return;
    }

    let load_time = t0.elapsed().as_secs_f64();
    let config = engine.config();
    let mode_str = config
        .inference_mode
        .map(|m| format!("{}", m))
        .unwrap_or_else(|| "Unknown".to_string());

    eprintln!(
        "[Engine] Ready in {:.2}s ({} layers, {} experts, mode={})",
        load_time, config.num_layers, config.num_experts, mode_str,
    );

    let model_info = ModelInfo {
        model_name: config.model_name.clone(),
        architecture: config.architecture.clone(),
        num_layers: config.num_layers,
        num_experts: config.num_experts,
        num_experts_per_tok: config.num_experts_per_tok,
        hidden_size: config.hidden_size,
        vocab_size: config.vocab_size,
        inference_mode: mode_str,
        chat_template: engine.chat_template().name().to_string(),
    };

    // Auto-detect tokenizer in model directory
    let tokenizer = {
        let model_dir = std::path::Path::new(model_path).parent();
        let candidate = model_dir.map(|d| d.join("tokenizer.json"));
        match candidate {
            Some(p) if p.exists() => match Tokenizer::from_file(&p) {
                Ok(t) => {
                    eprintln!("[Engine] Tokenizer loaded: {}", p.display());
                    Some(t)
                }
                Err(e) => {
                    eprintln!("[Engine] Tokenizer load failed: {}", e);
                    None
                }
            },
            _ => {
                eprintln!("[Engine] No tokenizer.json found in model directory");
                None
            }
        }
    };

    let _ = ready_tx.send(Ok(model_info));

    // Track cached token IDs for KV-cache reuse
    let mut cached_ids: Vec<u32> = Vec::new();
    let mut rng = rand::thread_rng();

    // Command loop
    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            EngineCommand::ChatCompletion {
                messages,
                max_tokens,
                sampling,
                token_tx,
            } => {
                handle_chat_completion(
                    &mut engine,
                    tokenizer.as_ref(),
                    &messages,
                    max_tokens,
                    &sampling,
                    &token_tx,
                    &mut cached_ids,
                    &mut rng,
                );
            }
            EngineCommand::GetModelInfo { reply } => {
                let config = engine.config();
                let mode_str = config
                    .inference_mode
                    .map(|m| format!("{}", m))
                    .unwrap_or_else(|| "Unknown".to_string());
                let info = ModelInfo {
                    model_name: config.model_name.clone(),
                    architecture: config.architecture.clone(),
                    num_layers: config.num_layers,
                    num_experts: config.num_experts,
                    num_experts_per_tok: config.num_experts_per_tok,
                    hidden_size: config.hidden_size,
                    vocab_size: config.vocab_size,
                    inference_mode: mode_str,
                    chat_template: engine.chat_template().name().to_string(),
                };
                let _ = reply.send(info);
            }
            EngineCommand::Tokenize { text, reply } => {
                let result = match &tokenizer {
                    Some(tok) => tok
                        .encode(&text)
                        .map_err(|e| format!("{}", e)),
                    None => Err("No tokenizer loaded".to_string()),
                };
                let _ = reply.send(result);
            }
        }
    }

    eprintln!("[Engine] Command channel closed, shutting down");
}

fn handle_chat_completion(
    engine: &mut Engine,
    tokenizer: Option<&Tokenizer>,
    messages: &[ChatMessage],
    max_tokens: usize,
    sampling: &SamplingParams,
    token_tx: &tokio::sync::mpsc::UnboundedSender<TokenEvent>,
    cached_ids: &mut Vec<u32>,
    rng: &mut impl Rng,
) {
    let tokenizer = match tokenizer {
        Some(t) => t,
        None => {
            let _ = token_tx.send(TokenEvent::Error("No tokenizer loaded".into()));
            return;
        }
    };

    // Format messages using chat template
    let prompt = engine.chat_template().apply(messages);

    // Encode
    let prompt_ids = match tokenizer.encode(&prompt) {
        Ok(ids) => ids,
        Err(e) => {
            let _ = token_tx.send(TokenEvent::Error(format!("Encode failed: {}", e)));
            return;
        }
    };

    let prompt_token_count = prompt_ids.len();

    // Fresh generation each time (no KV-cache reuse across requests for simplicity)
    engine.clear_cache();
    cached_ids.clear();

    let t0 = Instant::now();

    // Prefill
    let new_ids = &prompt_ids;
    let prefill_logits = if !new_ids.is_empty() {
        let input = match Tensor::from_vec(
            new_ids.to_vec(),
            (1, new_ids.len()),
            engine.device(),
        ) {
            Ok(t) => t,
            Err(e) => {
                let _ = token_tx.send(TokenEvent::Error(format!("Tensor error: {}", e)));
                return;
            }
        };
        match engine.forward(&input, true) {
            Ok(logits) => {
                cached_ids.extend_from_slice(new_ids);
                let seq_len = new_ids.len();
                match logits.i((0, seq_len - 1)) {
                    Ok(t) => t.to_vec1::<f32>().ok(),
                    Err(_) => None,
                }
            }
            Err(e) => {
                let _ = token_tx.send(TokenEvent::Error(format!("Prefill failed: {}", e)));
                engine.clear_cache();
                cached_ids.clear();
                return;
            }
        }
    } else {
        let _ = token_tx.send(TokenEvent::Error("Empty prompt".into()));
        return;
    };

    // Generate tokens
    let mut all_ids = prompt_ids.clone();
    let mut generated = Vec::new();
    let mut next_logits: Option<Vec<f32>> = prefill_logits;

    for _step in 0..max_tokens {
        let logits_vec = match next_logits.take() {
            Some(lv) => lv,
            None => break,
        };

        let next_token = sampling.sample(&logits_vec, &generated, rng);

        cached_ids.push(next_token);
        generated.push(next_token);
        all_ids.push(next_token);

        // Incremental decode
        let full_text = tokenizer.decode(&all_ids).unwrap_or_default();
        let prev_text = tokenizer.decode(&all_ids[..all_ids.len() - 1]).unwrap_or_default();
        let start = prev_text.len();
        if start < full_text.len() && full_text.is_char_boundary(start) {
            let new_text = &full_text[start..];
            if token_tx.send(TokenEvent::Token(new_text.to_string())).is_err() {
                // Client disconnected
                break;
            }
        }

        // EOS check
        if engine.chat_template().is_eos(next_token) {
            break;
        }

        if generated.len() >= max_tokens {
            break;
        }

        // Decode next token
        let input = match Tensor::from_vec(vec![next_token], (1, 1), engine.device()) {
            Ok(t) => t,
            Err(e) => {
                let _ = token_tx.send(TokenEvent::Error(format!("Tensor error: {}", e)));
                break;
            }
        };
        match engine.forward(&input, true) {
            Ok(logits) => {
                next_logits = logits
                    .i((0, 0))
                    .ok()
                    .and_then(|t: Tensor| t.to_vec1::<f32>().ok());
            }
            Err(e) => {
                let _ = token_tx.send(TokenEvent::Error(format!("Decode failed: {}", e)));
                break;
            }
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let _ = token_tx.send(TokenEvent::Done {
        generated_tokens: generated.len(),
        prompt_tokens: prompt_token_count,
        elapsed_s: elapsed,
    });

    eprintln!(
        "[Engine] Generated {} tokens in {:.2}s ({:.1} tok/s)",
        generated.len(),
        elapsed,
        generated.len() as f64 / elapsed,
    );
}
