use rmcp::{
    ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
};

use crate::engine_handle::{EngineHandle, TokenEvent};
use moe_stream_core::chat_template::ChatMessage;
use moe_stream_core::model::SamplingParams;

/// MCP tool input: generate text from a raw prompt.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct GenerateRequest {
    #[schemars(description = "The text prompt to generate from")]
    pub prompt: String,
    #[schemars(description = "Maximum number of tokens to generate (default: 512)")]
    pub max_tokens: Option<usize>,
    #[schemars(description = "Sampling temperature (0.0 = greedy, default: 0.0)")]
    pub temperature: Option<f32>,
}

/// MCP tool input: chat with the model using message history.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ChatRequest {
    #[schemars(description = "Chat messages array with 'role' and 'content' fields")]
    pub messages: Vec<ChatMessageInput>,
    #[schemars(description = "Maximum number of tokens to generate (default: 512)")]
    pub max_tokens: Option<usize>,
    #[schemars(description = "Sampling temperature (0.0 = greedy, default: 0.0)")]
    pub temperature: Option<f32>,
}

/// A chat message for the MCP chat tool.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ChatMessageInput {
    #[schemars(description = "Message role: 'system', 'user', or 'assistant'")]
    pub role: String,
    #[schemars(description = "Message content")]
    pub content: String,
}

/// MCP tool input: tokenize text.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct TokenizeRequest {
    #[schemars(description = "Text to tokenize")]
    pub text: String,
}

/// The MCP server wrapping a moe-stream EngineHandle.
#[derive(Debug, Clone)]
pub struct MoeStreamMcp {
    engine: EngineHandle,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl MoeStreamMcp {
    pub fn new(engine: EngineHandle) -> Self {
        Self {
            engine,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Generate text from a raw prompt using the local MoE model")]
    async fn generate(&self, Parameters(req): Parameters<GenerateRequest>) -> String {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: req.prompt,
        }];
        let max_tokens = req.max_tokens.unwrap_or(512);
        let temperature = req.temperature.unwrap_or(0.0);
        let sampling = SamplingParams {
            temperature,
            top_p: if temperature > 0.0 { 0.9 } else { 1.0 },
            repetition_penalty: 1.0,
        };

        let mut rx = self.engine.chat_completion(messages, max_tokens, sampling);
        let mut result = String::new();
        while let Some(event) = rx.recv().await {
            match event {
                TokenEvent::Token(text) => result.push_str(&text),
                TokenEvent::Done { .. } => break,
                TokenEvent::Error(msg) => {
                    return format!("Error: {}", msg);
                }
            }
        }
        result
    }

    #[tool(description = "Chat with the local MoE model using message history")]
    async fn chat(&self, Parameters(req): Parameters<ChatRequest>) -> String {
        let messages: Vec<ChatMessage> = req
            .messages
            .into_iter()
            .map(|m| ChatMessage {
                role: m.role,
                content: m.content,
            })
            .collect();
        let max_tokens = req.max_tokens.unwrap_or(512);
        let temperature = req.temperature.unwrap_or(0.0);
        let sampling = SamplingParams {
            temperature,
            top_p: if temperature > 0.0 { 0.9 } else { 1.0 },
            repetition_penalty: 1.0,
        };

        let mut rx = self.engine.chat_completion(messages, max_tokens, sampling);
        let mut result = String::new();
        while let Some(event) = rx.recv().await {
            match event {
                TokenEvent::Token(text) => result.push_str(&text),
                TokenEvent::Done { .. } => break,
                TokenEvent::Error(msg) => {
                    return format!("Error: {}", msg);
                }
            }
        }
        result
    }

    #[tool(description = "Get information about the loaded model")]
    async fn model_info(&self) -> String {
        let info = self.engine.model_info();
        serde_json::json!({
            "name": info.model_name,
            "architecture": info.architecture,
            "layers": info.num_layers,
            "experts": info.num_experts,
            "experts_per_token": info.num_experts_per_tok,
            "hidden_size": info.hidden_size,
            "vocab_size": info.vocab_size,
            "inference_mode": info.inference_mode,
            "chat_template": info.chat_template,
        })
        .to_string()
    }

    #[tool(description = "Tokenize text and return token IDs and count")]
    async fn tokenize(&self, Parameters(req): Parameters<TokenizeRequest>) -> String {
        match self.engine.tokenize(req.text).await {
            Ok(ids) => {
                serde_json::json!({
                    "token_ids": ids,
                    "count": ids.len(),
                })
                .to_string()
            }
            Err(e) => format!("Error: {}", e),
        }
    }
}

#[tool_handler]
impl ServerHandler for MoeStreamMcp {
    fn get_info(&self) -> ServerInfo {
        let info = self.engine.model_info();
        ServerInfo {
            instructions: Some(format!(
                "moe-stream inference server. Model: {} ({}), Mode: {}",
                info.model_name, info.architecture, info.inference_mode,
            )),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}
