use std::convert::Infallible;

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use crate::engine_handle::{EngineHandle, TokenEvent};
use crate::types::*;
use moe_stream_core::chat_template::ChatMessage;
use moe_stream_core::model::SamplingParams;

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(handle): State<EngineHandle>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if req.stream {
        handle_streaming(handle, req).await.into_response()
    } else {
        handle_non_streaming(handle, req).await.into_response()
    }
}

async fn handle_streaming(
    handle: EngineHandle,
    req: ChatCompletionRequest,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let (messages, max_tokens, sampling) = parse_request(&req);
    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    let model_name = handle.model_info().model_name.clone();

    let token_rx = handle.chat_completion(messages, max_tokens, sampling);
    let rx_stream = UnboundedReceiverStream::new(token_rx);

    let completion_id_clone = completion_id.clone();
    let model_name_clone = model_name.clone();

    // First chunk: role
    let first_chunk = ChatCompletionChunk {
        id: completion_id.clone(),
        object: "chat.completion.chunk",
        created,
        model: model_name.clone(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: Some("assistant".to_string()),
                content: None,
            },
            finish_reason: None,
        }],
    };

    let first_event = Event::default()
        .data(serde_json::to_string(&first_chunk).unwrap());

    let token_stream = rx_stream.map(move |event| {
        let chunk = match event {
            TokenEvent::Token(text) => ChatCompletionChunk {
                id: completion_id_clone.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_name_clone.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: Some(text),
                    },
                    finish_reason: None,
                }],
            },
            TokenEvent::Done { .. } => ChatCompletionChunk {
                id: completion_id_clone.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_name_clone.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
            },
            TokenEvent::Error(msg) => ChatCompletionChunk {
                id: completion_id_clone.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_name_clone.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: Some(format!("[Error: {}]", msg)),
                    },
                    finish_reason: Some("stop".to_string()),
                }],
            },
        };
        Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()))
    });

    // Chain: first role chunk, then token stream, then [DONE]
    let done_stream = tokio_stream::once(Ok(Event::default().data("[DONE]")));
    let full_stream = tokio_stream::once(Ok(first_event))
        .chain(token_stream)
        .chain(done_stream);

    Sse::new(full_stream)
}

async fn handle_non_streaming(
    handle: EngineHandle,
    req: ChatCompletionRequest,
) -> Json<ChatCompletionResponse> {
    let (messages, max_tokens, sampling) = parse_request(&req);
    let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    let model_name = handle.model_info().model_name.clone();

    let mut token_rx = handle.chat_completion(messages, max_tokens, sampling);

    let mut full_text = String::new();
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;

    while let Some(event) = token_rx.recv().await {
        match event {
            TokenEvent::Token(text) => {
                full_text.push_str(&text);
            }
            TokenEvent::Done {
                generated_tokens,
                prompt_tokens: pt,
                ..
            } => {
                completion_tokens = generated_tokens;
                prompt_tokens = pt;
                break;
            }
            TokenEvent::Error(msg) => {
                full_text = format!("[Error: {}]", msg);
                break;
            }
        }
    }

    Json(ChatCompletionResponse {
        id: completion_id,
        object: "chat.completion",
        created,
        model: model_name,
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: full_text,
            },
            finish_reason: Some("stop".to_string()),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
}

fn parse_request(req: &ChatCompletionRequest) -> (Vec<ChatMessage>, usize, SamplingParams) {
    let messages: Vec<ChatMessage> = req
        .messages
        .iter()
        .map(|m| ChatMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();

    let max_tokens = req.max_tokens;

    let temperature = req.temperature.unwrap_or(0.0);
    let top_p = req.top_p.unwrap_or(if temperature > 0.0 { 0.9 } else { 1.0 });
    let repetition_penalty = req.repetition_penalty.unwrap_or(1.0);

    let sampling = SamplingParams {
        temperature,
        top_p,
        repetition_penalty,
    };

    (messages, max_tokens, sampling)
}
