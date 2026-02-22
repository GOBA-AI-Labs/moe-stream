use axum::extract::State;
use axum::Json;

use crate::engine_handle::EngineHandle;
use crate::types::HealthResponse;

/// GET /health
pub async fn health(State(handle): State<EngineHandle>) -> Json<HealthResponse> {
    let info = handle.model_info();
    Json(HealthResponse {
        status: "ok",
        model: info.model_name.clone(),
        architecture: info.architecture.clone(),
        inference_mode: info.inference_mode.clone(),
        num_layers: info.num_layers,
        num_experts: info.num_experts,
    })
}
