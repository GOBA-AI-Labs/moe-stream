use axum::extract::State;
use axum::Json;

use crate::engine_handle::EngineHandle;
use crate::types::{ModelListResponse, ModelObject};

/// GET /v1/models
pub async fn list_models(State(handle): State<EngineHandle>) -> Json<ModelListResponse> {
    let info = handle.model_info();
    Json(ModelListResponse {
        object: "list",
        data: vec![ModelObject {
            id: info.model_name.clone(),
            object: "model",
            created: 0,
            owned_by: "local".to_string(),
        }],
    })
}
