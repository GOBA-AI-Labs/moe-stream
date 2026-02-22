mod engine_handle;
mod mcp;
mod routes;
mod types;

use axum::Router;
use axum::routing::{get, post};
use clap::Parser;
use rmcp::ServiceExt;
use tower_http::cors::CorsLayer;

use crate::engine_handle::EngineHandle;
use crate::mcp::tools::MoeStreamMcp;
use moe_stream_core::config::DevicePreference;

#[derive(Parser)]
#[command(name = "moe-stream-server")]
#[command(about = "OpenAI-compatible HTTP server for moe-stream inference")]
struct Cli {
    /// Path to GGUF model file
    #[arg(long)]
    model: String,

    /// Server port
    #[arg(long, default_value = "11434")]
    port: u16,

    /// Server host
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Device preference: auto, gpu, cpu
    #[arg(long, default_value = "auto")]
    device: String,

    /// Max sequence length (context window)
    #[arg(long, default_value = "4096")]
    max_seq_len: usize,

    /// Enable MCP server over stdio (can run alongside HTTP)
    #[arg(long)]
    mcp_stdio: bool,
}

#[tokio::main]
async fn main() {
    env_logger::init();
    let cli = Cli::parse();

    let device_preference = match cli.device.as_str() {
        "gpu" => DevicePreference::Gpu,
        "cpu" => DevicePreference::Cpu,
        _ => DevicePreference::Auto,
    };

    eprintln!("=== moe-stream-server ===");
    eprintln!("Model: {}", cli.model);
    eprintln!("Device: {}", cli.device);
    eprintln!("Context: {} tokens", cli.max_seq_len);
    eprintln!();

    // Spawn engine on dedicated thread (blocks until model is loaded)
    let handle = EngineHandle::spawn(&cli.model, device_preference, cli.max_seq_len)
        .expect("Failed to initialize engine");

    let info = handle.model_info();
    eprintln!("Model: {} ({})", info.model_name, info.architecture);
    eprintln!("Mode: {}", info.inference_mode);
    eprintln!("Template: {}", info.chat_template);
    eprintln!();

    if cli.mcp_stdio && cli.port == 11434 {
        // MCP-only mode (no HTTP)
        run_mcp_stdio(handle).await;
    } else if cli.mcp_stdio {
        // Both HTTP and MCP: spawn MCP on a separate task
        let mcp_handle = handle.clone();
        let http_handle = handle;
        let http_task = tokio::spawn(run_http(http_handle, cli.host, cli.port));
        let mcp_task = tokio::spawn(run_mcp_stdio(mcp_handle));

        tokio::select! {
            r = http_task => { if let Err(e) = r { eprintln!("HTTP task error: {}", e); } }
            r = mcp_task => { if let Err(e) = r { eprintln!("MCP task error: {}", e); } }
        }
    } else {
        // HTTP-only mode (default)
        run_http(handle, cli.host, cli.port).await;
    }
}

async fn run_http(handle: EngineHandle, host: String, port: u16) {
    let app = Router::new()
        .route("/v1/chat/completions", post(routes::chat_completions::chat_completions))
        .route("/v1/models", get(routes::models::list_models))
        .route("/health", get(routes::health::health))
        .layer(CorsLayer::permissive())
        .with_state(handle);

    let addr = format!("{}:{}", host, port);
    eprintln!("HTTP listening on http://{}", addr);
    eprintln!("  POST /v1/chat/completions");
    eprintln!("  GET  /v1/models");
    eprintln!("  GET  /health");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind");
    axum::serve(listener, app)
        .await
        .expect("Server error");
}

async fn run_mcp_stdio(handle: EngineHandle) {
    eprintln!("MCP stdio server starting...");

    let transport = rmcp::transport::stdio();
    let service = MoeStreamMcp::new(handle)
        .serve(transport)
        .await
        .expect("MCP server init failed");

    eprintln!("MCP stdio server ready");
    service.waiting().await.expect("MCP server error");
    eprintln!("MCP stdio server stopped");
}
