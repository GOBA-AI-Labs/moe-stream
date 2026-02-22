# moe-stream

SSD-streaming MoE inference engine for consumer hardware. Run 80B parameter Mixture-of-Experts models on a 24GB Apple Silicon Mac.

## Highlights

- **3 inference modes** -- GPU Resident, GPU Hybrid, SSD Streaming, auto-selected based on model size vs system RAM
- **Layer-adaptive pruned model support** -- handles models with different expert counts per layer (`experts_per_layer` metadata)
- **OpenAI-compatible HTTP server** -- `POST /v1/chat/completions` (SSE streaming + JSON)
- **MCP server** -- expose inference as MCP tools for AI agent integration
- **MCP client + Agent runtime** -- connect to external MCP servers, skills/hooks
- **Tauri desktop app** -- native macOS GUI with React frontend
- **Metal GPU kernels** -- fused MXFP4 matvec, Q5_0/Q8_0 attention, RoPE, RMSNorm on Apple Silicon
- **Q4 quantized matmul** -- skip dequantization, compute directly on Q4 weights for +79% speedup
- **JSONL server mode** -- persistent stdin/stdout server for benchmarking pipelines

## Supported Models

| Model | Architecture | Params | Speed (24GB M4 Pro) |
|-------|-------------|--------|---------------------|
| Qwen3-Coder-Next 80B | 36 DeltaNet + 12 Attention, 512 experts top-10 + shared | 80B total / 3B active | ~2.1 tok/s (SSD Streaming) |
| Qwen3-30B-A3B | 48 Attention, 128 experts top-8 | 30B total / 3B active | ~55 tok/s (GPU Resident) |
| GPT-OSS-20B | 24 layers, 32 experts top-8, MXFP4 | 20B total | ~17 tok/s (GPU Resident) |

### PrunedHub Models

| Model | Size | MMLU | Notes |
|-------|------|------|-------|
| [PrunedHub-GPT-OSS-20B-28x](https://huggingface.co/goba-ai-labs/PrunedHub-GPT-OSS-20B-28x) | 10.4 GB | 78% | Lossless pruning, GPU-resident on 24GB Mac |
| [PrunedHub-GPT-OSS-20B-27x-Zerobias](https://huggingface.co/goba-ai-labs/PrunedHub-GPT-OSS-20B-27x-Zerobias) | 9.4 GB | 77% | Zerobias cliff recovery |
| [PrunedHub-Qwen3-30B-A3B-JP-80pct](https://huggingface.co/goba-ai-labs/PrunedHub-Qwen3-30B-A3B-JP-80pct) | 14.0 GB | 74% | Japanese-aware pruning |
| [PrunedHub-Qwen3-Coder-Next-50pct](https://huggingface.co/goba-ai-labs/PrunedHub-Qwen3-Coder-Next-50pct) | 24.4 GB | 72% | 80B model, 50% experts kept |

## Quick Start

### Prerequisites

- Rust 1.75+ (install via [rustup](https://rustup.rs/))
- macOS with Apple Silicon (Metal support) or Linux (CPU-only, CUDA planned)
- A GGUF model file

### Build

```bash
# CLI binary (macOS, recommended)
cargo build --release -p moe-stream-core --bin moe-stream --features metal,accelerate

# HTTP/MCP server
cargo build --release -p moe-stream-server --features metal,accelerate

# Agent CLI (MCP client + skills/hooks)
cargo build --release -p moe-stream-agent --features metal,accelerate

# Desktop app (Tauri + React)
cd moe-stream-desktop-ui && npm install && cd ..
cargo build --release -p moe-stream-desktop --features metal,accelerate

# Linux (CPU-only)
cargo build --release -p moe-stream-core --bin moe-stream
```

### CLI Inference

```bash
./target/release/moe-stream path/to/model.gguf 100 \
  --prompt "def fibonacci(n):" --stream
```

### OpenAI-Compatible Server

```bash
# Start the HTTP server
./target/release/moe-stream-server --model path/to/model.gguf --port 11434

# Test with curl
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"Hello!"}],"stream":true}'

# Connect from any OpenAI-compatible client
export OPENAI_BASE_URL=http://localhost:11434/v1
```

### MCP Server

```bash
./target/release/moe-stream-server --model path/to/model.gguf --mcp-stdio
```

### Agent CLI

```bash
# Interactive agent with MCP tools and skills
./target/release/moe-stream-agent --model path/to/model.gguf

# With MCP server connections
./target/release/moe-stream-agent --model path/to/model.gguf \
  --mcp-config .moe-stream/mcp.json
```

### Desktop App

```bash
cd moe-stream-desktop
cargo tauri dev --features metal,accelerate
```

## Architecture

### Inference Modes

| Mode | Condition | Strategy |
|------|-----------|----------|
| **GPU Resident** | GGUF < 80% RAM + Metal | All weights in Metal GPU memory |
| **GPU Hybrid** | GGUF 80-90% RAM | Attention on GPU, experts from SSD |
| **SSD Streaming** | GGUF > 90% RAM | Minimal resident memory, experts from NVMe |

Override with `--device gpu` or `--device cpu`.

### GPU Resident Architecture

```
  +--------+    +----------------------------+    +---------+
  |  GGUF  +--->|    Metal GPU Compute       +--->| Output  |
  |  (load |    | (embed, attention, experts,|    | Tokens  |
  |  once) |    |  norms, LM head -- all GPU)|    +---------+
  +--------+    +----------------------------+
```

### SSD Streaming Architecture

```
                    +------------------+
                    |   Metal GPU      |
                    |  (embed + LM     |
                    |   head only)     |
                    +--------+---------+
                             |
  +--------+    +------------v-----------+    +---------+
  |  GGUF  +--->|    CPU Compute         +--->| Output  |
  |  mmap  |    | (MoE experts, DeltaNet,|    | Tokens  |
  |  NVMe  |    |  Attention, Norms)     |    +---------+
  +--------+    +------------------------+
```

### Platform Architecture

```
moe-stream/
├── moe-stream-core/        # Pure Rust inference engine (no HTTP/async deps)
├── moe-stream-server/      # OpenAI HTTP + MCP server (axum + tokio)
├── moe-stream-agent/       # MCP client + Agent runtime + CLI
├── moe-stream-desktop/     # Tauri v2 GUI (Engine direct-linked)
├── moe-stream-desktop-ui/  # React + Vite frontend
└── moe-stream-python/      # PyO3 bindings (Engine, generate, chat)
```

## Performance

Measured on Apple M4 Pro, 24GB unified memory, internal NVMe SSD.

| Configuration | Size | Mode | Decode Speed |
|---------------|------|------|--------------|
| GPT-OSS-20B Pruned (MXFP4) | 10.4 GB | GPU Resident | ~20 tok/s |
| 30B-A3B Pruned-80% (Q4_K_M) | 14.0 GB | GPU Resident | ~55 tok/s |
| 80B 50% pruned (Q4_K_M) | 24.4 GB | SSD Streaming | ~2.1 tok/s |
| 80B Q4_K_M Original | ~48 GB | SSD Streaming | ~0.6 tok/s |

## Project Structure

```
moe-stream/
├── moe-stream-core/           # Pure Rust inference engine
│   ├── src/
│   │   ├── lib.rs              # Public API
│   │   ├── config.rs           # Model config + architecture dispatch
│   │   ├── chat_template.rs    # Chat template handling (8 formats)
│   │   ├── gguf/               # Custom GGUF reader (mmap + expert slicing)
│   │   ├── model/
│   │   │   ├── engine.rs       # Main engine (load, generate, preload)
│   │   │   ├── layer.rs        # Hybrid layer (DeltaNet / Attention / MoE)
│   │   │   ├── deltanet.rs     # DeltaNet SSM forward pass
│   │   │   ├── cache.rs        # Resident weight storage
│   │   │   └── kv_cache.rs     # KV-cache for attention layers
│   │   ├── ops/                # Activation, attention, norm operations
│   │   ├── metal/              # Apple Metal GPU compute (feature-gated)
│   │   │   ├── mod.rs          # MXFP4, Q5_0/Q8_0, RoPE, RMSNorm kernels
│   │   │   ├── mxfp4.metal     # MXFP4 4-bit matvec shader
│   │   │   ├── fused_ops.metal # Fused RoPE + RMSNorm shader
│   │   │   └── quantized_attn.metal  # Q5_0/Q8_0 attention shader
│   │   └── tokenizer.rs        # Tokenizer wrapper
│   └── src/bin/
│       └── moe-stream.rs       # CLI binary + JSONL server
├── moe-stream-server/          # OpenAI-compatible HTTP + MCP server
│   └── src/
│       ├── main.rs             # CLI entry (--model, --port, --mcp-stdio)
│       ├── engine_handle.rs    # Engine on dedicated thread, channel bridge
│       ├── types.rs            # OpenAI-compatible request/response types
│       ├── routes/             # HTTP endpoints (chat, models, health)
│       └── mcp/                # MCP server (stdio transport)
├── moe-stream-agent/           # MCP client + Agent runtime
│   └── src/
│       ├── main.rs             # Agent CLI entry
│       ├── mcp_client.rs       # MCP server process management
│       ├── tool_registry.rs    # Tool aggregation (built-in + MCP)
│       ├── tool_call_parser.rs # Model output <tool_call> extraction
│       ├── skill_loader.rs     # .moe-stream/skills/*.md parsing
│       ├── agent_def.rs        # .moe-stream/agents/*.md parsing
│       ├── runtime.rs          # Agent runtime (generate + tool loop)
│       └── cli/                # Interactive REPL + commands
├── moe-stream-desktop/         # Tauri v2 native desktop app
│   └── src/
│       ├── lib.rs              # Tauri commands (generate, model_info, agents)
│       └── engine_handle.rs    # Engine thread bridge for Tauri
├── moe-stream-desktop-ui/      # React + Vite frontend
│   └── src/
│       ├── App.tsx             # Main app component
│       ├── components/         # Chat, ModelSelector, ToolPanel, etc.
│       └── hooks/              # useEngine, useChat, etc.
├── moe-stream-python/          # PyO3 bindings (Engine, generate, chat)
├── docs/                       # Technical documentation
└── scripts/                    # Benchmark scripts
```

## Key Design: EngineHandle

Engine is `&mut self` + non-Send (Metal handles), so it runs on a dedicated OS thread with channel-based communication:

```rust
enum EngineCommand {
    ChatCompletion { messages, params, token_tx, cancel_rx },
    GetModelInfo { reply: oneshot::Sender<ModelInfo> },
}
```

Server, Agent, and Desktop all share this pattern via `EngineHandle`.

## Links

- [GOBA AI Labs](https://goba-ai-labs.github.io) -- project website
- [PrunedHub Models](https://huggingface.co/GOBA-AI-Labs) -- pre-pruned models on HuggingFace
- [docs/CLI.md](docs/CLI.md) -- CLI and server API reference
- [docs/TECHNOLOGY.md](docs/TECHNOLOGY.md) -- detailed technical overview
- [docs/GPU_AUTO_SELECT.md](docs/GPU_AUTO_SELECT.md) -- inference mode auto-selection

## CUDA (Linux)

CUDA support is available as a feature flag. It uses candle's CUDA backend -- no custom CUDA kernels.

### Requirements

- Linux x86_64
- CUDA Toolkit 12.0+ (`nvcc` on PATH)
- NVIDIA driver 525+
- cuDNN 8+ (optional, for attention acceleration)

### Build

```bash
# Build with CUDA support
cargo build --release -p moe-stream-core --bin moe-stream --features cuda
cargo build --release -p moe-stream-server --features cuda

# Check CUDA detection
./target/release/moe-stream --help
```

### Inference Mode Selection

On CUDA, the inference mode auto-detection uses **GPU VRAM** (not system RAM):

| GGUF Size vs VRAM | Mode |
|-------------------|------|
| < 80% VRAM | GPU Resident (all weights on CUDA) |
| 80-90% VRAM | GPU Hybrid (attention on GPU, experts on CPU/SSD) |
| > 90% VRAM | SSD Streaming (CPU + SSD) |

Override with `--device gpu` (force GPU Resident) or `--device cpu` (force CPU).

### Notes

- CUDA + Metal cannot be enabled simultaneously (compile-time exclusive)
- GPU Resident mode requires the full model to fit in VRAM
- Performance depends on GPU model; A100/H100 expected >100 tok/s for 20B models

## License

Dual-licensed under your choice of:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)
