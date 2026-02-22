# moe-stream

SSD-streaming MoE inference engine for consumer hardware. Run 80B parameter Mixture-of-Experts models on a 24GB Apple Silicon Mac.

## Why moe-stream?

- **Layer-adaptive pruned model support** -- handles models with different expert counts per layer (`experts_per_layer` metadata), which llama.cpp does not currently support
- **3 inference modes** -- GPU Resident, GPU Hybrid, SSD Streaming, auto-selected based on model size vs system RAM
- **OpenAI-compatible HTTP server** -- `POST /v1/chat/completions` (SSE streaming + JSON), connect from any OpenAI client
- **MCP server** -- expose inference as MCP tools for AI agent integration
- **Metal GPU kernels** -- fused MXFP4 matvec, Q5_0/Q8_0 attention, RoPE, RMSNorm on Apple Silicon
- **Q4 quantized matmul** -- skip dequantization, compute directly on Q4 weights for +79% speedup
- **JSONL server mode** -- persistent stdin/stdout server for benchmarking pipelines

## Supported Models

| Model | Architecture | Params | Speed (24GB M4 Pro) |
|-------|-------------|--------|---------------------|
| Qwen3-Coder-Next 80B | 36 DeltaNet + 12 Attention, 512 experts top-10 + shared | 80B total / 3B active | ~2.1 tok/s (SSD Streaming) |
| Qwen3-30B-A3B | 48 Attention, 128 experts top-8 | 30B total / 3B active | ~55 tok/s (GPU Resident) |
| GPT-OSS-20B | 24 layers, 32 experts top-8, MXFP4 | 20B total | ~17 tok/s (GPU Resident) |

Additional model architectures (Llama, Mistral, DeepSeek, etc.) can be added via the `ModelAdapter` trait in `config.rs`.

### PrunedHub Models

moe-stream is the recommended inference engine for [PrunedHub](https://huggingface.co/GOBA-AI-Labs) models, which use layer-adaptive expert pruning:

| Model | Size | MMLU | Notes |
|-------|------|------|-------|
| [PrunedHub-GPT-OSS-20B-28x](https://huggingface.co/goba-ai-labs/PrunedHub-GPT-OSS-20B-28x) | 10.4 GB | 78% | Lossless pruning, GPU-resident on 24GB Mac |
| [PrunedHub-GPT-OSS-20B-27x-Zerobias](https://huggingface.co/goba-ai-labs/PrunedHub-GPT-OSS-20B-27x-Zerobias) | 9.4 GB | 77% | Zerobias cliff recovery |
| [PrunedHub-Qwen3-30B-A3B-JP-80pct](https://huggingface.co/goba-ai-labs/PrunedHub-Qwen3-30B-A3B-JP-80pct) | 14.0 GB | 74% | Japanese-aware pruning |
| [PrunedHub-Qwen3-Coder-Next-50pct](https://huggingface.co/goba-ai-labs/PrunedHub-Qwen3-Coder-Next-50pct) | 24.4 GB | 72% | 80B model, 50% experts kept |

These models have varying expert counts per layer and **require moe-stream** for inference (llama.cpp does not support `experts_per_layer`).

## Quick Start

### Prerequisites

- Rust 1.75+ (install via [rustup](https://rustup.rs/))
- macOS with Apple Silicon (Metal support) or Linux (CPU-only, CUDA planned)
- A GGUF model file

### Build

```bash
# macOS (recommended: Metal GPU + vecLib acceleration)
cargo build --release -p moe-stream-core --bin moe-stream --features metal,accelerate

# Build the HTTP/MCP server
cargo build --release -p moe-stream-server --features metal,accelerate

# Linux (CPU-only)
cargo build --release -p moe-stream-core --bin moe-stream

# Linux with CUDA (experimental)
cargo build --release -p moe-stream-core --bin moe-stream --features cuda
```

### CLI Inference

```bash
# Interactive generation
./target/release/moe-stream path/to/model.gguf 100 \
  --prompt "def fibonacci(n):" --stream

# JSONL server (stdin/stdout)
./target/release/moe-stream path/to/model.gguf --server
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

### MCP Server (AI Agent Integration)

```bash
# Start as MCP server (stdio transport)
./target/release/moe-stream-server --model path/to/model.gguf --mcp-stdio
```

Add to your MCP client configuration (e.g., `.claude/mcp.json`):
```json
{
  "mcpServers": {
    "moe-stream": {
      "command": "./target/release/moe-stream-server",
      "args": ["--model", "path/to/model.gguf", "--mcp-stdio"]
    }
  }
}
```

Place a `tokenizer.json` in the same directory as your GGUF file for automatic tokenizer detection, or pass `--tokenizer path/to/tokenizer.json` explicitly.

See [docs/CLI.md](docs/CLI.md) for the full CLI reference.

### Pre-built Binaries

Pre-built macOS and Linux binaries are available on the [Releases](https://github.com/GOBA-AI-Labs/moe-stream/releases) page. No Rust toolchain required.

## Architecture

### Inference Modes

The engine automatically selects the inference mode based on model size vs system RAM:

| Mode | Condition | Strategy |
|------|-----------|----------|
| **GPU Resident** | GGUF < 80% RAM + Metal | All weights in Metal GPU memory |
| **GPU Hybrid** | GGUF 80-90% RAM | Attention on GPU, experts from SSD |
| **SSD Streaming** | GGUF > 90% RAM | Minimal resident memory, experts from NVMe |

Override with `--device gpu` or `--device cpu`.

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

### GPU Resident Architecture

```
  +--------+    +----------------------------+    +---------+
  |  GGUF  +--->|    Metal GPU Compute       +--->| Output  |
  |  (load |    | (embed, attention, experts,|    | Tokens  |
  |  once) |    |  norms, LM head -- all GPU)|    +---------+
  +--------+    +----------------------------+
```

Custom Metal kernels for MXFP4 (4-bit MX format), Q5_0/Q8_0 quantized attention, fused RoPE, and fused RMSNorm eliminate CPU-GPU transfer overhead.

## Performance

Measured on Apple M4 Pro, 24GB unified memory, internal NVMe SSD.

| Configuration | Size | Mode | Decode Speed |
|---------------|------|------|--------------|
| GPT-OSS-20B Pruned (MXFP4) | 10.4 GB | GPU Resident | ~17 tok/s |
| 30B-A3B Pruned-80% (Q4_K_M) | 14.0 GB | GPU Resident | ~55 tok/s |
| 80B 50% pruned (Q4_K_M) | 24.4 GB | SSD Streaming | ~2.1 tok/s |
| 80B Q4_K_M Original | ~48 GB | SSD Streaming | ~0.6 tok/s |

## `experts_per_layer` Support

moe-stream supports the `experts_per_layer` GGUF metadata field, enabling inference on layer-adaptive pruned models where each layer retains a different number of experts.

Standard MoE inference engines assume a uniform expert count across all layers. Layer-adaptive pruning produces models where some layers retain all experts (important layers) while others are aggressively pruned. moe-stream reads the per-layer expert count from the GGUF metadata and correctly routes tokens to the available experts in each layer.

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
└── docs/                       # Technical documentation
```

## Links

- [GOBA AI Labs](https://goba-ai-labs.github.io) -- project website
- [PrunedHub Models](https://huggingface.co/GOBA-AI-Labs) -- pre-pruned models on HuggingFace
- [docs/CLI.md](docs/CLI.md) -- CLI and server API reference
- [docs/TECHNOLOGY.md](docs/TECHNOLOGY.md) -- detailed technical overview

## License

Dual-licensed under your choice of:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Ensure your changes build: `cargo build --release --features metal,accelerate`
4. Run tests: `cargo test`
5. Open a pull request

When reporting bugs, please include your hardware (especially RAM size), OS version, and the model you are using.
