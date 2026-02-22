# moe-stream

SSD-streaming MoE inference engine for consumer hardware. Run 80B parameter Mixture-of-Experts models on a 24GB Apple Silicon Mac.

## Why moe-stream?

- **Layer-adaptive pruned model support** -- handles models with different expert counts per layer (`experts_per_layer` metadata), which llama.cpp does not currently support
- **SSD streaming for 80B models on 24GB hardware** -- stream models from NVMe with only ~4GB resident in memory
- **Metal GPU + CPU hybrid inference** -- CPU for MoE expert compute (avoids Metal kernel launch overhead), Metal for embedding and LM head
- **Q4 quantized matmul** -- skip dequantization, compute directly on Q4 weights for +79% speedup
- **Server mode** -- persistent JSONL-over-stdin/stdout server for integration with benchmarking and evaluation pipelines

## Supported Models

| Model | Architecture | Params | Speed (24GB M4 Pro) |
|-------|-------------|--------|---------------------|
| Qwen3-Coder-Next 80B | 36 DeltaNet + 12 Attention, 512 experts top-10 + shared | 80B total / 3B active | ~2.1 tok/s (Q4 matmul) |
| Qwen3-30B-A3B | 48 Attention, 128 experts top-8 | 30B total / 3B active | ~55 tok/s (GPU-resident) |
| GPT-OSS-20B | 24 layers, 32 experts top-8, MXFP4 | 20B total | ~55 tok/s (GPU-resident) |

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
# macOS (recommended)
cargo build --release -p moe-stream-core --features accelerate

# macOS with Metal GPU support
cargo build --release -p moe-stream-core --features metal,accelerate

# Linux (CPU-only)
cargo build --release -p moe-stream-core

# Linux with CUDA (experimental)
cargo build --release -p moe-stream-core --features cuda
```

### Generate Text

```bash
# Basic generation (auto-preloads all resident weights)
cargo run --release -p moe-stream-core --example generate -- \
  path/to/model.gguf

# With prompt and streaming output
cargo run --release -p moe-stream-core --example generate -- \
  path/to/model.gguf 100 \
  --prompt "def fibonacci(n):" --stream

# Run as a persistent JSONL server
cargo run --release -p moe-stream-core --example generate -- \
  path/to/model.gguf --server
```

Place a `tokenizer.json` in the same directory as your GGUF file for automatic tokenizer detection, or pass `--tokenizer path/to/tokenizer.json` explicitly.

See [docs/CLI.md](docs/CLI.md) for the full CLI reference.

### Pre-built Binaries

Pre-built macOS and Linux binaries are available on the [Releases](https://github.com/GOBA-AI-Labs/moe-stream/releases) page. No Rust toolchain required.

## Architecture

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

**Why CPU for MoE decode?** During single-token decode, weight matrices are small (e.g., `[1, 2048] x [768, 2048]`). Metal kernel launch overhead (~10-50us per operation, hundreds of ops per step) exceeds the actual compute time. CPU avoids this overhead entirely and reads directly from mmap'd memory.

**Weight residency strategy:**
- **Always resident (~4GB):** embedding table, LM head, MoE gate weights, attention weights
- **Streamed from SSD:** MoE expert weights (~44GB), DeltaNet weights

The engine automatically selects the inference mode based on model size:
- **GpuResident** (model < 80% RAM): All weights in Metal GPU memory, maximum speed
- **GpuHybrid** (80-90% RAM): Attention on GPU, experts streamed from SSD
- **SsdStreaming** (model > 90% RAM): Minimal resident memory, experts from NVMe SSD

The engine uses `madvise(MADV_WILLNEED)` to batch-prefetch all selected expert slices before sequential processing, allowing the NVMe controller to queue parallel reads.

## Performance

Measured on Apple M4 Pro, 24GB unified memory, internal NVMe SSD.

| Configuration | Size | Resident Memory | Decode Speed |
|---------------|------|----------------|--------------|
| GPT-OSS-20B Pruned, GPU-resident | 10.4 GB | ~10 GB | ~55 tok/s |
| 30B-A3B Pruned-80%, GPU-resident | 14.0 GB | ~14 GB | ~55 tok/s |
| 80B Q4_K_M Original, SSD streaming | ~48 GB | ~4 GB | ~0.6 tok/s |
| 80B 50% pruned, Q4 matmul, SSD streaming | 24.4 GB | ~4 GB | ~2.1 tok/s |

## `experts_per_layer` Support

moe-stream supports the `experts_per_layer` GGUF metadata field, enabling inference on layer-adaptive pruned models where each layer retains a different number of experts.

Standard MoE inference engines assume a uniform expert count across all layers. Layer-adaptive pruning produces models where some layers retain all experts (important layers) while others are aggressively pruned. moe-stream reads the per-layer expert count from the GGUF metadata and correctly routes tokens to the available experts in each layer.

## Python Bindings

> PyO3/maturin-based Python bindings (`pip install moe-stream`) are planned for a future release. For now, the Rust CLI and JSONL server mode are recommended. The server protocol makes integration with Python scripts straightforward -- see [docs/CLI.md](docs/CLI.md) for the server API.

## Project Structure

```
moe-stream/
├── moe-stream-core/          # Pure Rust inference engine
│   ├── src/
│   │   ├── lib.rs             # Public API
│   │   ├── config.rs          # Model config + type dispatch
│   │   ├── chat_template.rs   # Chat template handling
│   │   ├── gguf/              # Custom GGUF reader (mmap + expert slicing)
│   │   ├── model/
│   │   │   ├── engine.rs      # Main engine (load, generate, preload)
│   │   │   ├── layer.rs       # Hybrid layer (DeltaNet / Attention / MoE)
│   │   │   ├── deltanet.rs    # DeltaNet SSM forward pass
│   │   │   ├── cache.rs       # Resident weight storage
│   │   │   └── kv_cache.rs    # KV-cache for attention layers
│   │   ├── ops/               # Activation, attention, norm operations
│   │   ├── metal/             # Apple Metal GPU compute (feature-gated)
│   │   └── tokenizer.rs       # Tokenizer wrapper
│   └── examples/
│       └── generate.rs        # CLI generation tool + JSONL server
└── docs/                      # Technical documentation
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
3. Ensure your changes build: `cargo build --release -p moe-stream-core`
4. Run tests: `cargo test -p moe-stream-core`
5. Open a pull request

When reporting bugs, please include your hardware (especially RAM size), OS version, and the model you are using.
