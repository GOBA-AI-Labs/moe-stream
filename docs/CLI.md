# CLI Reference

The `generate` example provides both one-shot text generation and a persistent JSONL server mode.

## Usage

```
generate <gguf_path> [max_tokens] [OPTIONS]
```

## Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `<gguf_path>` | Yes | -- | Path to the GGUF model file |
| `[max_tokens]` | No | 20 | Maximum number of tokens to generate (one-shot mode only) |

## Options

### Weight Preloading

| Flag | Description |
|------|-------------|
| `--preload-gates` | Keep MoE gate weights resident in memory (~50 MB). Speeds up routing. |
| `--preload-attn` | Keep attention weights resident in memory (~1.3 GB for 80B). Reduces SSD reads for attention layers. |
| `--preload-dn` | Keep DeltaNet weights resident in memory (~4.9 GB for 80B). **Not recommended on 24GB systems** -- causes memory pressure that degrades MoE expert I/O. |
| `--preload` | Enable all preloading (gates + attention + DeltaNet). Only use on systems with sufficient RAM. |

### Input

| Flag | Description |
|------|-------------|
| `--prompt "text"` | Text prompt to generate from. Requires a tokenizer. |
| `--tokens 1,2,3` | Provide raw token IDs directly (comma-separated). |
| `--tokenizer path` | Path to `tokenizer.json`. Auto-detected if present in the same directory as the GGUF file. |
| `--hello` | Use a built-in "Hello" prompt (token 9707). |
| `--chat` | Use a built-in chat-format prompt. |
| `--think` | Use a built-in thinking-mode prompt. |
| `--comment` | Use a built-in code-comment prompt. |

### Output

| Flag | Description |
|------|-------------|
| `--stream` | Stream tokens to stdout as they are generated (one-shot mode). Requires a tokenizer. |

### Memory Management

| Flag | Description |
|------|-------------|
| `--ram-budget N` | Pin N GB of MoE expert layers in RAM using mlock (Q4 format, no dequantization). Pages are guaranteed to stay in physical RAM. |
| `--ram-budget auto` | Automatically compute optimal budget (15% of system RAM). On 24 GB → ~3.9 GB (6/48 layers). |

### Engine

| Flag | Description |
|------|-------------|
| `--max-layers N` | Limit inference to the first N layers. Useful for debugging. |

### Server Mode

| Flag | Description |
|------|-------------|
| `--server` | Run as a persistent JSONL server (stdin/stdout). Requires a tokenizer. |

## Examples

### One-Shot Generation (80B)

```bash
cargo run --release -p moe-stream-core --example generate -- \
  models/Qwen3-Coder-Next-Q4_K_M-official/Qwen3-Coder-Next-Q4_K_M.gguf 100 \
  --preload-gates --preload-attn \
  --prompt "def fibonacci(n):" --stream
```

### One-Shot Generation (30B)

```bash
cargo run --release -p moe-stream-core --example generate -- \
  models/Qwen3-Coder-30B-A3B-Q4_K_M.gguf 50 \
  --preload-gates --preload-attn \
  --prompt "Write a Python function to sort a list." --stream
```

### Server Mode

Start the server:

```bash
cargo run --release -p moe-stream-core --example generate -- \
  models/Qwen3-Coder-Next-Q4_K_M-official/Qwen3-Coder-Next-Q4_K_M.gguf \
  --server --preload-gates --preload-attn
```

The server emits `{"ready": true}` on stdout when initialization is complete.

Send requests as JSONL on stdin:

```json
{"prompt": "def fibonacci(n):", "max_tokens": 100}
```

Responses stream as JSONL on stdout:

```json
{"token": "\n"}
{"token": "    "}
{"token": "if"}
...
{"done": true, "tokens": 42, "elapsed": 65.8}
```

If a request fails, an error response is returned:

```json
{"error": "Invalid JSON: ..."}
```

The server keeps the engine and preloaded weights in memory between requests. Send new requests on the same stdin connection for fast repeated generation.

## Performance Tuning

### Recommended Flags by System RAM

| System RAM | Recommended Flags |
|------------|-------------------|
| 24 GB | `--preload-gates --preload-attn` (default) |
| 24 GB + server | `--server --ram-budget auto` |
| 48+ GB | `--preload` (all weights resident) |

### Tips

- **Always use `--preload-gates`** -- gate weights are small (~50 MB) and eliminate SSD reads for MoE routing.
- **Use `--preload-attn` on 24GB+** -- attention weights add ~1.3 GB but avoid streaming them from SSD every step.
- **Avoid `--preload-dn` on 24GB** -- DeltaNet weights (~4.9 GB for 80B) push total resident memory too high, causing page cache pressure that slows down MoE expert streaming.
- **Use `--stream`** for interactive use -- see output as it is generated rather than waiting for completion.
- **Use `--server` for integration** -- avoids model reload overhead between requests.

### Hybrid RAM/SSD Mode (--ram-budget)

Pins a portion of MoE expert tensor pages in physical RAM using `mlock`. Experts stay in Q4 quantized format (no dequantization overhead, no extra memory). The OS cannot evict pinned pages, guaranteeing zero page-fault I/O for those layers.

**When to use:**
- **Server mode**: One-time mlock cost at startup, guaranteed performance for all subsequent requests.
- **Memory pressure**: When other applications compete for RAM, mlock'd pages are eviction-proof.
- **Slower storage**: SATA SSDs or HDDs benefit more than NVMe.

**When NOT to use:**
- **Single-shot generation on NVMe**: macOS page cache warms up within 3-5 tokens. On NVMe SSDs, the page cache alone is sufficient and mlock adds startup overhead without measurable benefit.

**Budget guidelines:**

| System RAM | Auto Budget | Pinned Layers (80B) | Page Cache Left |
|------------|-------------|---------------------|-----------------|
| 16 GB | 2.4 GB | ~4 | ~4.3 GB |
| 24 GB | 3.6 GB | ~6 | ~10.9 GB |
| 32 GB | 4.8 GB | ~8 | ~17.9 GB |
| 48 GB | 7.2 GB | ~13 | ~31.5 GB |
| 64+ GB | 9.6 GB | ~17 | ~45 GB |

**How it works:**
1. At startup, the engine calls `mlock()` on mmap'd GGUF pages for selected layers
2. Pages are forced into physical RAM (page-faulted from SSD if needed)
3. During inference, pinned layers read from RAM (zero I/O), remaining layers stream from SSD
4. No dequantization at pin time -- Q4 format preserved, dequant happens on-the-fly during matmul (same as SSD path)

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RUST_LOG` | Set log level (e.g., `RUST_LOG=info` or `RUST_LOG=debug`). Uses `env_logger`. |
