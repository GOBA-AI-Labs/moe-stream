# CLI Reference

## Binaries

moe-stream provides two binaries:

| Binary | Purpose |
|--------|---------|
| `moe-stream` | CLI inference + JSONL server |
| `moe-stream-server` | OpenAI-compatible HTTP server + MCP server |

## moe-stream (CLI)

### Usage

```
moe-stream <gguf_path> [max_tokens] [OPTIONS]
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `<gguf_path>` | Yes | -- | Path to the GGUF model file |
| `[max_tokens]` | No | 20 | Maximum number of tokens to generate |

### Options

#### Device Selection

| Flag | Description |
|------|-------------|
| `--device auto` | Auto-select mode based on model size vs system RAM (default) |
| `--device gpu` | Force GPU Resident mode (all weights on Metal GPU) |
| `--device cpu` | Force CPU mode (SSD Streaming) |

See [GPU_AUTO_SELECT.md](GPU_AUTO_SELECT.md) for details on automatic mode selection.

#### Weight Preloading (SSD Streaming mode)

These flags are relevant when running in SSD Streaming mode. In GPU Resident mode, all weights are on GPU by default.

| Flag | Description |
|------|-------------|
| `--preload-gates` | Keep MoE gate weights resident in memory (~50 MB) |
| `--preload-attn` | Keep attention weights resident in memory (~1.3 GB for 80B) |
| `--preload-dn` | Keep DeltaNet weights resident in memory (~4.9 GB for 80B). **Not recommended on 24GB** |
| `--preload` | Enable all preloading (gates + attention + DeltaNet) |

#### Input

| Flag | Description |
|------|-------------|
| `--prompt "text"` | Text prompt to generate from. Requires a tokenizer. |
| `--tokens 1,2,3` | Provide raw token IDs directly (comma-separated) |
| `--tokenizer path` | Path to `tokenizer.json`. Auto-detected if in same directory as GGUF |
| `--hello` | Use a built-in "Hello" prompt |
| `--chat` | Use a built-in chat-format prompt |
| `--think` | Use a built-in thinking-mode prompt |

#### Output

| Flag | Description |
|------|-------------|
| `--stream` | Stream tokens to stdout as they are generated |

#### Memory Management

| Flag | Description |
|------|-------------|
| `--ram-budget N` | Pin N GB of MoE expert layers in RAM using mlock |
| `--ram-budget auto` | Automatically compute optimal budget (15% of system RAM) |

#### Engine

| Flag | Description |
|------|-------------|
| `--max-layers N` | Limit inference to the first N layers (debugging) |

#### Server Mode

| Flag | Description |
|------|-------------|
| `--server` | Run as a persistent JSONL server (stdin/stdout) |

### Examples

#### One-Shot Generation

```bash
./target/release/moe-stream path/to/model.gguf 100 \
  --prompt "def fibonacci(n):" --stream
```

#### JSONL Server Mode

```bash
./target/release/moe-stream path/to/model.gguf --server
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

---

## moe-stream-server (HTTP + MCP)

### Usage

```
moe-stream-server --model <gguf_path> [OPTIONS]
```

### Options

| Flag | Description |
|------|-------------|
| `--model path` | Path to the GGUF model file (required) |
| `--port N` | HTTP server port (default: 11434) |
| `--host addr` | HTTP server bind address (default: 127.0.0.1) |
| `--tokenizer path` | Path to `tokenizer.json` |
| `--mcp-stdio` | Enable MCP server on stdio transport |
| `--device auto\|gpu\|cpu` | Override inference mode |

### OpenAI-Compatible HTTP API

#### POST /v1/chat/completions

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true,
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

**Request fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | -- | Model name (any value accepted) |
| `messages` | array | -- | Chat messages `[{role, content}]` |
| `stream` | bool | false | SSE streaming response |
| `max_tokens` | int | 200 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature |
| `top_p` | float | 0.9 | Top-p (nucleus) sampling |

**Streaming response** (SSE):
```
data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"index":0}]}
data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop","index":0}]}
data: [DONE]
```

**Non-streaming response** (JSON):
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "choices": [{
    "message": {"role": "assistant", "content": "Hello! How can I help?"},
    "finish_reason": "stop",
    "index": 0
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
}
```

#### GET /v1/models

Returns available models:
```json
{"object": "list", "data": [{"id": "local", "object": "model"}]}
```

#### GET /health

Returns server health:
```json
{"status": "ok"}
```

### MCP Server (AI Agent Integration)

Start in MCP stdio mode:
```bash
./target/release/moe-stream-server --model path/to/model.gguf --mcp-stdio
```

Add to `.claude/mcp.json`:
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

**Available MCP tools:**

| Tool | Input | Output |
|------|-------|--------|
| `generate` | `{prompt, max_tokens?, temperature?}` | `{text, tokens, elapsed_s}` |
| `chat` | `{messages: [{role, content}], max_tokens?}` | `{response, tokens}` |
| `model_info` | `{}` | `{name, architecture, layers, experts, mode}` |
| `tokenize` | `{text}` | `{token_ids, count}` |

### Examples

```bash
# HTTP server
./target/release/moe-stream-server --model model.gguf --port 11434

# HTTP + MCP simultaneous
./target/release/moe-stream-server --model model.gguf --port 11434 --mcp-stdio

# Connect from OpenAI-compatible clients
export OPENAI_BASE_URL=http://localhost:11434/v1
```

---

## Performance Tuning

### By Inference Mode

| Mode | Tuning |
|------|--------|
| GPU Resident | No tuning needed. All weights on GPU. |
| SSD Streaming | Use `--preload-gates --preload-attn` for best performance |
| SSD Streaming + Server | Add `--ram-budget auto` for mlock-pinned expert layers |

### RAM Budget (SSD Streaming only)

Pins MoE expert pages in physical RAM using `mlock`. Pages stay in Q4 format (no dequantization overhead).

| System RAM | Auto Budget | Pinned Layers (80B) |
|------------|-------------|---------------------|
| 24 GB | 3.6 GB | ~6 |
| 32 GB | 4.8 GB | ~8 |
| 48 GB | 7.2 GB | ~13 |
| 64+ GB | 9.6 GB | ~17 |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RUST_LOG` | Log level: `info`, `debug`, `trace` |
| `QUANTIZED_MATMUL` | Set to `1` for Q4 fused matmul (+79% speedup in SSD Streaming) |
