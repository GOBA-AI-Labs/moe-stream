# moe-stream: Technical Overview

## What is it?

moe-stream is an inference engine designed to run large MoE (Mixture of Experts) AI models on consumer hardware. It enables running 80B parameter models on a 24GB MacBook.

**Core idea**: MoE models use only 2-3% of their total parameters per token. moe-stream exploits this sparsity with three inference modes: GPU Resident (all weights on Metal GPU), GPU Hybrid, and SSD Streaming (expert weights loaded on-demand from NVMe).

```
Traditional: 80B model -> 160GB RAM required -> not feasible for most users
moe-stream:  80B MoE   -> GPU Resident (if fits) or SSD Streaming -> runs on a MacBook
```

---

## Technology Summary

| Technology | What it does | Impact |
|-----------|-------------|--------|
| GPU Resident Mode | All weights on Metal GPU for small models | ~17-55 tok/s |
| SSD Streaming | Loads only selected experts from SSD | 80B on 24GB Mac |
| Custom Metal Kernels | MXFP4 matvec, Q5_0/Q8_0 attention, fused RoPE/RMSNorm | No CPU-GPU transfer overhead |
| mmap + madvise Batch | Leverages NVMe parallelism for expert loading | Maximum SSD bandwidth |
| Dynamic K | Adjusts expert count based on routing entropy | +12.7% speedup |
| Q4 Quantized MatMul | Computes directly on Q4 weights (skip dequant) | +79% speedup |
| Hybrid Architecture | DeltaNet (linear) + Attention (quadratic) | 80B O(n) inference |
| experts_per_layer | Per-layer variable expert counts | Layer-adaptive pruned model support |
| Auto Config | Auto-detects model properties from GGUF | Zero configuration |

---

## 1. Inference Modes

### GPU Resident (< 80% system RAM)

All weights loaded into Metal GPU memory at startup. Zero SSD I/O during inference.

```
  +--------+    +----------------------------+    +---------+
  |  GGUF  +--->|    Metal GPU Compute       +--->| Output  |
  |  (load |    | (embed, attention, experts,|    | Tokens  |
  |  once) |    |  norms, LM head -- all GPU)|    +---------+
  +--------+    +----------------------------+
```

Custom Metal kernels handle all quantization types natively on GPU:
- **MXFP4**: Custom `mxfp4.metal` shader (4-bit MX format matvec)
- **Q5_0/Q8_0**: Custom `quantized_attn.metal` shader for attention
- **Q4_K, Q6_K, etc.**: candle QMatMul (native Metal)
- **Fused ops**: RoPE + RMSNorm in single kernel dispatch (`fused_ops.metal`)

Multi-row batched wrappers handle prefill (seq_len > 1) by dispatching per-row Metal kernel calls within a single command buffer.

### GPU Hybrid (80-90% system RAM)

Attention weights reside on GPU. MoE expert weights stream from SSD on demand.

### SSD Streaming (> 90% system RAM)

Embedding and LM head on Metal GPU. Expert weights read from mmap'd GGUF via NVMe.

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

---

## 2. Expert Slicing -- Precision I/O for MoE

### MoE Model Structure

```
80B MoE Model (Qwen3-Coder-Next)
+-- Embedding:     600MB  (resident)
+-- LM Head:       600MB  (resident)
+-- Attention:    1.7GB   (resident)
+-- Router Gates:   50MB  (resident)
+-- Experts:     45.5GB   <- 94% of total
    +-- 48 layers x 512 experts x 3 weights (gate/up/down)
```

**Only top-10/512 experts = 2% of total are used per token.**

### Implementation

Expert weights in GGUF are stored as stacked tensors `[n_experts, intermediate, hidden]`. Given an expert_idx, only that slice is read from the mmap'd file.

```
GGUF file (48.5GB, mmap'd)
|
+- blk.0.ffn_gate_exps.weight [512, 2048, 4096]
|   +- expert 0:   offset 0x...000, 0.84MB
|   +- expert 1:   offset 0x...340, 0.84MB    <- only this is read
|   +- ...
|   +- expert 511: offset 0x...FFF, 0.84MB
```

---

## 3. mmap + madvise Batch -- NVMe Parallelism

`madvise(WILLNEED)` tells the OS to prefetch address ranges, allowing the NVMe controller to handle page faults in parallel.

```
1. Router computation -> identify top-10 experts (CPU, < 1ms)

2. Batch madvise for 10 experts x 3 weights = 30 slices
   +- madvise(expert_3_gate,  WILLNEED)  -+
   +- madvise(expert_3_up,    WILLNEED)   | NVMe fetches
   +- madvise(expert_3_down,  WILLNEED)   | in parallel
   +- madvise(expert_17_gate, WILLNEED)   |
   +- ...                                 |
   +- madvise(expert_201_down, WILLNEED) -+

3. Process experts sequentially (already paged in)
```

---

## 4. Custom Metal GPU Kernels

### MXFP4 Matvec (`mxfp4.metal`)

4-bit MX format (Microscaling) matrix-vector multiply. Ported from llama.cpp's `kernel_mul_mv_mxfp4_f32`. Handles E8M0 exponent decode + LUT lookup + simdgroup reduction entirely on GPU.

### Quantized Attention (`quantized_attn.metal`)

Q5_0 and Q8_0 quantized attention matvec. Eliminates CPU dequantization for attention Q/K/V/O projections in GPU Resident mode.

### Fused Operations (`fused_ops.metal`)

RoPE (Rotary Position Embedding) and RMSNorm in single kernel dispatch, avoiding multiple GPU-CPU round-trips.

### Batched Prefill

Metal matvec kernels process single rows. For prefill (seq_len > 1), batched wrapper functions loop over rows and dispatch all operations within a single Metal command buffer, leveraging GPU automatic batching.

---

## 5. Dynamic K -- Entropy-based Expert Selection

"High-confidence tokens need fewer experts; uncertain tokens need more"

```
"The capital of France is" -> low entropy -> K=2 sufficient -> fast
"The best way to"          -> high entropy -> K=10 needed   -> accurate
```

Algorithm:
```
1. Router logits -> softmax -> P
2. Entropy H = -sum(p * ln(p))
3. Normalize: H_norm = H / ln(n_experts)
4. K = round(k_min + H_norm * (k_max - k_min))
```

Result: **+12.7% speedup**, average K from 10 to 7.2 (28% fewer expert computations).

---

## 6. Q4 Quantized MatMul

Skip dequantization entirely. Compute matrix-vector products directly on Q4-quantized weights using integer arithmetic.

Set `QUANTIZED_MATMUL=1` to enable. Result: **+79% speedup** in SSD Streaming mode (1.16 -> 2.07 tok/s). Greedy output is bit-identical to dequantized path.

---

## 7. `experts_per_layer` -- Layer-Adaptive Pruned Models

Standard MoE engines assume uniform expert count across all layers. moe-stream reads the `experts_per_layer` GGUF metadata field to support models where each layer retains a different number of experts.

This enables inference on layer-adaptive pruned models (e.g., PrunedHub models) where important layers keep all experts while others are aggressively pruned.

---

## 8. Hybrid Architecture -- DeltaNet + Attention

### Qwen3-Coder-Next 80B Structure

```
48 layers:
  Layer 0:  DeltaNet  (linear attention, O(n))
  Layer 1:  DeltaNet
  Layer 2:  DeltaNet
  Layer 3:  Attention (full attention, O(n^2))  <- every 4th layer
  ...
  Layer 47: Attention

36 DeltaNet + 12 Attention layers
```

DeltaNet layers use linear-time recurrent computation with sequential state vector updates, Conv1D for local patterns, and gated output.

---

## 9. Auto Config -- GGUF Metadata-driven

Opening a GGUF file automatically configures all settings:

```
From GGUF metadata:
  architecture, hidden_size, num_layers, num_experts,
  num_experts_per_tok, experts_per_layer, ...

Inferred from model family:
  norm_topk_prob, attention bias, chat template format

From system environment:
  inference_mode (GPU Resident / GPU Hybrid / SSD Streaming)
```

---

## Performance Summary

Measured on Apple M4 Pro, 24GB unified memory, internal NVMe SSD.

| Model | Size | Mode | Speed |
|-------|------|------|-------|
| GPT-OSS-20B Pruned (MXFP4) | 10.4 GB | GPU Resident | ~17 tok/s |
| 30B-A3B Pruned-80% (Q4_K_M) | 14.0 GB | GPU Resident | ~55 tok/s |
| 80B 50% pruned (Q4_K_M) | 24.4 GB | SSD Streaming | ~2.1 tok/s |
| 80B Q4_K_M Original | ~48 GB | SSD Streaming | ~0.6 tok/s |

### Bottleneck Analysis (SSD Streaming)

```
Per-token breakdown (80B):
  SSD I/O (expert loading):     ~60%  <- primary bottleneck
  Dequantization (Q4 -> F32):   ~25%  (eliminated with QUANTIZED_MATMUL=1)
  Matrix operations (CPU):      ~10%
  Other (RoPE, Norm, etc.):      ~5%
```

---

## OpenAI-Compatible Server

`moe-stream-server` provides an HTTP API compatible with OpenAI clients:

- `POST /v1/chat/completions` -- SSE streaming and non-streaming JSON
- `GET /v1/models` -- model listing
- `GET /health` -- server health check

The engine runs on a dedicated OS thread (non-Send due to Metal handles) with channel-based communication to async HTTP handlers.

See [CLI.md](CLI.md) for the full server API reference.

---

## MCP Server

`moe-stream-server --mcp-stdio` exposes inference as MCP tools (stdio transport) for AI agent integration. Tools: `generate`, `chat`, `model_info`, `tokenize`.

---

## Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Language | Rust | Memory safety + C-level performance |
| ML Framework | candle-core | Rust-native + Metal support |
| GPU | Custom Metal shaders | MXFP4, Q5_0/Q8_0, fused RoPE/RMSNorm |
| I/O | memmap2 + libc madvise | OS page cache + NVMe parallelism |
| HTTP | axum + tokio | Async HTTP server |
| MCP | rust-mcp-server | AI agent tool protocol |
| Model Format | GGUF | Industry standard, built-in quantization |
