# MoE SSD Streaming Engine

## Overview

A streaming inference engine for large-scale MoE (Mixture of Experts) models on consumer hardware with limited GPU/unified memory (e.g., 24GB).

**Key Innovation**: Existing approaches (AirLLM, etc.) load all experts per layer from disk. This engine **loads the router first, determines top-k selection, then loads only the selected experts from SSD**. Since MoE models are 97% expert weights, loading only the top-k provides massive memory reduction.

### Comparison with Existing Approaches

| Approach | Strategy | MoE Layer Memory | Notes |
|----------|----------|-----------------|-------|
| HuggingFace default | Load entire model to GPU | All experts resident | 30B requires 61GB |
| AirLLM | Layer-by-layer SSD→GPU | All experts loaded | Not MoE-optimized |
| **moe-stream** | Layer-by-layer + top-k selection | **Top-k only** | **94% reduction** |

### Performance (Qwen3-Coder-30B-A3B, Apple M4 Pro 24GB)

| Metric | Value |
|--------|-------|
| Peak memory | ~2.5 GB |
| Layer processing time | 0.2-0.5s |
| Decode speed (all resident + router opt) | **0.45 tok/s** |
| Decode speed (embed + attention resident) | 0.23 tok/s |
| Decode speed (KV-cache only) | 0.10 tok/s |
| Resident memory (all preloaded) | **2.87 GB** (Embed 1.16 + Attn 1.69 + Router 0.02) |

---

## Architecture

```
Input Tokens
    |
    v
[Embedding] <- Resident in memory (~593MB)
    |
    v --- Layer 0-47 (loop) ---
    |
    +- [Input LayerNorm]
    +- [Self-Attention]     <- Resident (Q/K/V/O proj, ~36MB/layer)
    |   +- GQA, RoPE, Scaled Dot-Product Attention
    +- [Residual Connection]
    +- [Post-Attention LayerNorm]
    |
    +- [MoE Router Gate]    <- Resident (~512KB) * Key innovation
    |   +- top-k selection -> identify required expert IDs
    +- [Expert MLP x top-k] <- Only selected experts loaded from SSD
    |   +- SwiGLU: gate_proj, up_proj, down_proj (~9MB each)
    +- [Shared Expert]      <- If present (e.g., Qwen3-Coder-Next)
    +- [Residual Connection]
    |
    +- [Memory release]     <- Expert weights freed after use
    |
    ----------------------------------------
    |
    v
[Final RMSNorm]
    |
    v
[LM Head] <- Resident (~593MB)
    |
    v
Logits Output
```

---

## Weight Residency Strategy

### Adaptive Resident Mode (Default)

```
Resident (GPU/unified memory):
  Embedding + LM Head + Norm:  ~1.16 GB
  Attention weights (all layers): ~1.69 GB
  Router Gates (all layers):      ~0.02 GB
  RoPE cache:                     ~0.001 GB
  -> Total resident:              ~2.87 GB

Per-layer temporary memory:
  1 expert:                       ~0.009 GB (gate+up+down proj)
  -> top-8 loop max:              ~0.01 GB

Memory budget calculation:
  total_mem - max(4GB, 35%) - 0.7GB = budget
  24GB Mac: 24 - 8.4 - 0.7 = 14.9GB -> 2.87GB resident OK
  16GB Mac: 16 - 5.6 - 0.7 = 9.7GB  -> 2.87GB resident OK
   8GB Mac:  8 - 4.0 - 0.7 = 3.3GB  -> 2.87GB resident (tight)

Total peak: ~2.9 GB
```

### Minimal Mode (low-memory fallback)

```
Resident: RoPE cache only (~0.001 GB)
Temporary: Attention + Router + Expert per layer (~0.032 GB)
Embedding/LM Head: load -> compute -> free (~0.6 GB peak)

Total peak: ~2.5 GB
```

### Comparison

```
Qwen3-Coder-30B-A3B BF16:       57 GB -> impossible on 24GB Mac
moe-stream (adaptive resident):  ~2.9 GB -> fits easily (12% of 24GB)
moe-stream (minimal mode):       ~2.5 GB -> works on 8GB Mac
```

---

## I/O Optimization

The engine uses `mmap` + `madvise(MADV_WILLNEED)` for batch prefetching of selected expert slices. This leverages NVMe SSD parallelism to queue multiple reads concurrently.

### Batch Prefetch Flow

```
1. Router computation -> identify top-k experts (CPU, < 1ms)

2. Batch madvise for k experts x 3 weights = 3k slices
   +- madvise(expert_3_gate,  WILLNEED)  -+
   +- madvise(expert_3_up,    WILLNEED)   | NVMe fetches
   +- madvise(expert_3_down,  WILLNEED)   | in parallel
   +- madvise(expert_17_gate, WILLNEED)   |
   +- ...                                 |
   +- madvise(expert_201_down, WILLNEED) -+

3. Sequential expert computation (data already paged in)
```

### I/O Optimization Results

| Approach | Result | Why |
|----------|--------|-----|
| **mmap + madvise batch** | **Fastest** | Maximizes NVMe parallelism |
| CPU routing | +10% | Eliminates Metal->CPU sync barrier |
| F_NOCACHE pread | -13% | Serial I/O loses NVMe parallelism |
| MADV_FREE eviction | +/-0% | Kernel LRU already optimal |
| Expert LRU cache | -20% | Memory pressure on 24GB |
| Async I/O thread | -21% | Contention with NVMe |
| Speculative madvise | -4% | Page cache contention |

**Conclusion**: On 24GB + 48.5GB model, mmap + madvise batch is optimal. SSD bandwidth is the bottleneck.

---

## ModelAdapter Pattern

The engine uses a `ModelAdapter` trait to abstract differences between MoE architectures. The streaming engine itself is model-agnostic.

### Supported Adapters

| Adapter | Architecture | Features | Status |
|---------|-------------|----------|--------|
| Qwen3MoE | `qwen2moe` | Standard MoE, 128 experts, top-8 | Implemented |
| Qwen3Next | `qwen3_next` | DeltaNet hybrid, 512 experts, shared expert | Implemented |
| GPT-OSS | `gpt-oss` | 32 experts, MXFP4 quantization | Implemented |
| DeepSeek | `deepseek_v3` | MLA attention, 671B | Planned |
| Mixtral | `mixtral` | Sparse MoE | Planned |

### Auto-detection

```
Open GGUF -> read metadata -> detect architecture -> select adapter -> initialize engine
```

---

## Expert Prefetch (Planned)

### Concept

Predict which experts the next MoE layer will need and prefetch them from SSD during the current layer's attention computation (which uses resident weights, ~50ms).

```
Layer N:
  [Attention ~50ms]    <- Resident, while computing: prefetch Layer N+1 experts from SSD
  [MoE Expert compute] <- Prefetched experts already in memory -> zero I/O wait

Layer N+1:
  [Attention ~50ms]    <- Simultaneously prefetch Layer N+2 experts
  [MoE Expert compute] <- Zero I/O wait again
```

**Important**: Prefetch does not affect output quality. On prediction miss, experts are loaded normally from SSD. Logits are identical with prefetch ON/OFF.

### Prediction Accuracy vs Speed

| Expert Prediction Accuracy | Estimated tok/s |
|---------------------------|-----------------|
| 0% (all miss = prefetch OFF) | 0.45 |
| 50% | ~0.7 |
| 70% (target) | ~1.0 |
| 90% | ~2.0 |
| 100% (ideal) | ~3-5 |
