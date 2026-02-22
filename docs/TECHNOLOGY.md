# moe-stream: Technical Overview

## What is it?

moe-stream is an inference engine designed to run large AI models on consumer hardware.
It enables running 80B parameter models on a 24GB MacBook.

**Core idea**: MoE (Mixture of Experts) models use only 2-3% of their total parameters to generate each token. The unused 97% can stay on SSD -- only the needed parts are loaded.

```
Traditional: 80B model -> 160GB RAM required -> not feasible for most users
moe-stream:  80B MoE   -> 4GB RAM + SSD      -> runs on a MacBook
```

---

## Technology Summary

| Technology | What it does | Impact |
|-----------|-------------|--------|
| Expert Slicing | Loads only selected experts from SSD | 94% memory reduction |
| mmap + madvise Batch | Leverages NVMe parallelism for expert loading | Maximum SSD bandwidth utilization |
| RAM Resident Mode | Keeps all weights in RAM for smaller models | Zero SSD I/O |
| Dynamic K | Adjusts expert count based on routing entropy | +12.7% speedup |
| CPU/Metal Hybrid | Selects CPU or GPU based on operation size | +10% speedup |
| Hybrid Architecture | Supports DeltaNet (linear) + Attention (quadratic) | 80B O(n) inference |
| Auto Config | Auto-detects model properties from GGUF metadata | Zero configuration |
| GGUF Dequant | On-the-fly F32 conversion for 13+ quantization formats | 1/4 storage |

---

## 1. Expert Slicing -- Precision I/O for MoE

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

### Existing Approaches vs moe-stream

```
AirLLM approach:     Load entire layer -> 512 experts x 0.84MB = 430MB/layer
moe-stream approach: Load top-10 only  -> 10 experts x 0.84MB = 8.4MB/layer

Reduction: 98%
```

### Implementation

Expert weights in a GGUF file are stored as stacked tensors `[n_experts, intermediate, hidden]`.
Given an expert_idx, only that slice is read from the mmap'd file.

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

## 2. mmap + madvise Batch -- Leveraging NVMe Parallelism

### Why mmap?

SSD reads are delegated to the OS page cache.
`madvise(WILLNEED)` tells the OS "this address range will be used soon",
allowing the NVMe SSD to handle page faults **in parallel**.

### Batch Prefetch Flow

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

### I/O Optimization Results

| Approach | Result | Why |
|----------|--------|-----|
| **mmap + madvise batch** | **Fastest** | Maximizes NVMe parallelism |
| CPU routing | +10% | Eliminates Metal-to-CPU sync barrier |
| F_NOCACHE pread | -13% | Serial I/O loses NVMe parallelism |
| MADV_FREE eviction | +/-0% | Kernel LRU already optimal |
| Expert LRU cache | -20% | Memory pressure on 24GB |
| Async I/O thread | -21% | Contention with NVMe |
| Speculative madvise | -4% | Page cache contention |

**Conclusion**: On 24GB + 48.5GB model, mmap + madvise batch is the optimal approach. SSD bandwidth is the bottleneck.

---

## 3. RAM Resident Mode -- Keep Everything in Memory

### Auto-detection

```
Compute total F32-expanded weight size:
  expert_f32 = layers x experts x 3 x intermediate x hidden x 4 bytes

Compare against 75% of system RAM:
  Fits     -> RAM Resident (all weights resident, zero SSD I/O)
  Too large -> SSD Streaming (only experts streamed from SSD)
```

### 3-tier Expert Retrieval

```
When an expert is needed:

1. In resident.experts[layer][expert]?   -> RAM Resident mode, use immediately
   | not found
2. In expert_cache.get(layer, expert)?   -> LRU cache hit
   | not found
3. Load from SSD mmap + dequant          -> SSD Streaming
```

---

## 4. Dynamic K -- Entropy-based Adaptive Expert Selection

### Concept

"High-confidence tokens need fewer experts; uncertain tokens need more experts"

```
"The capital of France is" -> next token is almost certainly "Paris"
  -> low entropy -> K=2 is sufficient -> fast

"The best way to" -> many possible continuations
  -> high entropy -> K=10 needed -> accurate
```

### Algorithm

```
1. Router logits -> softmax -> probability distribution P
2. Entropy H = -sum(p * ln(p))
3. Normalize: H_norm = H / ln(n_experts)   <- 0.0 to 1.0
4. K = round(k_min + H_norm * (k_max - k_min))
```

### Results

- **+12.7% speedup** on 80B model (0.59 -> 0.64+ tok/s)
- Quality: 5/5 test cases PASS (output matches llama.cpp)
- Average K: fixed 10 -> dynamic 7.2 (28% fewer expert computations)

---

## 5. CPU/Metal Hybrid -- Size-based Dispatch

### Why CPU is Faster for MoE Decode

Matrix sizes during single-token generation:

```
Expert MLP:  [1, 2048] x [2048, 768]  <- tiny
Attention:   [1, 2048] x [2048, 2048] <- small

Metal kernel launch: 10-50us x hundreds of ops = several ms
CPU compute time: < 1ms
-> Metal overhead exceeds actual compute time
```

### Dispatch Strategy

```
Metal (GPU):
  +-- Embedding lookup:  [151936, 2048] x token_id -> large
  +-- LM Head:           [2048, 151936] x hidden   -> large

CPU:
  +-- Router gate:       [1, 2048] x [2048, 512]   -> medium, avoids sync
  +-- Expert MLP:        [1, 2048] x [2048, 768]   -> small
  +-- Attention QKV:     [1, 2048] x [2048, 512]   -> small (resident)
  +-- RMSNorm:           element-wise               -> small
  +-- RoPE:              element-wise               -> small
```

**CPU routing is especially important**: Placing gate weights on CPU eliminates one Metal-to-CPU sync barrier (waiting for GPU-to-CPU hidden state transfer). +10% speed improvement.

---

## 6. Hybrid Architecture -- DeltaNet + Attention

### Qwen3-Coder-Next 80B Structure

```
All 48 layers:
  Layer 0:  DeltaNet  (linear attention, O(n))
  Layer 1:  DeltaNet
  Layer 2:  DeltaNet
  Layer 3:  Attention (full attention, O(n^2)) <- every 4th layer
  Layer 4:  DeltaNet
  ...
  Layer 47: Attention

36 DeltaNet + 12 Attention layers
```

### DeltaNet (State Space Model)

- **Linear time complexity O(n)**: proportional to sequence length (vs quadratic for Attention)
- Recurrent computation with sequential state vector updates
- Conv1D captures local patterns
- Gated output controls information flow

### Partial RoPE

```
Of head_dim = 256:
  First 64 dimensions: RoPE (Rotary Position Embedding) applied
  Remaining 192 dimensions: unchanged

-> Balances positional information in the hybrid DeltaNet + Attention design
```

---

## 7. Auto Config -- GGUF Metadata-driven Configuration

### Zero-configuration Auto-detection

Simply opening a GGUF file automatically determines all settings:

```
Extracted from GGUF metadata:
  +-- architecture:         "qwen2moe" / "qwen3_next"
  +-- hidden_size:          2048 / 4096
  +-- num_layers:           24 / 48
  +-- num_experts:          60 / 128 / 512
  +-- num_experts_per_tok:  4 / 8 / 10
  +-- ...

Inferred from model name:
  +-- norm_topk_prob:  Qwen1.5/2 -> false, Qwen3+ -> true
  +-- attention bias:  Qwen1.5 -> yes, Qwen3 -> no

Determined from system environment:
  +-- inference_mode:  system_ram vs F32 expanded size
```

### Why This Matters

A single misconfiguration like `norm_topk_prob` produces garbage output.
By auto-detecting model family differences, users don't need to configure anything.

---

## 8. GGUF Dequantization -- 13+ Quantization Formats

### Supported Formats

```
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K    <- K-quant family
Q4_0, Q4_1, Q5_0, Q5_1            <- Legacy formats
Q8_0, Q8_1                         <- High-precision quants
F16, F32                           <- Unquantized
```

### On-the-fly Conversion

```
On SSD: Q4_K_M (4.5 bits/weight) -> 48.5GB for 80B model
At compute time: Convert to F32 (32 bits/weight) for operations
Conversion: Block-level (256 elements/block) for efficient decoding
```

Storage savings: **~7x** compared to F32

---

## Performance Summary

### Measured on 24GB M4 Pro MacBook

| Model | Parameters | Resident RAM | Speed | Mode |
|-------|-----------|-------------|-------|------|
| Qwen3-Coder-Next 80B | 80B (~3B active per token) | 4GB | ~2.1 tok/s | SSD Streaming |
| Qwen3-30B-A3B | 30B (active 3B) | 4GB | 0.75 tok/s | SSD Streaming |
| GPT-OSS-20B Pruned | 20B | ~10GB | ~55 tok/s | GPU Resident |

### Comparison

```
llama.cpp (Qwen1.5-MoE, fully RAM-resident): 102 tok/s
moe-stream (same model, SSD Streaming):       1.12 tok/s
-> SSD streaming is ~100x slower. But it can run models that don't fit in RAM.

llama.cpp (80B, 24GB Mac): Cannot run (out of memory)
moe-stream (80B, 24GB Mac): ~2.1 tok/s
-> Slow, but it works. Much better than not running at all.
```

### Bottleneck Analysis

```
Per-token breakdown (80B, SSD streaming mode):
  SSD I/O (expert loading):     ~60%  <- primary bottleneck
  Dequantization (Q4 -> F32):   ~25%
  Matrix operations (CPU):      ~10%
  Other (RoPE, Norm, etc.):      ~5%
```

---

## Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Language | Rust | Memory safety + C-level performance + no Python dependency |
| ML Framework | candle-core | Rust-native + Metal support |
| I/O | memmap2 + libc madvise | OS page cache + NVMe parallelism |
| GPU | candle-metal-kernels | Apple Silicon Metal |
| Model Format | GGUF | Industry standard, built-in quantization |
| Python Bindings | PyO3 + maturin | Planned for future release |
