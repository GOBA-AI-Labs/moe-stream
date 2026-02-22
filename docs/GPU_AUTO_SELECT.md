# GPU Auto-Select: Inference Mode Selection

## Overview

moe-stream automatically selects the optimal inference mode based on GGUF file size, system RAM, and Metal GPU availability. No manual configuration needed.

## Inference Modes

| Mode | Condition | Strategy |
|------|-----------|----------|
| **GPU Resident** | `has_metal && gguf_size < system_ram * 0.80` | All weights in Metal GPU memory |
| **GPU Hybrid** | `has_metal && gguf_size < system_ram * 0.90` | Attention on GPU, experts streamed from SSD |
| **SSD Streaming** | Otherwise | Minimal resident memory, experts from NVMe |

### Mode 1: GPU Resident

All weights (including experts) are loaded into Metal GPU memory at startup. Zero SSD I/O during inference. This is the fastest mode.

**Example**: GPT-OSS-20B Pruned (10.4 GB) on 24GB Mac -> GPU Resident (~17 tok/s)

**Metal GPU kernels used:**

| Quantization | GPU Kernel |
|-------------|------------|
| Q4_K, Q5_K, Q6_K, Q8_0, etc. | candle QMatMul (native Metal) |
| MXFP4 (type 39) | Custom Metal shader (`mxfp4.metal`) |
| Q5_0, Q8_0 (attention) | Custom Metal shader (`quantized_attn.metal`) |
| F16, F32 | Standard Metal matmul |

Custom Metal kernels include fused RoPE and fused RMSNorm to eliminate CPU-GPU transfer overhead.

### Mode 2: GPU Hybrid

Attention weights reside on GPU, MoE expert weights stream from SSD. Balances GPU memory with I/O for models that almost fit in RAM.

**Example**: 80B model at ~21 GB on 24GB Mac -> GPU Hybrid

### Mode 3: SSD Streaming

Embedding and LM head on Metal GPU. All other weights (experts, attention, norms) computed on CPU with mmap'd SSD access. Uses `madvise(WILLNEED)` for NVMe parallel prefetch.

**Example**: 80B Q4_K_M (48.5 GB) on 24GB Mac -> SSD Streaming (~2.1 tok/s)

## Auto-Selection Logic

```rust
fn determine_inference_mode(gguf_file_size: u64, system_ram: u64, has_metal: bool) -> InferenceMode {
    if has_metal && gguf_file_size < (system_ram as f64 * 0.80) as u64 {
        InferenceMode::GpuResident
    } else if has_metal && gguf_file_size < (system_ram as f64 * 0.90) as u64 {
        InferenceMode::GpuHybrid
    } else {
        InferenceMode::SsdStreaming
    }
}
```

## Manual Override

```bash
# Force GPU (all weights on Metal)
moe-stream path/to/model.gguf 100 --device gpu

# Force CPU (SSD Streaming even if model fits in RAM)
moe-stream path/to/model.gguf 100 --device cpu
```

## Examples by Model Size (24GB Mac)

| Model | GGUF Size | Auto Mode | Speed |
|-------|-----------|-----------|-------|
| GPT-OSS-20B Pruned | 10.4 GB | GPU Resident | ~17 tok/s |
| 30B-A3B Pruned-80% | 14.0 GB | GPU Resident | ~55 tok/s |
| 80B 50% Pruned | 24.4 GB | SSD Streaming | ~2.1 tok/s |
| 80B Original | ~48 GB | SSD Streaming | ~0.6 tok/s |

## Supported Quantization Types

| Type | ID | GPU matmul |
|------|----|------------|
| F32 | 0 | Native Metal |
| F16 | 1 | Native Metal |
| Q4_0 / Q4_1 | 2 / 3 | QTensor Metal |
| Q5_0 / Q5_1 | 6 / 7 | Custom Metal kernel / QTensor |
| Q8_0 / Q8_1 | 8 / 9 | Custom Metal kernel / QTensor |
| Q2_K / Q3_K | 10 / 11 | QTensor Metal |
| Q4_K / Q5_K / Q6_K | 12 / 13 / 14 | QTensor Metal |
| Q8_K | 15 | QTensor Metal |
| BF16 | 30 | via F32 |
| MXFP4 | 39 | Custom Metal kernel |
