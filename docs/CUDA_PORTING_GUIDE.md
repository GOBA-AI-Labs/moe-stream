# CUDA Porting Guide

## Goal

Port moe-stream to Linux + NVIDIA GPU (CUDA).
Currently macOS + Metal only. Target: benchmark on A100 80GB or similar.

## Current Architecture

```
moe-stream-core/
+-- src/
|   +-- config.rs       # StreamingConfig (device preference, etc.)
|   +-- model/
|   |   +-- engine.rs   # Engine::open() with Device::new_metal(0) -> change needed
|   |   +-- layer.rs    # Attention + MoE forward (compute_device switching)
|   |   +-- deltanet.rs # DeltaNet SSM (all CPU) -> no changes needed
|   |   +-- cache.rs    # Resident weights + Expert LRU
|   |   +-- kv_cache.rs # KV-cache
|   +-- ops/
|   |   +-- activation.rs # SiLU, sigmoid
|   |   +-- attention.rs  # RoPE, scaled dot-product attention
|   |   +-- norm.rs       # RMSNorm
|   +-- gguf/
|   |   +-- reader.rs   # mmap + madvise + F_NOCACHE + variable expert GGUF
|   |   +-- dequant.rs  # Q4_K_M/Q8_0 dequantization (CPU)
|   +-- lib.rs
+-- examples/
|   +-- generate.rs     # CLI + JSONL server mode
+-- Cargo.toml
```

## Changes Required

### 1. Cargo.toml -- Add CUDA Feature (Low effort)

```toml
[features]
default = []
metal = ["candle-core/metal", "candle-nn/metal"]
accelerate = ["candle-core/accelerate"]
cuda = ["candle-core/cuda", "candle-nn/cuda"]
mkl = ["candle-core/mkl"]
```

Build commands:
```bash
# macOS (current)
cargo build --release --features metal,accelerate

# Linux + CUDA
cargo build --release --features cuda

# Linux CPU only (fallback)
cargo build --release --features mkl
# or
cargo build --release  # no features = pure CPU
```

### 2. engine.rs -- Device Initialization (Low effort)

Current:
```rust
let device = Device::new_metal(0).unwrap_or_else(|_| {
    log::warn!("Metal not available, falling back to CPU");
    Device::Cpu
});
```

Proposed:
```rust
let device = {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0).unwrap_or_else(|e| {
            log::warn!("CUDA not available ({}), falling back to CPU", e);
            Device::Cpu
        })
    }
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0).unwrap_or_else(|e| {
            log::warn!("Metal not available ({}), falling back to CPU", e);
            Device::Cpu
        })
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        log::info!("No GPU feature enabled, using CPU");
        Device::Cpu
    }
};
```

### 3. reader.rs -- F_NOCACHE (Medium effort)

`libc::fcntl(fd, libc::F_NOCACHE, 1)` is **macOS-specific**. Does not exist on Linux.

Linux alternative:
```rust
#[cfg(target_os = "macos")]
unsafe { libc::fcntl(nocache_file.as_raw_fd(), libc::F_NOCACHE, 1); }

#[cfg(target_os = "linux")]
unsafe { libc::posix_fadvise(nocache_file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED); }
// Note: POSIX_FADV_DONTNEED evicts existing cache but doesn't prevent future caching
// like F_NOCACHE does. O_DIRECT is the true equivalent but has alignment constraints.
// Impact is minimal when using VRAM-resident mode.
```

### 4. engine.rs -- RAM Detection (Low effort)

macOS uses `sysctl`, Linux uses `/proc/meminfo`:
```rust
#[cfg(not(target_os = "macos"))]
{
    if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
        for line in content.lines() {
            if line.starts_with("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Ok(kb) = parts[1].parse::<u64>() {
                    return Some(kb * 1024);
                }
            }
        }
    }
    None
}
```

### 5. madvise (No changes needed)

`madvise(MADV_WILLNEED)` and `madvise(MADV_FREE)` work on Linux as-is.
`MADV_FREE` requires Linux 4.5+ (Ubuntu 22.04+ is compatible).

### 6. DeltaNet (No changes needed)

All computation in `deltanet.rs` runs on `Device::Cpu`. No Metal/CUDA dependency.

### 7. Attention / MoE Routing (Low-medium effort)

`layer.rs` compute switches between `self.device` (Metal or CUDA) and CPU.
Controlled by the `config.gpu_compute` flag.

Metal-specific workarounds that are compatible with CUDA:
- `rsqrt()` not supported -> `sqrt()?.recip()?` -- works on CUDA too
- 3D x 2D matmul -> flatten workaround -- works on CUDA too
- Non-contiguous tensor matmul -> `.contiguous()` -- also needed on CUDA
- `scatter_add` non-contiguous -> CPU loop -- CUDA has native scatter_add but CPU loop works
- `top_k` -> CPU sort+select -- same on CUDA

**Conclusion: Metal workarounds work as-is on CUDA. Optimization can come later.**

## Effort Summary

| Change | File | Effort | Description |
|--------|------|--------|-------------|
| CUDA feature | Cargo.toml | Low | Add one line |
| Device init | engine.rs | Low | Add cfg branches |
| F_NOCACHE -> Linux | reader.rs | Medium | posix_fadvise or O_DIRECT |
| RAM detection | engine.rs | Low | /proc/meminfo parsing |
| **DeltaNet** | deltanet.rs | **None** | All CPU |
| **Attention/MoE** | layer.rs, ops/ | **None** | candle handles CUDA dispatch |
| **GGUF reader** | reader.rs | **None** | mmap is POSIX |

## A100 80GB Optimization (Optional)

On A100 80GB, the entire model (27.4 GB) fits in VRAM. SSD streaming is unnecessary.

The existing `ram_resident` mode loads all experts to `Device::Cpu`.
For CUDA, with `gpu_compute = true` + `ram_resident = true`, experts would be loaded
to `Device::Cuda(0)` for VRAM residency.

**Caveat**: 27.4 GB Q4_K_M dequantized to F32 expands to ~80-110 GB, which may not fit
in A100 80GB VRAM.

Options:
1. F16 dequant (half the size) -- candle CUDA supports F16 matmul
2. On-demand dequant (CPU RAM resident + CUDA transfer per layer)
3. Only non-MoE weights (embed, lm_head, attention) in VRAM

## Testing

### Phase 1: WSL2 + RTX 3060/3070

```bash
# WSL2 Ubuntu 22.04 + CUDA toolkit
sudo apt install -y build-essential pkg-config libssl-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

git clone https://github.com/GOBA-AI-Labs/moe-stream.git
cd moe-stream

cargo build --release -p moe-stream-core --features cuda --example generate

./target/release/examples/generate models/small-test.gguf 50 --prompt "Hello"
```

Verification:
- [ ] CUDA build succeeds
- [ ] Small model inference runs
- [ ] Output matches macOS version

### Phase 2: Cloud GPU

```bash
git clone https://github.com/GOBA-AI-Labs/moe-stream.git
cd moe-stream

cargo build --release -p moe-stream-core --features cuda --example generate

./target/release/examples/generate path/to/model.gguf 2048 \
  --prompt "def fibonacci(n):" --preload-gates --preload-attn
```
