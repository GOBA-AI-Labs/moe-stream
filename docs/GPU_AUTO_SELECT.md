# moe-stream: 全MoE対応 + GPU/CPU自動選択 設計書

## ビジョン

moe-stream は**全てのMoEモデルを最速で推論できるエンジン**。
モデルサイズに関係なく、利用可能なハードウェアリソースを自動判定し、
最適な推論パス (GPU全常駐 / CPU RAM常駐 / SSD streaming) を自動選択する。

llama.cpp が対応できない pruned MoE モデル (variable expert count) も含め、
全ての MoE GGUF を推論可能にする。

## 背景: なぜ moe-stream が必要か

- llama.cpp は pruned MoE (layer ごとに expert 数が異なる) に非対応
- pruning で expert を削ったモデルは moe-stream でしか推論できない
- moe-stream は SSD streaming で 80B MoE を 24GB Mac で動かせる唯一のエンジン
- ただし現状は GPU 推論が不完全 (手動フラグ、MXFP4 Metal 未実装)

## 推論モード (自動選択)

### Mode 1: GPU Resident

**条件**: `has_metal && gguf_file_size < system_ram * 0.60`

全 weights (experts 含む) を Metal GPU に常駐。CPU↔GPU 転送ゼロ。
Apple Silicon unified memory では system_ram = GPU VRAM。

**量子化タイプ別の GPU ロード方法:**

| 量子化 | candle QTensor 対応 | GPU ロード方法 | メモリ比 |
|--------|---------------------|----------------|----------|
| Q4_K, Q4_0, Q5_K, Q6_K, Q8_0 等 | ✅ | QTensor → Metal (native quantized matmul) | 1x |
| MXFP4 (type 39) | ❌ | **カスタム Metal kernel** (llama.cpp 移植) | 1x |
| F16 | ✅ | Tensor → Metal (native) | 1x |
| F32 | ✅ | Tensor → Metal (native) | 1x |

**MXFP4 Metal Kernel**: llama.cpp の `kernel_mul_mv_mxfp4_f32` を移植。
- `.metal` shader ファイルを作成
- candle の Metal backend からカスタム kernel をディスパッチ
- dequant を GPU 上で実行、F32 中間バッファ不要
- Fallback: MXFP4 → F16 dequant → GPU 常駐 (kernel 完成前の暫定パス)

### Mode 2: RAM Resident

**条件**: `f32_estimate < system_ram * 0.75`

全 expert を起動時に F32 dequant → CPU メモリ常駐。
小さいモデル向け。既存の `ram_resident` モード。

### Mode 3: SSD Streaming

**条件**: 上記いずれにも収まらない

Expert weights は mmap + オンデマンド dequant。
Attention/norms/gates のみ CPU 常駐。
`QUANTIZED_MATMUL=1` で Q4/MXFP4 fused matmul (CPU、スレッド並列)。

## 自動選択ロジック

```rust
fn determine_inference_mode(gguf_file_size: u64, system_ram: u64, has_metal: bool) -> InferenceMode {
    if has_metal && gguf_file_size < (system_ram as f64 * 0.60) as u64 {
        return InferenceMode::GpuResident;
    }

    let f32_estimate = estimate_f32_size_from_config(&config);
    if f32_estimate < (system_ram as f64 * 0.75) as u64 {
        return InferenceMode::RamResident;
    }

    InferenceMode::SsdStreaming
}
```

## CLI

```bash
# 自動 (デフォルト) — 推奨
moe-stream generate model.gguf 100 --prompt "Hello"

# GPU 強制
moe-stream generate model.gguf 100 --prompt "Hello" --device gpu

# CPU 強制
moe-stream generate model.gguf 100 --prompt "Hello" --device cpu

# 後方互換
moe-stream generate model.gguf 100 --prompt "Hello" --gpu-compute  # = --device gpu
```

## 実装タスク一覧

### T1: InferenceMode + 自動選択 [config.rs, engine.rs]

1. `config.rs`: `InferenceMode` enum (GpuResident / RamResident / SsdStreaming)
2. `config.rs`: `device_override: Option<DevicePreference>` (Auto / Gpu / Cpu)
3. `engine.rs` `open()`: GGUF file size + Metal + system RAM → mode 自動決定
4. ログ出力: 選択されたモードと理由

### T2: GPU Resident Expert Preloading [engine.rs]

1. `preload_experts_gpu()`: 全 expert を GPU device に load
   - Q4_K 等: `expert_slice_as_qtensor()` → QMatMul → Metal
   - MXFP4: MXFP4 Metal kernel (T4) 経由、or Fallback で F16 dequant → Metal
   - F16/F32: そのまま Metal device に load
2. `preload_attention()`: GPU Resident 時は Metal device に load (現状は CPU)
3. `preload_norms()`: GPU Resident 時は Metal device に load
4. `preload_weights()`: mode に応じて適切な preload を呼び分け

### T3: layer.rs GPU Resident パス [layer.rs]

1. `run_moe()`: GPU Resident 時、expert_cache → GPU tensor → GPU matmul (CPU fallback なし)
2. `run_attention()`: GPU Resident 時、attention weights を GPU で直接使用
3. hidden_states は全レイヤーで GPU 上に保持 (CPU 転送なし)
4. output_data accumulation も GPU 上で実行

### T4: MXFP4 Metal Kernel [新規ファイル]

llama.cpp から移植:
- `ggml-metal.metal:8499-8590` → `moe-stream-core/src/metal/mxfp4.metal`
- `kernel_mul_mv_mxfp4_f32`: MXFP4 × F32 matrix-vector multiply
- E8M0 exponent decode + LUT lookup + simdgroup reduction
- Rust 側: `candle_core::metal` API でカスタム kernel dispatch

**llama.cpp 参照コード:**
- `/Users/to/Documents/productCodes/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` (shader)
- `/Users/to/Documents/productCodes/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.cpp` (dispatch)
- `kvalues_mxfp4_f[16]`, `e8m0_to_fp32()`, `kernel_mul_mv_mxfp4_f32()`

### T5: CLI + generate.rs 更新

1. `--device {auto|gpu|cpu}` arg 追加
2. `--gpu-compute` を `--device gpu` のエイリアスとして維持
3. デフォルトは `auto` (T1 の自動選択)
4. 選択された mode をログ出力

### T6: テスト・検証

1. GPT-OSS-20B GPU Resident: Token IDs が F32 CPU パスと一致
2. Qwen3-80B v7 SSD Streaming: 既存と同一動作
3. GPT-OSS pruned (9.7 GB) GPU Resident: 動作確認
4. 速度ベンチマーク: GPU Resident vs llama.cpp

## 対応する量子化タイプ (全対応)

| Type | ID | moe-stream 対応 | GPU matmul |
|------|----|----------------|------------|
| F32 | 0 | ✅ dequant | ✅ native |
| F16 | 1 | ✅ dequant | ✅ native |
| Q4_0 | 2 | ✅ dequant + QMatMul | ✅ QTensor Metal |
| Q4_1 | 3 | ✅ dequant + QMatMul | ✅ QTensor Metal |
| Q5_0 | 6 | ✅ dequant + QMatMul | ✅ QTensor Metal |
| Q5_1 | 7 | ✅ dequant + QMatMul | ✅ QTensor Metal |
| Q8_0 | 8 | ✅ dequant + QMatMul | ✅ QTensor Metal |
| Q8_1 | 9 | ✅ dequant | ✅ via F32 |
| Q2_K | 10 | ✅ dequant + QMatMul | ✅ QTensor Metal |
| Q3_K | 11 | ✅ dequant + QMatMul | ✅ QTensor Metal |
| Q4_K | 12 | ✅ dequant + QMatMul | ✅ QTensor Metal |
| Q5_K | 13 | ✅ dequant + QMatMul | ✅ QTensor Metal |
| Q6_K | 14 | ✅ dequant + QMatMul | ✅ QTensor Metal |
| Q8_K | 15 | ✅ dequant | ✅ via F32 |
| BF16 | 30 | ✅ dequant | ✅ via F32 |
| MXFP4 | 39 | ✅ dequant + CPU fused | 🔧 Metal kernel (T4) |

## ファイル変更一覧

| ファイル | 変更内容 |
|----------|----------|
| `src/config.rs` | InferenceMode enum, device_override |
| `src/model/engine.rs` | 自動モード選択, preload_experts_gpu(), preload to Metal |
| `src/model/layer.rs` | GPU Resident expert matmul パス |
| `src/metal/mxfp4.metal` | **新規**: MXFP4 Metal compute shader |
| `src/metal/mod.rs` | **新規**: Metal kernel dispatch |
| `examples/generate.rs` | --device arg |

## 期待される性能

| モデル | サイズ | Mode | 現状 | 目標 |
|--------|--------|------|------|------|
| GPT-OSS-20B | 11.7 GB | GPU Resident | 2.3 tok/s | **~55 tok/s** |
| GPT-OSS pruned | 9.7 GB | GPU Resident | N/A | **~55 tok/s** |
| Qwen3-80B v7 | 27.7 GB | SSD Streaming | 2.1 tok/s | 2.1 tok/s |
| Qwen3-80B original | 45 GB | SSD Streaming | 0.64 tok/s | 0.64 tok/s |
