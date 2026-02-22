#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Fused RMSNorm kernel
// =============================================================================
//
// Computes: output[row, i] = input[row, i] * rsqrt(mean(input[row]^2) + eps) * weight[i]
// in a single kernel dispatch, replacing 7 candle tensor ops.
//
// Uses threadgroup reduction for the variance computation.
// Each threadgroup processes one row (independent normalization).
// Dispatch: threadgroups=(num_rows, 1, 1), threads_per_group=(256, 1, 1).

struct RmsNormArgs {
    int hidden_dim;
    float eps;
};

kernel void rms_norm_kernel(
    device const float* input    [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    device float*       output   [[buffer(2)]],
    constant RmsNormArgs& args   [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint row_idx [[threadgroup_position_in_grid]]
) {
    const int hidden_dim = args.hidden_dim;
    const float eps = args.eps;
    const int row_offset = (int)row_idx * hidden_dim;

    // Phase 1: Compute sum of squares (parallel reduction)
    threadgroup float shared_sum[256];
    float local_sum = 0.0f;
    for (int i = (int)tid; i < hidden_dim; i += (int)tg_size) {
        float val = input[row_offset + i];
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Binary tree reduction
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase 2: Normalize and apply weight
    float inv_rms = rsqrt(shared_sum[0] / float(hidden_dim) + eps);
    for (int i = (int)tid; i < hidden_dim; i += (int)tg_size) {
        output[row_offset + i] = input[row_offset + i] * inv_rms * weight[i];
    }
}

// =============================================================================
// Fused RoPE kernel (half-rotate convention)
// =============================================================================
//
// Applies rotary position embedding to a tensor:
//   x1 = x[..., :half_dim]
//   x2 = x[..., half_dim:]
//   output[..., :half_dim]     = x1 * cos - x2 * sin
//   output[..., half_dim:]     = x2 * cos + x1 * sin
//
// Input shape:  [num_heads, head_dim] (flattened from [batch, seq, heads, dim])
// Cos/Sin shape: [half_dim] (already sliced to current position)
// Output shape: [num_heads, head_dim]
//
// Each thread handles one (x1, x2) pair across one head.

struct RopeArgs {
    int num_heads;
    int head_dim;
};

kernel void rope_kernel(
    device const float* input     [[buffer(0)]],
    device const float* cos_vals  [[buffer(1)]],
    device const float* sin_vals  [[buffer(2)]],
    device float*       output    [[buffer(3)]],
    constant RopeArgs& args       [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const int head_dim = args.head_dim;
    const int half_dim = head_dim / 2;
    const int total_pairs = args.num_heads * half_dim;

    if ((int)gid >= total_pairs) return;

    int head = (int)gid / half_dim;
    int pair_idx = (int)gid % half_dim;

    int base = head * head_dim;
    float x1 = input[base + pair_idx];
    float x2 = input[base + half_dim + pair_idx];
    float c = cos_vals[pair_idx];
    float s = sin_vals[pair_idx];

    output[base + pair_idx]            = x1 * c - x2 * s;
    output[base + half_dim + pair_idx] = x2 * c + x1 * s;
}

// =============================================================================
// GPU-side Softmax + Top-K routing kernel
// =============================================================================
//
// Replaces the CPU routing path (to_vec1 GPU sync + CPU softmax + top-k).
// Eliminates 1 GPU↔CPU sync per MoE layer (24 syncs/token → 0).
//
// Two modes:
//   softmax_weight=1 (GPT-OSS): select top-K by raw logits, then softmax on selected
//   softmax_weight=0 (Qwen):    softmax over ALL experts, then select top-K
//
// Also performs offset lookup: for each selected expert, looks up byte offsets
// from pre-uploaded offset tables and writes them to output buffers for the
// fused MoE kernels to consume directly (no CPU round-trip).
//
// Dispatch: threadgroups=(1,1,1), threads_per_group=(256,1,1).
// Single threadgroup processes all experts (n_experts <= 512).

struct SoftmaxTopkArgs {
    int n_experts;       // total number of experts (e.g. 28)
    int top_k;           // number to select (e.g. 4)
    int softmax_weight;  // 1 = GPT-OSS mode, 0 = standard
    int norm_topk_prob;  // 1 = renormalize selected to sum=1
};

kernel void softmax_topk_kernel(
    device const float*    gate_logits     [[buffer(0)]],  // [n_experts]
    device const float*    dummy_mask      [[buffer(1)]],  // [n_experts] (0 or -inf)
    device int32_t*        out_indices     [[buffer(2)]],  // [top_k] output
    device float*          out_weights     [[buffer(3)]],  // [top_k] output
    device const uint64_t* all_gate_offsets [[buffer(4)]], // [n_experts] offset table
    device const uint64_t* all_up_offsets  [[buffer(5)]],  // [n_experts] offset table
    device const uint64_t* all_down_offsets [[buffer(6)]], // [n_experts] offset table
    device uint64_t*       out_gate_offsets [[buffer(7)]], // [top_k] output
    device uint64_t*       out_up_offsets  [[buffer(8)]],  // [top_k] output
    device uint64_t*       out_down_offsets [[buffer(9)]], // [top_k] output
    constant SoftmaxTopkArgs& args         [[buffer(10)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const int ne = args.n_experts;
    const int K = args.top_k;

    // Load logits + apply dummy mask into shared memory
    threadgroup float logits[512];
    for (int i = (int)tid; i < ne; i += (int)tg_size) {
        logits[i] = gate_logits[i] + dummy_mask[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Top-K selection using shared memory
    // Mark selected experts with -inf to exclude from subsequent passes
    threadgroup int   selected_idx[16];    // K <= 16
    threadgroup float selected_logit[16];

    if (args.softmax_weight) {
        // === GPT-OSS mode: select top-K by raw logits first ===
        threadgroup float max_vals[256];
        threadgroup int   max_ids[256];

        for (int k = 0; k < K; k++) {
            // Each thread finds max among its assigned experts
            float local_max = -INFINITY;
            int   local_id  = -1;
            for (int i = (int)tid; i < ne; i += (int)tg_size) {
                if (logits[i] > local_max) {
                    local_max = logits[i];
                    local_id  = i;
                }
            }
            max_vals[tid] = local_max;
            max_ids[tid]  = local_id;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Reduce to find global max
            for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
                if (tid < stride && max_vals[tid + stride] > max_vals[tid]) {
                    max_vals[tid] = max_vals[tid + stride];
                    max_ids[tid]  = max_ids[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0) {
                selected_idx[k]   = max_ids[0];
                selected_logit[k] = max_vals[0];
                // Mask out selected expert
                logits[max_ids[0]] = -INFINITY;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Softmax over selected top-K logits only
        if (tid == 0) {
            float max_val = -INFINITY;
            for (int k = 0; k < K; k++) {
                if (selected_logit[k] > max_val) max_val = selected_logit[k];
            }
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                selected_logit[k] = exp(selected_logit[k] - max_val);
                sum += selected_logit[k];
            }
            for (int k = 0; k < K; k++) {
                int eidx = selected_idx[k];
                out_indices[k]      = eidx;
                out_weights[k]      = selected_logit[k] / sum;
                out_gate_offsets[k]  = all_gate_offsets[eidx];
                out_up_offsets[k]    = all_up_offsets[eidx];
                out_down_offsets[k]  = all_down_offsets[eidx];
            }
        }
    } else {
        // === Standard mode: softmax over ALL experts, then select top-K ===

        // Find max (parallel reduction)
        threadgroup float reduce_buf[256];
        float local_max = -INFINITY;
        for (int i = (int)tid; i < ne; i += (int)tg_size) {
            if (logits[i] > local_max) local_max = logits[i];
        }
        reduce_buf[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
            if (tid < stride && reduce_buf[tid + stride] > reduce_buf[tid]) {
                reduce_buf[tid] = reduce_buf[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float global_max = reduce_buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute exp and sum (parallel reduction)
        threadgroup float probs[512];
        float local_sum = 0.0f;
        for (int i = (int)tid; i < ne; i += (int)tg_size) {
            float p = exp(logits[i] - global_max);
            probs[i] = p;
            local_sum += p;
        }
        reduce_buf[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
            if (tid < stride) reduce_buf[tid] += reduce_buf[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float total_sum = reduce_buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Normalize to probabilities
        for (int i = (int)tid; i < ne; i += (int)tg_size) {
            probs[i] /= total_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Select top-K by probability (iterative, K is small)
        threadgroup float max_vals[256];
        threadgroup int   max_ids[256];

        for (int k = 0; k < K; k++) {
            float local_max_p = -1.0f;
            int   local_id    = -1;
            for (int i = (int)tid; i < ne; i += (int)tg_size) {
                if (probs[i] > local_max_p) {
                    local_max_p = probs[i];
                    local_id    = i;
                }
            }
            max_vals[tid] = local_max_p;
            max_ids[tid]  = local_id;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
                if (tid < stride && max_vals[tid + stride] > max_vals[tid]) {
                    max_vals[tid] = max_vals[tid + stride];
                    max_ids[tid]  = max_ids[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0) {
                selected_idx[k]   = max_ids[0];
                selected_logit[k] = max_vals[0];  // probability
                probs[max_ids[0]] = -1.0f;  // exclude
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Write results
        if (tid == 0) {
            // Optionally renormalize (norm_topk_prob)
            float topk_sum = 0.0f;
            if (args.norm_topk_prob) {
                for (int k = 0; k < K; k++) topk_sum += selected_logit[k];
            }

            for (int k = 0; k < K; k++) {
                int eidx = selected_idx[k];
                out_indices[k] = eidx;
                out_weights[k] = args.norm_topk_prob
                    ? selected_logit[k] / topk_sum
                    : selected_logit[k];
                out_gate_offsets[k]  = all_gate_offsets[eidx];
                out_up_offsets[k]    = all_up_offsets[eidx];
                out_down_offsets[k]  = all_down_offsets[eidx];
            }
        }
    }
}
