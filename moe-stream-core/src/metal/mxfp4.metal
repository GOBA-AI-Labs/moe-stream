// MXFP4 matrix-vector multiply Metal kernel for moe-stream.
// Ported from llama.cpp's kernel_mul_mv_mxfp4_f32 (ggml-metal.metal).
//
// MXFP4 format: 32 elements per block = 1 byte E8M0 exponent + 16 bytes packed nibbles = 17 bytes.
// Low nibble qs[j] & 0x0F -> element[j], high nibble qs[j] >> 4 -> element[j+16].
//
// This kernel computes: dst[row] = dot(weight_row[row], input_vec) for each row.
// weight is (out_features x in_features) in MXFP4 format.
// input is (in_features,) in F32.
// dst is (out_features,) in F32.

#include <metal_stdlib>
using namespace metal;

// MXFP4 block: 32 elements packed in 17 bytes
#define QK_MXFP4 32
#define MXFP4_BLOCK_SIZE 17

// E2M1 values * 2 (the doubling cancels with e8m0 half-scale in llama.cpp's convention).
// We use llama.cpp's LUT which has values already halved: {0, .5, 1, 1.5, 2, 3, 4, 6, ...}
// Combined with e8m0_to_fp32 (full scale), this gives the correct result.
constant float kvalues_mxfp4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Convert E8M0 exponent byte to f32 scale factor.
// E8M0 is 8-bit unsigned exponent-only: value = 2^(e - 127).
// For e == 0: returns 2^(-126) (smallest positive normal in fp32 convention).
static inline float e8m0_to_fp32(uint8_t x) {
    uint32_t bits;
    if (x == 0) {
        bits = 0x00400000;  // 2^(-126) * 2 = 2^(-125) -- matches llama.cpp
    } else {
        bits = (uint32_t)x << 23;
    }
    return as_type<float>(bits);
}

// Kernel arguments struct (simplified for matrix-vector multiply).
// weight: (out_features x in_features) MXFP4
// input:  (in_features,) F32
// dst:    (out_features,) F32
struct MxfpMvArgs {
    int32_t  out_features;    // number of output rows (ne01 in llama.cpp)
    int32_t  in_features;     // number of input columns (ne00 in llama.cpp)
    uint64_t weight_stride;   // bytes per row in weight matrix (nb01)
};

// Number of rows processed per simdgroup
#define NR0 2
// Number of simdgroups per threadgroup
#define NSG 2

// MXFP4 x F32 matrix-vector multiply kernel.
//
// Grid: ((out_features + NR0*NSG - 1) / (NR0*NSG), 1, 1)
// Threadgroup: (32, NSG, 1)
// Shared memory: 32 * sizeof(float) = 128 bytes
//
// Each simdgroup processes NR0 rows. Each thread in the simdgroup processes
// a subset of the blocks along the row, accumulating partial dot products,
// then simd_sum reduces across the simdgroup.
[[kernel]]
void mxfp4_matvec_f32(
        constant MxfpMvArgs & args       [[buffer(0)]],
        device const char   * weights    [[buffer(1)]],
        device const float  * input      [[buffer(2)]],
        device       float  * dst        [[buffer(3)]],
        threadgroup  char   * shmem      [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    threadgroup float * shmem_f32 = (threadgroup float *)shmem;

    const int r0 = tgpig.x;  // threadgroup index along output rows
    const int first_row = (r0 * NSG + sgitg) * NR0;

    // Load LUT into shared memory (32 threads write 16 unique values, first 16 are used)
    shmem_f32[tiisg] = kvalues_mxfp4[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int nb = args.in_features / QK_MXFP4;  // blocks per row
    const int ns01 = (int)(args.weight_stride / MXFP4_BLOCK_SIZE);  // stride in blocks

    const short ix = tiisg / 2;   // 0..15 block index within chunk
    const short it = tiisg % 2;   // 0 or 1: which half of 16 nibbles

    float4 yl[4];
    float sumf[NR0] = {0.0f};

    device const float * yb = input + ix * QK_MXFP4 + it * 8;

    for (int ib = ix; ib < nb && ib < ns01; ib += 16) {
        device const float4 * y4 = (device const float4 *)yb;

        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; row++) {
            // Compute pointer to the MXFP4 block for this row
            device const uint8_t * block_ptr = (device const uint8_t *)(weights + (uint64_t)(first_row + row) * args.weight_stride + (uint64_t)ib * MXFP4_BLOCK_SIZE);
            uint8_t e = block_ptr[0];
            device const uint8_t * q2 = block_ptr + 1 + 8 * it;

            float4 acc1 = yl[0] * float4(shmem_f32[q2[0] & 0x0F], shmem_f32[q2[1] & 0x0F], shmem_f32[q2[2] & 0x0F], shmem_f32[q2[3] & 0x0F]);
            float4 acc2 = yl[1] * float4(shmem_f32[q2[0] >> 4],   shmem_f32[q2[1] >> 4],   shmem_f32[q2[2] >> 4],   shmem_f32[q2[3] >> 4]);
            float4 acc3 = yl[2] * float4(shmem_f32[q2[4] & 0x0F], shmem_f32[q2[5] & 0x0F], shmem_f32[q2[6] & 0x0F], shmem_f32[q2[7] & 0x0F]);
            float4 acc4 = yl[3] * float4(shmem_f32[q2[4] >> 4],   shmem_f32[q2[5] >> 4],   shmem_f32[q2[6] >> 4],   shmem_f32[q2[7] >> 4]);

            acc1 = (acc1 + acc3) + (acc2 + acc4);

            sumf[row] += e8m0_to_fp32(e) * ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
        }

        yb += 16 * QK_MXFP4;
    }

    // Reduce across simdgroup and write result
    for (int row = 0; row < NR0 && first_row + row < args.out_features; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[first_row + row] = sum_all;
        }
    }
}

// ============================================================================
// Batched multi-expert MXFP4 matvec kernel.
//
// Processes up to MAX_BATCH_EXPERTS expert matmuls in a single dispatch using
// the Z grid dimension for expert indexing.
//
// This is the key optimization: instead of 4 separate kernel dispatches per
// projection (one per active expert), we dispatch once with depth=n_experts.
//
// Grid:  ((out_features + NR0*NSG - 1) / (NR0*NSG), 1, n_experts)
// TG:    (32, NSG, 1)
//
// Expert weights are passed as a single packed buffer. The expert_offsets
// array provides byte offsets into this buffer for each expert.
// Output: dst[expert_idx * out_features + row]
// ============================================================================

#define MAX_BATCH_EXPERTS 8

struct MxfpBatchedArgs {
    int32_t  out_features;     // rows per expert weight matrix
    int32_t  in_features;      // columns per expert weight matrix
    uint64_t weight_stride;    // bytes per row in weight matrix
    int32_t  n_experts;        // number of active experts (depth of Z grid)
    uint64_t expert_offsets[MAX_BATCH_EXPERTS]; // byte offset into packed weight buffer for each expert
};

[[kernel]]
void mxfp4_batched_experts_matvec_f32(
        constant MxfpBatchedArgs & args      [[buffer(0)]],
        device const char        * weights   [[buffer(1)]],   // packed expert weights
        device const float       * input     [[buffer(2)]],   // shared input vector
        device       float       * dst       [[buffer(3)]],   // output [n_experts, out_features]
        threadgroup  char        * shmem     [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    const int expert_idx = tgpig.z;
    if (expert_idx >= args.n_experts) return;

    // Expert-specific weight pointer
    device const char * expert_weights = weights + args.expert_offsets[expert_idx];

    threadgroup float * shmem_f32 = (threadgroup float *)shmem;

    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    shmem_f32[tiisg] = kvalues_mxfp4[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int nb = args.in_features / QK_MXFP4;
    const int ns01 = (int)(args.weight_stride / MXFP4_BLOCK_SIZE);

    const short ix = tiisg / 2;
    const short it = tiisg % 2;

    float4 yl[4];
    float sumf[NR0] = {0.0f};

    device const float * yb = input + ix * QK_MXFP4 + it * 8;

    for (int ib = ix; ib < nb && ib < ns01; ib += 16) {
        device const float4 * y4 = (device const float4 *)yb;

        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; row++) {
            device const uint8_t * block_ptr = (device const uint8_t *)(expert_weights + (uint64_t)(first_row + row) * args.weight_stride + (uint64_t)ib * MXFP4_BLOCK_SIZE);
            uint8_t e = block_ptr[0];
            device const uint8_t * q2 = block_ptr + 1 + 8 * it;

            float4 acc1 = yl[0] * float4(shmem_f32[q2[0] & 0x0F], shmem_f32[q2[1] & 0x0F], shmem_f32[q2[2] & 0x0F], shmem_f32[q2[3] & 0x0F]);
            float4 acc2 = yl[1] * float4(shmem_f32[q2[0] >> 4],   shmem_f32[q2[1] >> 4],   shmem_f32[q2[2] >> 4],   shmem_f32[q2[3] >> 4]);
            float4 acc3 = yl[2] * float4(shmem_f32[q2[4] & 0x0F], shmem_f32[q2[5] & 0x0F], shmem_f32[q2[6] & 0x0F], shmem_f32[q2[7] & 0x0F]);
            float4 acc4 = yl[3] * float4(shmem_f32[q2[4] >> 4],   shmem_f32[q2[5] >> 4],   shmem_f32[q2[6] >> 4],   shmem_f32[q2[7] >> 4]);

            acc1 = (acc1 + acc3) + (acc2 + acc4);

            sumf[row] += e8m0_to_fp32(e) * ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
        }

        yb += 16 * QK_MXFP4;
    }

    // Write to expert-indexed output: dst[expert_idx * out_features + row]
    const int dst_offset = expert_idx * args.out_features;
    for (int row = 0; row < NR0 && first_row + row < args.out_features; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[dst_offset + first_row + row] = sum_all;
        }
    }
}

// ============================================================================
// Batched multi-expert MXFP4 matvec with per-expert input vectors.
//
// Same as mxfp4_batched_experts_matvec_f32, but each expert reads from its
// own region of the input buffer: input[expert_idx * in_features ... ].
//
// Used for the down projection where each expert has its own intermediate
// vector (from the SwiGLU activation).
//
// Grid:  ((out_features + NR0*NSG - 1) / (NR0*NSG), 1, n_experts)
// TG:    (32, NSG, 1)
// ============================================================================

[[kernel]]
void mxfp4_batched_experts_per_input_matvec_f32(
        constant MxfpBatchedArgs & args      [[buffer(0)]],
        device const char        * weights   [[buffer(1)]],   // packed expert weights
        device const float       * input     [[buffer(2)]],   // per-expert input [n_experts, in_features]
        device       float       * dst       [[buffer(3)]],   // output [n_experts, out_features]
        threadgroup  char        * shmem     [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    const int expert_idx = tgpig.z;
    if (expert_idx >= args.n_experts) return;

    // Expert-specific weight and input pointers
    device const char * expert_weights = weights + args.expert_offsets[expert_idx];
    device const float * expert_input = input + expert_idx * args.in_features;

    threadgroup float * shmem_f32 = (threadgroup float *)shmem;

    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    shmem_f32[tiisg] = kvalues_mxfp4[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int nb = args.in_features / QK_MXFP4;
    const int ns01 = (int)(args.weight_stride / MXFP4_BLOCK_SIZE);

    const short ix = tiisg / 2;
    const short it = tiisg % 2;

    float4 yl[4];
    float sumf[NR0] = {0.0f};

    device const float * yb = expert_input + ix * QK_MXFP4 + it * 8;

    for (int ib = ix; ib < nb && ib < ns01; ib += 16) {
        device const float4 * y4 = (device const float4 *)yb;

        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; row++) {
            device const uint8_t * block_ptr = (device const uint8_t *)(expert_weights + (uint64_t)(first_row + row) * args.weight_stride + (uint64_t)ib * MXFP4_BLOCK_SIZE);
            uint8_t e = block_ptr[0];
            device const uint8_t * q2 = block_ptr + 1 + 8 * it;

            float4 acc1 = yl[0] * float4(shmem_f32[q2[0] & 0x0F], shmem_f32[q2[1] & 0x0F], shmem_f32[q2[2] & 0x0F], shmem_f32[q2[3] & 0x0F]);
            float4 acc2 = yl[1] * float4(shmem_f32[q2[0] >> 4],   shmem_f32[q2[1] >> 4],   shmem_f32[q2[2] >> 4],   shmem_f32[q2[3] >> 4]);
            float4 acc3 = yl[2] * float4(shmem_f32[q2[4] & 0x0F], shmem_f32[q2[5] & 0x0F], shmem_f32[q2[6] & 0x0F], shmem_f32[q2[7] & 0x0F]);
            float4 acc4 = yl[3] * float4(shmem_f32[q2[4] >> 4],   shmem_f32[q2[5] >> 4],   shmem_f32[q2[6] >> 4],   shmem_f32[q2[7] >> 4]);

            acc1 = (acc1 + acc3) + (acc2 + acc4);

            sumf[row] += e8m0_to_fp32(e) * ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
        }

        yb += 16 * QK_MXFP4;
    }

    // Write to expert-indexed output: dst[expert_idx * out_features + row]
    const int dst_offset = expert_idx * args.out_features;
    for (int row = 0; row < NR0 && first_row + row < args.out_features; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[dst_offset + first_row + row] = sum_all;
        }
    }
}

// ============================================================================
// Fused OAI SwiGLU + weighted accumulation kernel.
//
// For each expert e in [0, n_experts):
//   gate_c = clamp(gate[e], -inf, limit)
//   up_c   = clamp(up[e], -limit, limit)
//   activated = gate_c * sigmoid(alpha * gate_c)
//   expert_out = down[e] (already computed externally)
//   accum += routing_weight[e] * expert_out
//
// But this kernel handles the SwiGLU part: given gate and up outputs for all
// experts, compute the intermediate (activated) values and store them.
// The down projection + accumulation is separate.
//
// Input:  gate_outputs[n_experts * intermediate_dim]
//         up_outputs[n_experts * intermediate_dim]
//         expert_biases_gate/up (optional, handled externally)
// Output: activated[n_experts * intermediate_dim]
//
// Grid:  (ceil(intermediate_dim / 256), n_experts, 1)
// TG:    (256, 1, 1)
// ============================================================================

struct SwiGluOaiArgs {
    int32_t dim;            // intermediate_dim
    int32_t n_experts;
    float   alpha;          // 1.702 for GPT-OSS
    float   limit;          // 7.0 for GPT-OSS
};

[[kernel]]
void mxfp4_fused_swiglu_oai_f32(
        constant SwiGluOaiArgs & args         [[buffer(0)]],
        device const float     * gate_outputs [[buffer(1)]],  // [n_experts, dim]
        device const float     * up_outputs   [[buffer(2)]],  // [n_experts, dim]
        device       float     * dst          [[buffer(3)]],  // [n_experts, dim]
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint   tid   [[thread_index_in_threadgroup]]) {

    const int col = tgpig.x * 256 + tid;
    const int expert_idx = tgpig.y;

    if (col >= args.dim || expert_idx >= args.n_experts) return;

    const int idx = expert_idx * args.dim + col;

    float gate = gate_outputs[idx];
    float up   = up_outputs[idx];

    // clamp gate to [-inf, limit] and up to [-limit, limit]
    float gate_c = min(gate, args.limit);
    float up_c   = clamp(up, -args.limit, args.limit);

    // OAI SwiGLU: gate_c * sigmoid(alpha * gate_c) * (up_c + 1.0)
    float sig = 1.0f / (1.0f + exp(-args.alpha * gate_c));
    float activated = gate_c * sig * (up_c + 1.0f);

    dst[idx] = activated;
}

// ============================================================================
// Fused standard SwiGLU (non-OAI) activation for all experts.
//
// Standard SwiGLU: activated = silu(gate) * up
// where silu(x) = x * sigmoid(x)
//
// Input:  gate_outputs[n_experts * intermediate_dim]
//         up_outputs[n_experts * intermediate_dim]
// Output: activated[n_experts * intermediate_dim]
//
// Grid:  (ceil(intermediate_dim / 256), n_experts, 1)
// TG:    (256, 1, 1)
// ============================================================================

struct SwiGluStdArgs {
    int32_t dim;            // intermediate_dim
    int32_t n_experts;
};

[[kernel]]
void mxfp4_fused_swiglu_std_f32(
        constant SwiGluStdArgs & args          [[buffer(0)]],
        device const float     * gate_outputs  [[buffer(1)]],  // [n_experts, dim]
        device const float     * up_outputs    [[buffer(2)]],  // [n_experts, dim]
        device       float     * dst           [[buffer(3)]],  // [n_experts, dim]
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint   tid   [[thread_index_in_threadgroup]]) {

    const int col = tgpig.x * 256 + tid;
    const int expert_idx = tgpig.y;

    if (col >= args.dim || expert_idx >= args.n_experts) return;

    const int idx = expert_idx * args.dim + col;

    float gate = gate_outputs[idx];
    float up   = up_outputs[idx];

    // Standard SwiGLU: silu(gate) * up = gate * sigmoid(gate) * up
    float sig = 1.0f / (1.0f + exp(-gate));
    float activated = gate * sig * up;

    dst[idx] = activated;
}

// ============================================================================
// Batched expert bias addition.
//
// Adds per-expert bias vectors to batched expert outputs in-place.
// bias_data contains all experts' biases packed contiguously.
// expert_indices maps the batch index to the actual expert index for bias lookup.
//
// Input/Output: data[n_experts * dim] (modified in-place)
// Bias:         bias_data[total_experts * dim] (all expert biases, indexed by expert_indices)
// Indices:      expert_indices[n_experts] (maps batch slot -> real expert index)
//
// Grid:  (ceil(dim / 256), n_experts, 1)
// TG:    (256, 1, 1)
// ============================================================================

struct BiasAddArgs {
    int32_t dim;            // feature dimension
    int32_t n_experts;      // number of active experts in batch
};

[[kernel]]
void mxfp4_batched_bias_add_f32(
        constant BiasAddArgs     & args            [[buffer(0)]],
        device       float       * data            [[buffer(1)]],  // [n_experts, dim] (in-place)
        device const float       * bias_data       [[buffer(2)]],  // [total_experts, dim]
        device const int32_t     * expert_indices  [[buffer(3)]],  // [n_experts]
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint   tid   [[thread_index_in_threadgroup]]) {

    const int col = tgpig.x * 256 + tid;
    const int batch_idx = tgpig.y;

    if (col >= args.dim || batch_idx >= args.n_experts) return;

    const int real_expert = expert_indices[batch_idx];
    const int data_idx = batch_idx * args.dim + col;
    const int bias_idx = real_expert * args.dim + col;

    data[data_idx] += bias_data[bias_idx];
}

// ============================================================================
// Fused weighted accumulation: sum over experts with routing weights.
//
// Input:  expert_outputs[n_experts * hidden_dim] (after down projection)
//         routing_weights[n_experts]
// Output: accum[hidden_dim] += sum_e(routing_weights[e] * expert_outputs[e])
//
// Grid:  (ceil(hidden_dim / 256), 1, 1)
// TG:    (256, 1, 1)
// ============================================================================

struct AccumArgs {
    int32_t dim;            // hidden_dim
    int32_t n_experts;
};

[[kernel]]
void mxfp4_fused_weighted_accum_f32(
        constant AccumArgs     & args            [[buffer(0)]],
        device const float     * expert_outputs  [[buffer(1)]],  // [n_experts, dim]
        device const float     * routing_weights [[buffer(2)]],  // [n_experts]
        device       float     * dst             [[buffer(3)]],  // [dim]
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint   tid   [[thread_index_in_threadgroup]]) {

    const int col = tgpig.x * 256 + tid;
    if (col >= args.dim) return;

    float sum = 0.0f;
    for (int e = 0; e < args.n_experts; ++e) {
        sum += routing_weights[e] * expert_outputs[e * args.dim + col];
    }
    dst[col] = sum;
}

// ============================================================================
// Fused Gate+Up projection + Bias + SwiGLU activation kernel.
//
// Replaces 5 separate dispatches:
//   gate_matvec + gate_bias + up_matvec + up_bias + swiglu
// with a single kernel that:
//   1. Reads the input vector ONCE
//   2. Computes both gate and up dot products simultaneously
//   3. Adds biases inline
//   4. Applies SwiGLU activation
//   5. Writes directly to the activated output
//
// This halves input vector reads and eliminates 4 intermediate buffer writes.
//
// Grid:  ((out_features + NR0*NSG - 1) / (NR0*NSG), 1, n_experts)
// TG:    (32, NSG, 1)
// ============================================================================

struct MxfpFusedGateUpArgs {
    int32_t  out_features;     // intermediate_dim (gate/up output rows)
    int32_t  in_features;      // hidden_dim (input vector length)
    uint64_t weight_stride;    // bytes per row in weight matrix
    int32_t  n_experts;        // number of active experts
    int32_t  use_oai_swiglu;   // 1 = OAI SwiGLU, 0 = standard SwiGLU
    float    alpha;            // OAI SwiGLU alpha (1.702)
    float    limit;            // OAI SwiGLU limit (7.0)
    int32_t  has_bias;         // 1 = add bias, 0 = no bias
    // Offsets moved to device buffer bindings (buffer 8/9) for GPU-side routing.
};

[[kernel]]
void mxfp4_fused_gate_up_swiglu_f32(
        constant MxfpFusedGateUpArgs & args         [[buffer(0)]],
        device const char            * gate_weights  [[buffer(1)]],   // packed gate expert weights
        device const char            * up_weights    [[buffer(2)]],   // packed up expert weights
        device const float           * input         [[buffer(3)]],   // shared input vector [in_features]
        device       float           * dst           [[buffer(4)]],   // output [n_experts, out_features]
        device const float           * gate_bias     [[buffer(5)]],   // [total_experts, out_features] or null
        device const float           * up_bias       [[buffer(6)]],   // [total_experts, out_features] or null
        device const int32_t         * expert_indices [[buffer(7)]],  // [n_experts] maps batch→real expert
        device const uint64_t        * gate_offsets  [[buffer(8)]],   // [n_experts] byte offsets into gate weights
        device const uint64_t        * up_offsets    [[buffer(9)]],   // [n_experts] byte offsets into up weights
        threadgroup  char            * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    const int expert_idx = tgpig.z;
    if (expert_idx >= args.n_experts) return;

    // Expert-specific weight pointers (offsets from device buffer)
    device const char * gate_exp = gate_weights + gate_offsets[expert_idx];
    device const char * up_exp = up_weights + up_offsets[expert_idx];

    threadgroup float * shmem_f32 = (threadgroup float *)shmem;

    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    shmem_f32[tiisg] = kvalues_mxfp4[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int nb = args.in_features / QK_MXFP4;
    const int ns01 = (int)(args.weight_stride / MXFP4_BLOCK_SIZE);

    const short ix = tiisg / 2;
    const short it = tiisg % 2;

    float4 yl[4];
    float gate_sumf[NR0] = {0.0f};
    float up_sumf[NR0] = {0.0f};

    device const float * yb = input + ix * QK_MXFP4 + it * 8;

    for (int ib = ix; ib < nb && ib < ns01; ib += 16) {
        // Read input ONCE (shared between gate and up)
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; row++) {
            // Gate projection
            {
                device const uint8_t * block_ptr = (device const uint8_t *)(gate_exp + (uint64_t)(first_row + row) * args.weight_stride + (uint64_t)ib * MXFP4_BLOCK_SIZE);
                uint8_t e = block_ptr[0];
                device const uint8_t * q2 = block_ptr + 1 + 8 * it;

                float4 acc1 = yl[0] * float4(shmem_f32[q2[0] & 0x0F], shmem_f32[q2[1] & 0x0F], shmem_f32[q2[2] & 0x0F], shmem_f32[q2[3] & 0x0F]);
                float4 acc2 = yl[1] * float4(shmem_f32[q2[0] >> 4],   shmem_f32[q2[1] >> 4],   shmem_f32[q2[2] >> 4],   shmem_f32[q2[3] >> 4]);
                float4 acc3 = yl[2] * float4(shmem_f32[q2[4] & 0x0F], shmem_f32[q2[5] & 0x0F], shmem_f32[q2[6] & 0x0F], shmem_f32[q2[7] & 0x0F]);
                float4 acc4 = yl[3] * float4(shmem_f32[q2[4] >> 4],   shmem_f32[q2[5] >> 4],   shmem_f32[q2[6] >> 4],   shmem_f32[q2[7] >> 4]);
                acc1 = (acc1 + acc3) + (acc2 + acc4);
                gate_sumf[row] += e8m0_to_fp32(e) * ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
            }

            // Up projection (same input, different weights)
            {
                device const uint8_t * block_ptr = (device const uint8_t *)(up_exp + (uint64_t)(first_row + row) * args.weight_stride + (uint64_t)ib * MXFP4_BLOCK_SIZE);
                uint8_t e = block_ptr[0];
                device const uint8_t * q2 = block_ptr + 1 + 8 * it;

                float4 acc1 = yl[0] * float4(shmem_f32[q2[0] & 0x0F], shmem_f32[q2[1] & 0x0F], shmem_f32[q2[2] & 0x0F], shmem_f32[q2[3] & 0x0F]);
                float4 acc2 = yl[1] * float4(shmem_f32[q2[0] >> 4],   shmem_f32[q2[1] >> 4],   shmem_f32[q2[2] >> 4],   shmem_f32[q2[3] >> 4]);
                float4 acc3 = yl[2] * float4(shmem_f32[q2[4] & 0x0F], shmem_f32[q2[5] & 0x0F], shmem_f32[q2[6] & 0x0F], shmem_f32[q2[7] & 0x0F]);
                float4 acc4 = yl[3] * float4(shmem_f32[q2[4] >> 4],   shmem_f32[q2[5] >> 4],   shmem_f32[q2[6] >> 4],   shmem_f32[q2[7] >> 4]);
                acc1 = (acc1 + acc3) + (acc2 + acc4);
                up_sumf[row] += e8m0_to_fp32(e) * ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
            }
        }

        yb += 16 * QK_MXFP4;
    }

    // Reduce across simdgroup, apply bias + SwiGLU, and write
    const int dst_offset = expert_idx * args.out_features;
    const int real_expert = expert_indices[expert_idx];

    for (int row = 0; row < NR0 && first_row + row < args.out_features; ++row) {
        float gate_val = simd_sum(gate_sumf[row]);
        float up_val = simd_sum(up_sumf[row]);

        if (tiisg == 0) {
            // Add biases
            if (args.has_bias) {
                gate_val += gate_bias[real_expert * args.out_features + first_row + row];
                up_val += up_bias[real_expert * args.out_features + first_row + row];
            }

            // Apply SwiGLU
            float activated;
            if (args.use_oai_swiglu) {
                float gate_c = min(gate_val, args.limit);
                float up_c = clamp(up_val, -args.limit, args.limit);
                float sig = 1.0f / (1.0f + exp(-args.alpha * gate_c));
                activated = gate_c * sig * (up_c + 1.0f);
            } else {
                float sig = 1.0f / (1.0f + exp(-gate_val));
                activated = gate_val * sig * up_val;
            }

            dst[dst_offset + first_row + row] = activated;
        }
    }
}

// ============================================================================
// Fused Down projection + Bias + Weighted Accumulation kernel.
//
// Replaces 3 separate dispatches:
//   down_matvec + down_bias + weighted_accum
// with a single kernel that:
//   1. For each output row, iterates over all active experts
//   2. Computes down projection dot product
//   3. Adds bias inline
//   4. Accumulates with routing weight directly
//   5. Writes final result to output
//
// Eliminates the intermediate down_buf.
//
// Grid:  ((out_features + NR0*NSG - 1) / (NR0*NSG), 1, 1)
// TG:    (32, NSG, 1)
// ============================================================================

struct MxfpFusedDownAccumArgs {
    int32_t  out_features;     // hidden_dim (down output rows)
    int32_t  in_features;      // intermediate_dim (down input = swiglu output)
    uint64_t weight_stride;    // bytes per row in down weight matrix
    int32_t  n_experts;        // number of active experts
    int32_t  has_bias;         // 1 = add bias, 0 = no bias
    // Offsets moved to device buffer binding (buffer 7) for GPU-side routing.
};

[[kernel]]
void mxfp4_fused_down_accum_f32(
        constant MxfpFusedDownAccumArgs & args          [[buffer(0)]],
        device const char               * down_weights  [[buffer(1)]],   // packed down expert weights
        device const float              * inputs        [[buffer(2)]],   // [n_experts, in_features]
        device const float              * routing_weights [[buffer(3)]], // [n_experts]
        device       float              * dst           [[buffer(4)]],   // [out_features]
        device const float              * down_bias     [[buffer(5)]],   // [total_experts, out_features]
        device const int32_t            * expert_indices [[buffer(6)]],  // [n_experts]
        device const uint64_t           * down_offsets  [[buffer(7)]],   // [n_experts] byte offsets
        threadgroup  char               * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    threadgroup float * shmem_f32 = (threadgroup float *)shmem;

    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    shmem_f32[tiisg] = kvalues_mxfp4[tiisg % 16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int nb = args.in_features / QK_MXFP4;
    const int ns01 = (int)(args.weight_stride / MXFP4_BLOCK_SIZE);

    const short ix = tiisg / 2;
    const short it = tiisg % 2;

    // Accumulate across all experts
    float accum[NR0] = {0.0f};

    for (int e = 0; e < args.n_experts; e++) {
        device const char * down_exp = down_weights + down_offsets[e];
        device const float * exp_input = inputs + e * args.in_features;
        float rw = routing_weights[e];

        float sumf[NR0] = {0.0f};
        device const float * yb = exp_input + ix * QK_MXFP4 + it * 8;

        for (int ib = ix; ib < nb && ib < ns01; ib += 16) {
            device const float4 * y4 = (device const float4 *)yb;

            float4 yl[4];
            yl[0] = y4[0];
            yl[1] = y4[4];
            yl[2] = y4[1];
            yl[3] = y4[5];

            for (short row = 0; row < NR0; row++) {
                device const uint8_t * block_ptr = (device const uint8_t *)(down_exp + (uint64_t)(first_row + row) * args.weight_stride + (uint64_t)ib * MXFP4_BLOCK_SIZE);
                uint8_t e_scale = block_ptr[0];
                device const uint8_t * q2 = block_ptr + 1 + 8 * it;

                float4 acc1 = yl[0] * float4(shmem_f32[q2[0] & 0x0F], shmem_f32[q2[1] & 0x0F], shmem_f32[q2[2] & 0x0F], shmem_f32[q2[3] & 0x0F]);
                float4 acc2 = yl[1] * float4(shmem_f32[q2[0] >> 4],   shmem_f32[q2[1] >> 4],   shmem_f32[q2[2] >> 4],   shmem_f32[q2[3] >> 4]);
                float4 acc3 = yl[2] * float4(shmem_f32[q2[4] & 0x0F], shmem_f32[q2[5] & 0x0F], shmem_f32[q2[6] & 0x0F], shmem_f32[q2[7] & 0x0F]);
                float4 acc4 = yl[3] * float4(shmem_f32[q2[4] >> 4],   shmem_f32[q2[5] >> 4],   shmem_f32[q2[6] >> 4],   shmem_f32[q2[7] >> 4]);

                acc1 = (acc1 + acc3) + (acc2 + acc4);

                sumf[row] += e8m0_to_fp32(e_scale) * ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
            }

            yb += 16 * QK_MXFP4;
        }

        // Add bias + weighted accumulate
        const int real_expert = expert_indices[e];
        for (int row = 0; row < NR0; row++) {
            float down_val = simd_sum(sumf[row]);
            if (args.has_bias && tiisg == 0) {
                down_val += down_bias[real_expert * args.out_features + first_row + row];
            }
            // Only thread 0 has the correct reduced value, but we need it for accum
            // Broadcast from thread 0 to all threads (not needed since only thread 0 writes)
            if (tiisg == 0) {
                accum[row] += rw * down_val;
            }
        }
    }

    // Write final accumulated result
    for (int row = 0; row < NR0 && first_row + row < args.out_features; ++row) {
        if (tiisg == 0) {
            dst[first_row + row] = accum[row];
        }
    }
}
