// Quantized attention weight matvec Metal kernels for moe-stream.
//
// Q5_0 and Q8_0 matrix-vector multiply kernels for attention weights
// (Q/K/V/O projections) that are stored in GGUF quantized format.
// These avoid dequantizing to F32 before matmul, saving ~6x bandwidth.
//
// IMPORTANT: Q5_0 blocks are 22 bytes (not power-of-2), so qh (uint32_t at
// offset +2) is often NOT 4-byte aligned. We read qh byte-by-byte to avoid
// unaligned access issues on Apple GPU.

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Q5_0 format: 32 elements per block = 22 bytes
//   - d:     F16 scale (2 bytes)
//   - qh[4]: 5th bit of each quant (4 bytes = 32 bits, one per element)
//   - qs[16]: low 4 bits packed as nibbles (16 bytes, 2 elements per byte)
//
// Dequantized value: (((qs_nibble | (qh_bit << 4)) & 0x1F) - 16) * d
// ============================================================================

#define QK5_0 32
#define Q5_0_BLOCK_SIZE 22  // sizeof(half) + 4 + QK5_0/2

// ============================================================================
// Q8_0 format: 32 elements per block = 34 bytes
//   - d:     F16 scale (2 bytes)
//   - qs[32]: int8 quants (32 bytes)
//
// Dequantized value: qs[i] * d
// ============================================================================

#define QK8_0 32
#define Q8_0_BLOCK_SIZE 34  // sizeof(half) + QK8_0

// Number of rows processed per simdgroup
#define NR0 2
// Number of simdgroups per threadgroup
#define NSG 2

// ============================================================================
// Args struct (same pattern as MxfpMvArgs in mxfp4.metal)
// ============================================================================

struct QuantizedMvArgs {
    int32_t  out_features;    // number of output rows
    int32_t  in_features;     // number of input columns
    uint64_t weight_stride;   // bytes per row in weight matrix
};

// Helper: read uint32_t from potentially unaligned address (byte-by-byte).
// Q5_0 blocks are 22 bytes, so the qh field at offset +2 alternates between
// 4-byte aligned (odd blocks) and 2-byte aligned (even blocks).
inline uint32_t read_u32_unaligned(device const uint8_t * ptr) {
    return uint32_t(ptr[0])
         | (uint32_t(ptr[1]) << 8)
         | (uint32_t(ptr[2]) << 16)
         | (uint32_t(ptr[3]) << 24);
}

// ============================================================================
// Q5_0 x F32 matrix-vector multiply kernel
//
// Grid: ((out_features + NR0*NSG - 1) / (NR0*NSG), 1, 1)
// Threadgroup: (32, NSG, 1)
//
// Each simdgroup processes NR0 rows. Each thread processes a subset of blocks
// along the row, accumulating partial dot products. simd_sum reduces across
// the simdgroup.
// ============================================================================

[[kernel]]
void q5_0_matvec_f32(
        constant QuantizedMvArgs & args       [[buffer(0)]],
        device const char        * weights    [[buffer(1)]],
        device const float       * input      [[buffer(2)]],
        device       float       * dst        [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    const int nb = args.in_features / QK5_0;  // blocks per row

    float sumf[NR0] = {0.0f};

    // Each thread processes blocks: tiisg, tiisg+32, tiisg+64, ...
    for (int ib = tiisg; ib < nb; ib += 32) {
        // Load input vector for this block (32 floats)
        device const float * y = input + ib * QK5_0;

        for (short row = 0; row < NR0; row++) {
            int cur_row = first_row + row;
            if (cur_row >= args.out_features) break;

            // Pointer to the Q5_0 block for this row
            device const uint8_t * block_ptr = (device const uint8_t *)(
                weights + (uint64_t)cur_row * args.weight_stride + (uint64_t)ib * Q5_0_BLOCK_SIZE);

            // Parse Q5_0 block:
            //   bytes [0..1]: F16 scale 'd'
            //   bytes [2..5]: qh (4 bytes = 32 high bits) — READ BYTE-BY-BYTE (alignment!)
            //   bytes [6..21]: qs (16 bytes = 32 nibbles)
            float d = float(*(device const half *)block_ptr);
            uint32_t qh = read_u32_unaligned(block_ptr + 2);
            device const uint8_t * qs = block_ptr + 6;

            float acc = 0.0f;

            // Process 32 elements: qs has nibbles (low 4 bits), qh has 5th bit
            for (short j = 0; j < 16; j++) {
                uint8_t q_byte = qs[j];

                // Low nibble -> element j
                int x0 = (int(q_byte & 0x0F) | (int((qh >> j) & 1) << 4)) - 16;
                acc += y[j] * float(x0);

                // High nibble -> element j+16
                int x1 = (int(q_byte >> 4) | (int((qh >> (j + 16)) & 1) << 4)) - 16;
                acc += y[j + 16] * float(x1);
            }

            sumf[row] += d * acc;
        }
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
// Q8_0 x F32 matrix-vector multiply kernel
//
// Grid: ((out_features + NR0*NSG - 1) / (NR0*NSG), 1, 1)
// Threadgroup: (32, NSG, 1)
//
// Same dispatch pattern. Q8_0 is simpler: each element is just qs[i] * d.
// Q8_0 blocks are 34 bytes — the F16 'd' at offset 0 is always 2-byte aligned
// and qs at offset 2 are single bytes, so no alignment issues.
// ============================================================================

[[kernel]]
void q8_0_matvec_f32(
        constant QuantizedMvArgs & args       [[buffer(0)]],
        device const char        * weights    [[buffer(1)]],
        device const float       * input      [[buffer(2)]],
        device       float       * dst        [[buffer(3)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    const int nb = args.in_features / QK8_0;  // blocks per row

    float sumf[NR0] = {0.0f};

    // Each thread processes blocks: tiisg, tiisg+32, tiisg+64, ...
    for (int ib = tiisg; ib < nb; ib += 32) {
        device const float * y = input + ib * QK8_0;

        for (short row = 0; row < NR0; row++) {
            int cur_row = first_row + row;
            if (cur_row >= args.out_features) break;

            // Pointer to the Q8_0 block for this row
            device const uint8_t * block_ptr = (device const uint8_t *)(
                weights + (uint64_t)cur_row * args.weight_stride + (uint64_t)ib * Q8_0_BLOCK_SIZE);

            // Parse Q8_0 block:
            //   bytes [0..1]: F16 scale 'd'
            //   bytes [2..33]: qs (32 int8 quants)
            float d = float(*(device const half *)block_ptr);
            device const int8_t * qs = (device const int8_t *)(block_ptr + 2);

            float acc = 0.0f;

            // Process 32 elements in groups of 4 for better ILP
            for (short j = 0; j < 32; j += 4) {
                acc += y[j + 0] * float(qs[j + 0]);
                acc += y[j + 1] * float(qs[j + 1]);
                acc += y[j + 2] * float(qs[j + 2]);
                acc += y[j + 3] * float(qs[j + 3]);
            }

            sumf[row] += d * acc;
        }
    }

    // Reduce across simdgroup and write result
    for (int row = 0; row < NR0 && first_row + row < args.out_features; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[first_row + row] = sum_all;
        }
    }
}
