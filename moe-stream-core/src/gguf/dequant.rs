//! Q2_K, Q3_K, Q4_K, Q5_K and Q6_K dequantization for GGUF tensors.
//!
//! Direct port of fast_dequant.c with Rust safety + potential NEON SIMD.
//! Block formats match ggml/llama.cpp specification.
#![allow(clippy::needless_range_loop)]

use half::f16;

/// Bytes per Q2_K block (256 elements per block)
pub const Q2K_BLOCK_SIZE: usize = 84;
/// Bytes per Q3_K block (256 elements per block)
pub const Q3K_BLOCK_SIZE: usize = 110;
/// Bytes per Q4_K block (256 elements per block)
pub const Q4K_BLOCK_SIZE: usize = 144;
/// Bytes per Q5_K block (256 elements per block)
pub const Q5K_BLOCK_SIZE: usize = 176;
/// Bytes per Q6_K block (256 elements per block)
pub const Q6K_BLOCK_SIZE: usize = 210;
/// Elements per quantized block (K-quants)
pub const BLOCK_ELEMENTS: usize = 256;

// === Legacy quantization types (32 elements per block) ===
/// Elements per legacy block (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1)
pub const LEGACY_BLOCK_ELEMENTS: usize = 32;
/// Bytes per Q4_0 block: d(2B fp16) + qs(16B packed 4-bit) = 18B
pub const Q4_0_BLOCK_SIZE: usize = 18;
/// Bytes per Q4_1 block: d(2B fp16) + min(2B fp16) + qs(16B packed 4-bit) = 20B
pub const Q4_1_BLOCK_SIZE: usize = 20;
/// Bytes per Q5_0 block: d(2B fp16) + qh(4B high bits) + qs(16B low 4-bit) = 22B
pub const Q5_0_BLOCK_SIZE: usize = 22;
/// Bytes per Q5_1 block: d(2B fp16) + min(2B fp16) + qh(4B) + qs(16B) = 24B
pub const Q5_1_BLOCK_SIZE: usize = 24;
/// Bytes per Q8_0 block: d(2B fp16) + qs(32B int8) = 34B
pub const Q8_0_BLOCK_SIZE: usize = 34;
/// Bytes per Q8_1 block: d(2B fp16) + s(2B fp16) + qs(32B int8) = 36B
pub const Q8_1_BLOCK_SIZE: usize = 36;

/// Convert a raw u16 (IEEE 754 half-precision) to f32.
#[inline(always)]
fn fp16_to_f32(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

/// Read a little-endian u16 from a byte slice.
#[inline(always)]
fn read_u16_le(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

/// Dequantize Q3_K block data to f32.
///
/// Q3_K format: 110 bytes per 256 elements.
/// Layout: hmask (32B) + qs (64B) + scales (12B) + d (2B fp16).
///
/// This is a 3-bit quantization with complex bit packing.
pub fn dequantize_q3k(block_data: &[u8], n_elements: usize) -> Vec<f32> {
    assert!(n_elements.is_multiple_of(BLOCK_ELEMENTS));
    let n_blocks = n_elements / BLOCK_ELEMENTS;
    assert!(block_data.len() >= n_blocks * Q3K_BLOCK_SIZE);

    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;

    let mut output = vec![0.0f32; n_elements];

    for block in 0..n_blocks {
        let x = &block_data[block * Q3K_BLOCK_SIZE..];
        let y = &mut output[block * BLOCK_ELEMENTS..];

        let hm = &x[0..32];
        let qs = &x[32..96];
        let scales_raw = &x[96..108];
        let d = fp16_to_f32(read_u16_le(x, 108));

        // Unpack scales using aux array method from ggml
        let mut aux = [0u32; 4];
        aux[0] = u32::from_le_bytes([scales_raw[0], scales_raw[1], scales_raw[2], scales_raw[3]]);
        aux[1] = u32::from_le_bytes([scales_raw[4], scales_raw[5], scales_raw[6], scales_raw[7]]);
        let tmp = u32::from_le_bytes([scales_raw[8], scales_raw[9], scales_raw[10], scales_raw[11]]);

        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

        // Interpret aux as 16 int8 scales
        let scales: &[i8; 16] = unsafe { std::mem::transmute(&aux) };

        let mut q_offset = 0usize;
        let mut out_idx = 0usize;
        let mut us = 1u8;

        // Process 256 elements in two 128-element halves
        for _half in 0..2 {
            let mut shift = 0u8;

            for _j in 0..4 {
                let scale_idx = out_idx / 16;
                let dl = d * (scales[scale_idx] as i32 - 32) as f32;

                for l in 0..16 {
                    let q_low = (qs[q_offset + l] >> shift) & 3;
                    let h_bit = if hm[l] & us != 0 { 0i8 } else { 4i8 };
                    let q = (q_low as i8) - h_bit;
                    y[out_idx] = dl * q as f32;
                    out_idx += 1;
                }

                let ml = d * (scales[scale_idx + 1] as i32 - 32) as f32;
                for l in 0..16 {
                    let q_low = (qs[q_offset + 16 + l] >> shift) & 3;
                    let h_bit = if hm[16 + l] & us != 0 { 0i8 } else { 4i8 };
                    let q = (q_low as i8) - h_bit;
                    y[out_idx] = ml * q as f32;
                    out_idx += 1;
                }

                shift += 2;
            }

            q_offset += 32;
            us <<= 1;
            if us == 0 {
                us = 1;
            }
        }
    }

    output
}

/// Dequantize Q2_K block data to f32.
///
/// Q2_K format: 84 bytes per 256 elements.
/// Layout: d (fp16, 2B) + dmin (fp16, 2B) + scales (16B) + qs (64B packed 2-bit).
///
/// # Arguments
/// * `block_data` - Raw Q2_K block bytes (must be `n_blocks * 84` bytes)
/// * `n_elements` - Total output elements (must be `n_blocks * 256`)
///
/// # Returns
/// Vec<f32> of dequantized values
pub fn dequantize_q2k(block_data: &[u8], n_elements: usize) -> Vec<f32> {
    assert!(n_elements.is_multiple_of(BLOCK_ELEMENTS));
    let n_blocks = n_elements / BLOCK_ELEMENTS;
    assert!(block_data.len() >= n_blocks * Q2K_BLOCK_SIZE);

    let mut output = vec![0.0f32; n_elements];

    for block in 0..n_blocks {
        let x = &block_data[block * Q2K_BLOCK_SIZE..];
        let y = &mut output[block * BLOCK_ELEMENTS..];

        let d = fp16_to_f32(read_u16_le(x, 0));
        let dmin = fp16_to_f32(read_u16_le(x, 2));
        let scales = &x[4..20]; // 16 bytes of 4-bit scale/min pairs
        let qs_base = &x[20..84]; // 64 bytes of packed 2-bit values

        let mut is = 0usize;
        // Process two 128-element halves
        for n in (0..256).step_by(128) {
            let mut q_offset = 0usize;
            let mut shift = 0u8;

            for _j in 0..4 {
                // First 16 elements of this 32-element chunk
                let dl1 = d * (scales[is] & 0x0F) as f32;
                let ml1 = dmin * (scales[is] >> 4) as f32;
                for l in 0..16 {
                    let q_val = (qs_base[q_offset + l] >> shift) & 3;
                    y[n + l] = dl1 * q_val as f32 - ml1;
                }

                // Second 16 elements of this 32-element chunk
                let dl2 = d * (scales[is + 1] & 0x0F) as f32;
                let ml2 = dmin * (scales[is + 1] >> 4) as f32;
                for l in 0..16 {
                    let q_val = (qs_base[q_offset + 16 + l] >> shift) & 3;
                    y[n + 16 + l] = dl2 * q_val as f32 - ml2;
                }

                // Move to next 32-element output position
                shift += 2;
                if shift == 8 {
                    shift = 0;
                    q_offset += 32;
                }
                is += 2;
            }
        }
    }

    output
}

/// Extract 6-bit scale and min values for Q4_K sub-blocks.
///
/// Matches `get_scale_min_k4` from ggml-quants.c.
#[inline(always)]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Dequantize Q4_K block data to f32.
///
/// # Arguments
/// * `block_data` - Raw Q4_K block bytes (must be `n_blocks * 144` bytes)
/// * `n_elements` - Total output elements (must be `n_blocks * 256`)
///
/// # Returns
/// Vec<f32> of dequantized values
pub fn dequantize_q4k(block_data: &[u8], n_elements: usize) -> Vec<f32> {
    assert!(n_elements.is_multiple_of(BLOCK_ELEMENTS));
    let n_blocks = n_elements / BLOCK_ELEMENTS;
    assert!(block_data.len() >= n_blocks * Q4K_BLOCK_SIZE);

    let mut output = vec![0.0f32; n_elements];

    for block in 0..n_blocks {
        let x = &block_data[block * Q4K_BLOCK_SIZE..];
        let y = &mut output[block * BLOCK_ELEMENTS..];

        let d = fp16_to_f32(read_u16_le(x, 0));
        let dmin = fp16_to_f32(read_u16_le(x, 2));
        let scales = &x[4..16]; // 12 bytes of packed 6-bit scales/mins
        let qs = &x[16..144]; // 128 bytes of packed 4-bit values

        let mut is = 0usize;
        let mut out_idx = 0usize;

        for j in 0..4usize {
            let (sc0, m0) = get_scale_min_k4(is, scales);
            let d1 = d * sc0 as f32;
            let m1f = dmin * m0 as f32;

            let (sc1, m1) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc1 as f32;
            let m2f = dmin * m1 as f32;

            // Low nibbles → sub-block (is)
            for l in 0..32usize {
                y[out_idx] = d1 * (qs[j * 32 + l] & 0x0F) as f32 - m1f;
                out_idx += 1;
            }
            // High nibbles → sub-block (is+1)
            for l in 0..32usize {
                y[out_idx] = d2 * (qs[j * 32 + l] >> 4) as f32 - m2f;
                out_idx += 1;
            }

            is += 2;
        }
    }

    output
}

/// Dequantize Q5_K block data to f32.
///
/// Q5_K format: 176 bytes per 256 elements.
/// Layout: d (2B fp16) + dmin (2B fp16) + scales (12B packed 6-bit) + qh (32B high bits) + qs (128B low 4-bit).
///
/// 5-bit quantization: low 4 bits from qs + high 1 bit from qh.
/// qh layout: each byte qh[l] stores 8 high bits for element position l across all 8 sub-blocks.
/// Bit 2*j = high bit for low-nibble sub-block j, bit 2*j+1 = high bit for high-nibble sub-block j.
pub fn dequantize_q5k(block_data: &[u8], n_elements: usize) -> Vec<f32> {
    assert!(n_elements.is_multiple_of(BLOCK_ELEMENTS));
    let n_blocks = n_elements / BLOCK_ELEMENTS;
    assert!(block_data.len() >= n_blocks * Q5K_BLOCK_SIZE);

    let mut output = vec![0.0f32; n_elements];

    for block in 0..n_blocks {
        let x = &block_data[block * Q5K_BLOCK_SIZE..];
        let y = &mut output[block * BLOCK_ELEMENTS..];

        let d = fp16_to_f32(read_u16_le(x, 0));
        let dmin = fp16_to_f32(read_u16_le(x, 2));
        let scales = &x[4..16];   // 12 bytes of packed 6-bit scales/mins
        let qh = &x[16..48];      // 32 bytes: qh[l] has 8 high bits for position l
        let qs = &x[48..176];     // 128 bytes of low 4-bit values

        let mut is = 0usize;
        let mut out_idx = 0usize;
        let mut u1: u8 = 1;   // bit mask for low-nibble high bit (shifts <<2 per j)
        let mut u2: u8 = 2;   // bit mask for high-nibble high bit (shifts <<2 per j)

        for j in 0..4usize {
            let (sc0, m0) = get_scale_min_k4(is, scales);
            let d1 = d * sc0 as f32;
            let m1f = dmin * m0 as f32;

            let (sc1, m1) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc1 as f32;
            let m2f = dmin * m1 as f32;

            // Low nibbles + high bit → sub-block (is)
            for l in 0..32usize {
                let low4 = (qs[j * 32 + l] & 0x0F) as u32;
                let high = if qh[l] & u1 != 0 { 16u32 } else { 0u32 };
                y[out_idx] = d1 * (low4 + high) as f32 - m1f;
                out_idx += 1;
            }
            // High nibbles + high bit → sub-block (is+1)
            for l in 0..32usize {
                let low4 = ((qs[j * 32 + l] >> 4) & 0x0F) as u32;
                let high = if qh[l] & u2 != 0 { 16u32 } else { 0u32 };
                y[out_idx] = d2 * (low4 + high) as f32 - m2f;
                out_idx += 1;
            }

            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    output
}

/// Dequantize Q6_K block data to f32.
///
/// # Arguments
/// * `block_data` - Raw Q6_K block bytes (must be `n_blocks * 210` bytes)
/// * `n_elements` - Total output elements (must be `n_blocks * 256`)
///
/// # Returns
/// Vec<f32> of dequantized values
pub fn dequantize_q6k(block_data: &[u8], n_elements: usize) -> Vec<f32> {
    assert!(n_elements.is_multiple_of(BLOCK_ELEMENTS));
    let n_blocks = n_elements / BLOCK_ELEMENTS;
    assert!(block_data.len() >= n_blocks * Q6K_BLOCK_SIZE);

    let mut output = vec![0.0f32; n_elements];

    for block in 0..n_blocks {
        let x = &block_data[block * Q6K_BLOCK_SIZE..];
        let y = &mut output[block * BLOCK_ELEMENTS..];

        // Q6_K layout: ql[128] + qh[64] + scales[16] + d[2]
        let mut ql = &x[0..128];
        let mut qh = &x[128..192];
        let sc: &[i8] = unsafe {
            std::slice::from_raw_parts(x[192..208].as_ptr() as *const i8, 16)
        };
        let d = fp16_to_f32(read_u16_le(x, 208));

        // Process two 128-element chunks
        let mut y_offset = 0usize;
        for n in (0..256).step_by(128) {
            for l in 0..32usize {
                let is = n / 16 + l / 16; // scale index

                // Reconstruct 6-bit values from ql (lower 4 bits) + qh (upper 2 bits)
                let q1 = ((ql[l] & 0x0F) as i32 | ((qh[l] & 3) as i32) << 4) - 32;
                let q2 = ((ql[l + 32] & 0x0F) as i32 | (((qh[l] >> 2) & 3) as i32) << 4) - 32;
                let q3 = ((ql[l] >> 4) as i32 | (((qh[l] >> 4) & 3) as i32) << 4) - 32;
                let q4 = ((ql[l + 32] >> 4) as i32 | (((qh[l] >> 6) & 3) as i32) << 4) - 32;

                y[y_offset + l] = d * sc[is] as f32 * q1 as f32;
                y[y_offset + l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                y[y_offset + l + 64] = d * sc[is + 4] as f32 * q3 as f32;
                y[y_offset + l + 96] = d * sc[is + 6] as f32 * q4 as f32;
            }
            y_offset += 128;
            ql = &ql[64..]; // Advance ql by 64
            qh = &qh[32..]; // Advance qh by 32
        }
    }

    output
}

/// Dequantize Q4_0 block data to f32.
///
/// Q4_0 format: 18 bytes per 32 elements.
/// Layout: d (2B fp16) + qs (16B packed 4-bit unsigned).
/// Dequantization: output[i] = d * ((qs[i] & 0xF) - 8)
pub fn dequantize_q40(block_data: &[u8], n_elements: usize) -> Vec<f32> {
    assert!(n_elements.is_multiple_of(LEGACY_BLOCK_ELEMENTS));
    let n_blocks = n_elements / LEGACY_BLOCK_ELEMENTS;
    assert!(block_data.len() >= n_blocks * Q4_0_BLOCK_SIZE);

    let mut output = vec![0.0f32; n_elements];

    for block in 0..n_blocks {
        let x = &block_data[block * Q4_0_BLOCK_SIZE..];
        let y = &mut output[block * LEGACY_BLOCK_ELEMENTS..];

        let d = fp16_to_f32(read_u16_le(x, 0));
        let qs = &x[2..18];

        for j in 0..16 {
            let lo = (qs[j] & 0x0F) as i32 - 8;
            let hi = (qs[j] >> 4) as i32 - 8;
            y[j] = d * lo as f32;
            y[j + 16] = d * hi as f32;
        }
    }

    output
}

/// Dequantize Q4_1 block data to f32.
///
/// Q4_1 format: 20 bytes per 32 elements.
/// Layout: d (2B fp16) + min (2B fp16) + qs (16B packed 4-bit unsigned).
/// Dequantization: output[i] = d * qs[i] + min
pub fn dequantize_q41(block_data: &[u8], n_elements: usize) -> Vec<f32> {
    assert!(n_elements.is_multiple_of(LEGACY_BLOCK_ELEMENTS));
    let n_blocks = n_elements / LEGACY_BLOCK_ELEMENTS;
    assert!(block_data.len() >= n_blocks * Q4_1_BLOCK_SIZE);

    let mut output = vec![0.0f32; n_elements];

    for block in 0..n_blocks {
        let x = &block_data[block * Q4_1_BLOCK_SIZE..];
        let y = &mut output[block * LEGACY_BLOCK_ELEMENTS..];

        let d = fp16_to_f32(read_u16_le(x, 0));
        let min = fp16_to_f32(read_u16_le(x, 2));
        let qs = &x[4..20];

        for j in 0..16 {
            y[j] = d * (qs[j] & 0x0F) as f32 + min;
            y[j + 16] = d * (qs[j] >> 4) as f32 + min;
        }
    }

    output
}

/// Dequantize Q5_0 block data to f32.
///
/// Q5_0 format: 22 bytes per 32 elements.
/// Layout: d (2B fp16) + qh (4B high bits) + qs (16B low 4-bit).
/// Dequantization: 5-bit = low 4 bits + 1 high bit, output = d * (q5 - 16)
pub fn dequantize_q50(block_data: &[u8], n_elements: usize) -> Vec<f32> {
    assert!(n_elements.is_multiple_of(LEGACY_BLOCK_ELEMENTS));
    let n_blocks = n_elements / LEGACY_BLOCK_ELEMENTS;
    assert!(block_data.len() >= n_blocks * Q5_0_BLOCK_SIZE);

    let mut output = vec![0.0f32; n_elements];

    for block in 0..n_blocks {
        let x = &block_data[block * Q5_0_BLOCK_SIZE..];
        let y = &mut output[block * LEGACY_BLOCK_ELEMENTS..];

        let d = fp16_to_f32(read_u16_le(x, 0));
        let qh = u32::from_le_bytes([x[2], x[3], x[4], x[5]]);
        let qs = &x[6..22];

        for j in 0..16 {
            let xh_0 = ((qh >> j) & 1) as u8;
            let xh_1 = ((qh >> (j + 16)) & 1) as u8;
            let lo = ((qs[j] & 0x0F) | (xh_0 << 4)) as i32 - 16;
            let hi = ((qs[j] >> 4) | (xh_1 << 4)) as i32 - 16;
            y[j] = d * lo as f32;
            y[j + 16] = d * hi as f32;
        }
    }

    output
}

/// Dequantize Q5_1 block data to f32.
///
/// Q5_1 format: 24 bytes per 32 elements.
/// Layout: d (2B fp16) + min (2B fp16) + qh (4B) + qs (16B).
/// Dequantization: 5-bit = low 4 + 1 high, output = d * q5 + min
pub fn dequantize_q51(block_data: &[u8], n_elements: usize) -> Vec<f32> {
    assert!(n_elements.is_multiple_of(LEGACY_BLOCK_ELEMENTS));
    let n_blocks = n_elements / LEGACY_BLOCK_ELEMENTS;
    assert!(block_data.len() >= n_blocks * Q5_1_BLOCK_SIZE);

    let mut output = vec![0.0f32; n_elements];

    for block in 0..n_blocks {
        let x = &block_data[block * Q5_1_BLOCK_SIZE..];
        let y = &mut output[block * LEGACY_BLOCK_ELEMENTS..];

        let d = fp16_to_f32(read_u16_le(x, 0));
        let min = fp16_to_f32(read_u16_le(x, 2));
        let qh = u32::from_le_bytes([x[4], x[5], x[6], x[7]]);
        let qs = &x[8..24];

        for j in 0..16 {
            let xh_0 = ((qh >> j) & 1) as u8;
            let xh_1 = ((qh >> (j + 16)) & 1) as u8;
            let lo = (qs[j] & 0x0F) | (xh_0 << 4);
            let hi = (qs[j] >> 4) | (xh_1 << 4);
            y[j] = d * lo as f32 + min;
            y[j + 16] = d * hi as f32 + min;
        }
    }

    output
}

/// Dequantize Q8_0 block data to f32.
///
/// Q8_0 format: 34 bytes per 32 elements.
/// Layout: d (2B fp16 scale) + qs (32B signed int8 values).
/// Dequantization: output[i] = d * qs[i]
pub fn dequantize_q80(block_data: &[u8], n_elements: usize) -> Vec<f32> {
    assert!(n_elements.is_multiple_of(LEGACY_BLOCK_ELEMENTS));
    let n_blocks = n_elements / LEGACY_BLOCK_ELEMENTS;
    assert!(block_data.len() >= n_blocks * Q8_0_BLOCK_SIZE);

    let mut output = vec![0.0f32; n_elements];

    for block in 0..n_blocks {
        let x = &block_data[block * Q8_0_BLOCK_SIZE..];
        let y = &mut output[block * LEGACY_BLOCK_ELEMENTS..];

        let d = fp16_to_f32(read_u16_le(x, 0));
        let qs = &x[2..34];

        for i in 0..LEGACY_BLOCK_ELEMENTS {
            y[i] = d * (qs[i] as i8) as f32;
        }
    }

    output
}

/// Dequantize Q8_1 block data to f32.
///
/// Q8_1 format: 36 bytes per 32 elements.
/// Layout: d (2B fp16) + s (2B fp16, sum for dot product) + qs (32B int8).
/// Dequantization: output[i] = d * qs[i]
pub fn dequantize_q81(block_data: &[u8], n_elements: usize) -> Vec<f32> {
    assert!(n_elements.is_multiple_of(LEGACY_BLOCK_ELEMENTS));
    let n_blocks = n_elements / LEGACY_BLOCK_ELEMENTS;
    assert!(block_data.len() >= n_blocks * Q8_1_BLOCK_SIZE);

    let mut output = vec![0.0f32; n_elements];

    for block in 0..n_blocks {
        let x = &block_data[block * Q8_1_BLOCK_SIZE..];
        let y = &mut output[block * LEGACY_BLOCK_ELEMENTS..];

        let d = fp16_to_f32(read_u16_le(x, 0));
        // skip s at offset 2..4 (only used for dot product)
        let qs = &x[4..36];

        for i in 0..LEGACY_BLOCK_ELEMENTS {
            y[i] = d * (qs[i] as i8) as f32;
        }
    }

    output
}

/// Convert f16 (IEEE 754) raw bytes to f32.
pub fn convert_f16_to_f32(data: &[u8], n_elements: usize) -> Vec<f32> {
    assert!(data.len() >= n_elements * 2);
    let mut output = vec![0.0f32; n_elements];
    for (i, val) in output.iter_mut().enumerate().take(n_elements) {
        *val = fp16_to_f32(read_u16_le(data, i * 2));
    }
    output
}


/// MXFP4 block size: 17 bytes per 32 elements.
/// Layout: e (1B E8M0 shared exponent) + qs (16B packed 4-bit).
pub const MXFP4_BLOCK_ELEMENTS: usize = 32;
pub const MXFP4_BLOCK_SIZE: usize = 17; // 1 + 32/2

/// MXFP4 lookup table: 4-bit nibble -> E2M1 value doubled (signed).
/// Matches llama.cpp kvalues_mxfp4: {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}.
/// These are multiplied by e8m0_to_f32_half (which is e8m0/2) to cancel the doubling.
const MXFP4_VALUES: [f32; 16] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0,
    -0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0,
];

/// Convert E8M0 exponent byte to f32 scale (half).
/// E8M0 is an 8-bit unsigned exponent-only format.
/// Result = 2^(e-128) for e >= 2, with denormal handling for e < 2.
#[inline]
fn e8m0_to_f32_half(e: u8) -> f32 {
    if e < 2 {
        // Denormal: 2^(-128) << e
        f32::from_bits(0x00200000u32 << e)
    } else {
        // Normalized: 2^(e-128) = exponent field = (e-1) in IEEE 754
        f32::from_bits(((e as u32).wrapping_sub(1)) << 23)
    }
}

/// Dequantize MXFP4 block data to f32.
///
/// MXFP4 format: 17 bytes per 32 elements.
/// Layout: e (1B E8M0) + qs (16B packed nibbles).
/// Low nibble qs[j] & 0x0F -> element [j], high nibble qs[j] >> 4 -> element [j+16].
/// Dequantization: output[i] = MXFP4_VALUES[nibble] * e8m0_to_f32_half(e)
pub fn dequantize_mxfp4(block_data: &[u8], n_elements: usize) -> Vec<f32> {
    assert!(n_elements.is_multiple_of(MXFP4_BLOCK_ELEMENTS));
    let n_blocks = n_elements / MXFP4_BLOCK_ELEMENTS;
    assert!(block_data.len() >= n_blocks * MXFP4_BLOCK_SIZE);

    let mut output = vec![0.0f32; n_elements];

    for block in 0..n_blocks {
        let x = &block_data[block * MXFP4_BLOCK_SIZE..];
        let y = &mut output[block * MXFP4_BLOCK_ELEMENTS..];

        let d = e8m0_to_f32_half(x[0]);
        let qs = &x[1..17];

        for j in 0..16 {
            let lo = MXFP4_VALUES[(qs[j] & 0x0F) as usize];
            let hi = MXFP4_VALUES[(qs[j] >> 4) as usize];
            y[j] = lo * d;
            y[j + 16] = hi * d;
        }
    }

    output
}

/// Fused MXFP4 matrix-vector multiply: output = W @ x (no F32 intermediate).
///
/// W is [out_features, in_features] stored in MXFP4 format (row-major).
/// x is [in_features] as f32.
/// Returns [out_features] as f32.
///
/// This fuses dequantization into the dot product: each MXFP4 block
/// (32 elements = 17 bytes) is decoded on the fly during accumulation,
/// avoiding the full F32 materialization of the weight matrix.
/// Output rows are computed in parallel across available CPU cores.
pub fn mxfp4_matvec_mul(
    weight_data: &[u8],
    input: &[f32],
    out_features: usize,
    in_features: usize,
) -> Vec<f32> {
    debug_assert!(in_features.is_multiple_of(MXFP4_BLOCK_ELEMENTS));
    let blocks_per_row = in_features / MXFP4_BLOCK_ELEMENTS;
    let row_bytes = blocks_per_row * MXFP4_BLOCK_SIZE;

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get().min(out_features))
        .unwrap_or(1);
    let chunk_size = out_features.div_ceil(num_threads);

    let mut output = vec![0.0f32; out_features];

    std::thread::scope(|s| {
        let handles: Vec<_> = output
            .chunks_mut(chunk_size)
            .enumerate()
            .map(|(chunk_idx, out_chunk)| {
                let start_row = chunk_idx * chunk_size;
                s.spawn(move || {
                    for (local_i, out_val) in out_chunk.iter_mut().enumerate() {
                        let i = start_row + local_i;
                        let row_start = i * row_bytes;
                        let mut acc = 0.0f32;

                        for b in 0..blocks_per_row {
                            let block_offset = row_start + b * MXFP4_BLOCK_SIZE;
                            let d = e8m0_to_f32_half(weight_data[block_offset]);
                            let qs = &weight_data[block_offset + 1..block_offset + 17];
                            let input_offset = b * MXFP4_BLOCK_ELEMENTS;

                            for j in 0..16 {
                                let lo = MXFP4_VALUES[(qs[j] & 0x0F) as usize];
                                let hi = MXFP4_VALUES[(qs[j] >> 4) as usize];
                                acc += lo * d * input[input_offset + j];
                                acc += hi * d * input[input_offset + j + 16];
                            }
                        }

                        *out_val = acc;
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    });

    output
}

/// Fused MXFP4 batched matmul: output = input @ W^T (no F32 intermediate).
///
/// W is [out_features, in_features] stored in MXFP4 format (row-major).
/// input is [num_tokens, in_features] as flat f32.
/// Returns [num_tokens, out_features] as flat f32.
///
/// Delegates to `mxfp4_matvec_mul` per token (each call is thread-parallel).
/// For single-token decode, prefer `mxfp4_matvec_mul` directly.
pub fn mxfp4_matmul(
    weight_data: &[u8],
    input: &[f32],
    num_tokens: usize,
    out_features: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(num_tokens * out_features);

    for t in 0..num_tokens {
        let input_row = &input[t * in_features..(t + 1) * in_features];
        let row_out = mxfp4_matvec_mul(weight_data, input_row, out_features, in_features);
        output.extend_from_slice(&row_out);
    }

    output
}

/// Fused Q5_0 matrix-vector multiply (CPU fallback for Metal CustomOp1).
///
/// Computes `output = W @ input` where W is [out_features, in_features] in Q5_0 format.
/// Fuses dequantization with dot product to avoid materializing the full F32 weight matrix.
pub fn dequant_matvec_q5_0(
    weight_data: &[u8],
    input: &[f32],
    out_features: usize,
    in_features: usize,
) -> Vec<f32> {
    debug_assert!(in_features.is_multiple_of(LEGACY_BLOCK_ELEMENTS));
    let blocks_per_row = in_features / LEGACY_BLOCK_ELEMENTS;
    let row_bytes = blocks_per_row * Q5_0_BLOCK_SIZE;
    let mut output = vec![0.0f32; out_features];

    for i in 0..out_features {
        let row_start = i * row_bytes;
        let mut acc = 0.0f32;

        for b in 0..blocks_per_row {
            let block_offset = row_start + b * Q5_0_BLOCK_SIZE;
            let x = &weight_data[block_offset..];
            let y = &input[b * LEGACY_BLOCK_ELEMENTS..];

            let d = fp16_to_f32(read_u16_le(x, 0));
            let qh = u32::from_le_bytes([x[2], x[3], x[4], x[5]]);
            let qs = &x[6..22];

            let mut block_acc = 0.0f32;
            for j in 0..16 {
                let xh_0 = ((qh >> j) & 1) as u8;
                let xh_1 = ((qh >> (j + 16)) & 1) as u8;
                let lo = ((qs[j] & 0x0F) | (xh_0 << 4)) as i32 - 16;
                let hi = ((qs[j] >> 4) | (xh_1 << 4)) as i32 - 16;
                block_acc += y[j] * lo as f32;
                block_acc += y[j + 16] * hi as f32;
            }
            acc += d * block_acc;
        }
        output[i] = acc;
    }
    output
}

/// Fused Q8_0 matrix-vector multiply (CPU fallback for Metal CustomOp1).
///
/// Computes `output = W @ input` where W is [out_features, in_features] in Q8_0 format.
pub fn dequant_matvec_q8_0(
    weight_data: &[u8],
    input: &[f32],
    out_features: usize,
    in_features: usize,
) -> Vec<f32> {
    debug_assert!(in_features.is_multiple_of(LEGACY_BLOCK_ELEMENTS));
    let blocks_per_row = in_features / LEGACY_BLOCK_ELEMENTS;
    let row_bytes = blocks_per_row * Q8_0_BLOCK_SIZE;
    let mut output = vec![0.0f32; out_features];

    for i in 0..out_features {
        let row_start = i * row_bytes;
        let mut acc = 0.0f32;

        for b in 0..blocks_per_row {
            let block_offset = row_start + b * Q8_0_BLOCK_SIZE;
            let x = &weight_data[block_offset..];
            let y = &input[b * LEGACY_BLOCK_ELEMENTS..];

            let d = fp16_to_f32(read_u16_le(x, 0));
            let qs = &x[2..34];

            let mut block_acc = 0.0f32;
            for j in 0..32 {
                block_acc += y[j] * (qs[j] as i8) as f32;
            }
            acc += d * block_acc;
        }
        output[i] = acc;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp16_conversion() {
        // 1.0 in fp16 = 0x3C00
        assert_eq!(fp16_to_f32(0x3C00), 1.0);
        // 0.0 in fp16 = 0x0000
        assert_eq!(fp16_to_f32(0x0000), 0.0);
        // -1.0 in fp16 = 0xBC00
        assert_eq!(fp16_to_f32(0xBC00), -1.0);
    }

    #[test]
    fn test_q4k_single_block() {
        // Create a minimal Q4_K block with known values
        let mut block = vec![0u8; Q4K_BLOCK_SIZE];
        // d = 1.0 (fp16: 0x3C00), dmin = 0.0
        block[0] = 0x00;
        block[1] = 0x3C;
        block[2] = 0x00;
        block[3] = 0x00;
        // scales all = 1 (packed in first 4 bytes of scales field)
        block[4] = 1;
        block[5] = 1;
        block[6] = 1;
        block[7] = 1;

        let result = dequantize_q4k(&block, BLOCK_ELEMENTS);
        assert_eq!(result.len(), BLOCK_ELEMENTS);
        // First element: d * sc * (qs[0] & 0x0F) - dmin * m = 1.0 * 1 * 0 - 0 = 0
        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn test_q6k_single_block() {
        let block = vec![0u8; Q6K_BLOCK_SIZE];
        let result = dequantize_q6k(&block, BLOCK_ELEMENTS);
        assert_eq!(result.len(), BLOCK_ELEMENTS);
    }
}
