//! Custom Metal kernel dispatch for MXFP4 matrix-vector multiply.
//!
//! This module loads and dispatches the MXFP4 Metal compute shader ported from llama.cpp.
//! Used for GPU Resident inference of MXFP4-quantized models (e.g., GPT-OSS-20B).
//!
//! ## Pipeline Integration (v2)
//!
//! The `Mxfp4MatmulOp` implements candle's `CustomOp1` trait, encoding MXFP4 kernels
//! onto candle's shared command buffer. This eliminates per-expert GPU synchronization:
//! - Old: 312 `wait_until_completed()` per decode step (~780ms overhead)
//! - New: 0 explicit syncs; candle batches all kernels automatically
//!
//! The output stays on GPU as a proper candle Tensor (StorageModePrivate), enabling
//! zero-copy flow: MXFP4 matmul → SiLU → mul → MXFP4 matmul → add, all pipelined.

#[cfg(feature = "metal")]
mod inner {
    use candle_core::{Device, Result};
    use metal::{Buffer, CompileOptions, ComputePipelineState, MTLSize};
    use std::ffi::c_void;
    use std::sync::{Arc, Mutex};

    /// MXFP4 block: 32 elements, 17 bytes (1 E8M0 + 16 packed nibbles).
    pub const MXFP4_BLOCK_ELEMENTS: usize = 32;
    pub const MXFP4_BLOCK_SIZE: usize = 17;

    /// Q5_0 block: 32 elements, 22 bytes (d: F16(2) + qh: u32(4) + qs: 16 nibbles(16)).
    pub const Q5_0_BLOCK_ELEMENTS: usize = 32;
    pub const Q5_0_BLOCK_SIZE: usize = 22;

    /// Q8_0 block: 32 elements, 34 bytes (d: F16(2) + qs: 32 int8(32)).
    pub const Q8_0_BLOCK_ELEMENTS: usize = 32;
    pub const Q8_0_BLOCK_SIZE: usize = 34;

    /// Kernel constants matching the Metal shader.
    const NR0: usize = 2;
    const NSG: usize = 2;
    const SHMEM_SIZE: usize = 32 * std::mem::size_of::<f32>(); // 128 bytes

    /// The Metal shader source, embedded at compile time.
    const MXFP4_SHADER_SOURCE: &str = include_str!("mxfp4.metal");

    /// Quantized attention (Q5_0/Q8_0) shader source.
    const QUANTIZED_ATTN_SHADER_SOURCE: &str = include_str!("quantized_attn.metal");

    /// Fused ops (RMSNorm, RoPE) shader source.
    const FUSED_OPS_SHADER_SOURCE: &str = include_str!("fused_ops.metal");

    /// Args struct matching the Metal shader's MxfpMvArgs.
    /// Must be repr(C) to match the Metal struct layout exactly.
    #[repr(C)]
    struct MxfpMvArgs {
        out_features: i32,
        in_features: i32,
        weight_stride: u64,
    }

    /// Maximum number of experts in a single batched dispatch.
    pub const MAX_BATCH_EXPERTS: usize = 8;

    /// Args struct matching the quantized_attn.metal QuantizedMvArgs.
    /// Same layout as MxfpMvArgs but used for Q5_0/Q8_0 kernels.
    #[repr(C)]
    struct QuantizedMvArgs {
        out_features: i32,
        in_features: i32,
        weight_stride: u64,
    }

    /// Quantized attention weight type (Q5_0 or Q8_0).
    #[derive(Clone, Copy, Debug)]
    pub enum QuantizedAttnType {
        Q5_0,
        Q8_0,
    }

    impl QuantizedAttnType {
        pub fn block_size(&self) -> usize {
            match self {
                Self::Q5_0 => Q5_0_BLOCK_SIZE,
                Self::Q8_0 => Q8_0_BLOCK_SIZE,
            }
        }
    }

    /// Cached pipeline states for Metal kernels.
    struct PipelineCache {
        /// Single-expert matvec kernel (MXFP4).
        matvec: Option<ComputePipelineState>,
        /// Fused gate+up+swiglu kernel (8 dispatches → 2).
        fused_gate_up_swiglu: Option<ComputePipelineState>,
        /// Fused down+accum kernel.
        fused_down_accum: Option<ComputePipelineState>,
        /// Q5_0 attention matvec kernel.
        q5_0_matvec: Option<ComputePipelineState>,
        /// Q8_0 attention matvec kernel.
        q8_0_matvec: Option<ComputePipelineState>,
        /// Fused RMSNorm kernel.
        rms_norm: Option<ComputePipelineState>,
        /// Fused RoPE kernel.
        rope: Option<ComputePipelineState>,
        /// GPU-side softmax+topk routing kernel.
        softmax_topk: Option<ComputePipelineState>,
    }

    static PIPELINES: Mutex<PipelineCache> = Mutex::new(PipelineCache {
        matvec: None,
        fused_gate_up_swiglu: None,
        fused_down_accum: None,
        q5_0_matvec: None,
        q8_0_matvec: None,
        rms_norm: None,
        rope: None,
        softmax_topk: None,
    });

    // Q5_0/Q8_0 pipeline entries are in the unified PipelineCache above.

    /// Compile the MXFP4 shader library (shared across all kernels).
    fn compile_library(device: &metal::DeviceRef) -> Result<metal::Library> {
        let options = CompileOptions::new();
        device
            .new_library_with_source(MXFP4_SHADER_SOURCE, &options)
            .map_err(|e| candle_core::Error::Msg(format!("MXFP4 Metal shader compile error: {e}")))
    }

    /// Get or create a compute pipeline for a named kernel function.
    fn get_or_create_pipeline(
        device: &metal::DeviceRef,
        cache: &mut Option<ComputePipelineState>,
        function_name: &str,
    ) -> Result<ComputePipelineState> {
        if let Some(pipeline) = cache.as_ref() {
            return Ok(pipeline.clone());
        }
        let library = compile_library(device)?;
        let function = library
            .get_function(function_name, None)
            .map_err(|e| candle_core::Error::Msg(format!("Metal function '{}' not found: {e}", function_name)))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| candle_core::Error::Msg(format!("Metal pipeline '{}' creation failed: {e}", function_name)))?;
        *cache = Some(pipeline.clone());
        Ok(pipeline)
    }

    /// Get or create the compute pipeline for the single-expert MXFP4 matvec kernel.
    fn get_pipeline(device: &metal::DeviceRef) -> Result<ComputePipelineState> {
        let mut guard = PIPELINES.lock().map_err(|e| {
            candle_core::Error::Msg(format!("MXFP4 pipeline lock poisoned: {e}"))
        })?;
        get_or_create_pipeline(device, &mut guard.matvec, "mxfp4_matvec_f32")
    }

    /// Get or create the fused gate+up+swiglu pipeline.
    fn get_fused_gate_up_swiglu_pipeline(device: &metal::DeviceRef) -> Result<ComputePipelineState> {
        let mut guard = PIPELINES.lock().map_err(|e| {
            candle_core::Error::Msg(format!("MXFP4 pipeline lock poisoned: {e}"))
        })?;
        get_or_create_pipeline(device, &mut guard.fused_gate_up_swiglu, "mxfp4_fused_gate_up_swiglu_f32")
    }

    /// Get or create the fused down+accum pipeline.
    fn get_fused_down_accum_pipeline(device: &metal::DeviceRef) -> Result<ComputePipelineState> {
        let mut guard = PIPELINES.lock().map_err(|e| {
            candle_core::Error::Msg(format!("MXFP4 pipeline lock poisoned: {e}"))
        })?;
        get_or_create_pipeline(device, &mut guard.fused_down_accum, "mxfp4_fused_down_accum_f32")
    }

    /// Compile the quantized attention shader library (Q5_0, Q8_0 kernels).
    fn compile_quantized_attn_library(device: &metal::DeviceRef) -> Result<metal::Library> {
        let options = CompileOptions::new();
        device
            .new_library_with_source(QUANTIZED_ATTN_SHADER_SOURCE, &options)
            .map_err(|e| candle_core::Error::Msg(format!("Quantized attention Metal shader compile error: {e}")))
    }

    /// Get or create a compute pipeline for a quantized attention kernel.
    fn get_or_create_quantized_attn_pipeline(
        device: &metal::DeviceRef,
        cache: &mut Option<ComputePipelineState>,
        function_name: &str,
    ) -> Result<ComputePipelineState> {
        if let Some(pipeline) = cache.as_ref() {
            return Ok(pipeline.clone());
        }
        let library = compile_quantized_attn_library(device)?;
        let function = library
            .get_function(function_name, None)
            .map_err(|e| candle_core::Error::Msg(format!("Metal function '{}' not found: {e}", function_name)))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| candle_core::Error::Msg(format!("Metal pipeline '{}' creation failed: {e}", function_name)))?;
        *cache = Some(pipeline.clone());
        Ok(pipeline)
    }

    /// Get or create the Q5_0 matvec pipeline.
    fn get_q5_0_pipeline(device: &metal::DeviceRef) -> Result<ComputePipelineState> {
        let mut guard = PIPELINES.lock().map_err(|e| {
            candle_core::Error::Msg(format!("Pipeline lock poisoned: {e}"))
        })?;
        get_or_create_quantized_attn_pipeline(device, &mut guard.q5_0_matvec, "q5_0_matvec_f32")
    }

    /// Get or create the Q8_0 matvec pipeline.
    fn get_q8_0_pipeline(device: &metal::DeviceRef) -> Result<ComputePipelineState> {
        let mut guard = PIPELINES.lock().map_err(|e| {
            candle_core::Error::Msg(format!("Pipeline lock poisoned: {e}"))
        })?;
        get_or_create_quantized_attn_pipeline(device, &mut guard.q8_0_matvec, "q8_0_matvec_f32")
    }

    // =========================================================================
    // Fused ops (RMSNorm, RoPE) pipeline management
    // =========================================================================

    /// Compile the fused ops shader library.
    fn compile_fused_ops_library(device: &metal::DeviceRef) -> Result<metal::Library> {
        let options = CompileOptions::new();
        device
            .new_library_with_source(FUSED_OPS_SHADER_SOURCE, &options)
            .map_err(|e| candle_core::Error::Msg(format!("Fused ops Metal shader compile error: {e}")))
    }

    /// Get or create a compute pipeline from the fused ops library.
    fn get_or_create_fused_pipeline(
        device: &metal::DeviceRef,
        cache: &mut Option<ComputePipelineState>,
        function_name: &str,
    ) -> Result<ComputePipelineState> {
        if let Some(pipeline) = cache.as_ref() {
            return Ok(pipeline.clone());
        }
        let library = compile_fused_ops_library(device)?;
        let function = library
            .get_function(function_name, None)
            .map_err(|e| candle_core::Error::Msg(format!("Metal function '{}' not found: {e}", function_name)))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| candle_core::Error::Msg(format!("Metal pipeline '{}' creation failed: {e}", function_name)))?;
        *cache = Some(pipeline.clone());
        Ok(pipeline)
    }

    fn get_rms_norm_pipeline(device: &metal::DeviceRef) -> Result<ComputePipelineState> {
        let mut guard = PIPELINES.lock().map_err(|e| {
            candle_core::Error::Msg(format!("Pipeline lock poisoned: {e}"))
        })?;
        get_or_create_fused_pipeline(device, &mut guard.rms_norm, "rms_norm_kernel")
    }

    fn get_rope_pipeline(device: &metal::DeviceRef) -> Result<ComputePipelineState> {
        let mut guard = PIPELINES.lock().map_err(|e| {
            candle_core::Error::Msg(format!("Pipeline lock poisoned: {e}"))
        })?;
        get_or_create_fused_pipeline(device, &mut guard.rope, "rope_kernel")
    }

    fn get_softmax_topk_pipeline(device: &metal::DeviceRef) -> Result<ComputePipelineState> {
        let mut guard = PIPELINES.lock().map_err(|e| {
            candle_core::Error::Msg(format!("Pipeline lock poisoned: {e}"))
        })?;
        get_or_create_fused_pipeline(device, &mut guard.softmax_topk, "softmax_topk_kernel")
    }

    /// Args struct matching the Metal shader's RmsNormArgs.
    #[repr(C)]
    struct RmsNormArgs {
        hidden_dim: i32,
        eps: f32,
    }

    /// Args struct matching the Metal shader's RopeArgs.
    #[repr(C)]
    struct RopeArgs {
        num_heads: i32,
        head_dim: i32,
    }

    /// Fused RMSNorm on Metal GPU.
    ///
    /// Replaces 7 candle ops (to_dtype, sqr, mean_keepdim, add, sqrt, recip, broadcast_mul×2)
    /// with a single Metal kernel dispatch. Encodes onto candle's shared command buffer.
    ///
    /// Input: x [*, hidden_dim] (F32, on Metal)
    /// Weight: [hidden_dim] (F32, on Metal)
    /// Output: [*, hidden_dim] (F32, on Metal)
    pub fn fused_rms_norm_metal(
        device: &Device,
        input: &candle_core::Tensor,
        weight: &candle_core::Tensor,
        eps: f32,
    ) -> Result<candle_core::Tensor> {
        let metal_device = match device {
            Device::Metal(m) => m,
            _ => return Err(candle_core::Error::Msg("fused_rms_norm_metal requires Metal device".into())),
        };

        let hidden_dim = input.dim(candle_core::D::Minus1)?;
        let elem_count = input.elem_count();

        // How many "rows" to normalize (batch dimension)
        let num_rows = elem_count / hidden_dim;

        let pipeline = get_rms_norm_pipeline(metal_device.device())?;

        // Extract input Metal buffer
        let (input_storage, input_layout) = input.storage_and_layout();
        let input_buffer = match &*input_storage {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => return Err(candle_core::Error::Msg("RMSNorm input must be on Metal device".into())),
        };
        let input_offset = (input_layout.start_offset() * std::mem::size_of::<f32>()) as u64;

        // Extract weight Metal buffer
        let (weight_storage, weight_layout) = weight.storage_and_layout();
        let weight_buffer = match &*weight_storage {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => return Err(candle_core::Error::Msg("RMSNorm weight must be on Metal device".into())),
        };
        let weight_offset = (weight_layout.start_offset() * std::mem::size_of::<f32>()) as u64;

        // Allocate output buffer
        let output_buffer = metal_device.new_buffer(
            elem_count,
            candle_core::DType::F32,
            "rms_norm_output",
        )?;

        let args = RmsNormArgs {
            hidden_dim: hidden_dim as i32,
            eps,
        };

        let threads_per_group = 256u64;

        // Single encoder, one threadgroup per row (each threadgroup independently normalizes one row)
        let cb = metal_device.command_buffer()?;
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(input_buffer), input_offset);
        encoder.set_buffer(1, Some(weight_buffer), weight_offset);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<RmsNormArgs>() as u64,
            &args as *const RmsNormArgs as *const c_void,
        );

        let grid_size = MTLSize::new(num_rows as u64, 1, 1);
        let group_size = MTLSize::new(threads_per_group, 1, 1);
        encoder.dispatch_thread_groups(grid_size, group_size);
        encoder.end_encoding();

        // Drop storage guards before creating output tensor
        drop(input_storage);
        drop(weight_storage);

        // Wrap output buffer as a candle Tensor via BufferToTensorOp
        let output_op = BufferToTensorOp {
            source_buffer: output_buffer,
            elem_count,
        };
        let dummy = candle_core::Tensor::zeros(1, candle_core::DType::F32, device)?;
        let output = dummy.apply_op1_no_bwd(&output_op)?;

        // Reshape to match input shape
        output.reshape(input.shape())
    }

    /// Fused RoPE (rotary position embedding) on Metal GPU.
    ///
    /// Replaces ~12 candle ops (narrow×4, broadcast_mul×4, broadcast_sub/add×2, cat×2)
    /// with a single Metal kernel dispatch per tensor. Encodes onto candle's shared
    /// command buffer.
    ///
    /// Input: x [num_heads, head_dim] flattened (single-token decode: [num_heads * head_dim])
    /// Cos: [half_dim] (already sliced to current position)
    /// Sin: [half_dim]
    /// Output: [num_heads, head_dim] flattened
    pub fn fused_rope_metal(
        device: &Device,
        x: &candle_core::Tensor,
        cos: &candle_core::Tensor,
        sin: &candle_core::Tensor,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<candle_core::Tensor> {
        let metal_device = match device {
            Device::Metal(m) => m,
            _ => return Err(candle_core::Error::Msg("fused_rope_metal requires Metal device".into())),
        };

        let half_dim = head_dim / 2;
        let total_pairs = num_heads * half_dim;
        let elem_count = num_heads * head_dim;

        let pipeline = get_rope_pipeline(metal_device.device())?;

        // Extract Metal buffers
        let (x_storage, x_layout) = x.storage_and_layout();
        let x_buffer = match &*x_storage {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => return Err(candle_core::Error::Msg("RoPE input must be on Metal device".into())),
        };
        let x_offset = (x_layout.start_offset() * std::mem::size_of::<f32>()) as u64;

        let (cos_storage, cos_layout) = cos.storage_and_layout();
        let cos_buffer = match &*cos_storage {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => return Err(candle_core::Error::Msg("RoPE cos must be on Metal device".into())),
        };
        let cos_offset = (cos_layout.start_offset() * std::mem::size_of::<f32>()) as u64;

        let (sin_storage, sin_layout) = sin.storage_and_layout();
        let sin_buffer = match &*sin_storage {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => return Err(candle_core::Error::Msg("RoPE sin must be on Metal device".into())),
        };
        let sin_offset = (sin_layout.start_offset() * std::mem::size_of::<f32>()) as u64;

        // Allocate output buffer
        let output_buffer = metal_device.new_buffer(
            elem_count,
            candle_core::DType::F32,
            "rope_output",
        )?;

        let args = RopeArgs {
            num_heads: num_heads as i32,
            head_dim: head_dim as i32,
        };

        // Dispatch: one thread per (head, pair) combination
        let threads_per_group = 256u64;
        let num_groups = ((total_pairs as u64) + threads_per_group - 1) / threads_per_group;

        let cb = metal_device.command_buffer()?;
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(x_buffer), x_offset);
        encoder.set_buffer(1, Some(cos_buffer), cos_offset);
        encoder.set_buffer(2, Some(sin_buffer), sin_offset);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<RopeArgs>() as u64,
            &args as *const RopeArgs as *const c_void,
        );

        let grid_size = MTLSize::new(num_groups, 1, 1);
        let group_size = MTLSize::new(threads_per_group, 1, 1);
        encoder.dispatch_thread_groups(grid_size, group_size);
        encoder.end_encoding();

        // Drop storage guards before creating output tensor
        drop(x_storage);
        drop(cos_storage);
        drop(sin_storage);

        // Wrap output buffer as a candle Tensor
        let output_op = BufferToTensorOp {
            source_buffer: output_buffer,
            elem_count,
        };
        let dummy = candle_core::Tensor::zeros(1, candle_core::DType::F32, device)?;
        let output = dummy.apply_op1_no_bwd(&output_op)?;

        output.reshape(x.shape())
    }

    // =========================================================================

    /// Encode the MXFP4 matvec kernel dispatch onto a command buffer.
    ///
    /// This is the core dispatch logic shared by both sync and async paths.
    /// `weight_offset` is a byte offset into the weight buffer (for packed buffers).
    fn encode_mxfp4_matvec(
        command_buffer: &metal::CommandBufferRef,
        pipeline: &ComputePipelineState,
        weight_buffer: &Buffer,
        weight_offset: u64,
        input_buffer: &Buffer,
        input_offset: u64,
        output_buffer: &Buffer,
        out_features: usize,
        in_features: usize,
    ) {
        let blocks_per_row = in_features / MXFP4_BLOCK_ELEMENTS;
        let weight_stride = (blocks_per_row * MXFP4_BLOCK_SIZE) as u64;

        let args = MxfpMvArgs {
            out_features: out_features as i32,
            in_features: in_features as i32,
            weight_stride,
        };

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_bytes(
            0,
            std::mem::size_of::<MxfpMvArgs>() as u64,
            &args as *const MxfpMvArgs as *const c_void,
        );
        encoder.set_buffer(1, Some(weight_buffer), weight_offset);
        encoder.set_buffer(2, Some(input_buffer), input_offset);
        encoder.set_buffer(3, Some(output_buffer), 0);

        encoder.set_threadgroup_memory_length(0, SHMEM_SIZE as u64);

        let rows_per_tg = NR0 * NSG; // 4 rows per threadgroup
        let n_threadgroups = (out_features + rows_per_tg - 1) / rows_per_tg;

        let thread_group_count = MTLSize {
            width: n_threadgroups as u64,
            height: 1,
            depth: 1,
        };
        let thread_group_size = MTLSize {
            width: 32,
            height: NSG as u64,
            depth: 1,
        };

        encoder.use_resource(weight_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(input_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(output_buffer, metal::MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
    }

    // =========================================================================
    // Fused kernel args structs (matching the Metal shader)
    // =========================================================================

    /// Args for the fused gate+up+swiglu kernel.
    /// Offsets are passed via separate device buffers (buffer 8/9) for GPU-side routing.
    #[repr(C)]
    struct MxfpFusedGateUpArgs {
        out_features: i32,
        in_features: i32,
        weight_stride: u64,
        n_experts: i32,
        use_oai_swiglu: i32,
        alpha: f32,
        limit: f32,
        has_bias: i32,
    }

    /// Args for the fused down+accum kernel.
    /// Offsets are passed via separate device buffer (buffer 7) for GPU-side routing.
    #[repr(C)]
    struct MxfpFusedDownAccumArgs {
        out_features: i32,
        in_features: i32,
        weight_stride: u64,
        n_experts: i32,
        has_bias: i32,
    }

    /// Args for the softmax+topk GPU routing kernel.
    #[repr(C)]
    struct SoftmaxTopkArgs {
        n_experts: i32,
        top_k: i32,
        softmax_weight: i32,
        norm_topk_prob: i32,
    }

    /// Encode a fused gate+up+bias+swiglu dispatch.
    /// Offsets are passed as device buffers (buffer 8/9), supporting both CPU-computed
    /// and GPU-computed (softmax_topk output) offsets.
    fn encode_fused_gate_up_swiglu(
        command_buffer: &metal::CommandBufferRef,
        pipeline: &ComputePipelineState,
        gate_weights: &Buffer,
        up_weights: &Buffer,
        gate_offsets_buf: &Buffer,
        up_offsets_buf: &Buffer,
        input_buffer: &Buffer,
        input_offset: u64,
        output_buffer: &Buffer,
        gate_bias: Option<&Buffer>,
        up_bias: Option<&Buffer>,
        expert_idx_buf: &Buffer,
        out_features: usize,
        in_features: usize,
        n_experts: usize,
        use_oai_swiglu: bool,
        alpha: f32,
        limit: f32,
    ) {
        let blocks_per_row = in_features / MXFP4_BLOCK_ELEMENTS;
        let weight_stride = (blocks_per_row * MXFP4_BLOCK_SIZE) as u64;

        let args = MxfpFusedGateUpArgs {
            out_features: out_features as i32,
            in_features: in_features as i32,
            weight_stride,
            n_experts: n_experts as i32,
            use_oai_swiglu: if use_oai_swiglu { 1 } else { 0 },
            alpha,
            limit,
            has_bias: if gate_bias.is_some() { 1 } else { 0 },
        };

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_bytes(
            0,
            std::mem::size_of::<MxfpFusedGateUpArgs>() as u64,
            &args as *const MxfpFusedGateUpArgs as *const c_void,
        );
        encoder.set_buffer(1, Some(gate_weights), 0);
        encoder.set_buffer(2, Some(up_weights), 0);
        encoder.set_buffer(3, Some(input_buffer), input_offset);
        encoder.set_buffer(4, Some(output_buffer), 0);

        // Bias buffers (set even if unused — kernel checks has_bias flag)
        if let Some(gb) = gate_bias {
            encoder.set_buffer(5, Some(gb), 0);
        }
        if let Some(ub) = up_bias {
            encoder.set_buffer(6, Some(ub), 0);
        }
        encoder.set_buffer(7, Some(expert_idx_buf), 0);
        encoder.set_buffer(8, Some(gate_offsets_buf), 0);
        encoder.set_buffer(9, Some(up_offsets_buf), 0);

        encoder.set_threadgroup_memory_length(0, SHMEM_SIZE as u64);

        let rows_per_tg = NR0 * NSG;
        let n_threadgroups_x = (out_features + rows_per_tg - 1) / rows_per_tg;

        let thread_group_count = MTLSize {
            width: n_threadgroups_x as u64,
            height: 1,
            depth: n_experts as u64,
        };
        let thread_group_size = MTLSize {
            width: 32,
            height: NSG as u64,
            depth: 1,
        };

        encoder.use_resource(gate_weights, metal::MTLResourceUsage::Read);
        encoder.use_resource(up_weights, metal::MTLResourceUsage::Read);
        encoder.use_resource(input_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(output_buffer, metal::MTLResourceUsage::Write);
        if let Some(gb) = gate_bias {
            encoder.use_resource(gb, metal::MTLResourceUsage::Read);
        }
        if let Some(ub) = up_bias {
            encoder.use_resource(ub, metal::MTLResourceUsage::Read);
        }
        encoder.use_resource(expert_idx_buf, metal::MTLResourceUsage::Read);
        encoder.use_resource(gate_offsets_buf, metal::MTLResourceUsage::Read);
        encoder.use_resource(up_offsets_buf, metal::MTLResourceUsage::Read);
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
    }

    /// Encode a fused down+bias+weighted_accum dispatch.
    /// Offsets are passed as device buffer (buffer 7), supporting both CPU-computed
    /// and GPU-computed (softmax_topk output) offsets.
    fn encode_fused_down_accum(
        command_buffer: &metal::CommandBufferRef,
        pipeline: &ComputePipelineState,
        down_weights: &Buffer,
        down_offsets_buf: &Buffer,
        input_buffer: &Buffer,  // [n_experts, in_features] (swiglu output)
        routing_weights: &Buffer,
        output_buffer: &Buffer,
        down_bias: Option<&Buffer>,
        expert_idx_buf: &Buffer,
        out_features: usize,
        in_features: usize,
        n_experts: usize,
    ) {
        let blocks_per_row = in_features / MXFP4_BLOCK_ELEMENTS;
        let weight_stride = (blocks_per_row * MXFP4_BLOCK_SIZE) as u64;

        let args = MxfpFusedDownAccumArgs {
            out_features: out_features as i32,
            in_features: in_features as i32,
            weight_stride,
            n_experts: n_experts as i32,
            has_bias: if down_bias.is_some() { 1 } else { 0 },
        };

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_bytes(
            0,
            std::mem::size_of::<MxfpFusedDownAccumArgs>() as u64,
            &args as *const MxfpFusedDownAccumArgs as *const c_void,
        );
        encoder.set_buffer(1, Some(down_weights), 0);
        encoder.set_buffer(2, Some(input_buffer), 0);
        encoder.set_buffer(3, Some(routing_weights), 0);
        encoder.set_buffer(4, Some(output_buffer), 0);
        if let Some(db) = down_bias {
            encoder.set_buffer(5, Some(db), 0);
        }
        encoder.set_buffer(6, Some(expert_idx_buf), 0);
        encoder.set_buffer(7, Some(down_offsets_buf), 0);

        encoder.set_threadgroup_memory_length(0, SHMEM_SIZE as u64);

        let rows_per_tg = NR0 * NSG;
        let n_threadgroups_x = (out_features + rows_per_tg - 1) / rows_per_tg;

        // No Z dimension — single output, accumulates across experts inside kernel
        let thread_group_count = MTLSize {
            width: n_threadgroups_x as u64,
            height: 1,
            depth: 1,
        };
        let thread_group_size = MTLSize {
            width: 32,
            height: NSG as u64,
            depth: 1,
        };

        encoder.use_resource(down_weights, metal::MTLResourceUsage::Read);
        encoder.use_resource(input_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(routing_weights, metal::MTLResourceUsage::Read);
        encoder.use_resource(output_buffer, metal::MTLResourceUsage::Write);
        if let Some(db) = down_bias {
            encoder.use_resource(db, metal::MTLResourceUsage::Read);
        }
        encoder.use_resource(expert_idx_buf, metal::MTLResourceUsage::Read);
        encoder.use_resource(down_offsets_buf, metal::MTLResourceUsage::Read);
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
    }

    // =========================================================================
    // CustomOp1 implementation: MXFP4 matmul integrated into candle's pipeline
    // =========================================================================

    /// MXFP4 matrix-vector multiply as a candle CustomOp1.
    ///
    /// Encodes the MXFP4 kernel onto candle's shared command buffer without any
    /// explicit GPU synchronization. The output is allocated as a GPU-private buffer
    /// and returned as a proper MetalStorage, enabling fully pipelined execution:
    ///
    ///   MXFP4(gate) → SiLU → mul(up) → MXFP4(down) → add
    ///
    /// All operations share candle's command buffer, which batches up to 50 compute
    /// encoders before auto-committing. No per-expert pipeline drain.
    pub struct Mxfp4MatmulOp {
        /// MXFP4 weight data on GPU (pre-uploaded, shared across calls).
        weight_buffer: Arc<Buffer>,
        /// Byte offset into weight_buffer (for packed buffer access).
        weight_offset: u64,
        /// Number of output features (rows in weight matrix).
        out_features: usize,
        /// Number of input features (columns in weight matrix, must be multiple of 32).
        in_features: usize,
    }

    impl Mxfp4MatmulOp {
        /// Create a new MXFP4 matmul operation.
        ///
        /// The weight buffer must already be on the Metal GPU (use `upload_mxfp4_weights`).
        pub fn new(weight_buffer: Arc<Buffer>, out_features: usize, in_features: usize) -> Self {
            debug_assert_eq!(
                in_features % MXFP4_BLOCK_ELEMENTS, 0,
                "in_features must be a multiple of 32"
            );
            Self { weight_buffer, weight_offset: 0, out_features, in_features }
        }

        /// Create a new MXFP4 matmul operation with a byte offset into the weight buffer.
        /// Used for packed per-layer buffers where each expert starts at a different offset.
        pub fn new_with_offset(weight_buffer: Arc<Buffer>, weight_offset: u64, out_features: usize, in_features: usize) -> Self {
            debug_assert_eq!(
                in_features % MXFP4_BLOCK_ELEMENTS, 0,
                "in_features must be a multiple of 32"
            );
            Self { weight_buffer, weight_offset, out_features, in_features }
        }
    }

    impl candle_core::CustomOp1 for Mxfp4MatmulOp {
        fn name(&self) -> &'static str {
            "mxfp4-matmul"
        }

        fn cpu_fwd(
            &self,
            storage: &candle_core::CpuStorage,
            layout: &candle_core::Layout,
        ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
            // CPU fallback: extract f32 data and use the CPU MXFP4 path
            // This should not normally be called in the GPU Resident path
            use candle_core::CpuStorage;
            let input_data = match storage {
                CpuStorage::F32(data) => data,
                _ => return Err(candle_core::Error::Msg(
                    "Mxfp4MatmulOp: expected F32 input".into(),
                )),
            };

            // Handle layout offset for the input
            let start = layout.start_offset();
            let input_slice = &input_data[start..start + self.in_features];

            // Use CPU MXFP4 path (need to get raw weight bytes from the Metal buffer)
            // This is a sync fallback - read from managed buffer
            let weight_ptr = self.weight_buffer.contents() as *const u8;
            let blocks_per_row = self.in_features / MXFP4_BLOCK_ELEMENTS;
            let weight_bytes_len = self.out_features * blocks_per_row * MXFP4_BLOCK_SIZE;
            let weight_data = unsafe { std::slice::from_raw_parts(weight_ptr, weight_bytes_len) };

            let output = crate::gguf::dequant::mxfp4_matvec_mul(
                weight_data, input_slice, self.out_features, self.in_features,
            );
            let shape = candle_core::Shape::from((1, self.out_features));
            Ok((CpuStorage::F32(output), shape))
        }

        fn metal_fwd(
            &self,
            storage: &candle_core::MetalStorage,
            layout: &candle_core::Layout,
        ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
            use candle_core::backend::BackendStorage;

            let device = storage.device().clone();
            let pipeline = get_pipeline(device.device())?;

            // Input buffer from the candle tensor storage
            let input_buffer = storage.buffer();
            let input_offset = (layout.start_offset() * std::mem::size_of::<f32>()) as u64;

            // Allocate output buffer as GPU-private (no CPU read-back needed).
            // candle's allocator handles buffer reuse automatically.
            let output_buffer = device.new_buffer(
                self.out_features,
                candle_core::DType::F32,
                "mxfp4_output",
            )?;

            // Get candle's shared command buffer (batched, no sync).
            // This is the key difference from the old mxfp4_matmul_metal_gpu():
            // we encode onto candle's command buffer, not our own.
            let command_buffer = device.command_buffer()?;
            encode_mxfp4_matvec(
                &command_buffer,
                &pipeline,
                &self.weight_buffer,
                self.weight_offset,
                input_buffer,
                input_offset,
                &output_buffer,
                self.out_features,
                self.in_features,
            );

            // NO wait_until_completed()! The kernel is just enqueued.
            // candle will sync when it actually needs the data (e.g., to_cpu).

            let out_storage = candle_core::MetalStorage::new(
                output_buffer,
                device,
                self.out_features,
                candle_core::DType::F32,
            );
            let shape = candle_core::Shape::from((1, self.out_features));
            Ok((out_storage, shape))
        }
    }

    /// Perform MXFP4 matrix-vector multiply on Metal GPU, returning a Tensor on GPU.
    ///
    /// **Pipeline-integrated (no sync)**: Uses candle's CustomOp1 to encode the MXFP4
    /// kernel onto the shared command buffer. No explicit `wait_until_completed()`.
    /// The output stays on GPU as a private-storage Tensor.
    ///
    /// This replaces the old sync-per-expert approach which called
    /// `wait_until_completed()` after every matmul (312 syncs/step = ~780ms overhead).
    ///
    /// - `weight_buffer`: Pre-uploaded MXFP4 weight data (Arc for shared ownership).
    /// - `input`: Input tensor on Metal GPU (F32, shape [1, in_features] or [in_features]).
    /// - `out_features`: Number of output rows.
    /// - `in_features`: Number of input columns (must be a multiple of 32).
    ///
    /// Returns: F32 Tensor on Metal GPU with shape [1, out_features].
    pub fn mxfp4_matmul_metal_gpu(
        weight_buffer: &Arc<Buffer>,
        input: &candle_core::Tensor,
        out_features: usize,
        in_features: usize,
    ) -> Result<candle_core::Tensor> {
        let op = Mxfp4MatmulOp::new(Arc::clone(weight_buffer), out_features, in_features);
        input.apply_op1_no_bwd(&op)
    }

    /// Batched MXFP4 matmul for prefill (seq_len > 1).
    ///
    /// The MXFP4 kernel is a matvec (single row). For multi-row inputs (prefill),
    /// we iterate over rows and enqueue all kernels on candle's shared command buffer
    /// (no per-row sync). For single-row inputs (decode), delegates directly.
    pub fn mxfp4_matmul_metal_gpu_batched(
        weight_buffer: &Arc<Buffer>,
        input: &candle_core::Tensor,
        out_features: usize,
        in_features: usize,
    ) -> Result<candle_core::Tensor> {
        use candle_core::IndexOp;
        let n_rows = input.dim(0)?;
        if n_rows <= 1 {
            return mxfp4_matmul_metal_gpu(weight_buffer, input, out_features, in_features);
        }
        let mut rows = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let row = input.i(i)?.contiguous()?;
            rows.push(mxfp4_matmul_metal_gpu(weight_buffer, &row, out_features, in_features)?);
        }
        candle_core::Tensor::cat(&rows, 0)
    }

    /// Perform MXFP4 matrix-vector multiply with a byte offset into the weight buffer.
    /// Used for packed per-layer buffers (Mxfp4Packed variant).
    pub fn mxfp4_matmul_metal_gpu_offset(
        weight_buffer: &Arc<Buffer>,
        weight_offset: u64,
        input: &candle_core::Tensor,
        out_features: usize,
        in_features: usize,
    ) -> Result<candle_core::Tensor> {
        let op = Mxfp4MatmulOp::new_with_offset(Arc::clone(weight_buffer), weight_offset, out_features, in_features);
        input.apply_op1_no_bwd(&op)
    }

    /// Batched MXFP4 matmul with offset for prefill (seq_len > 1).
    pub fn mxfp4_matmul_metal_gpu_offset_batched(
        weight_buffer: &Arc<Buffer>,
        weight_offset: u64,
        input: &candle_core::Tensor,
        out_features: usize,
        in_features: usize,
    ) -> Result<candle_core::Tensor> {
        use candle_core::IndexOp;
        let n_rows = input.dim(0)?;
        if n_rows <= 1 {
            return mxfp4_matmul_metal_gpu_offset(weight_buffer, weight_offset, input, out_features, in_features);
        }
        let mut rows = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let row = input.i(i)?.contiguous()?;
            rows.push(mxfp4_matmul_metal_gpu_offset(weight_buffer, weight_offset, &row, out_features, in_features)?);
        }
        candle_core::Tensor::cat(&rows, 0)
    }

    // =========================================================================
    // Quantized attention (Q5_0/Q8_0) Metal kernel dispatch
    // =========================================================================

    /// Encode a quantized attention matvec dispatch (Q5_0 or Q8_0).
    fn encode_quantized_attn_matvec(
        command_buffer: &metal::CommandBufferRef,
        pipeline: &ComputePipelineState,
        weight_buffer: &Buffer,
        input_buffer: &Buffer,
        input_offset: u64,
        output_buffer: &Buffer,
        out_features: usize,
        in_features: usize,
        quant_type: QuantizedAttnType,
    ) {
        let block_size = quant_type.block_size();
        let blocks_per_row = in_features / 32; // Both Q5_0 and Q8_0 use 32 elements per block
        let weight_stride = (blocks_per_row * block_size) as u64;

        let args = QuantizedMvArgs {
            out_features: out_features as i32,
            in_features: in_features as i32,
            weight_stride,
        };

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_bytes(
            0,
            std::mem::size_of::<QuantizedMvArgs>() as u64,
            &args as *const QuantizedMvArgs as *const c_void,
        );
        encoder.set_buffer(1, Some(weight_buffer), 0);
        encoder.set_buffer(2, Some(input_buffer), input_offset);
        encoder.set_buffer(3, Some(output_buffer), 0);

        // Q5_0/Q8_0 kernels don't use shared memory (unlike MXFP4)
        let rows_per_tg = NR0 * NSG; // 4 rows per threadgroup
        let n_threadgroups = (out_features + rows_per_tg - 1) / rows_per_tg;

        let thread_group_count = MTLSize {
            width: n_threadgroups as u64,
            height: 1,
            depth: 1,
        };
        let thread_group_size = MTLSize {
            width: 32,
            height: NSG as u64,
            depth: 1,
        };

        encoder.use_resource(weight_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(input_buffer, metal::MTLResourceUsage::Read);
        encoder.use_resource(output_buffer, metal::MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
    }

    /// Quantized attention matvec as a candle CustomOp1.
    ///
    /// Encodes Q5_0 or Q8_0 matvec onto candle's shared command buffer,
    /// avoiding dequant to F32 and reducing bandwidth by ~6x (Q5_0) or ~4x (Q8_0).
    pub struct QuantizedAttnMatmulOp {
        weight_buffer: Arc<Buffer>,
        out_features: usize,
        in_features: usize,
        quant_type: QuantizedAttnType,
    }

    impl candle_core::CustomOp1 for QuantizedAttnMatmulOp {
        fn name(&self) -> &'static str {
            match self.quant_type {
                QuantizedAttnType::Q5_0 => "q5_0-attn-matmul",
                QuantizedAttnType::Q8_0 => "q8_0-attn-matmul",
            }
        }

        fn cpu_fwd(
            &self,
            _storage: &candle_core::CpuStorage,
            _layout: &candle_core::Layout,
        ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
            Err(candle_core::Error::Msg(
                "QuantizedAttnMatmulOp: CPU not supported (use QMatMul instead)".into(),
            ))
        }

        fn metal_fwd(
            &self,
            storage: &candle_core::MetalStorage,
            layout: &candle_core::Layout,
        ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
            use candle_core::backend::BackendStorage;

            let device = storage.device().clone();
            let pipeline = match self.quant_type {
                QuantizedAttnType::Q5_0 => get_q5_0_pipeline(device.device())?,
                QuantizedAttnType::Q8_0 => get_q8_0_pipeline(device.device())?,
            };

            let input_buffer = storage.buffer();
            let input_offset = (layout.start_offset() * std::mem::size_of::<f32>()) as u64;

            let output_buffer = device.new_buffer(
                self.out_features,
                candle_core::DType::F32,
                "quantized_attn_output",
            )?;

            let command_buffer = device.command_buffer()?;
            encode_quantized_attn_matvec(
                &command_buffer,
                &pipeline,
                &self.weight_buffer,
                input_buffer,
                input_offset,
                &output_buffer,
                self.out_features,
                self.in_features,
                self.quant_type,
            );

            let out_storage = candle_core::MetalStorage::new(
                output_buffer,
                device,
                self.out_features,
                candle_core::DType::F32,
            );
            let shape = candle_core::Shape::from((1, self.out_features));
            Ok((out_storage, shape))
        }
    }

    /// Perform Q5_0 or Q8_0 matrix-vector multiply on Metal GPU.
    ///
    /// Pipeline-integrated: encodes onto candle's shared command buffer (no sync).
    /// Returns F32 Tensor on Metal GPU with shape [1, out_features].
    pub fn quantized_attn_matmul_metal_gpu(
        weight_buffer: &Arc<Buffer>,
        input: &candle_core::Tensor,
        out_features: usize,
        in_features: usize,
        quant_type: QuantizedAttnType,
    ) -> Result<candle_core::Tensor> {
        let op = QuantizedAttnMatmulOp {
            weight_buffer: Arc::clone(weight_buffer),
            out_features,
            in_features,
            quant_type,
        };
        input.apply_op1_no_bwd(&op)
    }

    /// Batched quantized attention matmul for prefill (seq_len > 1).
    ///
    /// Same strategy as `mxfp4_matmul_metal_gpu_batched`: iterate over rows for
    /// multi-row inputs since the Q5_0/Q8_0 kernel is a matvec.
    pub fn quantized_attn_matmul_metal_gpu_batched(
        weight_buffer: &Arc<Buffer>,
        input: &candle_core::Tensor,
        out_features: usize,
        in_features: usize,
        quant_type: QuantizedAttnType,
    ) -> Result<candle_core::Tensor> {
        use candle_core::IndexOp;
        let n_rows = input.dim(0)?;
        if n_rows <= 1 {
            return quantized_attn_matmul_metal_gpu(weight_buffer, input, out_features, in_features, quant_type);
        }
        let mut rows = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let row = input.i(i)?.contiguous()?;
            rows.push(quantized_attn_matmul_metal_gpu(weight_buffer, &row, out_features, in_features, quant_type)?);
        }
        candle_core::Tensor::cat(&rows, 0)
    }

    // =========================================================================
    // Helper: Wrap a pre-computed Metal buffer as a candle Tensor via CustomOp1
    // =========================================================================

    /// CustomOp1 that ignores its input and returns a pre-existing Metal buffer as output.
    /// Used to convert raw Metal kernel output into a candle Tensor without
    /// Tensor::zeros() + blit copy (which can cause command buffer commit conflicts).
    struct BufferToTensorOp {
        source_buffer: Arc<Buffer>,
        elem_count: usize,
    }

    impl candle_core::CustomOp1 for BufferToTensorOp {
        fn name(&self) -> &'static str {
            "buffer-to-tensor"
        }

        fn cpu_fwd(
            &self,
            _storage: &candle_core::CpuStorage,
            _layout: &candle_core::Layout,
        ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
            Err(candle_core::Error::Msg("BufferToTensorOp: CPU not supported".into()))
        }

        fn metal_fwd(
            &self,
            _storage: &candle_core::MetalStorage,
            _layout: &candle_core::Layout,
        ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
            use candle_core::backend::BackendStorage;
            let device = _storage.device().clone();
            let out_storage = candle_core::MetalStorage::new(
                Arc::clone(&self.source_buffer),
                device,
                self.elem_count,
                candle_core::DType::F32,
            );
            let shape = candle_core::Shape::from((1, self.elem_count));
            Ok((out_storage, shape))
        }
    }

    // =========================================================================
    // High-level batched MoE forward pass for MXFP4 experts
    // =========================================================================

    /// Expert info for batched MoE dispatch.
    pub struct BatchedExpertInfo {
        /// Expert index in the model.
        pub expert_idx: usize,
        /// Routing weight for this expert.
        pub routing_weight: f32,
    }

    /// Pre-uploaded expert bias data for batched dispatch.
    ///
    /// Contains all experts' biases for a given projection (gate, up, or down)
    /// packed contiguously on the GPU: [num_total_experts, dim].
    pub struct ExpertBiasBuffer {
        /// Metal buffer containing all expert biases: [total_experts, dim].
        pub buffer: Arc<Buffer>,
        /// Total number of experts (determines buffer layout).
        pub total_experts: usize,
        /// Feature dimension per expert.
        pub dim: usize,
    }

    /// Pre-uploaded bias buffers for all three projections.
    pub struct ExpertBiasBuffers {
        pub gate: ExpertBiasBuffer,
        pub up: ExpertBiasBuffer,
        pub down: ExpertBiasBuffer,
    }

    /// Pre-allocated GPU buffers for batched MoE forward pass.
    ///
    /// Caching these avoids ~168 buffer allocations per token (7 buffers x 24 layers).
    /// Buffers are allocated for MAX_BATCH_EXPERTS to handle any expert count without reallocation.
    pub struct BatchedMoeBuffers {
        pub gate_buf: Arc<Buffer>,
        pub up_buf: Arc<Buffer>,
        pub swiglu_buf: Arc<Buffer>,
        pub down_buf: Arc<Buffer>,
        pub final_buf: Arc<Buffer>,
        /// Routing weights buffer (MAX_BATCH_EXPERTS f32s).
        pub rw_buf: Arc<Buffer>,
        /// Expert index buffer (MAX_BATCH_EXPERTS i32s).
        pub expert_idx_buf: Arc<Buffer>,
        /// Cached intermediate dim for sanity checks.
        pub intermediate_dim: usize,
        /// Cached hidden dim for sanity checks.
        pub hidden_dim: usize,
    }

    impl BatchedMoeBuffers {
        /// Pre-allocate all intermediate buffers for batched MoE dispatch.
        pub fn new(
            metal_device: &candle_core::MetalDevice,
            intermediate_dim: usize,
            hidden_dim: usize,
        ) -> Result<Self> {
            let max_e = MAX_BATCH_EXPERTS;
            let gate_buf = metal_device.new_buffer(max_e * intermediate_dim, candle_core::DType::F32, "moe_gate")?;
            let up_buf = metal_device.new_buffer(max_e * intermediate_dim, candle_core::DType::F32, "moe_up")?;
            let swiglu_buf = metal_device.new_buffer(max_e * intermediate_dim, candle_core::DType::F32, "moe_swiglu")?;
            let down_buf = metal_device.new_buffer(max_e * hidden_dim, candle_core::DType::F32, "moe_down")?;
            let final_buf = metal_device.new_buffer(hidden_dim, candle_core::DType::F32, "moe_out")?;
            // Allocate max-sized routing weight and expert index buffers.
            let rw_data = vec![0.0f32; max_e];
            let rw_buf = metal_device.new_buffer_with_data(&rw_data)?;
            let idx_data = vec![0i32; max_e];
            let expert_idx_buf = metal_device.new_buffer_with_data(&idx_data)?;
            Ok(Self {
                gate_buf, up_buf, swiglu_buf, down_buf, final_buf,
                rw_buf, expert_idx_buf,
                intermediate_dim, hidden_dim,
            })
        }

        /// Update routing weights in-place (CPU write to managed buffer).
        pub fn update_routing_weights(&self, weights: &[f32]) {
            let ptr = self.rw_buf.contents() as *mut f32;
            unsafe {
                std::ptr::copy_nonoverlapping(weights.as_ptr(), ptr, weights.len());
            }
            // Notify Metal that CPU modified this managed buffer range.
            let byte_len = (weights.len() * std::mem::size_of::<f32>()) as u64;
            self.rw_buf.did_modify_range(metal::NSRange::new(0, byte_len));
        }

        /// Update expert indices in-place (CPU write to managed buffer).
        pub fn update_expert_indices(&self, indices: &[i32]) {
            let ptr = self.expert_idx_buf.contents() as *mut i32;
            unsafe {
                std::ptr::copy_nonoverlapping(indices.as_ptr(), ptr, indices.len());
            }
            let byte_len = (indices.len() * std::mem::size_of::<i32>()) as u64;
            self.expert_idx_buf.did_modify_range(metal::NSRange::new(0, byte_len));
        }
    }

    /// Perform a complete batched MoE forward pass for MXFP4 experts on Metal GPU.
    ///
    /// Uses 2 fused Metal kernel dispatches (modeled after llama.cpp's approach):
    ///   1. Fused gate+up projection + bias + SwiGLU activation
    ///   2. Fused down projection + bias + weighted accumulation
    ///
    /// This replaces the previous 5-8 dispatch approach, eliminating:
    ///   - 6 dispatch overhead gaps
    ///   - 1 input vector read (gate+up share a single read)
    ///   - 4 intermediate buffer writes (gate_buf, up_buf, down_buf skipped)
    ///
    /// Returns: F32 Tensor on Metal GPU with shape [1, hidden_dim].
    pub fn batched_moe_forward_metal(
        device: &Device,
        input: &candle_core::Tensor,
        packed: &crate::model::cache::PackedMxfp4Layer,
        experts: &[BatchedExpertInfo],
        use_oai_swiglu: bool,
        alpha: f32,
        limit: f32,
        bias_buffers: Option<&ExpertBiasBuffers>,
        buffers: Option<&BatchedMoeBuffers>,
    ) -> Result<candle_core::Tensor> {
        let metal_device = match device {
            Device::Metal(m) => m,
            _ => return Err(candle_core::Error::Msg("batched_moe_forward_metal requires Metal device".into())),
        };

        let n_experts = experts.len();
        if n_experts == 0 || n_experts > MAX_BATCH_EXPERTS {
            return Err(candle_core::Error::Msg(format!(
                "batched_moe_forward: n_experts={} must be 1..{}", n_experts, MAX_BATCH_EXPERTS
            )));
        }

        let intermediate_dim = packed.gate.out_features;
        let hidden_dim = packed.gate.in_features;
        let down_out = packed.down.out_features;  // = hidden_dim
        let down_in = packed.down.in_features;    // = intermediate_dim

        // Get fused pipelines (cached after first call)
        let fused_gu_pl = get_fused_gate_up_swiglu_pipeline(metal_device.device())?;
        let fused_da_pl = get_fused_down_accum_pipeline(metal_device.device())?;

        // Extract input Metal buffer
        let (input_storage, input_layout) = input.storage_and_layout();
        let input_buffer = match &*input_storage {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => return Err(candle_core::Error::Msg("Input must be on Metal device".into())),
        };
        let input_offset = (input_layout.start_offset() * std::mem::size_of::<f32>()) as u64;

        // Build expert offsets for gate/up/down and upload as GPU buffers
        let gate_offsets_vec: Vec<u64> = experts.iter().map(|e| packed.gate.offsets[e.expert_idx]).collect();
        let up_offsets_vec: Vec<u64> = experts.iter().map(|e| packed.up.offsets[e.expert_idx]).collect();
        let down_offsets_vec: Vec<u64> = experts.iter().map(|e| packed.down.offsets[e.expert_idx]).collect();

        // Upload offsets as GPU buffers (device buffer bindings for fused kernels)
        let gate_offsets_buf = metal_device.new_buffer_with_data(&gate_offsets_vec)?;
        let up_offsets_buf = metal_device.new_buffer_with_data(&up_offsets_vec)?;
        let down_offsets_buf = metal_device.new_buffer_with_data(&down_offsets_vec)?;

        // Use pre-allocated buffers if available, otherwise allocate fresh ones.
        // Fused path only needs: swiglu_buf (intermediate), final_buf (output),
        // rw_buf (routing weights), expert_idx_buf (expert indices).
        let (swiglu_buf, final_buf, rw_buf, expert_idx_buf);
        let (swiglu_buf_ref, final_buf_ref, rw_buf_ref, expert_idx_buf_ref);

        if let Some(bufs) = buffers {
            let rw: Vec<f32> = experts.iter().map(|e| e.routing_weight).collect();
            bufs.update_routing_weights(&rw);
            let expert_idx_vec: Vec<i32> = experts.iter().map(|e| e.expert_idx as i32).collect();
            bufs.update_expert_indices(&expert_idx_vec);

            swiglu_buf_ref = &bufs.swiglu_buf;
            final_buf_ref = &bufs.final_buf;
            rw_buf_ref = &bufs.rw_buf;
            expert_idx_buf_ref = &bufs.expert_idx_buf;
        } else {
            swiglu_buf = metal_device.new_buffer(n_experts * intermediate_dim, candle_core::DType::F32, "moe_swiglu")?;
            final_buf = metal_device.new_buffer(down_out, candle_core::DType::F32, "moe_out")?;
            let rw: Vec<f32> = experts.iter().map(|e| e.routing_weight).collect();
            rw_buf = metal_device.new_buffer_with_data(&rw)?;
            let expert_idx_vec: Vec<i32> = experts.iter().map(|e| e.expert_idx as i32).collect();
            expert_idx_buf = metal_device.new_buffer_with_data(&expert_idx_vec)?;

            swiglu_buf_ref = &swiglu_buf;
            final_buf_ref = &final_buf;
            rw_buf_ref = &rw_buf;
            expert_idx_buf_ref = &expert_idx_buf;
        }

        // === Fused Dispatch 1: Gate + Up + Bias + SwiGLU ===
        // Reads input once, computes both projections, applies bias and activation inline.
        {
            let cb = metal_device.command_buffer()?;
            encode_fused_gate_up_swiglu(
                &cb,
                &fused_gu_pl,
                &packed.gate.buffer,
                &packed.up.buffer,
                &gate_offsets_buf,
                &up_offsets_buf,
                input_buffer,
                input_offset,
                swiglu_buf_ref,
                bias_buffers.map(|b| &*b.gate.buffer),
                bias_buffers.map(|b| &*b.up.buffer),
                expert_idx_buf_ref,
                intermediate_dim,
                hidden_dim,
                n_experts,
                use_oai_swiglu,
                alpha,
                limit,
            );
        }

        // Release input storage read lock
        drop(input_storage);

        // === Fused Dispatch 2: Down + Bias + Weighted Accumulation ===
        // Iterates over experts inside the kernel, accumulates directly to output.
        {
            let cb = metal_device.command_buffer()?;
            encode_fused_down_accum(
                &cb,
                &fused_da_pl,
                &packed.down.buffer,
                &down_offsets_buf,
                swiglu_buf_ref,
                rw_buf_ref,
                final_buf_ref,
                bias_buffers.map(|b| &*b.down.buffer),
                expert_idx_buf_ref,
                down_out,
                down_in,
                n_experts,
            );
        }

        // Create output tensor by wrapping the final buffer via CustomOp1.
        let output_op = BufferToTensorOp {
            source_buffer: Arc::clone(final_buf_ref),
            elem_count: down_out,
        };
        let dummy = candle_core::Tensor::zeros(1, candle_core::DType::F32, device)?;
        let output_tensor = dummy.apply_op1_no_bwd(&output_op)?;

        Ok(output_tensor)
    }

    /// Pre-uploaded offset tables for GPU-side routing (one per model).
    /// Allows the softmax_topk kernel to look up byte offsets for selected experts
    /// without CPU round-trip.
    pub struct GpuRoutingOffsets {
        pub all_gate_offsets: Arc<Buffer>,  // [total_experts] u64
        pub all_up_offsets: Arc<Buffer>,    // [total_experts] u64
        pub all_down_offsets: Arc<Buffer>,  // [total_experts] u64
        pub dummy_mask: Arc<Buffer>,        // [total_experts] f32 (0 or -inf)
    }

    /// Per-dispatch GPU buffers for routing output (reused across layers).
    pub struct GpuRoutingOutputBuffers {
        pub expert_indices: Arc<Buffer>,    // [top_k] i32
        pub routing_weights: Arc<Buffer>,   // [top_k] f32
        pub gate_offsets: Arc<Buffer>,      // [top_k] u64
        pub up_offsets: Arc<Buffer>,        // [top_k] u64
        pub down_offsets: Arc<Buffer>,      // [top_k] u64
    }

    /// Allocate GPU routing offset tables from PackedMxfp4Layer data.
    pub fn create_gpu_routing_offsets(
        device: &Device,
        packed: &crate::model::cache::PackedMxfp4Layer,
        dummy_experts: &[usize],
        total_experts: usize,
    ) -> Result<GpuRoutingOffsets> {
        let metal_device = match device {
            Device::Metal(m) => m,
            _ => return Err(candle_core::Error::Msg("Requires Metal device".into())),
        };
        let all_gate = metal_device.new_buffer_with_data(&packed.gate.offsets)?;
        let all_up = metal_device.new_buffer_with_data(&packed.up.offsets)?;
        let all_down = metal_device.new_buffer_with_data(&packed.down.offsets)?;

        let mut mask = vec![0.0f32; total_experts];
        for &idx in dummy_experts {
            if idx < total_experts {
                mask[idx] = f32::NEG_INFINITY;
            }
        }
        let dummy_mask = metal_device.new_buffer_with_data(&mask)?;

        Ok(GpuRoutingOffsets {
            all_gate_offsets: all_gate,
            all_up_offsets: all_up,
            all_down_offsets: all_down,
            dummy_mask,
        })
    }

    /// Allocate reusable GPU buffers for routing output.
    pub fn create_gpu_routing_output_buffers(
        device: &Device,
        top_k: usize,
    ) -> Result<GpuRoutingOutputBuffers> {
        let metal_device = match device {
            Device::Metal(m) => m,
            _ => return Err(candle_core::Error::Msg("Requires Metal device".into())),
        };
        Ok(GpuRoutingOutputBuffers {
            expert_indices: metal_device.new_buffer(top_k, candle_core::DType::I64, "rt_idx")?,
            routing_weights: metal_device.new_buffer(top_k, candle_core::DType::F32, "rt_wt")?,
            gate_offsets: metal_device.new_buffer(top_k, candle_core::DType::I64, "rt_go")?,
            up_offsets: metal_device.new_buffer(top_k, candle_core::DType::I64, "rt_uo")?,
            down_offsets: metal_device.new_buffer(top_k, candle_core::DType::I64, "rt_do")?,
        })
    }

    /// GPU-routed MoE forward: softmax+topk on GPU, no CPU sync.
    ///
    /// Eliminates the per-layer `to_vec1()` GPU sync that dominates decode time.
    /// Instead of: gate_matmul → GPU sync → CPU routing → GPU MoE dispatch
    /// Does:       gate_matmul → GPU softmax_topk → GPU MoE dispatch (0 syncs)
    ///
    /// Returns: F32 Tensor on Metal GPU with shape [1, hidden_dim].
    pub fn gpu_routed_moe_forward_metal(
        device: &Device,
        gate_logits: &candle_core::Tensor,  // [1, n_experts] on GPU (output of gate matmul)
        packed: &crate::model::cache::PackedMxfp4Layer,
        routing_offsets: &GpuRoutingOffsets,
        routing_out: &GpuRoutingOutputBuffers,
        input: &candle_core::Tensor,  // [1, hidden_dim] on GPU
        top_k: usize,
        n_experts: usize,
        use_oai_swiglu: bool,
        softmax_weight: bool,
        norm_topk_prob: bool,
        alpha: f32,
        limit: f32,
        bias_buffers: Option<&ExpertBiasBuffers>,
        moe_buffers: Option<&BatchedMoeBuffers>,
    ) -> Result<candle_core::Tensor> {
        let metal_device = match device {
            Device::Metal(m) => m,
            _ => return Err(candle_core::Error::Msg("gpu_routed_moe requires Metal device".into())),
        };

        let intermediate_dim = packed.gate.out_features;
        let hidden_dim = packed.gate.in_features;
        let down_out = packed.down.out_features;
        let down_in = packed.down.in_features;

        // Get pipelines
        let softmax_topk_pl = get_softmax_topk_pipeline(metal_device.device())?;
        let fused_gu_pl = get_fused_gate_up_swiglu_pipeline(metal_device.device())?;
        let fused_da_pl = get_fused_down_accum_pipeline(metal_device.device())?;

        // Extract gate_logits Metal buffer
        let (logits_storage, logits_layout) = gate_logits.storage_and_layout();
        let logits_buffer = match &*logits_storage {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => return Err(candle_core::Error::Msg("gate_logits must be on Metal".into())),
        };
        let logits_offset = (logits_layout.start_offset() * std::mem::size_of::<f32>()) as u64;

        // Extract input Metal buffer
        let (input_storage, input_layout) = input.storage_and_layout();
        let input_buffer = match &*input_storage {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => return Err(candle_core::Error::Msg("Input must be on Metal".into())),
        };
        let input_offset = (input_layout.start_offset() * std::mem::size_of::<f32>()) as u64;

        // === Dispatch 1: Softmax + Top-K routing on GPU ===
        {
            let args = SoftmaxTopkArgs {
                n_experts: n_experts as i32,
                top_k: top_k as i32,
                softmax_weight: if softmax_weight { 1 } else { 0 },
                norm_topk_prob: if norm_topk_prob { 1 } else { 0 },
            };
            let cb = metal_device.command_buffer()?;
            let encoder = cb.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&softmax_topk_pl);
            encoder.set_bytes(
                10,
                std::mem::size_of::<SoftmaxTopkArgs>() as u64,
                &args as *const SoftmaxTopkArgs as *const c_void,
            );
            encoder.set_buffer(0, Some(logits_buffer), logits_offset);
            encoder.set_buffer(1, Some(&routing_offsets.dummy_mask), 0);
            encoder.set_buffer(2, Some(&routing_out.expert_indices), 0);
            encoder.set_buffer(3, Some(&routing_out.routing_weights), 0);
            encoder.set_buffer(4, Some(&routing_offsets.all_gate_offsets), 0);
            encoder.set_buffer(5, Some(&routing_offsets.all_up_offsets), 0);
            encoder.set_buffer(6, Some(&routing_offsets.all_down_offsets), 0);
            encoder.set_buffer(7, Some(&routing_out.gate_offsets), 0);
            encoder.set_buffer(8, Some(&routing_out.up_offsets), 0);
            encoder.set_buffer(9, Some(&routing_out.down_offsets), 0);

            encoder.use_resource(logits_buffer, metal::MTLResourceUsage::Read);
            encoder.use_resource(&routing_offsets.dummy_mask, metal::MTLResourceUsage::Read);
            encoder.use_resource(&routing_out.expert_indices, metal::MTLResourceUsage::Write);
            encoder.use_resource(&routing_out.routing_weights, metal::MTLResourceUsage::Write);
            encoder.use_resource(&routing_offsets.all_gate_offsets, metal::MTLResourceUsage::Read);
            encoder.use_resource(&routing_offsets.all_up_offsets, metal::MTLResourceUsage::Read);
            encoder.use_resource(&routing_offsets.all_down_offsets, metal::MTLResourceUsage::Read);
            encoder.use_resource(&routing_out.gate_offsets, metal::MTLResourceUsage::Write);
            encoder.use_resource(&routing_out.up_offsets, metal::MTLResourceUsage::Write);
            encoder.use_resource(&routing_out.down_offsets, metal::MTLResourceUsage::Write);

            let tg_count = MTLSize { width: 1, height: 1, depth: 1 };
            let tg_size = MTLSize { width: 256, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(tg_count, tg_size);
            encoder.end_encoding();
        }

        // Release logits storage
        drop(logits_storage);

        // Use pre-allocated buffers if available
        let (swiglu_buf, final_buf);
        let (swiglu_buf_ref, final_buf_ref);

        if let Some(bufs) = moe_buffers {
            swiglu_buf_ref = &bufs.swiglu_buf;
            final_buf_ref = &bufs.final_buf;
        } else {
            swiglu_buf = metal_device.new_buffer(top_k * intermediate_dim, candle_core::DType::F32, "moe_swiglu")?;
            final_buf = metal_device.new_buffer(down_out, candle_core::DType::F32, "moe_out")?;
            swiglu_buf_ref = &swiglu_buf;
            final_buf_ref = &final_buf;
        }

        // === Dispatch 2: Fused gate+up+SwiGLU (reads routing from GPU buffers) ===
        {
            let cb = metal_device.command_buffer()?;
            encode_fused_gate_up_swiglu(
                &cb,
                &fused_gu_pl,
                &packed.gate.buffer,
                &packed.up.buffer,
                &routing_out.gate_offsets,
                &routing_out.up_offsets,
                input_buffer,
                input_offset,
                swiglu_buf_ref,
                bias_buffers.map(|b| &*b.gate.buffer),
                bias_buffers.map(|b| &*b.up.buffer),
                &routing_out.expert_indices,
                intermediate_dim,
                hidden_dim,
                top_k,  // n_experts = top_k (we always select exactly K)
                use_oai_swiglu,
                alpha,
                limit,
            );
        }

        // Release input storage read lock
        drop(input_storage);

        // === Dispatch 3: Fused down+accum (reads routing weights from GPU buffer) ===
        {
            let cb = metal_device.command_buffer()?;
            encode_fused_down_accum(
                &cb,
                &fused_da_pl,
                &packed.down.buffer,
                &routing_out.down_offsets,
                swiglu_buf_ref,
                &routing_out.routing_weights,
                final_buf_ref,
                bias_buffers.map(|b| &*b.down.buffer),
                &routing_out.expert_indices,
                down_out,
                down_in,
                top_k,
            );
        }

        // Create output tensor
        let output_op = BufferToTensorOp {
            source_buffer: Arc::clone(final_buf_ref),
            elem_count: down_out,
        };
        let dummy = candle_core::Tensor::zeros(1, candle_core::DType::F32, device)?;
        let output_tensor = dummy.apply_op1_no_bwd(&output_op)?;

        Ok(output_tensor)
    }

    /// Perform MXFP4 matrix-vector multiply on Metal GPU with synchronization.
    ///
    /// Computes: output = weight @ input, then waits for GPU completion and reads back to CPU.
    ///
    /// - `device`: candle Metal device.
    /// - `weight_buffer`: Raw MXFP4 data as a Metal buffer (out_features * weight_stride bytes).
    /// - `input_f32`: Input vector as F32 slice (length = in_features).
    /// - `out_features`: Number of output rows.
    /// - `in_features`: Number of input columns (must be a multiple of 32).
    ///
    /// Returns: F32 output vector (length = out_features).
    pub fn mxfp4_matmul_metal_sync(
        device: &Device,
        weight_buffer: &Buffer,
        input_f32: &[f32],
        out_features: usize,
        in_features: usize,
    ) -> Result<Vec<f32>> {
        let metal_device = match device {
            Device::Metal(m) => m,
            _ => {
                return Err(candle_core::Error::Msg(
                    "mxfp4_matmul_metal_sync requires Metal device".into(),
                ))
            }
        };

        assert_eq!(
            in_features % MXFP4_BLOCK_ELEMENTS,
            0,
            "in_features must be a multiple of 32"
        );
        assert_eq!(input_f32.len(), in_features, "input length mismatch");

        let pipeline = get_pipeline(metal_device.device())?;

        // Upload input to GPU
        let input_buffer = metal_device.new_buffer_with_data(input_f32)?;

        // Allocate output buffer (managed for CPU read-back)
        let output_size = (out_features * std::mem::size_of::<f32>()) as u64;
        let output_buffer = metal_device.new_buffer_managed(output_size)?;

        // Encode and dispatch
        let command_buffer = metal_device.command_buffer()?;
        encode_mxfp4_matvec(
            &command_buffer,
            &pipeline,
            weight_buffer,
            0,
            &input_buffer,
            0,
            &output_buffer,
            out_features,
            in_features,
        );

        // Wait for GPU completion
        metal_device.wait_until_completed()?;

        // Read back results
        let output_ptr = output_buffer.contents() as *const f32;
        let output = unsafe { std::slice::from_raw_parts(output_ptr, out_features) };

        Ok(output.to_vec())
    }

    /// Perform MXFP4 matrix-vector multiply using candle Tensor input on Metal GPU.
    ///
    /// The input tensor must be on the Metal device. The result is returned as a CPU Vec<f32>.
    /// This function synchronizes (waits for GPU completion) before returning.
    ///
    /// For full GPU-resident inference (no sync), the T2/T3 integration will use the raw
    /// buffer APIs and manage command buffer submission separately.
    pub fn mxfp4_matmul_metal_tensor(
        device: &Device,
        weight_buffer: &Buffer,
        input: &candle_core::Tensor,
        out_features: usize,
        in_features: usize,
    ) -> Result<Vec<f32>> {
        let metal_device = match device {
            Device::Metal(m) => m,
            _ => {
                return Err(candle_core::Error::Msg(
                    "mxfp4_matmul_metal_tensor requires Metal device".into(),
                ))
            }
        };

        assert_eq!(
            in_features % MXFP4_BLOCK_ELEMENTS,
            0,
            "in_features must be a multiple of 32"
        );

        let pipeline = get_pipeline(metal_device.device())?;

        // Extract Metal buffer from candle tensor
        let (storage, layout) = input.storage_and_layout();
        let input_buffer = match &*storage {
            candle_core::Storage::Metal(ms) => ms.buffer(),
            _ => {
                return Err(candle_core::Error::Msg(
                    "Input tensor must be on Metal device".into(),
                ))
            }
        };
        let input_offset = (layout.start_offset() * std::mem::size_of::<f32>()) as u64;

        // Allocate output buffer (managed for CPU read-back)
        let output_size = (out_features * std::mem::size_of::<f32>()) as u64;
        let output_buffer = metal_device.new_buffer_managed(output_size)?;

        // Encode and dispatch
        let command_buffer = metal_device.command_buffer()?;
        encode_mxfp4_matvec(
            &command_buffer,
            &pipeline,
            weight_buffer,
            0,
            input_buffer,
            input_offset,
            &output_buffer,
            out_features,
            in_features,
        );

        // Wait for GPU completion
        drop(storage); // Release storage read lock before waiting
        metal_device.wait_until_completed()?;

        // Read back results
        let output_ptr = output_buffer.contents() as *const f32;
        let output = unsafe { std::slice::from_raw_parts(output_ptr, out_features) };

        Ok(output.to_vec())
    }

    /// Upload raw MXFP4 weight data to a Metal buffer.
    ///
    /// Use this to preload expert weights onto the GPU for GPU Resident inference.
    /// The returned buffer can be passed to `mxfp4_matmul_metal_sync` or
    /// `mxfp4_matmul_metal_tensor`.
    pub fn upload_mxfp4_weights(device: &Device, data: &[u8]) -> Result<std::sync::Arc<Buffer>> {
        let metal_device = match device {
            Device::Metal(m) => m,
            _ => {
                return Err(candle_core::Error::Msg(
                    "upload_mxfp4_weights requires Metal device".into(),
                ))
            }
        };
        metal_device.new_buffer_with_data(data)
    }

    /// Upload raw quantized weight data (Q5_0 or Q8_0) to a Metal buffer.
    pub fn upload_quantized_weights(device: &Device, data: &[u8]) -> Result<std::sync::Arc<Buffer>> {
        let metal_device = match device {
            Device::Metal(m) => m,
            _ => {
                return Err(candle_core::Error::Msg(
                    "upload_quantized_weights requires Metal device".into(),
                ))
            }
        };
        metal_device.new_buffer_with_data(data)
    }

}

#[cfg(feature = "metal")]
pub use inner::*;
