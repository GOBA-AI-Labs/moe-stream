//! GGUF file reader with memory-mapped access and expert slicing.
//!
//! Supports random-access tensor loading from GGUF files with on-demand
//! dequantization. Key innovation: expert slicing from stacked tensors
//! reads only 0.84MB per expert instead of 108MB for all 128 experts.

use std::collections::HashMap;
use std::os::unix::io::AsRawFd;
use std::path::Path;

use memmap2::Mmap;
use thiserror::Error;

use super::dequant::{self, Q2K_BLOCK_SIZE, Q3K_BLOCK_SIZE, Q4K_BLOCK_SIZE, Q5K_BLOCK_SIZE, Q6K_BLOCK_SIZE, BLOCK_ELEMENTS};

// GGUF magic number
const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian

// GGUF metadata value types
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

/// GGML quantization types (matching ggml GGMLQuantizationType enum)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgmlQuantType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    IQ1_M = 24,
    BF16 = 30,
    MXFP4 = 39,
}

impl GgmlQuantType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            15 => Some(Self::Q8_K),
            30 => Some(Self::BF16),
            39 => Some(Self::MXFP4),
            _ => None,
        }
    }

    /// Bytes per element for non-block quantization types.
    pub fn element_size(&self) -> Option<usize> {
        match self {
            Self::F32 => Some(4),
            Self::F16 | Self::BF16 => Some(2),
            _ => None, // Block-quantized types use block_size()
        }
    }

    /// Bytes per block for block-quantized types. Returns (block_bytes, block_elements).
    pub fn block_info(&self) -> Option<(usize, usize)> {
        match self {
            Self::Q2_K => Some((Q2K_BLOCK_SIZE, BLOCK_ELEMENTS)),
            Self::Q3_K => Some((Q3K_BLOCK_SIZE, BLOCK_ELEMENTS)),
            Self::Q4_K => Some((Q4K_BLOCK_SIZE, BLOCK_ELEMENTS)),
            Self::Q5_K => Some((Q5K_BLOCK_SIZE, BLOCK_ELEMENTS)),
            Self::Q6_K => Some((Q6K_BLOCK_SIZE, BLOCK_ELEMENTS)),
            Self::Q4_0 => Some((dequant::Q4_0_BLOCK_SIZE, dequant::LEGACY_BLOCK_ELEMENTS)),
            Self::Q4_1 => Some((dequant::Q4_1_BLOCK_SIZE, dequant::LEGACY_BLOCK_ELEMENTS)),
            Self::Q5_0 => Some((dequant::Q5_0_BLOCK_SIZE, dequant::LEGACY_BLOCK_ELEMENTS)),
            Self::Q5_1 => Some((dequant::Q5_1_BLOCK_SIZE, dequant::LEGACY_BLOCK_ELEMENTS)),
            Self::Q8_0 => Some((dequant::Q8_0_BLOCK_SIZE, dequant::LEGACY_BLOCK_ELEMENTS)),
            Self::Q8_1 => Some((dequant::Q8_1_BLOCK_SIZE, dequant::LEGACY_BLOCK_ELEMENTS)),
            Self::MXFP4 => Some((dequant::MXFP4_BLOCK_SIZE, dequant::MXFP4_BLOCK_ELEMENTS)),
            _ => None,
        }
    }

    /// Calculate raw byte size for n_elements of this quant type.
    pub fn raw_size(&self, n_elements: usize) -> usize {
        if let Some(elem_size) = self.element_size() {
            n_elements * elem_size
        } else if let Some((block_bytes, block_elems)) = self.block_info() {
            let n_blocks = n_elements.div_ceil(block_elems);
            n_blocks * block_bytes
        } else {
            panic!("Unsupported quant type: {:?}", self);
        }
    }
}

/// Metadata about a single tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions in GGUF order (reversed from PyTorch/HuggingFace convention).
    /// GGUF stores [cols, rows, ...], PyTorch expects [rows, cols, ...].
    pub dimensions: Vec<u64>,
    pub quant_type: GgmlQuantType,
    /// Byte offset of tensor data within the file (absolute).
    pub data_offset: u64,
    /// Total number of elements.
    pub n_elements: u64,
}

impl TensorInfo {
    /// Shape in PyTorch convention (reversed from GGUF).
    pub fn pt_shape(&self) -> Vec<usize> {
        self.dimensions.iter().rev().map(|&d| d as usize).collect()
    }

    /// Raw byte size of this tensor's data.
    pub fn raw_size(&self) -> usize {
        self.quant_type.raw_size(self.n_elements as usize)
    }
}

/// Metadata value types
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
}

impl MetadataValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint8(v) => Some(*v as u32),
            Self::Uint16(v) => Some(*v as u32),
            Self::Uint32(v) => Some(*v),
            Self::Uint64(v) => Some(*v as u32),
            Self::Int32(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint64(v) => Some(*v),
            Self::Uint32(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            Self::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_u32_array(&self) -> Option<Vec<u32>> {
        match self {
            Self::Array(arr) => {
                let mut result = Vec::with_capacity(arr.len());
                for v in arr {
                    result.push(v.as_u32()?);
                }
                Some(result)
            }
            _ => None,
        }
    }
}

#[derive(Error, Debug)]
pub enum GgufError {
    #[error("IO error: {source}")]
    Io {
        #[from]
        source: std::io::Error,
    },
    #[error("Invalid GGUF magic: 0x{magic:08X}")]
    InvalidMagic { magic: u32 },
    #[error("Unsupported GGUF version: {version}")]
    UnsupportedVersion { version: u32 },
    #[error("Tensor not found: {name}")]
    TensorNotFound { name: String },
    #[error("Unsupported quant type: {quant_type}")]
    UnsupportedQuantType { quant_type: u32 },
    #[error("Expert index {idx} out of range for tensor with {num_experts} experts")]
    ExpertOutOfRange { idx: usize, num_experts: usize },
    #[error("Parse error: {msg}")]
    Parse { msg: String },
    #[error("Data out of bounds: offset {offset} + size {size} exceeds file size {file_size}")]
    OutOfBounds { offset: usize, size: usize, file_size: usize },
}

/// GGUF file reader with memory-mapped access.
pub struct GgufReader {
    mmap: Mmap,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: HashMap<String, TensorInfo>,
    _tensor_data_start: u64,
    /// Separate file descriptor with F_NOCACHE for expert reads.
    /// Bypasses OS page cache so expert I/O doesn't evict DeltaNet/Attention pages.
    nocache_file: std::fs::File,
}

impl GgufReader {
    /// Open a GGUF file and parse its header.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        let file = std::fs::File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut reader = BufReader::new(&mmap);

        // Parse header
        let magic = reader.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic { magic });
        }

        let version = reader.read_u32()?;
        if !(2..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion { version });
        }

        let n_tensors_raw = reader.read_u64()?;
        let n_metadata_raw = reader.read_u64()?;

        // Sanity check: reject absurdly large counts to prevent OOM
        const MAX_METADATA: u64 = 1_000_000;
        const MAX_TENSORS: u64 = 1_000_000;
        if n_metadata_raw > MAX_METADATA {
            return Err(GgufError::Parse { msg: format!(
                "Metadata count {} exceeds maximum {}", n_metadata_raw, MAX_METADATA
            ) });
        }
        if n_tensors_raw > MAX_TENSORS {
            return Err(GgufError::Parse { msg: format!(
                "Tensor count {} exceeds maximum {}", n_tensors_raw, MAX_TENSORS
            ) });
        }
        let n_tensors = n_tensors_raw as usize;
        let n_metadata = n_metadata_raw as usize;

        // Parse metadata
        let mut metadata = HashMap::with_capacity(n_metadata);
        for _ in 0..n_metadata {
            let key = reader.read_gguf_string()?;
            let value = reader.read_metadata_value()?;
            metadata.insert(key, value);
        }

        // Parse tensor infos
        let mut tensor_infos = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = reader.read_gguf_string()?;
            let n_dims = reader.read_u32()? as usize;
            let mut dimensions = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dimensions.push(reader.read_u64()?);
            }
            let quant_type_raw = reader.read_u32()?;
            let quant_type = GgmlQuantType::from_u32(quant_type_raw)
                .ok_or(GgufError::UnsupportedQuantType { quant_type: quant_type_raw })?;
            let offset = reader.read_u64()?;

            let n_elements: u64 = dimensions.iter().product();

            tensor_infos.push((
                name,
                dimensions,
                quant_type,
                offset,
                n_elements,
            ));
        }

        // Align to 32 bytes for tensor data start
        let header_end = reader.pos;
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32);
        let tensor_data_start = (header_end as u64).div_ceil(alignment) * alignment;

        // Build tensor index with absolute offsets
        let mut tensors = HashMap::with_capacity(tensor_infos.len());
        for (name, dimensions, quant_type, offset, n_elements) in tensor_infos {
            let info = TensorInfo {
                name: name.clone(),
                dimensions,
                quant_type,
                data_offset: tensor_data_start + offset,
                n_elements,
            };
            tensors.insert(name, info);
        }

        // Open a second fd with cache bypass for expert reads.
        // Expert data bypasses page cache → DeltaNet/Attention pages stay cached.
        let nocache_file = std::fs::File::open(path.as_ref())?;
        #[cfg(target_os = "macos")]
        unsafe {
            libc::fcntl(nocache_file.as_raw_fd(), libc::F_NOCACHE, 1);
        }
        #[cfg(target_os = "linux")]
        unsafe {
            libc::posix_fadvise(nocache_file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
        }

        Ok(Self {
            mmap,
            metadata,
            tensors,
            _tensor_data_start: tensor_data_start,
            nocache_file,
        })
    }

    /// Get raw bytes for a tensor.
    pub fn tensor_data(&self, name: &str) -> Result<&[u8], GgufError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound { name: name.to_string() })?;
        let start = info.data_offset as usize;
        let size = info.raw_size();
        let end = start.checked_add(size).ok_or_else(|| GgufError::OutOfBounds {
            offset: start, size, file_size: self.mmap.len(),
        })?;
        if end > self.mmap.len() {
            return Err(GgufError::OutOfBounds {
                offset: start, size, file_size: self.mmap.len(),
            });
        }
        Ok(&self.mmap[start..end])
    }

    /// Get raw bytes for a single expert slice from a stacked expert tensor.
    ///
    /// For stacked tensors with shape [n_experts, intermediate, cols], this reads
    /// only the bytes for expert `expert_idx`, which is ~0.84MB for Q4_K (vs 108MB
    /// for the full stacked tensor of 128 experts).
    pub fn expert_slice_data(
        &self,
        name: &str,
        expert_idx: usize,
    ) -> Result<(&[u8], Vec<usize>), GgufError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound { name: name.to_string() })?;

        // Stacked tensor shape in GGUF order: [cols, intermediate, n_experts]
        // PyTorch order: [n_experts, intermediate, cols]
        let pt_shape = info.pt_shape();
        if pt_shape.len() < 2 {
            return Err(GgufError::Parse { msg: format!(
                "Expert tensor {} has {} dims, expected >= 2",
                name,
                pt_shape.len()
            ) });
        }

        let num_experts = pt_shape[0];
        if expert_idx >= num_experts {
            return Err(GgufError::ExpertOutOfRange {
                idx: expert_idx,
                num_experts,
            });
        }

        // Calculate per-expert dimensions and byte size
        let expert_shape: Vec<usize> = pt_shape[1..].to_vec();
        let expert_elements: usize = expert_shape.iter().product();
        let bytes_per_expert = info.quant_type.raw_size(expert_elements);

        let expert_offset = expert_idx.checked_mul(bytes_per_expert).ok_or_else(|| GgufError::Parse {
            msg: format!("Expert offset overflow: {} * {}", expert_idx, bytes_per_expert),
        })?;
        let start = (info.data_offset as usize).checked_add(expert_offset).ok_or_else(|| GgufError::OutOfBounds {
            offset: info.data_offset as usize, size: expert_offset, file_size: self.mmap.len(),
        })?;
        let end = start.checked_add(bytes_per_expert).ok_or_else(|| GgufError::OutOfBounds {
            offset: start, size: bytes_per_expert, file_size: self.mmap.len(),
        })?;
        if end > self.mmap.len() {
            return Err(GgufError::OutOfBounds {
                offset: start, size: bytes_per_expert, file_size: self.mmap.len(),
            });
        }

        Ok((&self.mmap[start..end], expert_shape))
    }

    /// Dequantize a full tensor to f32.
    pub fn dequantize_tensor(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), GgufError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound { name: name.to_string() })?;
        let data = self.tensor_data(name)?;
        let n_elements = info.n_elements as usize;
        let pt_shape = info.pt_shape();

        let float_data = dequantize_raw(data, info.quant_type, n_elements)?;
        Ok((float_data, pt_shape))
    }

    /// Dequantize a single expert slice to f32.
    pub fn dequantize_expert(
        &self,
        name: &str,
        expert_idx: usize,
    ) -> Result<(Vec<f32>, Vec<usize>), GgufError> {
        let (data, expert_shape) = self.expert_slice_data(name, expert_idx)?;
        let info = self.tensors.get(name).unwrap();
        let n_elements: usize = expert_shape.iter().product();

        let float_data = dequantize_raw(data, info.quant_type, n_elements)?;
        Ok((float_data, expert_shape))
    }

    /// Get a single expert slice as a candle QTensor (no dequantization).
    ///
    /// This bypasses the dequant step entirely, constructing a QTensor directly
    /// from the mmap'd Q4_K (or other quantized) bytes. The QTensor can then be
    /// used with QMatMul for quantized matmul without the F32 intermediate.
    pub fn expert_slice_as_qtensor(
        &self,
        name: &str,
        expert_idx: usize,
        device: &candle_core::Device,
    ) -> Result<candle_core::quantized::QTensor, GgufError> {
        let (data, expert_shape) = self.expert_slice_data(name, expert_idx)?;
        let info = self.tensors.get(name).unwrap();
        let n_elements: usize = expert_shape.iter().product();

        // Map our GgmlQuantType to candle's GgmlDType
        let ggml_dtype = match info.quant_type {
            GgmlQuantType::F32 => candle_core::quantized::GgmlDType::F32,
            GgmlQuantType::F16 => candle_core::quantized::GgmlDType::F16,
            GgmlQuantType::Q4_0 => candle_core::quantized::GgmlDType::Q4_0,
            GgmlQuantType::Q4_1 => candle_core::quantized::GgmlDType::Q4_1,
            GgmlQuantType::Q5_0 => candle_core::quantized::GgmlDType::Q5_0,
            GgmlQuantType::Q5_1 => candle_core::quantized::GgmlDType::Q5_1,
            GgmlQuantType::Q8_0 => candle_core::quantized::GgmlDType::Q8_0,
            GgmlQuantType::Q8_1 => candle_core::quantized::GgmlDType::Q8_1,
            GgmlQuantType::Q2_K => candle_core::quantized::GgmlDType::Q2K,
            GgmlQuantType::Q3_K => candle_core::quantized::GgmlDType::Q3K,
            GgmlQuantType::Q4_K => candle_core::quantized::GgmlDType::Q4K,
            GgmlQuantType::Q5_K => candle_core::quantized::GgmlDType::Q5K,
            GgmlQuantType::Q6_K => candle_core::quantized::GgmlDType::Q6K,
            GgmlQuantType::Q8_K => candle_core::quantized::GgmlDType::Q8K,
            other => return Err(GgufError::UnsupportedQuantType { quant_type: other as u32 }),
        };

        // Validate size
        let block_size = ggml_dtype.block_size();
        if n_elements % block_size != 0 {
            return Err(GgufError::Parse {
                msg: format!(
                    "expert elements {} not divisible by block size {}",
                    n_elements, block_size
                ),
            });
        }

        candle_core::quantized::ggml_file::qtensor_from_ggml(
            ggml_dtype,
            data,
            expert_shape,
            device,
        )
        .map_err(|e| GgufError::Parse {
            msg: format!("Failed to create QTensor: {}", e),
        })
    }

    /// Get a full tensor as a candle QTensor (no dequantization).
    ///
    /// Like `expert_slice_as_qtensor` but for non-expert tensors (e.g. attention weights).
    /// Bypasses the dequant→F32→Tensor path, constructing a QTensor directly from mmap bytes.
    pub fn tensor_as_qtensor(
        &self,
        name: &str,
        device: &candle_core::Device,
    ) -> Result<candle_core::quantized::QTensor, GgufError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound { name: name.to_string() })?;
        let data = self.tensor_data(name)?;
        let pt_shape = info.pt_shape();
        let n_elements = info.n_elements as usize;

        let ggml_dtype = match info.quant_type {
            GgmlQuantType::F32 => candle_core::quantized::GgmlDType::F32,
            GgmlQuantType::F16 => candle_core::quantized::GgmlDType::F16,
            GgmlQuantType::Q4_0 => candle_core::quantized::GgmlDType::Q4_0,
            GgmlQuantType::Q4_1 => candle_core::quantized::GgmlDType::Q4_1,
            GgmlQuantType::Q5_0 => candle_core::quantized::GgmlDType::Q5_0,
            GgmlQuantType::Q5_1 => candle_core::quantized::GgmlDType::Q5_1,
            GgmlQuantType::Q8_0 => candle_core::quantized::GgmlDType::Q8_0,
            GgmlQuantType::Q8_1 => candle_core::quantized::GgmlDType::Q8_1,
            GgmlQuantType::Q2_K => candle_core::quantized::GgmlDType::Q2K,
            GgmlQuantType::Q3_K => candle_core::quantized::GgmlDType::Q3K,
            GgmlQuantType::Q4_K => candle_core::quantized::GgmlDType::Q4K,
            GgmlQuantType::Q5_K => candle_core::quantized::GgmlDType::Q5K,
            GgmlQuantType::Q6_K => candle_core::quantized::GgmlDType::Q6K,
            GgmlQuantType::Q8_K => candle_core::quantized::GgmlDType::Q8K,
            other => return Err(GgufError::UnsupportedQuantType { quant_type: other as u32 }),
        };

        let block_size = ggml_dtype.block_size();
        if n_elements % block_size != 0 {
            return Err(GgufError::Parse {
                msg: format!(
                    "tensor {} elements {} not divisible by block size {}",
                    name, n_elements, block_size
                ),
            });
        }

        candle_core::quantized::ggml_file::qtensor_from_ggml(
            ggml_dtype,
            data,
            pt_shape,
            device,
        )
        .map_err(|e| GgufError::Parse {
            msg: format!("Failed to create QTensor for {}: {}", name, e),
        })
    }

    /// Get a metadata value.
    pub fn get_metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }

    /// Check if this GGUF contains VQ-compressed expert weights.
    pub fn is_vq_model(&self) -> bool {
        self.metadata.contains_key("moe_compress.vq_k")
    }

    /// Get VQ block height (returns 0 if not a VQ model).
    pub fn vq_block_h(&self) -> usize {
        self.get_metadata("moe_compress.vq_block_h")
            .and_then(|v| v.as_u32())
            .unwrap_or(0) as usize
    }

    /// Get VQ block width (returns 0 if not a VQ model).
    pub fn vq_block_w(&self) -> usize {
        self.get_metadata("moe_compress.vq_block_w")
            .and_then(|v| v.as_u32())
            .unwrap_or(0) as usize
    }

    /// Get VQ codebook size K (returns 0 if not a VQ model).
    pub fn vq_k(&self) -> usize {
        self.get_metadata("moe_compress.vq_k")
            .and_then(|v| v.as_u32())
            .unwrap_or(0) as usize
    }

    /// Check if this VQ model uses per-expert codebooks (vs shared).
    pub fn is_vq_per_expert(&self) -> bool {
        self.get_metadata("moe_compress.vq_per_expert")
            .and_then(|v| v.as_u32())
            .unwrap_or(0) != 0
    }

    /// Get the total file size in bytes (mmap length).
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// Issue madvise(WILLNEED) for a tensor's mmap region to trigger async SSD readahead.
    /// This is non-blocking: the OS starts fetching pages in the background.
    pub fn prefetch_tensor(&self, name: &str) {
        if let Some(info) = self.tensors.get(name) {
            let start = info.data_offset as usize;
            let len = info.raw_size();
            self.madvise_willneed(start, len);
        }
    }

    /// Issue madvise(WILLNEED) for a single expert slice within a stacked tensor.
    pub fn prefetch_expert_slice(&self, name: &str, expert_idx: usize) {
        if let Some(info) = self.tensors.get(name) {
            let pt_shape = info.pt_shape();
            if pt_shape.len() >= 2 && expert_idx < pt_shape[0] {
                let expert_shape: Vec<usize> = pt_shape[1..].to_vec();
                let expert_elements: usize = expert_shape.iter().product();
                let bytes_per_expert = info.quant_type.raw_size(expert_elements);
                let start = info.data_offset as usize + expert_idx * bytes_per_expert;
                self.madvise_willneed(start, bytes_per_expert);
            }
        }
    }

    /// Internal: page-aligned madvise(MADV_WILLNEED) on the mmap region.
    fn madvise_willneed(&self, offset: usize, len: usize) {
        if len == 0 {
            return;
        }
        let page_size = 16384usize; // macOS ARM64 page size (16KB)
        let aligned_start = offset & !(page_size - 1);
        let aligned_end = (offset + len + page_size - 1) & !(page_size - 1);
        let aligned_len = aligned_end - aligned_start;

        // Safety: mmap is valid for the file's lifetime, madvise is advisory only
        unsafe {
            libc::madvise(
                self.mmap.as_ptr().add(aligned_start) as *mut libc::c_void,
                aligned_len,
                libc::MADV_WILLNEED,
            );
        }
    }

    /// Mark expert pages as reclaimable with MADV_FREE.
    /// The kernel will preferentially reclaim these pages when under memory pressure,
    /// while keeping un-marked pages (DeltaNet, Attention) in cache.
    /// This implements: "追い出すものと追い出さないものを見極めて残すべきものを残す"
    pub fn evict_expert_slice(&self, name: &str, expert_idx: usize) {
        if let Some(info) = self.tensors.get(name) {
            let pt_shape = info.pt_shape();
            if pt_shape.len() >= 2 && expert_idx < pt_shape[0] {
                let expert_shape: Vec<usize> = pt_shape[1..].to_vec();
                let expert_elements: usize = expert_shape.iter().product();
                let bytes_per_expert = info.quant_type.raw_size(expert_elements);
                let start = info.data_offset as usize + expert_idx * bytes_per_expert;
                self.madvise_free(start, bytes_per_expert);
            }
        }
    }

    /// Internal: page-aligned madvise(MADV_FREE) — mark pages as reclaimable.
    /// Unlike MADV_DONTNEED, pages are kept if there's no pressure (soft eviction hint).
    fn madvise_free(&self, offset: usize, len: usize) {
        if len == 0 {
            return;
        }
        let page_size = 16384usize;
        let aligned_start = offset & !(page_size - 1);
        let aligned_end = (offset + len + page_size - 1) & !(page_size - 1);
        let aligned_len = aligned_end - aligned_start;

        unsafe {
            libc::madvise(
                self.mmap.as_ptr().add(aligned_start) as *mut libc::c_void,
                aligned_len,
                libc::MADV_FREE,
            );
        }
    }

    /// Pin (mlock) a specific expert slice's mmap pages in RAM.
    /// Pages stay in Q4 format — no dequantization, no extra memory.
    /// The OS will not evict these pages, guaranteeing zero page-fault I/O.
    pub fn mlock_expert_slice(&self, name: &str, expert_idx: usize) -> Result<(), GgufError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound { name: name.to_string() })?;
        let pt_shape = info.pt_shape();
        if pt_shape.len() >= 2 && expert_idx < pt_shape[0] {
            let expert_shape: Vec<usize> = pt_shape[1..].to_vec();
            let expert_elements: usize = expert_shape.iter().product();
            let bytes_per_expert = info.quant_type.raw_size(expert_elements);
            let start = info.data_offset as usize + expert_idx * bytes_per_expert;
            self.mlock_range(start, bytes_per_expert);
        }
        Ok(())
    }

    /// Pin (mlock) a full tensor's mmap pages in RAM.
    pub fn mlock_tensor(&self, name: &str) -> Result<(), GgufError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound { name: name.to_string() })?;
        let start = info.data_offset as usize;
        let len = info.raw_size();
        self.mlock_range(start, len);
        Ok(())
    }

    /// Unlock (munlock) a specific expert slice's mmap pages, allowing eviction.
    pub fn munlock_expert_slice(&self, name: &str, expert_idx: usize) -> Result<(), GgufError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound { name: name.to_string() })?;
        let pt_shape = info.pt_shape();
        if pt_shape.len() >= 2 && expert_idx < pt_shape[0] {
            let expert_shape: Vec<usize> = pt_shape[1..].to_vec();
            let expert_elements: usize = expert_shape.iter().product();
            let bytes_per_expert = info.quant_type.raw_size(expert_elements);
            let start = info.data_offset as usize + expert_idx * bytes_per_expert;
            self.munlock_range(start, bytes_per_expert);
        }
        Ok(())
    }

    /// Internal: page-aligned mlock on the mmap region.
    /// Pins pages in physical RAM — the OS will not evict them.
    fn mlock_range(&self, offset: usize, len: usize) {
        if len == 0 {
            return;
        }
        let page_size = 16384usize; // macOS ARM64 page size (16KB)
        let aligned_start = offset & !(page_size - 1);
        let aligned_end = (offset + len + page_size - 1) & !(page_size - 1);
        let aligned_len = aligned_end - aligned_start;

        unsafe {
            libc::mlock(
                self.mmap.as_ptr().add(aligned_start) as *const libc::c_void,
                aligned_len,
            );
        }
    }

    /// Internal: page-aligned munlock on the mmap region.
    fn munlock_range(&self, offset: usize, len: usize) {
        if len == 0 {
            return;
        }
        let page_size = 16384usize;
        let aligned_start = offset & !(page_size - 1);
        let aligned_end = (offset + len + page_size - 1) & !(page_size - 1);
        let aligned_len = aligned_end - aligned_start;

        unsafe {
            libc::munlock(
                self.mmap.as_ptr().add(aligned_start) as *const libc::c_void,
                aligned_len,
            );
        }
    }

    /// Read expert slice via pread() through the F_NOCACHE file descriptor.
    /// Data goes directly SSD → user buffer, bypassing OS page cache entirely.
    /// This prevents expert reads from evicting DeltaNet/Attention pages.
    pub fn pread_expert_slice(
        &self,
        name: &str,
        expert_idx: usize,
        buf: &mut Vec<u8>,
    ) -> Result<Vec<usize>, GgufError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound { name: name.to_string() })?;

        let pt_shape = info.pt_shape();
        if pt_shape.len() < 2 {
            return Err(GgufError::Parse {
                msg: format!("Expert tensor {} has {} dims, expected >= 2", name, pt_shape.len()),
            });
        }

        let num_experts = pt_shape[0];
        if expert_idx >= num_experts {
            return Err(GgufError::ExpertOutOfRange {
                idx: expert_idx,
                num_experts,
            });
        }

        let expert_shape: Vec<usize> = pt_shape[1..].to_vec();
        let expert_elements: usize = expert_shape.iter().product();
        let bytes_per_expert = info.quant_type.raw_size(expert_elements);
        let file_offset = info.data_offset + (expert_idx * bytes_per_expert) as u64;

        buf.resize(bytes_per_expert, 0);

        let fd = self.nocache_file.as_raw_fd();
        let mut total_read = 0usize;
        while total_read < bytes_per_expert {
            let ret = unsafe {
                libc::pread(
                    fd,
                    buf[total_read..].as_mut_ptr() as *mut libc::c_void,
                    bytes_per_expert - total_read,
                    (file_offset + total_read as u64) as libc::off_t,
                )
            };
            if ret < 0 {
                let errno = std::io::Error::last_os_error();
                if errno.raw_os_error() == Some(libc::EINTR) {
                    continue;
                }
                return Err(GgufError::Io { source: errno });
            }
            if ret == 0 {
                return Err(GgufError::Parse {
                    msg: format!("pread: unexpected EOF at offset {}", file_offset + total_read as u64),
                });
            }
            total_read += ret as usize;
        }

        Ok(expert_shape)
    }

    /// Dequantize a single expert slice using F_NOCACHE bypass (pread → buffer → dequant).
    pub fn dequantize_expert_nocache(
        &self,
        name: &str,
        expert_idx: usize,
        buf: &mut Vec<u8>,
    ) -> Result<(Vec<f32>, Vec<usize>), GgufError> {
        let expert_shape = self.pread_expert_slice(name, expert_idx, buf)?;
        let info = self.tensors.get(name).unwrap();
        let n_elements: usize = expert_shape.iter().product();

        let float_data = dequantize_raw(buf, info.quant_type, n_elements)?;
        Ok((float_data, expert_shape))
    }
}

/// Dequantize raw bytes based on quantization type.
fn dequantize_raw(
    data: &[u8],
    quant_type: GgmlQuantType,
    n_elements: usize,
) -> Result<Vec<f32>, GgufError> {
    match quant_type {
        GgmlQuantType::F32 => {
            let mut output = vec![0.0f32; n_elements];
            for (i, val) in output.iter_mut().enumerate().take(n_elements) {
                let offset = i * 4;
                *val = f32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
            }
            Ok(output)
        }
        GgmlQuantType::F16 => Ok(dequant::convert_f16_to_f32(data, n_elements)),
        GgmlQuantType::BF16 => {
            // BF16: top 8 bits of exponent+sign same as f32, bottom 16 bits truncated
            let mut output = vec![0.0f32; n_elements];
            for i in 0..n_elements {
                let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
                let f32_bits = (bits as u32) << 16;
                output[i] = f32::from_bits(f32_bits);
            }
            Ok(output)
        }
        GgmlQuantType::Q2_K => Ok(dequant::dequantize_q2k(data, n_elements)),
        GgmlQuantType::Q3_K => Ok(dequant::dequantize_q3k(data, n_elements)),
        GgmlQuantType::Q4_K => Ok(dequant::dequantize_q4k(data, n_elements)),
        GgmlQuantType::Q5_K => Ok(dequant::dequantize_q5k(data, n_elements)),
        GgmlQuantType::Q6_K => Ok(dequant::dequantize_q6k(data, n_elements)),
        GgmlQuantType::Q4_0 => Ok(dequant::dequantize_q40(data, n_elements)),
        GgmlQuantType::Q4_1 => Ok(dequant::dequantize_q41(data, n_elements)),
        GgmlQuantType::Q5_0 => Ok(dequant::dequantize_q50(data, n_elements)),
        GgmlQuantType::Q5_1 => Ok(dequant::dequantize_q51(data, n_elements)),
        GgmlQuantType::Q8_0 => Ok(dequant::dequantize_q80(data, n_elements)),
        GgmlQuantType::Q8_1 => Ok(dequant::dequantize_q81(data, n_elements)),
        GgmlQuantType::MXFP4 => Ok(dequant::dequantize_mxfp4(data, n_elements)),
        other => Err(GgufError::UnsupportedQuantType { quant_type: other as u32 }),
    }
}

// ---- Internal buffer reader for GGUF header parsing ----

struct BufReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BufReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        if self.pos >= self.data.len() {
            return Err(GgufError::Parse { msg: "Unexpected EOF reading u8".into() });
        }
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, GgufError> {
        if self.pos + 2 > self.data.len() {
            return Err(GgufError::Parse { msg: "Unexpected EOF reading u16".into() });
        }
        let v = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    fn read_i16(&mut self) -> Result<i16, GgufError> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        if self.pos + 4 > self.data.len() {
            return Err(GgufError::Parse { msg: "Unexpected EOF reading u32".into() });
        }
        let v = u32::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    fn read_i32(&mut self) -> Result<i32, GgufError> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        if self.pos + 8 > self.data.len() {
            return Err(GgufError::Parse { msg: "Unexpected EOF reading u64".into() });
        }
        let v = u64::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
            self.data[self.pos + 4],
            self.data[self.pos + 5],
            self.data[self.pos + 6],
            self.data[self.pos + 7],
        ]);
        self.pos += 8;
        Ok(v)
    }

    fn read_i64(&mut self) -> Result<i64, GgufError> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32, GgufError> {
        let bits = self.read_u32()?;
        Ok(f32::from_bits(bits))
    }

    fn read_f64(&mut self) -> Result<f64, GgufError> {
        let bits = self.read_u64()?;
        Ok(f64::from_bits(bits))
    }

    fn read_bool(&mut self) -> Result<bool, GgufError> {
        Ok(self.read_u8()? != 0)
    }

    fn read_gguf_string(&mut self) -> Result<String, GgufError> {
        let len_u64 = self.read_u64()?;
        let len = usize::try_from(len_u64).map_err(|_| GgufError::Parse {
            msg: format!("String length {} exceeds platform address space", len_u64),
        })?;
        let end = self.pos.checked_add(len).ok_or_else(|| GgufError::Parse {
            msg: format!("String length {} causes offset overflow at pos {}", len, self.pos),
        })?;
        if end > self.data.len() {
            return Err(GgufError::Parse { msg: format!(
                "String length {} exceeds data at pos {}",
                len, self.pos
            ) });
        }
        let s = std::str::from_utf8(&self.data[self.pos..self.pos + len])
            .map_err(|e| GgufError::Parse { msg: format!("Invalid UTF-8: {}", e) })?
            .to_string();
        self.pos += len;
        Ok(s)
    }

    fn read_metadata_value(&mut self) -> Result<MetadataValue, GgufError> {
        let value_type = self.read_u32()?;
        self.read_typed_value(value_type)
    }

    fn read_typed_value(&mut self, value_type: u32) -> Result<MetadataValue, GgufError> {
        match value_type {
            GGUF_TYPE_UINT8 => Ok(MetadataValue::Uint8(self.read_u8()?)),
            GGUF_TYPE_INT8 => Ok(MetadataValue::Int8(self.read_i8()?)),
            GGUF_TYPE_UINT16 => Ok(MetadataValue::Uint16(self.read_u16()?)),
            GGUF_TYPE_INT16 => Ok(MetadataValue::Int16(self.read_i16()?)),
            GGUF_TYPE_UINT32 => Ok(MetadataValue::Uint32(self.read_u32()?)),
            GGUF_TYPE_INT32 => Ok(MetadataValue::Int32(self.read_i32()?)),
            GGUF_TYPE_FLOAT32 => Ok(MetadataValue::Float32(self.read_f32()?)),
            GGUF_TYPE_BOOL => Ok(MetadataValue::Bool(self.read_bool()?)),
            GGUF_TYPE_STRING => Ok(MetadataValue::String(self.read_gguf_string()?)),
            GGUF_TYPE_UINT64 => Ok(MetadataValue::Uint64(self.read_u64()?)),
            GGUF_TYPE_INT64 => Ok(MetadataValue::Int64(self.read_i64()?)),
            GGUF_TYPE_FLOAT64 => Ok(MetadataValue::Float64(self.read_f64()?)),
            GGUF_TYPE_ARRAY => {
                let elem_type = self.read_u32()?;
                let n_elems_raw = self.read_u64()?;
                const MAX_ARRAY_ELEMS: u64 = 10_000_000;
                if n_elems_raw > MAX_ARRAY_ELEMS {
                    return Err(GgufError::Parse { msg: format!(
                        "Array element count {} exceeds maximum {}", n_elems_raw, MAX_ARRAY_ELEMS
                    ) });
                }
                let n_elems = n_elems_raw as usize;
                let mut elems = Vec::with_capacity(n_elems);
                for _ in 0..n_elems {
                    elems.push(self.read_typed_value(elem_type)?);
                }
                Ok(MetadataValue::Array(elems))
            }
            other => Err(GgufError::Parse { msg: format!(
                "Unknown metadata type: {}",
                other
            ) }),
        }
    }
}
