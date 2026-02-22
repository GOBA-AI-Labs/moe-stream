//! Neural network operations for MoE inference.
//!
//! All operations work on candle Tensors and dispatch to Metal when available.

pub mod norm;
pub mod attention;
pub mod activation;

pub use norm::rms_norm;
pub use attention::{rotary_embedding, partial_rotary_embedding, scaled_dot_product_attention};
pub use activation::{silu_and_mul, swiglu_oai, sigmoid, softplus, l2_norm};
