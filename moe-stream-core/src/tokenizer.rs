//! Tokenizer wrapper using HuggingFace tokenizers crate.

use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    /// Load tokenizer from a tokenizer.json file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let inner = HfTokenizer::from_file(path.as_ref())
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
        Ok(Self { inner })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let encoding = self.inner.encode(text, false)
            .map_err(|e| format!("Encode failed: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String, Box<dyn std::error::Error>> {
        self.inner.decode(ids, true)
            .map_err(|e| format!("Decode failed: {}", e).into())
    }

    /// Decode a single token ID to text (for streaming).
    pub fn decode_token(&self, id: u32) -> Result<String, Box<dyn std::error::Error>> {
        self.decode(&[id])
    }
}
