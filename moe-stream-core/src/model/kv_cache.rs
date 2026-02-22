//! KV-cache for autoregressive generation.
//!
//! Stores key/value tensors for each layer, enabling O(1) decode per token
//! by reusing previous KV states.

use candle_core::{Result, Tensor};

/// Per-layer KV-cache entry.
struct LayerCache {
    key: Tensor,
    value: Tensor,
}

/// KV-cache for all layers.
pub struct KvCache {
    layers: Vec<Option<LayerCache>>,
}

impl KvCache {
    pub fn new(num_layers: usize) -> Self {
        let layers = (0..num_layers).map(|_| None).collect();
        Self { layers }
    }

    /// Get current sequence length from cached KV for a layer.
    pub fn seq_len(&self, layer_idx: usize) -> usize {
        self.layers[layer_idx]
            .as_ref()
            .map(|c| c.key.dim(2).unwrap_or(0))
            .unwrap_or(0)
    }

    /// Update cache: concatenate new K/V with existing cached K/V.
    /// Returns the full (cached) K and V tensors.
    ///
    /// new_k, new_v shape: [batch, kv_heads, new_seq_len, head_dim]
    pub fn update(
        &mut self,
        layer_idx: usize,
        new_k: &Tensor,
        new_v: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let entry = &mut self.layers[layer_idx];
        match entry {
            Some(cached) => {
                let k = Tensor::cat(&[&cached.key, new_k], 2)?;
                let v = Tensor::cat(&[&cached.value, new_v], 2)?;
                cached.key = k.clone();
                cached.value = v.clone();
                Ok((k, v))
            }
            None => {
                *entry = Some(LayerCache {
                    key: new_k.clone(),
                    value: new_v.clone(),
                });
                Ok((new_k.clone(), new_v.clone()))
            }
        }
    }

    /// Clear all cached KV states.
    pub fn clear(&mut self) {
        for entry in &mut self.layers {
            *entry = None;
        }
    }
}
