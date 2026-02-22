//! HuggingFace → GGUF tensor name mapping.
//!
//! Maps HuggingFace-style tensor names (e.g., `model.layers.0.self_attn.q_proj.weight`)
//! to GGUF tensor names (e.g., `blk.0.attn_q.weight`).
//! For stacked expert tensors, returns (gguf_name, expert_idx).

use std::collections::HashMap;

/// Resolved mapping target.
#[derive(Debug, Clone)]
pub enum MappingTarget {
    /// Direct tensor mapping.
    Direct(String),
    /// Stacked expert tensor: (gguf_name, expert_idx)
    ExpertSlice(String, usize),
}

/// Builds and stores HF→GGUF tensor name mappings.
pub struct NameMapper {
    mapping: HashMap<String, MappingTarget>,
}

impl NameMapper {
    /// Build name mapping for a model with the given number of layers and experts.
    pub fn build(num_layers: usize, num_experts: usize) -> Self {
        let mut mapping = HashMap::new();

        // Global tensors
        mapping.insert(
            "model.embed_tokens.weight".to_string(),
            MappingTarget::Direct("token_embd.weight".to_string()),
        );
        mapping.insert(
            "lm_head.weight".to_string(),
            MappingTarget::Direct("output.weight".to_string()),
        );
        mapping.insert(
            "model.norm.weight".to_string(),
            MappingTarget::Direct("output_norm.weight".to_string()),
        );

        // Per-layer mappings
        for i in 0..num_layers {
            let hf_prefix = format!("model.layers.{i}");
            let gguf_prefix = format!("blk.{i}");

            // Norms
            mapping.insert(
                format!("{hf_prefix}.input_layernorm.weight"),
                MappingTarget::Direct(format!("{gguf_prefix}.attn_norm.weight")),
            );
            mapping.insert(
                format!("{hf_prefix}.post_attention_layernorm.weight"),
                MappingTarget::Direct(format!("{gguf_prefix}.ffn_norm.weight")),
            );

            // Attention projections
            for (hf_name, gguf_name) in &[
                ("self_attn.q_proj", "attn_q"),
                ("self_attn.k_proj", "attn_k"),
                ("self_attn.v_proj", "attn_v"),
                ("self_attn.o_proj", "attn_output"),
            ] {
                mapping.insert(
                    format!("{hf_prefix}.{hf_name}.weight"),
                    MappingTarget::Direct(format!("{gguf_prefix}.{gguf_name}.weight")),
                );
            }

            // QK norms (Qwen3-specific)
            mapping.insert(
                format!("{hf_prefix}.self_attn.q_norm.weight"),
                MappingTarget::Direct(format!("{gguf_prefix}.attn_q_norm.weight")),
            );
            mapping.insert(
                format!("{hf_prefix}.self_attn.k_norm.weight"),
                MappingTarget::Direct(format!("{gguf_prefix}.attn_k_norm.weight")),
            );

            // MoE router gate
            mapping.insert(
                format!("{hf_prefix}.mlp.gate.weight"),
                MappingTarget::Direct(format!("{gguf_prefix}.ffn_gate_inp.weight")),
            );

            // Shared expert (Qwen3-Coder-Next)
            mapping.insert(
                format!("{hf_prefix}.mlp.shared_expert.gate_proj.weight"),
                MappingTarget::Direct(format!("{gguf_prefix}.ffn_gate_shexp.weight")),
            );
            mapping.insert(
                format!("{hf_prefix}.mlp.shared_expert.up_proj.weight"),
                MappingTarget::Direct(format!("{gguf_prefix}.ffn_up_shexp.weight")),
            );
            mapping.insert(
                format!("{hf_prefix}.mlp.shared_expert.down_proj.weight"),
                MappingTarget::Direct(format!("{gguf_prefix}.ffn_down_shexp.weight")),
            );
            mapping.insert(
                format!("{hf_prefix}.mlp.shared_expert_gate.weight"),
                MappingTarget::Direct(format!("{gguf_prefix}.ffn_gate_inp_shexp.weight")),
            );

            // Expert tensors (stacked in GGUF)
            for j in 0..num_experts {
                mapping.insert(
                    format!("{hf_prefix}.mlp.experts.{j}.gate_proj.weight"),
                    MappingTarget::ExpertSlice(
                        format!("{gguf_prefix}.ffn_gate_exps.weight"),
                        j,
                    ),
                );
                mapping.insert(
                    format!("{hf_prefix}.mlp.experts.{j}.up_proj.weight"),
                    MappingTarget::ExpertSlice(
                        format!("{gguf_prefix}.ffn_up_exps.weight"),
                        j,
                    ),
                );
                mapping.insert(
                    format!("{hf_prefix}.mlp.experts.{j}.down_proj.weight"),
                    MappingTarget::ExpertSlice(
                        format!("{gguf_prefix}.ffn_down_exps.weight"),
                        j,
                    ),
                );
            }

            // DeltaNet-specific weights (Qwen3-Coder-Next)
            // These use the same attention weight names in GGUF but represent different operations
            // The layer type (DeltaNet vs Attention) is determined by the ModelAdapter
        }

        Self { mapping }
    }

    /// Resolve a HuggingFace tensor name to a GGUF target.
    pub fn resolve(&self, hf_name: &str) -> Option<&MappingTarget> {
        self.mapping.get(hf_name)
    }

    /// Get all mapped HF names.
    pub fn hf_names(&self) -> impl Iterator<Item = &str> {
        self.mapping.keys().map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_mapping() {
        let mapper = NameMapper::build(48, 128);
        match mapper.resolve("model.embed_tokens.weight") {
            Some(MappingTarget::Direct(name)) => assert_eq!(name, "token_embd.weight"),
            other => panic!("Expected Direct, got {:?}", other),
        }
    }

    #[test]
    fn test_expert_mapping() {
        let mapper = NameMapper::build(48, 128);
        match mapper.resolve("model.layers.5.mlp.experts.42.gate_proj.weight") {
            Some(MappingTarget::ExpertSlice(name, idx)) => {
                assert_eq!(name, "blk.5.ffn_gate_exps.weight");
                assert_eq!(*idx, 42);
            }
            other => panic!("Expected ExpertSlice, got {:?}", other),
        }
    }

    #[test]
    fn test_attention_mapping() {
        let mapper = NameMapper::build(48, 128);
        match mapper.resolve("model.layers.0.self_attn.q_proj.weight") {
            Some(MappingTarget::Direct(name)) => assert_eq!(name, "blk.0.attn_q.weight"),
            other => panic!("Expected Direct, got {:?}", other),
        }
    }
}
