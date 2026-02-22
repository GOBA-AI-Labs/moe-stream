//! Extensible chat template system.
//!
//! Supports multiple chat formats (ChatML, Llama3, Mistral, Gemma, DeepSeekV3,
//! Phi, CommandR, Vicuna) with automatic detection from model name/architecture.

/// A chat message with role and content.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Supported chat template formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// Qwen / ChatML format: `<|im_start|>role\ncontent<|im_end|>`
    ChatML,
    /// Llama 3 format: `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`
    Llama3,
    /// Mistral format: `[INST] content [/INST]`
    Mistral,
    /// Gemma format: `<start_of_turn>role\ncontent<end_of_turn>`
    Gemma,
    /// DeepSeek V3 format: similar to ChatML with `<|User|>` / `<|Assistant|>` tags
    DeepSeekV3,
    /// Microsoft Phi-3/4 format: `<|system|>\ncontent<|end|>\n<|user|>\ncontent<|end|>\n<|assistant|>\n`
    Phi,
    /// Cohere Command-R format with `<|START_OF_TURN_TOKEN|>` / `<|END_OF_TURN_TOKEN|>` tags
    CommandR,
    /// Vicuna / LLaMA-2-Chat format: `USER: content\nASSISTANT: `
    Vicuna,
}

impl ChatTemplate {
    /// Auto-detect template from model name and architecture strings.
    ///
    /// Detection priority (first match wins):
    /// - "phi" -> Phi
    /// - "command" or "cohere" or "c4ai" -> CommandR
    /// - "vicuna" -> Vicuna
    /// - "dbrx" or "llama" -> Llama3
    /// - "mistral" or "mixtral" -> Mistral
    /// - "gemma" -> Gemma
    /// - "deepseek" -> DeepSeekV3
    /// - "arctic" / "jamba" / "olmo" / "grok" / "yi" -> ChatML
    /// - everything else -> ChatML (Qwen default)
    pub fn detect(model_name: &str, architecture: &str) -> Self {
        let name_lower = model_name.to_lowercase();
        let arch_lower = architecture.to_lowercase();

        if name_lower.contains("phi") || arch_lower.contains("phi") {
            ChatTemplate::Phi
        } else if name_lower.contains("command") || name_lower.contains("cohere")
            || name_lower.contains("c4ai")
            || arch_lower.contains("command") || arch_lower.contains("cohere")
            || arch_lower.contains("c4ai")
        {
            ChatTemplate::CommandR
        } else if name_lower.contains("vicuna") || arch_lower.contains("vicuna") {
            ChatTemplate::Vicuna
        } else if name_lower.contains("dbrx") || arch_lower.contains("dbrx")
            || name_lower.contains("llama") || arch_lower.contains("llama")
        {
            ChatTemplate::Llama3
        } else if name_lower.contains("mistral") || name_lower.contains("mixtral")
            || arch_lower.contains("mistral") || arch_lower.contains("mixtral")
        {
            ChatTemplate::Mistral
        } else if name_lower.contains("gemma") || arch_lower.contains("gemma") {
            ChatTemplate::Gemma
        } else if name_lower.contains("deepseek") || arch_lower.contains("deepseek") {
            ChatTemplate::DeepSeekV3
        } else {
            // ChatML covers: arctic, jamba, olmo, grok, yi, qwen, and others
            ChatTemplate::ChatML
        }
    }

    /// Apply this template to a list of messages, returning the formatted prompt string.
    ///
    /// The returned string includes the assistant turn prefix, ready for generation.
    pub fn apply(&self, messages: &[ChatMessage]) -> String {
        match self {
            ChatTemplate::ChatML => Self::apply_chatml(messages),
            ChatTemplate::Llama3 => Self::apply_llama3(messages),
            ChatTemplate::Mistral => Self::apply_mistral(messages),
            ChatTemplate::Gemma => Self::apply_gemma(messages),
            ChatTemplate::DeepSeekV3 => Self::apply_deepseek_v3(messages),
            ChatTemplate::Phi => Self::apply_phi(messages),
            ChatTemplate::CommandR => Self::apply_command_r(messages),
            ChatTemplate::Vicuna => Self::apply_vicuna(messages),
        }
    }

    /// Return the EOS token IDs for this template.
    pub fn eos_token_ids(&self) -> &'static [u32] {
        match self {
            // <|endoftext|> = 151643, <|im_end|> = 151645
            ChatTemplate::ChatML => &[151643, 151645],
            // <|end_of_text|> = 128001, <|eot_id|> = 128009
            ChatTemplate::Llama3 => &[128001, 128009],
            // </s> = 2
            ChatTemplate::Mistral => &[2],
            // <eos> = 1, <end_of_turn> = 107
            ChatTemplate::Gemma => &[1, 107],
            // <|end_of_sentence|> = 100001
            ChatTemplate::DeepSeekV3 => &[100001],
            // <|end|> = 32000, <|endoftext|> = 32007
            ChatTemplate::Phi => &[32000, 32007],
            // <|END_OF_TURN_TOKEN|> = 255001
            ChatTemplate::CommandR => &[255001],
            // </s> = 2
            ChatTemplate::Vicuna => &[2],
        }
    }

    /// Check if a token ID is an EOS token for this template.
    pub fn is_eos(&self, token_id: u32) -> bool {
        self.eos_token_ids().contains(&token_id)
    }

    /// Return a human-readable name for this template.
    pub fn name(&self) -> &'static str {
        match self {
            ChatTemplate::ChatML => "ChatML",
            ChatTemplate::Llama3 => "Llama3",
            ChatTemplate::Mistral => "Mistral",
            ChatTemplate::Gemma => "Gemma",
            ChatTemplate::DeepSeekV3 => "DeepSeekV3",
            ChatTemplate::Phi => "Phi",
            ChatTemplate::CommandR => "CommandR",
            ChatTemplate::Vicuna => "Vicuna",
        }
    }

    // --- Private format methods ---

    fn apply_chatml(messages: &[ChatMessage]) -> String {
        let mut out = String::new();
        for msg in messages {
            out.push_str("<|im_start|>");
            out.push_str(&msg.role);
            out.push('\n');
            out.push_str(&msg.content);
            out.push_str("<|im_end|>\n");
        }
        out.push_str("<|im_start|>assistant\n");
        out
    }

    fn apply_llama3(messages: &[ChatMessage]) -> String {
        let mut out = String::from("<|begin_of_text|>");
        for msg in messages {
            out.push_str("<|start_header_id|>");
            out.push_str(&msg.role);
            out.push_str("<|end_header_id|>\n\n");
            out.push_str(&msg.content);
            out.push_str("<|eot_id|>");
        }
        out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        out
    }

    fn apply_mistral(messages: &[ChatMessage]) -> String {
        let mut out = String::from("<s>");
        let mut i = 0;
        while i < messages.len() {
            let msg = &messages[i];
            match msg.role.as_str() {
                "system" | "user" => {
                    out.push_str("[INST] ");
                    out.push_str(&msg.content);
                    out.push_str(" [/INST]");
                    // If next message is assistant, append it
                    if i + 1 < messages.len() && messages[i + 1].role == "assistant" {
                        i += 1;
                        out.push_str(&messages[i].content);
                        out.push_str("</s>");
                    }
                }
                "assistant" => {
                    // Standalone assistant (shouldn't happen in normal flow)
                    out.push_str(&msg.content);
                    out.push_str("</s>");
                }
                _ => {
                    out.push_str(&msg.content);
                }
            }
            i += 1;
        }
        out
    }

    fn apply_gemma(messages: &[ChatMessage]) -> String {
        let mut out = String::new();
        for msg in messages {
            out.push_str("<start_of_turn>");
            // Gemma uses "user" and "model" roles
            let role = if msg.role == "assistant" { "model" } else { &msg.role };
            out.push_str(role);
            out.push('\n');
            out.push_str(&msg.content);
            out.push_str("<end_of_turn>\n");
        }
        out.push_str("<start_of_turn>model\n");
        out
    }

    fn apply_deepseek_v3(messages: &[ChatMessage]) -> String {
        let mut out = String::new();
        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    out.push_str("<|System|>");
                    out.push_str(&msg.content);
                    out.push_str("<|end\u{2581}of\u{2581}sentence|>");
                }
                "user" => {
                    out.push_str("<|User|>");
                    out.push_str(&msg.content);
                    out.push_str("<|end\u{2581}of\u{2581}sentence|>");
                }
                "assistant" => {
                    out.push_str("<|Assistant|>");
                    out.push_str(&msg.content);
                    out.push_str("<|end\u{2581}of\u{2581}sentence|>");
                }
                _ => {
                    out.push_str(&msg.content);
                }
            }
        }
        out.push_str("<|Assistant|>");
        out
    }

    fn apply_phi(messages: &[ChatMessage]) -> String {
        let mut out = String::new();
        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    out.push_str("<|system|>\n");
                    out.push_str(&msg.content);
                    out.push_str("<|end|>\n");
                }
                "user" => {
                    out.push_str("<|user|>\n");
                    out.push_str(&msg.content);
                    out.push_str("<|end|>\n");
                }
                "assistant" => {
                    out.push_str("<|assistant|>\n");
                    out.push_str(&msg.content);
                    out.push_str("<|end|>\n");
                }
                _ => {
                    out.push_str(&msg.content);
                }
            }
        }
        out.push_str("<|assistant|>\n");
        out
    }

    fn apply_command_r(messages: &[ChatMessage]) -> String {
        let mut out = String::new();
        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    out.push_str("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>");
                    out.push_str(&msg.content);
                    out.push_str("<|END_OF_TURN_TOKEN|>");
                }
                "user" => {
                    out.push_str("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>");
                    out.push_str(&msg.content);
                    out.push_str("<|END_OF_TURN_TOKEN|>");
                }
                "assistant" => {
                    out.push_str("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
                    out.push_str(&msg.content);
                    out.push_str("<|END_OF_TURN_TOKEN|>");
                }
                _ => {
                    out.push_str(&msg.content);
                }
            }
        }
        out.push_str("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
        out
    }

    fn apply_vicuna(messages: &[ChatMessage]) -> String {
        let mut out = String::new();
        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    out.push_str(&msg.content);
                    out.push('\n');
                }
                "user" => {
                    out.push_str("USER: ");
                    out.push_str(&msg.content);
                    out.push('\n');
                }
                "assistant" => {
                    out.push_str("ASSISTANT: ");
                    out.push_str(&msg.content);
                    out.push_str("</s>\n");
                }
                _ => {
                    out.push_str(&msg.content);
                }
            }
        }
        out.push_str("ASSISTANT: ");
        out
    }
}
