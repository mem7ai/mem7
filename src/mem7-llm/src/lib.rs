mod openai;
mod types;

pub use openai::OpenAICompatibleLlm;
pub use types::*;

use std::sync::Arc;

use async_trait::async_trait;
use mem7_config::LlmConfig;
use mem7_error::{Mem7Error, Result};

#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn chat_completion(
        &self,
        messages: &[LlmMessage],
        response_format: Option<&ResponseFormat>,
    ) -> Result<LlmResponse>;
}

/// Create an LLM client from config. All OpenAI-compatible providers
/// (OpenAI, Ollama, vLLM, LM Studio, DeepSeek, etc.) share the same client.
pub fn create_llm(config: &LlmConfig) -> Result<Arc<dyn LlmClient>> {
    match config.provider.as_str() {
        "openai" | "ollama" | "vllm" | "lmstudio" | "deepseek" => {
            Ok(Arc::new(OpenAICompatibleLlm::new(config.clone())))
        }
        other => Err(Mem7Error::Config(format!("unknown LLM provider: {other}"))),
    }
}
