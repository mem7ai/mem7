mod openai;

pub use openai::OpenAICompatibleEmbedding;

use std::sync::Arc;

use async_trait::async_trait;
use mem7_config::EmbeddingConfig;
use mem7_error::{Mem7Error, Result};

#[async_trait]
pub trait EmbeddingClient: Send + Sync {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

/// Create an embedding client from config. All OpenAI-compatible providers
/// (OpenAI, Ollama, vLLM, LM Studio, DeepSeek, etc.) share the same client.
pub fn create_embedding(config: &EmbeddingConfig) -> Result<Arc<dyn EmbeddingClient>> {
    match config.provider.as_str() {
        "openai" | "ollama" | "vllm" | "lmstudio" | "deepseek" => {
            Ok(Arc::new(OpenAICompatibleEmbedding::new(config.clone())))
        }
        other => Err(Mem7Error::Config(format!(
            "unknown embedding provider: {other}"
        ))),
    }
}
