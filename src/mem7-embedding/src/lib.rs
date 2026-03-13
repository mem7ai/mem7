mod openai;

#[cfg(feature = "fastembed")]
mod fastembed_provider;

pub use openai::OpenAICompatibleEmbedding;

#[cfg(feature = "fastembed")]
pub use fastembed_provider::FastEmbedClient;

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
        #[cfg(feature = "fastembed")]
        "fastembed" => Ok(Arc::new(FastEmbedClient::new(
            &config.model,
            config.cache_dir.as_deref(),
        )?)),
        #[cfg(not(feature = "fastembed"))]
        "fastembed" => Err(Mem7Error::Config(
            "fastembed provider requires the `fastembed` feature to be enabled".into(),
        )),
        other => Err(Mem7Error::Config(format!(
            "unknown embedding provider: {other}"
        ))),
    }
}
