mod client;

pub use client::OpenAICompatibleEmbedding;

use async_trait::async_trait;
use mem7_error::Result;

#[async_trait]
pub trait EmbeddingClient: Send + Sync {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}
