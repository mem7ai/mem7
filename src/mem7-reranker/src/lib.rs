mod cohere;
mod llm;

pub use cohere::CohereReranker;
pub use llm::LlmReranker;

use std::sync::Arc;

use async_trait::async_trait;
use mem7_config::RerankerConfig;
use mem7_error::{Mem7Error, Result};
use serde_json::Value;
use tracing::info;
use uuid::Uuid;

/// A document to be reranked.
#[derive(Debug, Clone)]
pub struct RerankDocument {
    pub id: Uuid,
    pub text: String,
    pub score: f32,
    pub payload: Value,
}

/// Result after reranking.
#[derive(Debug, Clone)]
pub struct RerankResult {
    pub id: Uuid,
    pub text: String,
    pub rerank_score: f32,
    pub original_score: f32,
    pub payload: Value,
}

#[async_trait]
pub trait RerankerClient: Send + Sync {
    async fn rerank(
        &self,
        query: &str,
        documents: &[RerankDocument],
        top_k: usize,
    ) -> Result<Vec<RerankResult>>;
}

/// Create a reranker from config.
pub fn create_reranker(config: &RerankerConfig) -> Result<Arc<dyn RerankerClient>> {
    match config.provider.as_str() {
        "cohere" => {
            let api_key = config.api_key.as_deref().ok_or_else(|| {
                Mem7Error::Config("reranker api_key is required for Cohere".into())
            })?;
            let model = config.model.as_deref().unwrap_or("rerank-v3.5");
            info!(provider = "cohere", model, "using Cohere reranker");
            Ok(Arc::new(CohereReranker::new(api_key, model)))
        }
        "llm" => {
            let base_url = config.base_url.as_deref().ok_or_else(|| {
                Mem7Error::Config("reranker base_url is required for LLM reranker".into())
            })?;
            let api_key = config.api_key.as_deref().unwrap_or("");
            let model = config.model.as_deref().ok_or_else(|| {
                Mem7Error::Config("reranker model is required for LLM reranker".into())
            })?;
            info!(provider = "llm", model, "using LLM reranker");
            Ok(Arc::new(LlmReranker::new(base_url, api_key, model)))
        }
        other => Err(Mem7Error::Config(format!(
            "unknown reranker provider: {other}"
        ))),
    }
}
