use async_trait::async_trait;
use mem7_config::EmbeddingConfig;
use mem7_error::{Mem7Error, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::EmbeddingClient;

/// An OpenAI-compatible embedding client that works with both OpenAI and vLLM.
pub struct OpenAICompatibleEmbedding {
    client: Client,
    config: EmbeddingConfig,
}

impl OpenAICompatibleEmbedding {
    pub fn new(config: EmbeddingConfig) -> Self {
        let client = Client::new();
        Self { client, config }
    }
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[async_trait]
impl EmbeddingClient for OpenAICompatibleEmbedding {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/embeddings", self.config.base_url.trim_end_matches('/'));

        let body = EmbeddingRequest {
            model: self.config.model.clone(),
            input: texts.to_vec(),
        };

        debug!(url = %url, model = %self.config.model, count = texts.len(), "sending embedding request");

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Mem7Error::Embedding(format!("HTTP {status}: {text}")));
        }

        let data: EmbeddingResponse = resp.json().await?;
        Ok(data.data.into_iter().map(|d| d.embedding).collect())
    }
}
