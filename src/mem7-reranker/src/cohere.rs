use async_trait::async_trait;
use mem7_error::{Mem7Error, Result};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::{RerankDocument, RerankResult, RerankerClient};

pub struct CohereReranker {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl CohereReranker {
    pub fn new(api_key: &str, model: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.to_string(),
            model: model.to_string(),
        }
    }
}

#[derive(Serialize)]
struct CohereRerankRequest {
    model: String,
    query: String,
    documents: Vec<String>,
    top_n: usize,
}

#[derive(Deserialize)]
struct CohereRerankResponse {
    results: Vec<CohereRerankResult>,
}

#[derive(Deserialize)]
struct CohereRerankResult {
    index: usize,
    relevance_score: f32,
}

#[async_trait]
impl RerankerClient for CohereReranker {
    async fn rerank(
        &self,
        query: &str,
        documents: &[RerankDocument],
        top_k: usize,
    ) -> Result<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let doc_texts: Vec<String> = documents.iter().map(|d| d.text.clone()).collect();

        let req = CohereRerankRequest {
            model: self.model.clone(),
            query: query.to_string(),
            documents: doc_texts,
            top_n: top_k,
        };

        let resp = self
            .client
            .post("https://api.cohere.com/v2/rerank")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&req)
            .send()
            .await
            .map_err(|e| Mem7Error::Reranker(format!("Cohere HTTP error: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            warn!(%status, %body, "Cohere rerank failed, falling back to original order");
            return Ok(fallback_results(documents, top_k));
        }

        let cohere_resp: CohereRerankResponse = resp
            .json()
            .await
            .map_err(|e| Mem7Error::Reranker(format!("Cohere response parse error: {e}")))?;

        let mut results: Vec<RerankResult> = cohere_resp
            .results
            .into_iter()
            .filter_map(|r| {
                documents.get(r.index).map(|doc| RerankResult {
                    id: doc.id,
                    text: doc.text.clone(),
                    rerank_score: r.relevance_score,
                    original_score: doc.score,
                    payload: doc.payload.clone(),
                })
            })
            .collect();

        results.sort_by(|a, b| {
            b.rerank_score
                .partial_cmp(&a.rerank_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);

        Ok(results)
    }
}

fn fallback_results(documents: &[RerankDocument], top_k: usize) -> Vec<RerankResult> {
    documents
        .iter()
        .take(top_k)
        .map(|doc| RerankResult {
            id: doc.id,
            text: doc.text.clone(),
            rerank_score: doc.score,
            original_score: doc.score,
            payload: doc.payload.clone(),
        })
        .collect()
}
