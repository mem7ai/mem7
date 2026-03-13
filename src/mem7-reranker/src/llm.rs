use async_trait::async_trait;
use mem7_config::LlmConfig;
use mem7_error::Result;
use mem7_llm::{LlmClient, LlmMessage, OpenAICompatibleLlm};
use tracing::warn;

use crate::{RerankDocument, RerankResult, RerankerClient};

pub struct LlmReranker {
    llm: OpenAICompatibleLlm,
}

impl LlmReranker {
    pub fn new(base_url: &str, api_key: &str, model: &str) -> Self {
        let config = LlmConfig {
            provider: "openai".into(),
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            temperature: 0.0,
            max_tokens: 50,
        };
        Self {
            llm: OpenAICompatibleLlm::new(config),
        }
    }
}

const SCORING_PROMPT: &str = "\
Rate the relevance of the following document to the query on a scale from 0.0 to 1.0.
Respond with ONLY a single number (e.g. 0.85). No explanation.

Query: {query}

Document: {document}

Score:";

#[async_trait]
impl RerankerClient for LlmReranker {
    async fn rerank(
        &self,
        query: &str,
        documents: &[RerankDocument],
        top_k: usize,
    ) -> Result<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let mut scored: Vec<RerankResult> = Vec::with_capacity(documents.len());

        for doc in documents {
            let prompt = SCORING_PROMPT
                .replace("{query}", query)
                .replace("{document}", &doc.text);

            let messages = vec![LlmMessage::user(prompt)];

            let score = match self.llm.chat_completion(&messages, None).await {
                Ok(resp) => parse_score(&resp.content),
                Err(e) => {
                    warn!(doc_id = %doc.id, error = %e, "LLM rerank scoring failed, using 0.5");
                    0.5
                }
            };

            scored.push(RerankResult {
                id: doc.id,
                text: doc.text.clone(),
                rerank_score: score,
                original_score: doc.score,
                payload: doc.payload.clone(),
            });
        }

        scored.sort_by(|a, b| {
            b.rerank_score
                .partial_cmp(&a.rerank_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);

        Ok(scored)
    }
}

fn parse_score(raw: &str) -> f32 {
    raw.trim()
        .parse::<f32>()
        .unwrap_or_else(|_| {
            raw.trim()
                .split(|c: char| !c.is_ascii_digit() && c != '.')
                .find_map(|s| s.parse::<f32>().ok())
                .unwrap_or(0.5)
        })
        .clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_score_clean() {
        assert!((parse_score("0.85") - 0.85).abs() < f32::EPSILON);
        assert!((parse_score(" 0.92 \n") - 0.92).abs() < f32::EPSILON);
    }

    #[test]
    fn test_parse_score_with_text() {
        let score = parse_score("The relevance score is 0.75");
        assert!((score - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_parse_score_clamped() {
        assert!((parse_score("1.5") - 1.0).abs() < f32::EPSILON);
        assert!((parse_score("-0.3") - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_parse_score_garbage() {
        assert!((parse_score("not a number") - 0.5).abs() < f32::EPSILON);
    }
}
