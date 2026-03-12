use async_trait::async_trait;
use mem7_config::LlmConfig;
use mem7_error::{Mem7Error, Result};
use reqwest::Client;
use tracing::debug;

use crate::{
    ChatCompletionRequest, ChatCompletionResponse, LlmClient, LlmMessage, LlmResponse,
    ResponseFormat,
};

/// An OpenAI-compatible LLM client that works with both OpenAI and vLLM.
pub struct OpenAICompatibleLlm {
    client: Client,
    config: LlmConfig,
}

impl OpenAICompatibleLlm {
    pub fn new(config: LlmConfig) -> Self {
        let client = Client::new();
        Self { client, config }
    }
}

#[async_trait]
impl LlmClient for OpenAICompatibleLlm {
    async fn chat_completion(
        &self,
        messages: &[LlmMessage],
        response_format: Option<&ResponseFormat>,
    ) -> Result<LlmResponse> {
        let url = format!("{}/chat/completions", self.config.base_url.trim_end_matches('/'));

        let body = ChatCompletionRequest {
            model: self.config.model.clone(),
            messages: messages.to_vec(),
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
            response_format: response_format.cloned(),
        };

        debug!(url = %url, model = %self.config.model, "sending chat completion request");

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
            return Err(Mem7Error::Llm(format!("HTTP {status}: {text}")));
        }

        let data: ChatCompletionResponse = resp.json().await?;
        let content = data
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .unwrap_or_default();

        Ok(LlmResponse { content })
    }
}
