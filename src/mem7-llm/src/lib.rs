mod client;
mod types;

pub use client::OpenAICompatibleLlm;
pub use types::*;

use async_trait::async_trait;
use mem7_error::Result;

#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn chat_completion(
        &self,
        messages: &[LlmMessage],
        response_format: Option<&ResponseFormat>,
    ) -> Result<LlmResponse>;
}
