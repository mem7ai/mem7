use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmMessage {
    pub role: String,
    pub content: String,
}

impl LlmMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub content: String,
}

/// Request body for OpenAI-compatible /v1/chat/completions.
#[derive(Debug, Serialize)]
pub(crate) struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<LlmMessage>,
    pub temperature: f32,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
}

/// OpenAI `response_format` parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
}

impl ResponseFormat {
    pub fn json() -> Self {
        Self {
            format_type: "json_object".into(),
        }
    }
}

/// Response body from OpenAI-compatible /v1/chat/completions.
#[derive(Debug, Deserialize)]
pub(crate) struct ChatCompletionResponse {
    pub choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Choice {
    pub message: ChoiceMessage,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChoiceMessage {
    pub content: Option<String>,
}
