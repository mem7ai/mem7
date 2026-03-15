use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};

/// A chat message sent to the LLM.
///
/// When `images` is empty, the message serializes as the standard
/// `{"role": "...", "content": "..."}` form.  When images are present
/// it emits the multi-modal content-array format required by vision
/// models (`[{"type":"text","text":"..."}, {"type":"image_url",...}]`).
#[derive(Debug, Clone, Deserialize)]
pub struct LlmMessage {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub images: Vec<String>,
}

impl Serialize for LlmMessage {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.images.is_empty() {
            let mut map = serializer.serialize_map(Some(2))?;
            map.serialize_entry("role", &self.role)?;
            map.serialize_entry("content", &self.content)?;
            map.end()
        } else {
            let mut parts: Vec<serde_json::Value> = Vec::new();
            if !self.content.is_empty() {
                parts.push(serde_json::json!({"type": "text", "text": &self.content}));
            }
            for url in &self.images {
                parts.push(serde_json::json!({"type": "image_url", "image_url": {"url": url}}));
            }
            let mut map = serializer.serialize_map(Some(2))?;
            map.serialize_entry("role", &self.role)?;
            map.serialize_entry("content", &parts)?;
            map.end()
        }
    }
}

impl LlmMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
            images: Vec::new(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
            images: Vec::new(),
        }
    }

    pub fn user_with_images(content: impl Into<String>, images: Vec<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
            images,
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
