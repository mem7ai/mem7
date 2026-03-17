use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A single memory item stored in the system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: Uuid,
    pub text: String,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub run_id: Option<String>,
    pub metadata: serde_json::Value,
    pub created_at: String,
    pub updated_at: String,
    pub score: Option<f32>,
    /// Last time this memory was accessed (written or retrieved).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_accessed_at: Option<String>,
    /// Number of times this memory has been retrieved (rehearsal count).
    #[serde(default)]
    pub access_count: u32,
}

/// A fact extracted by the LLM from a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub text: String,
}

/// The type of memory event recorded in the audit trail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum MemoryAction {
    Add,
    Update,
    Delete,
    None,
}

impl std::fmt::Display for MemoryAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add => write!(f, "ADD"),
            Self::Update => write!(f, "UPDATE"),
            Self::Delete => write!(f, "DELETE"),
            Self::None => write!(f, "NONE"),
        }
    }
}

/// A single entry in the memory audit trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvent {
    pub id: Uuid,
    pub memory_id: Uuid,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
    pub action: MemoryAction,
    pub created_at: String,
}

/// Result of an add() operation for a single memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryActionResult {
    pub id: Uuid,
    pub action: MemoryAction,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
}

/// A graph relation (subject-predicate-object triple).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRelation {
    pub source: String,
    pub relationship: String,
    pub destination: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

/// Aggregated result of a full add() call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddResult {
    pub results: Vec<MemoryActionResult>,
    #[serde(default)]
    pub relations: Vec<GraphRelation>,
}

/// Result of a search() call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub memories: Vec<MemoryItem>,
    #[serde(default)]
    pub relations: Vec<GraphRelation>,
}

/// Filter criteria for querying memories.
///
/// `user_id`, `agent_id`, and `run_id` are first-class equality filters on
/// top-level payload fields. `metadata` carries an optional JSON value that
/// is evaluated against the nested `payload.metadata` object using the
/// filter DSL (simple equality, operators like eq/gt/in, and AND/OR/NOT).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryFilter {
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// A chat message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Optional image URLs or base64 data URIs attached to this message.
    /// When `enable_vision` is set, these are sent to the LLM for description.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
}
