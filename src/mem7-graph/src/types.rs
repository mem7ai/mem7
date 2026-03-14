use serde::{Deserialize, Serialize};

/// An entity extracted from text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub name: String,
    pub entity_type: String,
    /// Vector embedding of the entity name, computed by the embedding model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    #[serde(default)]
    pub mentions: u32,
}

/// A relationship between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub source: String,
    pub relationship: String,
    pub destination: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    #[serde(default)]
    pub mentions: u32,
    #[serde(default = "default_valid")]
    pub valid: bool,
}

fn default_valid() -> bool {
    true
}

/// A single result from a graph search (subject-predicate-object triple).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSearchResult {
    pub source: String,
    pub relationship: String,
    pub destination: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}
