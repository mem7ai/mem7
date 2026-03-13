use serde::{Deserialize, Serialize};

/// An entity extracted from text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub name: String,
    pub entity_type: String,
}

/// A relationship between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub source: String,
    pub relationship: String,
    pub destination: String,
}

/// A single result from a graph search (subject-predicate-object triple).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSearchResult {
    pub source: String,
    pub relationship: String,
    pub destination: String,
}
