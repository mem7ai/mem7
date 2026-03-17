use mem7_core::{
    AddResult, GraphRelation, MemoryAction, MemoryActionResult, MemoryEvent, MemoryItem,
    SearchResult,
};
use mem7_error::Mem7Error;
use napi::bindgen_prelude::*;
use napi_derive::napi;

pub fn to_napi_err(e: Mem7Error) -> Error {
    Error::new(Status::GenericFailure, e.to_string())
}

// ── Input types ──────────────────────────────────────────────────────

#[napi(object)]
pub struct JsChatMessage {
    pub role: String,
    pub content: String,
}

// ── Output types ─────────────────────────────────────────────────────

#[napi(object)]
pub struct JsMemoryItem {
    pub id: String,
    pub text: String,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub run_id: Option<String>,
    pub metadata: String,
    pub created_at: String,
    pub updated_at: String,
    pub score: Option<f64>,
    pub last_accessed_at: Option<String>,
    pub access_count: u32,
}

impl From<MemoryItem> for JsMemoryItem {
    fn from(m: MemoryItem) -> Self {
        Self {
            id: m.id.to_string(),
            text: m.text,
            user_id: m.user_id,
            agent_id: m.agent_id,
            run_id: m.run_id,
            metadata: m.metadata.to_string(),
            created_at: m.created_at,
            updated_at: m.updated_at,
            score: m.score.map(|s| s as f64),
            last_accessed_at: m.last_accessed_at,
            access_count: m.access_count,
        }
    }
}

#[napi(string_enum)]
pub enum JsMemoryAction {
    Add,
    Update,
    Delete,
    None,
}

impl From<MemoryAction> for JsMemoryAction {
    fn from(a: MemoryAction) -> Self {
        match a {
            MemoryAction::Add => Self::Add,
            MemoryAction::Update => Self::Update,
            MemoryAction::Delete => Self::Delete,
            MemoryAction::None => Self::None,
        }
    }
}

#[napi(object)]
pub struct JsMemoryActionResult {
    pub id: String,
    pub action: JsMemoryAction,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
}

impl From<MemoryActionResult> for JsMemoryActionResult {
    fn from(r: MemoryActionResult) -> Self {
        Self {
            id: r.id.to_string(),
            action: r.action.into(),
            old_value: r.old_value,
            new_value: r.new_value,
        }
    }
}

#[napi(object)]
pub struct JsGraphRelation {
    pub source: String,
    pub relationship: String,
    pub destination: String,
    pub score: Option<f64>,
}

impl From<GraphRelation> for JsGraphRelation {
    fn from(r: GraphRelation) -> Self {
        Self {
            source: r.source,
            relationship: r.relationship,
            destination: r.destination,
            score: r.score.map(|s| s as f64),
        }
    }
}

#[napi(object)]
pub struct JsAddResult {
    pub results: Vec<JsMemoryActionResult>,
    pub relations: Vec<JsGraphRelation>,
}

impl From<AddResult> for JsAddResult {
    fn from(r: AddResult) -> Self {
        Self {
            results: r.results.into_iter().map(Into::into).collect(),
            relations: r.relations.into_iter().map(Into::into).collect(),
        }
    }
}

#[napi(object)]
pub struct JsSearchResult {
    pub memories: Vec<JsMemoryItem>,
    pub relations: Vec<JsGraphRelation>,
}

impl From<SearchResult> for JsSearchResult {
    fn from(r: SearchResult) -> Self {
        Self {
            memories: r.memories.into_iter().map(Into::into).collect(),
            relations: r.relations.into_iter().map(Into::into).collect(),
        }
    }
}

#[napi(object)]
pub struct JsMemoryEvent {
    pub id: String,
    pub memory_id: String,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
    pub action: JsMemoryAction,
    pub created_at: String,
}

impl From<MemoryEvent> for JsMemoryEvent {
    fn from(e: MemoryEvent) -> Self {
        Self {
            id: e.id.to_string(),
            memory_id: e.memory_id.to_string(),
            old_value: e.old_value,
            new_value: e.new_value,
            action: e.action.into(),
            created_at: e.created_at,
        }
    }
}
