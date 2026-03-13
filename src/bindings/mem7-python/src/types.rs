use mem7_core::{
    AddResult, GraphRelation, MemoryActionResult, MemoryEvent, MemoryItem, SearchResult,
};
use mem7_error::Mem7Error;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub fn to_py_err(e: Mem7Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

// ── Output types ─────────────────────────────────────────────────────

#[pyclass(get_all, from_py_object)]
#[derive(Clone)]
pub struct PyMemoryItem {
    pub id: String,
    pub text: String,
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub run_id: Option<String>,
    pub metadata: String,
    pub created_at: String,
    pub updated_at: String,
    pub score: Option<f32>,
}

#[pymethods]
impl PyMemoryItem {
    fn __repr__(&self) -> String {
        format!("MemoryItem(id='{}', text='{}')", self.id, self.text)
    }
}

impl From<MemoryItem> for PyMemoryItem {
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
            score: m.score,
        }
    }
}

#[pyclass(get_all, from_py_object)]
#[derive(Clone)]
pub struct PyMemoryActionResult {
    pub id: String,
    pub action: String,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
}

#[pymethods]
impl PyMemoryActionResult {
    fn __repr__(&self) -> String {
        format!(
            "MemoryActionResult(id='{}', action='{}')",
            self.id, self.action
        )
    }
}

impl From<MemoryActionResult> for PyMemoryActionResult {
    fn from(r: MemoryActionResult) -> Self {
        Self {
            id: r.id.to_string(),
            action: r.action.to_string(),
            old_value: r.old_value,
            new_value: r.new_value,
        }
    }
}

#[pyclass(get_all, from_py_object)]
#[derive(Clone)]
pub struct PyGraphRelation {
    pub source: String,
    pub relationship: String,
    pub destination: String,
}

#[pymethods]
impl PyGraphRelation {
    fn __repr__(&self) -> String {
        format!(
            "GraphRelation({} -[{}]-> {})",
            self.source, self.relationship, self.destination
        )
    }
}

impl From<GraphRelation> for PyGraphRelation {
    fn from(r: GraphRelation) -> Self {
        Self {
            source: r.source,
            relationship: r.relationship,
            destination: r.destination,
        }
    }
}

#[pyclass(get_all, from_py_object)]
#[derive(Clone)]
pub struct PyAddResult {
    pub results: Vec<PyMemoryActionResult>,
    pub relations: Vec<PyGraphRelation>,
}

#[pymethods]
impl PyAddResult {
    fn __repr__(&self) -> String {
        format!(
            "AddResult(count={}, relations={})",
            self.results.len(),
            self.relations.len()
        )
    }
}

impl From<AddResult> for PyAddResult {
    fn from(r: AddResult) -> Self {
        Self {
            results: r.results.into_iter().map(Into::into).collect(),
            relations: r.relations.into_iter().map(Into::into).collect(),
        }
    }
}

#[pyclass(get_all, from_py_object)]
#[derive(Clone)]
pub struct PySearchResult {
    pub memories: Vec<PyMemoryItem>,
    pub relations: Vec<PyGraphRelation>,
}

#[pymethods]
impl PySearchResult {
    fn __repr__(&self) -> String {
        format!(
            "SearchResult(count={}, relations={})",
            self.memories.len(),
            self.relations.len()
        )
    }
}

impl From<SearchResult> for PySearchResult {
    fn from(r: SearchResult) -> Self {
        Self {
            memories: r.memories.into_iter().map(Into::into).collect(),
            relations: r.relations.into_iter().map(Into::into).collect(),
        }
    }
}

#[pyclass(get_all, from_py_object)]
#[derive(Clone)]
pub struct PyMemoryEvent {
    pub id: String,
    pub memory_id: String,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
    pub action: String,
    pub created_at: String,
}

#[pymethods]
impl PyMemoryEvent {
    fn __repr__(&self) -> String {
        format!("MemoryEvent(id='{}', action='{}')", self.id, self.action)
    }
}

impl From<MemoryEvent> for PyMemoryEvent {
    fn from(e: MemoryEvent) -> Self {
        Self {
            id: e.id.to_string(),
            memory_id: e.memory_id.to_string(),
            old_value: e.old_value,
            new_value: e.new_value,
            action: e.action.to_string(),
            created_at: e.created_at,
        }
    }
}
