use std::sync::Arc;

use mem7_config::MemoryEngineConfig;
use mem7_core::ChatMessage;
use mem7_store::MemoryEngine;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::types::to_py_err;

/// Python wrapper around the Rust MemoryEngine.
/// All results are returned as JSON strings; the Python shell deserializes them.
#[pyclass]
pub struct PyMemoryEngine {
    inner: Arc<MemoryEngine>,
    rt: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl PyMemoryEngine {
    #[new]
    #[pyo3(signature = (config_json = None))]
    fn new(config_json: Option<&str>) -> PyResult<Self> {
        let config: MemoryEngineConfig = match config_json {
            Some(json) => serde_json::from_str(json)
                .map_err(|e| PyRuntimeError::new_err(format!("Invalid config: {e}")))?,
            None => MemoryEngineConfig::default(),
        };

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to build tokio runtime: {e}")))?;

        let engine = rt.block_on(MemoryEngine::new(config)).map_err(to_py_err)?;

        Ok(Self {
            inner: Arc::new(engine),
            rt: Arc::new(rt),
        })
    }

    /// Add memories from messages. Returns JSON string of AddResult.
    #[pyo3(signature = (messages, user_id=None, agent_id=None, run_id=None))]
    fn add(
        &self,
        messages: Vec<(String, String)>,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
    ) -> PyResult<String> {
        let msgs: Vec<ChatMessage> = messages
            .into_iter()
            .map(|(role, content)| ChatMessage { role, content })
            .collect();

        let engine = self.inner.clone();
        let result = self
            .rt
            .block_on(engine.add(
                &msgs,
                user_id.as_deref(),
                agent_id.as_deref(),
                run_id.as_deref(),
            ))
            .map_err(to_py_err)?;

        serde_json::to_string(&result)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {e}")))
    }

    /// Search memories by semantic similarity. Returns JSON string of SearchResult.
    #[pyo3(signature = (query, user_id=None, agent_id=None, run_id=None, limit=5))]
    fn search(
        &self,
        query: &str,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
        limit: usize,
    ) -> PyResult<String> {
        let engine = self.inner.clone();
        let result = self
            .rt
            .block_on(engine.search(
                query,
                user_id.as_deref(),
                agent_id.as_deref(),
                run_id.as_deref(),
                limit,
            ))
            .map_err(to_py_err)?;

        serde_json::to_string(&result)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {e}")))
    }

    /// Get a single memory by ID. Returns JSON string of MemoryItem.
    fn get(&self, memory_id: &str) -> PyResult<String> {
        let uuid = uuid::Uuid::parse_str(memory_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid UUID: {e}")))?;

        let item = self.rt.block_on(self.inner.get(uuid)).map_err(to_py_err)?;

        serde_json::to_string(&item)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {e}")))
    }

    /// List all memories matching filters. Returns JSON array string.
    #[pyo3(signature = (user_id=None, agent_id=None, run_id=None))]
    fn get_all(
        &self,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
    ) -> PyResult<String> {
        let items = self
            .rt
            .block_on(self.inner.get_all(
                user_id.as_deref(),
                agent_id.as_deref(),
                run_id.as_deref(),
            ))
            .map_err(to_py_err)?;

        serde_json::to_string(&items)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {e}")))
    }

    /// Update a memory's text.
    fn update(&self, memory_id: &str, new_text: &str) -> PyResult<()> {
        let uuid = uuid::Uuid::parse_str(memory_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid UUID: {e}")))?;

        self.rt
            .block_on(self.inner.update(uuid, new_text))
            .map_err(to_py_err)
    }

    /// Delete a memory by ID.
    fn delete(&self, memory_id: &str) -> PyResult<()> {
        let uuid = uuid::Uuid::parse_str(memory_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid UUID: {e}")))?;

        self.rt.block_on(self.inner.delete(uuid)).map_err(to_py_err)
    }

    /// Delete all memories matching filters.
    #[pyo3(signature = (user_id=None, agent_id=None, run_id=None))]
    fn delete_all(
        &self,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
    ) -> PyResult<()> {
        self.rt
            .block_on(self.inner.delete_all(
                user_id.as_deref(),
                agent_id.as_deref(),
                run_id.as_deref(),
            ))
            .map_err(to_py_err)
    }

    /// Get history for a memory. Returns JSON array string.
    fn history(&self, memory_id: &str) -> PyResult<String> {
        let uuid = uuid::Uuid::parse_str(memory_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid UUID: {e}")))?;

        let events = self
            .rt
            .block_on(self.inner.history(uuid))
            .map_err(to_py_err)?;

        serde_json::to_string(&events)
            .map_err(|e| PyRuntimeError::new_err(format!("Serialization error: {e}")))
    }

    /// Reset all data.
    fn reset(&self) -> PyResult<()> {
        self.rt.block_on(self.inner.reset()).map_err(to_py_err)
    }
}
