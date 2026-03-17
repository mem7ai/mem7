use std::sync::Arc;

use mem7_config::MemoryEngineConfig;
use mem7_core::ChatMessage;
use mem7_store::MemoryEngine;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::types::*;

/// Python wrapper around the Rust MemoryEngine.
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

    #[pyo3(signature = (messages, user_id=None, agent_id=None, run_id=None, metadata=None, infer=None))]
    fn add(
        &self,
        messages: Vec<(String, String)>,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
        metadata: Option<String>,
        infer: Option<bool>,
    ) -> PyResult<PyAddResult> {
        let msgs: Vec<ChatMessage> = messages
            .into_iter()
            .map(|(role, content)| ChatMessage {
                role,
                content,
                images: Vec::new(),
            })
            .collect();

        let meta_val: Option<serde_json::Value> = metadata
            .map(|s| serde_json::from_str(&s))
            .transpose()
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid metadata JSON: {e}")))?;

        let result = self
            .rt
            .block_on(self.inner.add(
                &msgs,
                user_id.as_deref(),
                agent_id.as_deref(),
                run_id.as_deref(),
                meta_val.as_ref(),
                infer.unwrap_or(true),
            ))
            .map_err(to_py_err)?;

        Ok(result.into())
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (query, user_id=None, agent_id=None, run_id=None, limit=5, filters=None, rerank=None, threshold=None, task_type=None))]
    fn search(
        &self,
        query: &str,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
        limit: usize,
        filters: Option<String>,
        rerank: Option<bool>,
        threshold: Option<f32>,
        task_type: Option<String>,
    ) -> PyResult<PySearchResult> {
        let filters_val: Option<serde_json::Value> = filters
            .map(|s| serde_json::from_str(&s))
            .transpose()
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid filters JSON: {e}")))?;

        let result = self
            .rt
            .block_on(self.inner.search(
                query,
                user_id.as_deref(),
                agent_id.as_deref(),
                run_id.as_deref(),
                limit,
                filters_val.as_ref(),
                rerank.unwrap_or(true),
                threshold,
                task_type.as_deref(),
            ))
            .map_err(to_py_err)?;

        Ok(result.into())
    }

    fn get(&self, memory_id: &str) -> PyResult<PyMemoryItem> {
        let uuid = uuid::Uuid::parse_str(memory_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid UUID: {e}")))?;

        let item = self.rt.block_on(self.inner.get(uuid)).map_err(to_py_err)?;
        Ok(item.into())
    }

    #[pyo3(signature = (user_id=None, agent_id=None, run_id=None, filters=None, limit=None))]
    fn get_all(
        &self,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
        filters: Option<String>,
        limit: Option<usize>,
    ) -> PyResult<Vec<PyMemoryItem>> {
        let filters_val: Option<serde_json::Value> = filters
            .map(|s| serde_json::from_str(&s))
            .transpose()
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid filters JSON: {e}")))?;

        let items = self
            .rt
            .block_on(self.inner.get_all(
                user_id.as_deref(),
                agent_id.as_deref(),
                run_id.as_deref(),
                filters_val.as_ref(),
                limit,
            ))
            .map_err(to_py_err)?;

        Ok(items.into_iter().map(Into::into).collect())
    }

    fn update(&self, memory_id: &str, new_text: &str) -> PyResult<()> {
        let uuid = uuid::Uuid::parse_str(memory_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid UUID: {e}")))?;

        self.rt
            .block_on(self.inner.update(uuid, new_text))
            .map_err(to_py_err)
    }

    fn delete(&self, memory_id: &str) -> PyResult<()> {
        let uuid = uuid::Uuid::parse_str(memory_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid UUID: {e}")))?;

        self.rt.block_on(self.inner.delete(uuid)).map_err(to_py_err)
    }

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

    fn history(&self, memory_id: &str) -> PyResult<Vec<PyMemoryEvent>> {
        let uuid = uuid::Uuid::parse_str(memory_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid UUID: {e}")))?;

        let events = self
            .rt
            .block_on(self.inner.history(uuid))
            .map_err(to_py_err)?;

        Ok(events.into_iter().map(Into::into).collect())
    }

    fn reset(&self) -> PyResult<()> {
        self.rt.block_on(self.inner.reset()).map_err(to_py_err)
    }
}
