use std::sync::Arc;

use mem7_config::MemoryEngineConfig;
use mem7_core::ChatMessage;
use mem7_store::MemoryEngine;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::types::*;

/// Async Python wrapper around the Rust MemoryEngine.
///
/// Every method returns a Python awaitable (coroutine) that runs on the
/// global tokio runtime managed by `pyo3-async-runtimes`.
///
/// Use the async classmethod `create` instead of `__init__` because
/// `MemoryEngine::new()` is async.
#[pyclass]
pub struct PyAsyncMemoryEngine {
    inner: Arc<MemoryEngine>,
}

#[pymethods]
impl PyAsyncMemoryEngine {
    #[classmethod]
    #[pyo3(signature = (config_json=None))]
    fn create<'py>(
        _cls: &Bound<'py, pyo3::types::PyType>,
        py: Python<'py>,
        config_json: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let config: MemoryEngineConfig = match config_json.as_deref() {
            Some(json) => serde_json::from_str(json)
                .map_err(|e| PyRuntimeError::new_err(format!("Invalid config: {e}")))?,
            None => MemoryEngineConfig::default(),
        };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let engine = MemoryEngine::new(config).await.map_err(to_py_err)?;
            Ok(PyAsyncMemoryEngine {
                inner: Arc::new(engine),
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (messages, user_id=None, agent_id=None, run_id=None, metadata=None, infer=None))]
    fn add<'py>(
        &self,
        py: Python<'py>,
        messages: Vec<(String, String)>,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
        metadata: Option<String>,
        infer: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
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

        let inner = self.inner.clone();
        let infer = infer.unwrap_or(true);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner
                .add(
                    &msgs,
                    user_id.as_deref(),
                    agent_id.as_deref(),
                    run_id.as_deref(),
                    meta_val.as_ref(),
                    infer,
                )
                .await
                .map_err(to_py_err)?;
            Ok(PyAddResult::from(result))
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (query, user_id=None, agent_id=None, run_id=None, limit=5, filters=None, rerank=None, threshold=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: String,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
        limit: usize,
        filters: Option<String>,
        rerank: Option<bool>,
        threshold: Option<f32>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let filters_val: Option<serde_json::Value> = filters
            .map(|s| serde_json::from_str(&s))
            .transpose()
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid filters JSON: {e}")))?;

        let inner = self.inner.clone();
        let rerank = rerank.unwrap_or(true);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner
                .search(
                    &query,
                    user_id.as_deref(),
                    agent_id.as_deref(),
                    run_id.as_deref(),
                    limit,
                    filters_val.as_ref(),
                    rerank,
                    threshold,
                )
                .await
                .map_err(to_py_err)?;
            Ok(PySearchResult::from(result))
        })
    }

    fn get<'py>(&self, py: Python<'py>, memory_id: String) -> PyResult<Bound<'py, PyAny>> {
        let uuid = uuid::Uuid::parse_str(&memory_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid UUID: {e}")))?;

        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let item = inner.get(uuid).await.map_err(to_py_err)?;
            Ok(PyMemoryItem::from(item))
        })
    }

    #[pyo3(signature = (user_id=None, agent_id=None, run_id=None, filters=None, limit=None))]
    fn get_all<'py>(
        &self,
        py: Python<'py>,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
        filters: Option<String>,
        limit: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let filters_val: Option<serde_json::Value> = filters
            .map(|s| serde_json::from_str(&s))
            .transpose()
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid filters JSON: {e}")))?;

        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let items = inner
                .get_all(
                    user_id.as_deref(),
                    agent_id.as_deref(),
                    run_id.as_deref(),
                    filters_val.as_ref(),
                    limit,
                )
                .await
                .map_err(to_py_err)?;
            Ok(items
                .into_iter()
                .map(PyMemoryItem::from)
                .collect::<Vec<_>>())
        })
    }

    fn update<'py>(
        &self,
        py: Python<'py>,
        memory_id: String,
        new_text: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let uuid = uuid::Uuid::parse_str(&memory_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid UUID: {e}")))?;

        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.update(uuid, &new_text).await.map_err(to_py_err)?;
            Ok(())
        })
    }

    fn delete<'py>(&self, py: Python<'py>, memory_id: String) -> PyResult<Bound<'py, PyAny>> {
        let uuid = uuid::Uuid::parse_str(&memory_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid UUID: {e}")))?;

        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.delete(uuid).await.map_err(to_py_err)?;
            Ok(())
        })
    }

    #[pyo3(signature = (user_id=None, agent_id=None, run_id=None))]
    fn delete_all<'py>(
        &self,
        py: Python<'py>,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .delete_all(user_id.as_deref(), agent_id.as_deref(), run_id.as_deref())
                .await
                .map_err(to_py_err)?;
            Ok(())
        })
    }

    fn history<'py>(&self, py: Python<'py>, memory_id: String) -> PyResult<Bound<'py, PyAny>> {
        let uuid = uuid::Uuid::parse_str(&memory_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid UUID: {e}")))?;

        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let events = inner.history(uuid).await.map_err(to_py_err)?;
            Ok(events
                .into_iter()
                .map(PyMemoryEvent::from)
                .collect::<Vec<_>>())
        })
    }

    fn reset<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.reset().await.map_err(to_py_err)?;
            Ok(())
        })
    }
}
