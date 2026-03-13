use std::sync::Arc;

use mem7_config::MemoryEngineConfig;
use mem7_core::ChatMessage;
use mem7_store::MemoryEngine;
use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::types::*;

#[napi(js_name = "MemoryEngine")]
pub struct JsMemoryEngine {
    inner: Arc<MemoryEngine>,
}

#[napi]
impl JsMemoryEngine {
    #[napi(factory)]
    pub async fn create(config_json: Option<String>) -> Result<Self> {
        let config: MemoryEngineConfig = match config_json {
            Some(json) => serde_json::from_str(&json)
                .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid config: {e}")))?,
            None => MemoryEngineConfig::default(),
        };

        let engine = MemoryEngine::new(config).await.map_err(to_napi_err)?;

        Ok(Self {
            inner: Arc::new(engine),
        })
    }

    #[napi]
    pub async fn add(
        &self,
        messages: Vec<JsChatMessage>,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
        metadata: Option<String>,
        infer: Option<bool>,
    ) -> Result<JsAddResult> {
        let msgs: Vec<ChatMessage> = messages
            .into_iter()
            .map(|m| ChatMessage {
                role: m.role,
                content: m.content,
            })
            .collect();

        let meta_val: Option<serde_json::Value> = metadata
            .map(|s| serde_json::from_str(&s))
            .transpose()
            .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid metadata JSON: {e}")))?;

        let result = self
            .inner
            .add(
                &msgs,
                user_id.as_deref(),
                agent_id.as_deref(),
                run_id.as_deref(),
                meta_val.as_ref(),
                infer.unwrap_or(true),
            )
            .await
            .map_err(to_napi_err)?;

        Ok(result.into())
    }

    #[allow(clippy::too_many_arguments)]
    #[napi]
    pub async fn search(
        &self,
        query: String,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
        limit: Option<u32>,
        filters: Option<String>,
        rerank: Option<bool>,
    ) -> Result<JsSearchResult> {
        let filters_val: Option<serde_json::Value> = filters
            .map(|s| serde_json::from_str(&s))
            .transpose()
            .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid filters JSON: {e}")))?;

        let result = self
            .inner
            .search(
                &query,
                user_id.as_deref(),
                agent_id.as_deref(),
                run_id.as_deref(),
                limit.unwrap_or(5) as usize,
                filters_val.as_ref(),
                rerank.unwrap_or(true),
            )
            .await
            .map_err(to_napi_err)?;

        Ok(result.into())
    }

    #[napi]
    pub async fn get(&self, memory_id: String) -> Result<JsMemoryItem> {
        let uuid = uuid::Uuid::parse_str(&memory_id)
            .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid UUID: {e}")))?;

        let item = self.inner.get(uuid).await.map_err(to_napi_err)?;
        Ok(item.into())
    }

    #[napi]
    pub async fn get_all(
        &self,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
        filters: Option<String>,
    ) -> Result<Vec<JsMemoryItem>> {
        let filters_val: Option<serde_json::Value> = filters
            .map(|s| serde_json::from_str(&s))
            .transpose()
            .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid filters JSON: {e}")))?;

        let items = self
            .inner
            .get_all(
                user_id.as_deref(),
                agent_id.as_deref(),
                run_id.as_deref(),
                filters_val.as_ref(),
            )
            .await
            .map_err(to_napi_err)?;

        Ok(items.into_iter().map(Into::into).collect())
    }

    #[napi]
    pub async fn update(&self, memory_id: String, new_text: String) -> Result<()> {
        let uuid = uuid::Uuid::parse_str(&memory_id)
            .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid UUID: {e}")))?;

        self.inner
            .update(uuid, &new_text)
            .await
            .map_err(to_napi_err)
    }

    #[napi]
    pub async fn delete(&self, memory_id: String) -> Result<()> {
        let uuid = uuid::Uuid::parse_str(&memory_id)
            .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid UUID: {e}")))?;

        self.inner.delete(uuid).await.map_err(to_napi_err)
    }

    #[napi]
    pub async fn delete_all(
        &self,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
    ) -> Result<()> {
        self.inner
            .delete_all(user_id.as_deref(), agent_id.as_deref(), run_id.as_deref())
            .await
            .map_err(to_napi_err)
    }

    #[napi]
    pub async fn history(&self, memory_id: String) -> Result<Vec<JsMemoryEvent>> {
        let uuid = uuid::Uuid::parse_str(&memory_id)
            .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid UUID: {e}")))?;

        let events = self.inner.history(uuid).await.map_err(to_napi_err)?;
        Ok(events.into_iter().map(Into::into).collect())
    }

    #[napi]
    pub async fn reset(&self) -> Result<()> {
        self.inner.reset().await.map_err(to_napi_err)
    }
}
