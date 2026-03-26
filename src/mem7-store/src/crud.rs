use mem7_core::{MemoryAction, MemoryEvent, MemoryFilter, MemoryItem};
use mem7_datetime::now_iso;
use mem7_error::{Mem7Error, Result};
use tracing::{info, instrument};
use uuid::Uuid;

use crate::engine::MemoryEngine;
use crate::payload::{payload_to_event_metadata, payload_to_memory_item};
use crate::require_scope;

impl MemoryEngine {
    #[instrument(skip(self))]
    pub async fn get(&self, memory_id: Uuid) -> Result<MemoryItem> {
        let entry = self
            .vector_index
            .get(&memory_id)
            .await?
            .ok_or_else(|| Mem7Error::NotFound(format!("memory {memory_id}")))?;

        Ok(payload_to_memory_item(memory_id, &entry.1, None))
    }

    #[instrument(skip(self, filters))]
    pub async fn get_all(
        &self,
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        filters: Option<&serde_json::Value>,
        limit: Option<usize>,
    ) -> Result<Vec<MemoryItem>> {
        require_scope("get_all", user_id, agent_id, run_id)?;
        let filter = MemoryFilter {
            metadata: filters.cloned(),
            ..MemoryFilter::from_session(user_id, agent_id, run_id)
        };

        let entries = self.vector_index.list(Some(&filter), limit).await?;

        Ok(entries
            .into_iter()
            .map(|(id, payload)| payload_to_memory_item(id, &payload, None))
            .collect())
    }

    #[instrument(skip(self, new_text))]
    pub async fn update(&self, memory_id: Uuid, new_text: &str) -> Result<()> {
        let entry = self
            .vector_index
            .get(&memory_id)
            .await?
            .ok_or_else(|| Mem7Error::NotFound(format!("memory {memory_id}")))?;

        let old_text = entry
            .1
            .get("text")
            .and_then(|v| v.as_str())
            .map(String::from);

        let vecs = self.embedder.embed(&[new_text.to_string()]).await?;
        let vec = vecs.into_iter().next().unwrap_or_default();

        let mut payload = entry.1.clone();
        payload["text"] = serde_json::Value::String(new_text.to_string());
        payload["updated_at"] = serde_json::Value::String(now_iso());
        let audit = payload_to_event_metadata(&payload);

        self.vector_index
            .update(&memory_id, Some(&vec), Some(payload))
            .await?;

        self.history
            .add_event(
                memory_id,
                old_text.as_deref(),
                Some(new_text),
                MemoryAction::Update,
                audit,
            )
            .await?;

        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn delete(&self, memory_id: Uuid) -> Result<()> {
        let entry = self.vector_index.get(&memory_id).await?;
        let old_text = entry
            .as_ref()
            .and_then(|(_, p)| p.get("text").and_then(|v| v.as_str()))
            .map(String::from);
        let audit = entry
            .as_ref()
            .map(|(_, payload)| {
                let mut metadata = payload_to_event_metadata(payload);
                metadata.is_deleted = true;
                metadata
            })
            .unwrap_or_else(|| mem7_core::MemoryEventMetadata {
                is_deleted: true,
                ..Default::default()
            });

        self.vector_index.delete(&memory_id).await?;

        self.history
            .add_event(
                memory_id,
                old_text.as_deref(),
                None,
                MemoryAction::Delete,
                audit,
            )
            .await?;

        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn delete_all(
        &self,
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
    ) -> Result<()> {
        require_scope("delete_all", user_id, agent_id, run_id).map_err(|_| {
            Mem7Error::Config(
                "delete_all requires at least one of user_id, agent_id, or run_id; use reset() to clear all memories"
                    .into(),
            )
        })?;

        let filter = MemoryFilter::from_session(user_id, agent_id, run_id);
        let entries = self.vector_index.list(Some(&filter), None).await?;

        if let Some(gp) = &self.graph_pipeline {
            gp.store().delete_all(&filter).await?;
        }

        for (id, payload) in entries {
            let old_text = payload
                .get("text")
                .and_then(|v| v.as_str())
                .map(String::from);
            let mut audit = payload_to_event_metadata(&payload);
            audit.is_deleted = true;
            self.vector_index.delete(&id).await?;
            self.history
                .add_event(id, old_text.as_deref(), None, MemoryAction::Delete, audit)
                .await?;
        }

        Ok(())
    }

    #[instrument(skip(self))]
    pub async fn history(&self, memory_id: Uuid) -> Result<Vec<MemoryEvent>> {
        self.history.get_history(memory_id).await
    }

    #[instrument(skip(self))]
    pub async fn reset(&self) -> Result<()> {
        self.vector_index.reset().await?;
        self.history.reset().await?;
        if let Some(gp) = &self.graph_pipeline {
            gp.store().reset().await?;
        }
        info!("MemoryEngine reset");
        Ok(())
    }
}
