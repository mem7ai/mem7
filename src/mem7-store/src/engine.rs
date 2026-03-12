use std::sync::Arc;

use mem7_config::MemoryEngineConfig;
use mem7_core::{
    AddResult, ChatMessage, MemoryAction, MemoryActionResult, MemoryEvent, MemoryFilter,
    MemoryItem, SearchResult, new_memory_id,
};
use mem7_embedding::{EmbeddingClient, OpenAICompatibleEmbedding};
use mem7_error::{Mem7Error, Result};
use mem7_history::SqliteHistory;
use mem7_llm::{LlmClient, OpenAICompatibleLlm};
use mem7_vector::{DistanceMetric, FlatIndex, UpstashVectorIndex, VectorIndex, VectorSearchResult};
use tracing::{debug, info};
use uuid::Uuid;

use crate::pipeline;

/// The core memory engine. Orchestrates the full add/search/get/update/delete/history pipeline.
pub struct MemoryEngine {
    llm: Arc<dyn LlmClient>,
    embedder: Arc<dyn EmbeddingClient>,
    vector_index: Arc<dyn VectorIndex>,
    history: Arc<SqliteHistory>,
    config: MemoryEngineConfig,
}

impl MemoryEngine {
    pub async fn new(config: MemoryEngineConfig) -> Result<Self> {
        let llm = Arc::new(OpenAICompatibleLlm::new(config.llm.clone())) as Arc<dyn LlmClient>;
        let embedder = Arc::new(OpenAICompatibleEmbedding::new(config.embedding.clone()))
            as Arc<dyn EmbeddingClient>;

        let vector_index: Arc<dyn VectorIndex> = match config.vector.provider.as_str() {
            "upstash" => {
                let url = config
                    .vector
                    .upstash_url
                    .as_deref()
                    .ok_or_else(|| Mem7Error::Config("upstash_url is required".into()))?;
                let token = config
                    .vector
                    .upstash_token
                    .as_deref()
                    .ok_or_else(|| Mem7Error::Config("upstash_token is required".into()))?;
                info!(namespace = %config.vector.collection_name, "using Upstash Vector");
                Arc::new(UpstashVectorIndex::new(
                    url,
                    token,
                    &config.vector.collection_name,
                ))
            }
            _ => {
                info!("using in-memory FlatIndex");
                Arc::new(FlatIndex::new(DistanceMetric::Cosine))
            }
        };

        let history = Arc::new(SqliteHistory::new(&config.history.db_path).await?);

        info!("MemoryEngine initialized");

        Ok(Self {
            llm,
            embedder,
            vector_index,
            history,
            config,
        })
    }

    /// Add memories from a conversation. Extracts facts, deduplicates, and stores.
    pub async fn add(
        &self,
        messages: &[ChatMessage],
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
    ) -> Result<AddResult> {
        let facts = pipeline::extract_facts(
            self.llm.as_ref(),
            messages,
            self.config.custom_fact_extraction_prompt.as_deref(),
        )
        .await?;

        if facts.is_empty() {
            return Ok(AddResult {
                results: Vec::new(),
            });
        }

        debug!(count = facts.len(), "extracted facts");

        let fact_texts: Vec<String> = facts.iter().map(|f| f.text.clone()).collect();
        let embeddings = self.embedder.embed(&fact_texts).await?;

        let filter = MemoryFilter {
            user_id: user_id.map(String::from),
            agent_id: agent_id.map(String::from),
            run_id: run_id.map(String::from),
        };
        let mut all_retrieved: Vec<(Uuid, String, f32)> = Vec::new();

        for embedding in &embeddings {
            let results = self
                .vector_index
                .search(embedding, 5, Some(&filter))
                .await?;
            for VectorSearchResult { id, score, payload } in results {
                if let Some(text) = payload.get("text").and_then(|v| v.as_str()) {
                    all_retrieved.push((id, text.to_string(), score));
                }
            }
        }

        let (update_resp, id_mapping) = pipeline::decide_memory_updates(
            self.llm.as_ref(),
            &facts,
            all_retrieved,
            self.config.custom_update_memory_prompt.as_deref(),
        )
        .await?;

        let now = chrono_now();
        let mut results = Vec::new();

        for decision in &update_resp.memory {
            match decision.event {
                MemoryAction::Add => {
                    let memory_id = new_memory_id();
                    let text = &decision.text;

                    let vecs = self.embedder.embed(std::slice::from_ref(text)).await?;
                    let vec = vecs.into_iter().next().unwrap_or_default();

                    let payload = serde_json::json!({
                        "text": text,
                        "user_id": user_id,
                        "agent_id": agent_id,
                        "run_id": run_id,
                        "created_at": now,
                        "updated_at": now,
                    });

                    self.vector_index.insert(memory_id, &vec, payload).await?;

                    self.history
                        .add_event(memory_id, None, Some(text), MemoryAction::Add)
                        .await?;

                    results.push(MemoryActionResult {
                        id: memory_id,
                        action: MemoryAction::Add,
                        old_value: None,
                        new_value: Some(text.clone()),
                    });
                }
                MemoryAction::Update => {
                    if let Some(real_id) = id_mapping.resolve(&decision.id) {
                        let text = &decision.text;
                        let old_text = decision.old_memory.as_deref();

                        let vecs = self.embedder.embed(std::slice::from_ref(text)).await?;
                        let vec = vecs.into_iter().next().unwrap_or_default();

                        let payload = serde_json::json!({
                            "text": text,
                            "user_id": user_id,
                            "agent_id": agent_id,
                            "run_id": run_id,
                            "updated_at": now,
                        });

                        self.vector_index
                            .update(&real_id, Some(&vec), Some(payload))
                            .await?;

                        self.history
                            .add_event(real_id, old_text, Some(text), MemoryAction::Update)
                            .await?;

                        results.push(MemoryActionResult {
                            id: real_id,
                            action: MemoryAction::Update,
                            old_value: old_text.map(String::from),
                            new_value: Some(text.clone()),
                        });
                    }
                }
                MemoryAction::Delete => {
                    if let Some(real_id) = id_mapping.resolve(&decision.id) {
                        let old_text = decision.old_memory.as_deref().or(Some(&decision.text));

                        self.vector_index.delete(&real_id).await?;

                        self.history
                            .add_event(real_id, old_text, None, MemoryAction::Delete)
                            .await?;

                        results.push(MemoryActionResult {
                            id: real_id,
                            action: MemoryAction::Delete,
                            old_value: old_text.map(String::from),
                            new_value: None,
                        });
                    }
                }
                MemoryAction::None => {}
            }
        }

        info!(count = results.len(), "memory operations completed");
        Ok(AddResult { results })
    }

    /// Search memories by semantic similarity.
    pub async fn search(
        &self,
        query: &str,
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        limit: usize,
    ) -> Result<SearchResult> {
        let vecs = self.embedder.embed(&[query.to_string()]).await?;
        let query_vec = vecs.into_iter().next().unwrap_or_default();

        let filter = MemoryFilter {
            user_id: user_id.map(String::from),
            agent_id: agent_id.map(String::from),
            run_id: run_id.map(String::from),
        };

        let results = self
            .vector_index
            .search(&query_vec, limit, Some(&filter))
            .await?;

        let memories = results
            .into_iter()
            .map(|r| payload_to_memory_item(r.id, &r.payload, Some(r.score)))
            .collect();

        Ok(SearchResult { memories })
    }

    /// Get a single memory by ID.
    pub async fn get(&self, memory_id: Uuid) -> Result<MemoryItem> {
        let entry = self
            .vector_index
            .get(&memory_id)
            .await?
            .ok_or_else(|| Mem7Error::NotFound(format!("memory {memory_id}")))?;

        Ok(payload_to_memory_item(memory_id, &entry.1, None))
    }

    /// List all memories matching the given filters.
    pub async fn get_all(
        &self,
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
    ) -> Result<Vec<MemoryItem>> {
        let filter = MemoryFilter {
            user_id: user_id.map(String::from),
            agent_id: agent_id.map(String::from),
            run_id: run_id.map(String::from),
        };

        let entries = self.vector_index.list(Some(&filter), None).await?;

        Ok(entries
            .into_iter()
            .map(|(id, payload)| payload_to_memory_item(id, &payload, None))
            .collect())
    }

    /// Update a memory's text directly.
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
        payload["updated_at"] = serde_json::Value::String(chrono_now());

        self.vector_index
            .update(&memory_id, Some(&vec), Some(payload))
            .await?;

        self.history
            .add_event(
                memory_id,
                old_text.as_deref(),
                Some(new_text),
                MemoryAction::Update,
            )
            .await?;

        Ok(())
    }

    /// Delete a memory by ID.
    pub async fn delete(&self, memory_id: Uuid) -> Result<()> {
        let entry = self.vector_index.get(&memory_id).await?;
        let old_text = entry
            .as_ref()
            .and_then(|(_, p)| p.get("text").and_then(|v| v.as_str()))
            .map(String::from);

        self.vector_index.delete(&memory_id).await?;

        self.history
            .add_event(memory_id, old_text.as_deref(), None, MemoryAction::Delete)
            .await?;

        Ok(())
    }

    /// Delete all memories matching the given filters.
    pub async fn delete_all(
        &self,
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
    ) -> Result<()> {
        let filter = MemoryFilter {
            user_id: user_id.map(String::from),
            agent_id: agent_id.map(String::from),
            run_id: run_id.map(String::from),
        };

        let entries = self.vector_index.list(Some(&filter), None).await?;
        for (id, _) in entries {
            self.vector_index.delete(&id).await?;
        }

        Ok(())
    }

    /// Get the change history for a memory.
    pub async fn history(&self, memory_id: Uuid) -> Result<Vec<MemoryEvent>> {
        self.history.get_history(memory_id).await
    }

    /// Reset all data (vector index + history).
    pub async fn reset(&self) -> Result<()> {
        self.vector_index.reset().await?;
        self.history.reset().await?;
        info!("MemoryEngine reset");
        Ok(())
    }
}

fn payload_to_memory_item(id: Uuid, payload: &serde_json::Value, score: Option<f32>) -> MemoryItem {
    MemoryItem {
        id,
        text: payload
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        user_id: payload
            .get("user_id")
            .and_then(|v| v.as_str())
            .map(String::from),
        agent_id: payload
            .get("agent_id")
            .and_then(|v| v.as_str())
            .map(String::from),
        run_id: payload
            .get("run_id")
            .and_then(|v| v.as_str())
            .map(String::from),
        metadata: payload
            .get("metadata")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
        created_at: payload
            .get("created_at")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        updated_at: payload
            .get("updated_at")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        score,
    }
}

fn chrono_now() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

fn days_to_ymd(days_since_epoch: u64) -> (u64, u64, u64) {
    let z = days_since_epoch + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
