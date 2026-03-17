use std::sync::Arc;

use mem7_config::MemoryEngineConfig;
use mem7_core::{
    AddResult, ChatMessage, GraphRelation, MemoryAction, MemoryActionResult, MemoryEvent,
    MemoryFilter, MemoryItem, SearchResult, new_memory_id,
};
use mem7_embedding::EmbeddingClient;
use mem7_error::{Mem7Error, Result};
use mem7_graph::GraphStore;
use mem7_history::SqliteHistory;
use mem7_llm::LlmClient;
use mem7_reranker::{RerankDocument, RerankerClient};
use mem7_vector::{VectorIndex, VectorSearchResult};
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

use crate::decay;
use crate::pipeline;
use crate::prompts::VISION_DESCRIBE_PROMPT;

/// The core memory engine. Orchestrates the full add/search/get/update/delete/history pipeline.
pub struct MemoryEngine {
    llm: Arc<dyn LlmClient>,
    embedder: Arc<dyn EmbeddingClient>,
    vector_index: Arc<dyn VectorIndex>,
    history: Arc<SqliteHistory>,
    reranker: Option<Arc<dyn RerankerClient>>,
    graph: Option<Arc<dyn GraphStore>>,
    graph_llm: Option<Arc<dyn LlmClient>>,
    config: MemoryEngineConfig,
}

impl MemoryEngine {
    pub async fn new(config: MemoryEngineConfig) -> Result<Self> {
        let llm = mem7_llm::create_llm(&config.llm)?;
        let embedder = mem7_embedding::create_embedding(&config.embedding)?;
        let vector_index = mem7_vector::create_vector_index(&config.vector)?;
        let history = Arc::new(SqliteHistory::new(&config.history.db_path).await?);
        let reranker = config
            .reranker
            .as_ref()
            .map(mem7_reranker::create_reranker)
            .transpose()?;

        let (graph, graph_llm) = if let Some(graph_cfg) = &config.graph {
            let store = mem7_graph::create_graph_store(graph_cfg).await?;
            let g_llm = graph_cfg
                .llm
                .as_ref()
                .map(mem7_llm::create_llm)
                .transpose()?
                .unwrap_or_else(|| llm.clone());
            (Some(store), Some(g_llm))
        } else {
            (None, None)
        };

        info!(
            reranker = reranker.is_some(),
            graph = graph.is_some(),
            "MemoryEngine initialized"
        );

        Ok(Self {
            llm,
            embedder,
            vector_index,
            history,
            reranker,
            graph,
            graph_llm,
            config,
        })
    }

    /// Add memories from a conversation.
    ///
    /// When `infer` is `true` (the default), the LLM extracts facts from the
    /// conversation, deduplicates them against existing memories, and decides
    /// whether to add, update, or delete.
    ///
    /// When `infer` is `false`, each message's content is stored directly as a
    /// new memory without any LLM processing — useful for importing raw text.
    ///
    /// `metadata` is an optional JSON object stored under `payload.metadata`.
    #[instrument(skip(self, messages, metadata), fields(msg_count = messages.len()))]
    pub async fn add(
        &self,
        messages: &[ChatMessage],
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        metadata: Option<&serde_json::Value>,
        infer: bool,
    ) -> Result<AddResult> {
        if infer {
            self.add_with_inference(messages, user_id, agent_id, run_id, metadata)
                .await
        } else {
            self.add_raw(messages, user_id, agent_id, run_id, metadata)
                .await
        }
    }

    /// Store raw message texts directly without LLM inference or deduplication.
    /// System messages are skipped. Each message's `role` is stored in the payload.
    async fn add_raw(
        &self,
        messages: &[ChatMessage],
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        metadata: Option<&serde_json::Value>,
    ) -> Result<AddResult> {
        let non_system: Vec<&ChatMessage> =
            messages.iter().filter(|m| m.role != "system").collect();

        if non_system.is_empty() {
            return Ok(AddResult {
                results: Vec::new(),
                relations: Vec::new(),
            });
        }

        let owned: Vec<String> = non_system.iter().map(|m| m.content.clone()).collect();
        let embeddings = self.embedder.embed(&owned).await?;

        let now = chrono_now();
        let mut results = Vec::new();

        for (msg, vec) in non_system.iter().zip(embeddings) {
            let memory_id = new_memory_id();

            let mut payload = serde_json::json!({
                "text": msg.content,
                "role": msg.role,
                "user_id": user_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "created_at": now,
                "updated_at": now,
                "last_accessed_at": now,
                "access_count": 0,
            });
            if let Some(meta) = metadata {
                payload["metadata"] = meta.clone();
            }

            self.vector_index.insert(memory_id, &vec, payload).await?;

            self.history
                .add_event(memory_id, None, Some(&msg.content), MemoryAction::Add)
                .await?;

            results.push(MemoryActionResult {
                id: memory_id,
                action: MemoryAction::Add,
                old_value: None,
                new_value: Some(msg.content.clone()),
            });
        }

        // Graph extraction for raw add (runs in parallel if graph is enabled)
        let relations = if self.graph.is_some() {
            let conversation = non_system
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            let filter = MemoryFilter {
                user_id: user_id.map(String::from),
                agent_id: agent_id.map(String::from),
                run_id: run_id.map(String::from),
                metadata: None,
            };
            self.add_graph(&conversation, &filter)
                .await
                .unwrap_or_else(|e| {
                    warn!(error = %e, "graph extraction failed during raw add");
                    Vec::new()
                })
        } else {
            Vec::new()
        };

        info!(count = results.len(), infer = false, "raw memories stored");
        Ok(AddResult { results, relations })
    }

    /// Full LLM-powered pipeline: extract facts, deduplicate, store.
    async fn add_with_inference(
        &self,
        messages: &[ChatMessage],
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        metadata: Option<&serde_json::Value>,
    ) -> Result<AddResult> {
        let messages = if self.config.llm.enable_vision {
            self.describe_images(messages).await?
        } else {
            messages.to_vec()
        };

        let conversation = messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        let filter = MemoryFilter {
            user_id: user_id.map(String::from),
            agent_id: agent_id.map(String::from),
            run_id: run_id.map(String::from),
            metadata: None,
        };

        // Run vector path and graph path concurrently
        let graph_enabled = self.graph.is_some();
        let graph_future = async {
            if graph_enabled {
                self.add_graph(&conversation, &filter).await
            } else {
                Ok(Vec::new())
            }
        };

        let vector_future =
            self.add_vector_with_inference(&messages, user_id, agent_id, run_id, metadata, &filter);

        let (vector_result, graph_result) = tokio::join!(vector_future, graph_future);

        let (results, _) = vector_result?;
        let relations = graph_result.unwrap_or_else(|e| {
            warn!(error = %e, "graph extraction failed");
            Vec::new()
        });

        info!(
            count = results.len(),
            relations = relations.len(),
            "memory operations completed"
        );
        Ok(AddResult { results, relations })
    }

    /// Vector-only inference pipeline, factored out for concurrent execution.
    async fn add_vector_with_inference(
        &self,
        messages: &[ChatMessage],
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        metadata: Option<&serde_json::Value>,
        filter: &MemoryFilter,
    ) -> Result<(Vec<MemoryActionResult>, ())> {
        let facts = pipeline::extract_facts(
            self.llm.as_ref(),
            messages,
            agent_id,
            self.config.custom_fact_extraction_prompt.as_deref(),
        )
        .await?;

        if facts.is_empty() {
            return Ok((Vec::new(), ()));
        }

        debug!(count = facts.len(), "extracted facts");

        let fact_texts: Vec<String> = facts.iter().map(|f| f.text.clone()).collect();
        let embeddings = self.embedder.embed(&fact_texts).await?;

        let mut all_retrieved: Vec<(Uuid, String, f32)> = Vec::new();

        let decay_cfg = self.config.decay.as_ref().filter(|d| d.enabled);

        for embedding in &embeddings {
            let results = self.vector_index.search(embedding, 5, Some(filter)).await?;
            for VectorSearchResult { id, score, payload } in results {
                if let Some(text) = payload.get("text").and_then(|v| v.as_str()) {
                    let effective_score = match decay_cfg {
                        Some(cfg) => {
                            let age = decay::age_from_payload(&payload);
                            let ac = decay::access_count_from_payload(&payload);
                            decay::apply_decay(score, age, ac, cfg)
                        }
                        None => score,
                    };
                    all_retrieved.push((id, text.to_string(), effective_score));
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

                    let mut payload = serde_json::json!({
                        "text": text,
                        "user_id": user_id,
                        "agent_id": agent_id,
                        "run_id": run_id,
                        "created_at": now,
                        "updated_at": now,
                        "last_accessed_at": now,
                        "access_count": 0,
                    });
                    if let Some(meta) = metadata {
                        payload["metadata"] = meta.clone();
                    }

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

                        let prev_ac = self
                            .vector_index
                            .get(&real_id)
                            .await
                            .ok()
                            .flatten()
                            .map(|(_, p)| decay::access_count_from_payload(&p))
                            .unwrap_or(0);

                        let mut payload = serde_json::json!({
                            "text": text,
                            "user_id": user_id,
                            "agent_id": agent_id,
                            "run_id": run_id,
                            "updated_at": now,
                            "last_accessed_at": now,
                            "access_count": prev_ac + 1,
                        });
                        if let Some(meta) = metadata {
                            payload["metadata"] = meta.clone();
                        }

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
                MemoryAction::None => {
                    if let Some(real_id) = id_mapping.resolve(&decision.id) {
                        let needs_update = agent_id.is_some() || run_id.is_some();
                        if needs_update
                            && let Ok(Some(entry)) = self.vector_index.get(&real_id).await
                        {
                            let mut payload = entry.1;
                            let mut changed = false;
                            if let Some(aid) = agent_id {
                                let cur = payload.get("agent_id").and_then(|v| v.as_str());
                                if cur != Some(aid) {
                                    payload["agent_id"] =
                                        serde_json::Value::String(aid.to_string());
                                    changed = true;
                                }
                            }
                            if let Some(rid) = run_id {
                                let cur = payload.get("run_id").and_then(|v| v.as_str());
                                if cur != Some(rid) {
                                    payload["run_id"] = serde_json::Value::String(rid.to_string());
                                    changed = true;
                                }
                            }
                            if changed {
                                payload["updated_at"] = serde_json::Value::String(now.clone());
                                let _ = self
                                    .vector_index
                                    .update(&real_id, None, Some(payload))
                                    .await;
                                debug!(id = %real_id, "updated session IDs on NONE action");
                            }
                        }
                    }
                }
            }
        }

        Ok((results, ()))
    }

    /// When `enable_vision` is set, send each message's images to the LLM and
    /// append the resulting description to the message's text content.
    async fn describe_images(&self, messages: &[ChatMessage]) -> Result<Vec<ChatMessage>> {
        let mut out = Vec::with_capacity(messages.len());
        for msg in messages {
            if msg.images.is_empty() {
                out.push(msg.clone());
                continue;
            }

            let llm_msg = mem7_llm::LlmMessage::user_with_images(
                VISION_DESCRIBE_PROMPT.to_string(),
                msg.images.clone(),
            );
            match self.llm.chat_completion(&[llm_msg], None).await {
                Ok(resp) => {
                    let mut enriched = msg.clone();
                    if enriched.content.is_empty() {
                        enriched.content = resp.content;
                    } else {
                        enriched.content = format!(
                            "{}\n[Image description: {}]",
                            enriched.content, resp.content
                        );
                    }
                    enriched.images.clear();
                    out.push(enriched);
                }
                Err(e) => {
                    warn!(error = %e, "vision description failed, using original text");
                    out.push(msg.clone());
                }
            }
        }
        Ok(out)
    }

    /// Extract entities and relations from conversation text, embed entity names,
    /// store them in the graph, then run conflict detection to soft-delete contradicted relations.
    async fn add_graph(
        &self,
        conversation: &str,
        filter: &MemoryFilter,
    ) -> Result<Vec<GraphRelation>> {
        let graph = self
            .graph
            .as_ref()
            .ok_or_else(|| Mem7Error::Config("graph store not configured".into()))?;
        let llm = self
            .graph_llm
            .as_ref()
            .ok_or_else(|| Mem7Error::Config("graph LLM not configured".into()))?;

        let custom_prompt = self
            .config
            .graph
            .as_ref()
            .and_then(|g| g.custom_prompt.as_deref());

        let mut entities =
            mem7_graph::extraction::extract_entities(llm.as_ref(), conversation, custom_prompt)
                .await?;

        debug!(count = entities.len(), "graph: extracted entities");

        if entities.is_empty() {
            return Ok(Vec::new());
        }

        let mut relations =
            mem7_graph::extraction::extract_relations(llm.as_ref(), conversation, &entities, None)
                .await?;

        debug!(count = relations.len(), "graph: extracted relations");

        // Embed entity names
        let entity_names: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();
        let embeddings = self.embedder.embed(&entity_names).await?;
        let now = chrono_now();

        for (entity, emb) in entities.iter_mut().zip(&embeddings) {
            entity.embedding = Some(emb.clone());
            entity.created_at = Some(now.clone());
        }
        for rel in &mut relations {
            rel.created_at = Some(now.clone());
        }

        // Store entities and relations
        graph.add_relations(&relations, &entities, filter).await?;

        // Conflict detection: search existing triples for involved entities
        let mut existing_triples = Vec::new();
        let mut seen_keys = std::collections::HashSet::new();
        for emb in &embeddings {
            let hits = graph
                .search_by_embedding(emb, filter, 0.5, 50)
                .await
                .unwrap_or_default();
            for hit in hits {
                let key = (
                    hit.source.clone(),
                    hit.relationship.clone(),
                    hit.destination.clone(),
                );
                if seen_keys.insert(key) {
                    existing_triples.push(hit);
                }
            }
        }

        if !existing_triples.is_empty() {
            let deletions = mem7_graph::extraction::extract_deletions(
                llm.as_ref(),
                &existing_triples,
                conversation,
            )
            .await
            .unwrap_or_else(|e| {
                warn!(error = %e, "graph conflict detection failed");
                Vec::new()
            });

            if !deletions.is_empty() {
                debug!(
                    count = deletions.len(),
                    "graph: soft-deleting contradicted relations"
                );
                graph.invalidate_relations(&deletions, filter).await?;
            }
        }

        Ok(relations
            .into_iter()
            .map(|r| GraphRelation {
                source: r.source,
                relationship: r.relationship,
                destination: r.destination,
                score: None,
            })
            .collect())
    }

    /// Search memories by semantic similarity.
    ///
    /// `filters` is an optional JSON object evaluated against `payload.metadata`
    /// using the filter DSL (simple equality, operators, AND/OR/NOT).
    ///
    /// When a reranker is configured and `rerank` is `true`, the engine
    /// over-fetches candidates by `top_k_multiplier` and then reranks them
    /// down to `limit`.
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip(self, filters))]
    pub async fn search(
        &self,
        query: &str,
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
        limit: usize,
        filters: Option<&serde_json::Value>,
        rerank: bool,
        threshold: Option<f32>,
    ) -> Result<SearchResult> {
        let vecs = self.embedder.embed(&[query.to_string()]).await?;
        let query_vec = vecs.into_iter().next().unwrap_or_default();

        let filter = MemoryFilter {
            user_id: user_id.map(String::from),
            agent_id: agent_id.map(String::from),
            run_id: run_id.map(String::from),
            metadata: filters.cloned(),
        };

        let should_rerank = rerank && self.reranker.is_some();
        let fetch_limit = if should_rerank {
            let multiplier = self
                .config
                .reranker
                .as_ref()
                .map(|r| r.top_k_multiplier)
                .unwrap_or(3);
            limit * multiplier
        } else {
            limit
        };

        // Run vector search and graph search concurrently
        let graph_filter = MemoryFilter {
            user_id: user_id.map(String::from),
            agent_id: agent_id.map(String::from),
            run_id: run_id.map(String::from),
            metadata: None,
        };

        let vector_future = self
            .vector_index
            .search(&query_vec, fetch_limit, Some(&filter));

        let graph_future = self.search_graph(query, &graph_filter, limit);

        let (results, graph_results) = tokio::join!(vector_future, graph_future);
        let results = results?;
        let graph_results = graph_results.unwrap_or_else(|e| {
            warn!(error = %e, "graph search failed");
            Vec::new()
        });

        let memories: Vec<MemoryItem> = if should_rerank && !results.is_empty() {
            let reranker = self.reranker.as_ref().unwrap();
            let docs: Vec<RerankDocument> = results
                .iter()
                .filter_map(|r| {
                    r.payload
                        .get("text")
                        .and_then(|v| v.as_str())
                        .map(|text| RerankDocument {
                            id: r.id,
                            text: text.to_string(),
                            score: r.score,
                            payload: r.payload.clone(),
                        })
                })
                .collect();

            match reranker.rerank(query, &docs, limit).await {
                Ok(reranked) => {
                    debug!(count = reranked.len(), "reranked results");
                    reranked
                        .into_iter()
                        .map(|r| {
                            let mut item =
                                payload_to_memory_item(r.id, &r.payload, Some(r.rerank_score));
                            item.score = Some(r.rerank_score);
                            item
                        })
                        .collect()
                }
                Err(e) => {
                    warn!(error = %e, "reranking failed, using original results");
                    results
                        .into_iter()
                        .take(limit)
                        .map(|r| payload_to_memory_item(r.id, &r.payload, Some(r.score)))
                        .collect()
                }
            }
        } else {
            results
                .into_iter()
                .map(|r| payload_to_memory_item(r.id, &r.payload, Some(r.score)))
                .collect()
        };

        let mut memories = if let Some(decay_cfg) = self.config.decay.as_ref().filter(|d| d.enabled)
        {
            let mut decayed: Vec<MemoryItem> = memories
                .into_iter()
                .map(|mut item| {
                    let age = decay::age_from_memory_item(
                        item.last_accessed_at.as_deref(),
                        &item.updated_at,
                        &item.created_at,
                    );
                    item.score = item
                        .score
                        .map(|s| decay::apply_decay(s, age, item.access_count, decay_cfg));
                    item
                })
                .collect();
            decayed.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            decayed
        } else {
            memories
        };

        if let Some(thresh) = threshold {
            memories.retain(|m| m.score.unwrap_or(0.0) >= thresh);
        }

        let relations: Vec<GraphRelation> = graph_results
            .into_iter()
            .map(|r| GraphRelation {
                source: r.source,
                relationship: r.relationship,
                destination: r.destination,
                score: r.score,
            })
            .collect();

        // Async rehearsal: strengthen retrieved memories and relations
        if self.config.decay.as_ref().is_some_and(|d| d.enabled)
            && (!memories.is_empty() || !relations.is_empty())
        {
            let vi = self.vector_index.clone();
            let graph = self.graph.clone();
            let mem_ids: Vec<Uuid> = memories.iter().map(|m| m.id).collect();
            let rel_triples: Vec<(String, String, String)> = relations
                .iter()
                .map(|r| {
                    (
                        r.source.clone(),
                        r.relationship.clone(),
                        r.destination.clone(),
                    )
                })
                .collect();
            let rehearsal_filter = MemoryFilter {
                user_id: user_id.map(String::from),
                agent_id: agent_id.map(String::from),
                run_id: run_id.map(String::from),
                metadata: None,
            };

            tokio::spawn(async move {
                let now = chrono_now();
                for mid in mem_ids {
                    if let Ok(Some((_, payload))) = vi.get(&mid).await {
                        let ac = payload
                            .get("access_count")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        let mut p = payload;
                        p["last_accessed_at"] = serde_json::Value::String(now.clone());
                        p["access_count"] = serde_json::json!(ac + 1);
                        let _ = vi.update(&mid, None, Some(p)).await;
                    }
                }
                if let Some(g) = graph
                    && !rel_triples.is_empty()
                {
                    let _ = g
                        .rehearse_relations(&rel_triples, &rehearsal_filter, &now)
                        .await;
                }
            });
        }

        Ok(SearchResult {
            memories,
            relations,
        })
    }

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
        let filter = MemoryFilter {
            user_id: user_id.map(String::from),
            agent_id: agent_id.map(String::from),
            run_id: run_id.map(String::from),
            metadata: filters.cloned(),
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

    #[instrument(skip(self))]
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

    #[instrument(skip(self))]
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
            metadata: None,
        };

        let entries = self.vector_index.list(Some(&filter), None).await?;
        for (id, _) in entries {
            self.vector_index.delete(&id).await?;
        }

        if let Some(graph) = &self.graph {
            graph.delete_all(&filter).await?;
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
        if let Some(graph) = &self.graph {
            graph.reset().await?;
        }
        info!("MemoryEngine reset");
        Ok(())
    }

    /// Semantic graph search: extract entities from query, embed them,
    /// cosine match graph nodes, 1-hop traversal, BM25 rerank.
    async fn search_graph(
        &self,
        query: &str,
        filter: &MemoryFilter,
        limit: usize,
    ) -> Result<Vec<mem7_graph::GraphSearchResult>> {
        let graph = match &self.graph {
            Some(g) => g,
            None => return Ok(Vec::new()),
        };
        let llm = match &self.graph_llm {
            Some(l) => l,
            None => return Ok(Vec::new()),
        };

        let custom_prompt = self
            .config
            .graph
            .as_ref()
            .and_then(|g| g.custom_prompt.as_deref());

        let entities =
            mem7_graph::extraction::extract_entities(llm.as_ref(), query, custom_prompt).await?;

        if entities.is_empty() {
            return Ok(Vec::new());
        }

        debug!(
            count = entities.len(),
            "graph search: extracted entities from query"
        );

        let entity_names: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();
        let embeddings = self.embedder.embed(&entity_names).await?;

        let mut all_results = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for emb in &embeddings {
            let hits = graph
                .search_by_embedding(emb, filter, 0.5, limit * 3)
                .await?;
            for hit in hits {
                let key = (
                    hit.source.clone(),
                    hit.relationship.clone(),
                    hit.destination.clone(),
                );
                if seen.insert(key) {
                    all_results.push(hit);
                }
            }
        }

        if all_results.is_empty() {
            return Ok(Vec::new());
        }

        let mut reranked = mem7_graph::bm25::rerank(&all_results, query, limit);
        debug!(count = reranked.len(), "graph search: BM25 reranked");

        if let Some(decay_cfg) = self.config.decay.as_ref().filter(|d| d.enabled) {
            for r in &mut reranked {
                let age = decay::age_from_option(
                    r.last_accessed_at.as_deref().or(r.created_at.as_deref()),
                );
                let ac = r.mentions.unwrap_or(0);
                r.score = r.score.map(|s| decay::apply_decay(s, age, ac, decay_cfg));
            }
            reranked.sort_by(|a, b| {
                b.score
                    .unwrap_or(0.0)
                    .partial_cmp(&a.score.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            reranked.truncate(limit);
        }

        Ok(reranked)
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
        last_accessed_at: payload
            .get("last_accessed_at")
            .and_then(|v| v.as_str())
            .map(String::from),
        access_count: payload
            .get("access_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32,
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
