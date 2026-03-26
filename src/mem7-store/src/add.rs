use std::collections::HashMap;

use mem7_core::{
    AddOptions, AddResult, ChatMessage, MemoryAction, MemoryActionResult, MemoryFilter,
    new_memory_id,
};
use mem7_datetime::now_iso;
use mem7_error::Result;
use mem7_vector::VectorSearchResult;
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

use crate::constants::*;
use crate::decay;
use crate::engine::MemoryEngine;
use crate::payload::{
    build_memory_payload, build_raw_memory_payload, build_update_payload, payload_to_event_metadata,
};
use crate::pipeline;
use crate::prompts::VISION_DESCRIBE_PROMPT;
use crate::require_scope;

impl MemoryEngine {
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
        let opts = AddOptions {
            user_id,
            agent_id,
            run_id,
            metadata,
            infer,
        };
        self.add_with_options(messages, &opts).await
    }

    /// Add memories using structured options.
    pub async fn add_with_options(
        &self,
        messages: &[ChatMessage],
        opts: &AddOptions<'_>,
    ) -> Result<AddResult> {
        require_scope("add", opts.user_id, opts.agent_id, opts.run_id)?;
        if opts.infer {
            self.add_with_inference(
                messages,
                opts.user_id,
                opts.agent_id,
                opts.run_id,
                opts.metadata,
            )
            .await
        } else {
            self.add_raw(
                messages,
                opts.user_id,
                opts.agent_id,
                opts.run_id,
                opts.metadata,
            )
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

        let now = now_iso();
        let mut results = Vec::new();

        for (msg, vec) in non_system.iter().zip(embeddings) {
            let memory_id = new_memory_id();

            let payload = build_raw_memory_payload(
                &msg.content,
                &msg.role,
                user_id,
                agent_id,
                run_id,
                metadata,
                &now,
            );
            let audit = payload_to_event_metadata(&payload);

            self.vector_index.insert(memory_id, &vec, payload).await?;

            self.history
                .add_event(
                    memory_id,
                    None,
                    Some(&msg.content),
                    MemoryAction::Add,
                    audit,
                )
                .await?;

            results.push(MemoryActionResult {
                id: memory_id,
                action: MemoryAction::Add,
                old_value: None,
                new_value: Some(msg.content.clone()),
            });
        }

        let relations = if let Some(gp) = &self.graph_pipeline {
            let conversation = non_system
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            let filter = MemoryFilter::from_session(user_id, agent_id, run_id);
            gp.add(&conversation, &filter).await.unwrap_or_else(|e| {
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

        let filter = MemoryFilter::from_session(user_id, agent_id, run_id);

        let graph_future = async {
            match &self.graph_pipeline {
                Some(gp) => gp.add(&conversation, &filter).await,
                None => Ok(Vec::new()),
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
            let results = self
                .vector_index
                .search(embedding, DEDUP_CANDIDATE_LIMIT, Some(filter))
                .await?;
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

        let fact_type_map: HashMap<&str, &str> = facts
            .iter()
            .map(|f| (f.text.as_str(), f.memory_type.as_str()))
            .collect();

        let now = now_iso();
        let mut results = Vec::new();

        for decision in &update_resp.memory {
            match decision.event {
                MemoryAction::Add => {
                    let memory_id = new_memory_id();
                    let text = &decision.text;

                    let vecs = self.embedder.embed(std::slice::from_ref(text)).await?;
                    let vec = vecs.into_iter().next().unwrap_or_default();

                    let mt = fact_type_map.get(text.as_str()).copied();
                    let payload =
                        build_memory_payload(text, user_id, agent_id, run_id, metadata, &now, mt);
                    let audit = payload_to_event_metadata(&payload);

                    self.vector_index.insert(memory_id, &vec, payload).await?;

                    self.history
                        .add_event(memory_id, None, Some(text), MemoryAction::Add, audit)
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

                        let existing_entry = self.vector_index.get(&real_id).await.ok().flatten();
                        let prev_ac = existing_entry
                            .as_ref()
                            .map(|(_, p)| decay::access_count_from_payload(p))
                            .unwrap_or(0);
                        let existing_mt = existing_entry
                            .as_ref()
                            .and_then(|(_, p)| p.get("memory_type").and_then(|v| v.as_str()));
                        let existing_created_at = existing_entry
                            .as_ref()
                            .and_then(|(_, p)| p.get("created_at").and_then(|v| v.as_str()));
                        let mt = existing_mt.or_else(|| fact_type_map.get(text.as_str()).copied());

                        let payload = build_update_payload(
                            text,
                            user_id,
                            agent_id,
                            run_id,
                            metadata,
                            existing_created_at,
                            &now,
                            prev_ac + 1,
                            mt,
                        );
                        let audit = payload_to_event_metadata(&payload);

                        self.vector_index
                            .update(&real_id, Some(&vec), Some(payload))
                            .await?;

                        self.history
                            .add_event(real_id, old_text, Some(text), MemoryAction::Update, audit)
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
                        let audit = self
                            .vector_index
                            .get(&real_id)
                            .await
                            .ok()
                            .flatten()
                            .map(|(_, payload)| {
                                let mut metadata = payload_to_event_metadata(&payload);
                                metadata.is_deleted = true;
                                metadata
                            })
                            .unwrap_or_else(|| mem7_core::MemoryEventMetadata {
                                is_deleted: true,
                                ..Default::default()
                            });

                        self.vector_index.delete(&real_id).await?;

                        self.history
                            .add_event(real_id, old_text, None, MemoryAction::Delete, audit)
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
                                if let Err(e) = self
                                    .vector_index
                                    .update(&real_id, None, Some(payload))
                                    .await
                                {
                                    warn!(id = %real_id, "failed to update session IDs: {e}");
                                } else {
                                    debug!(
                                        id = %real_id,
                                        "updated session IDs on NONE action"
                                    );
                                }
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
    pub(crate) async fn describe_images(
        &self,
        messages: &[ChatMessage],
    ) -> Result<Vec<ChatMessage>> {
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
}
