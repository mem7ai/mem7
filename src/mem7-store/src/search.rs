use mem7_core::{
    GraphRelation, MemoryFilter, MemoryItem, SearchOptions, SearchResult, sort_by_score_desc,
};
use mem7_error::Result;
use mem7_reranker::RerankDocument;
use tracing::{debug, instrument, warn};
use uuid::Uuid;

use crate::constants::*;
use crate::decay;
use crate::engine::{MemoryEngine, graph_result_to_relation};
use crate::payload::payload_to_memory_item;
use crate::rehearsal;

impl MemoryEngine {
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
        let opts = SearchOptions {
            user_id,
            agent_id,
            run_id,
            limit,
            filters,
            rerank,
            threshold,
        };
        self.search_with_options(query, &opts).await
    }

    /// Search memories using structured options.
    pub async fn search_with_options(
        &self,
        query: &str,
        opts: &SearchOptions<'_>,
    ) -> Result<SearchResult> {
        let vecs = self.embedder.embed(&[query.to_string()]).await?;
        let query_vec = vecs.into_iter().next().unwrap_or_default();

        let filter = MemoryFilter {
            metadata: opts.filters.cloned(),
            ..MemoryFilter::from_session(opts.user_id, opts.agent_id, opts.run_id)
        };

        let should_rerank = opts.rerank && self.reranker.is_some();
        let fetch_limit = if should_rerank {
            let multiplier = self
                .config
                .reranker
                .as_ref()
                .map(|r| r.top_k_multiplier)
                .unwrap_or(DEFAULT_RERANK_MULTIPLIER);
            opts.limit * multiplier
        } else {
            opts.limit
        };

        let graph_filter = MemoryFilter::from_session(opts.user_id, opts.agent_id, opts.run_id);

        let vector_future = self
            .vector_index
            .search(&query_vec, fetch_limit, Some(&filter));

        let decay_cfg = self.config.decay.as_ref().filter(|d| d.enabled);
        let graph_future = async {
            match &self.graph_pipeline {
                Some(gp) => gp.search(query, &graph_filter, opts.limit, decay_cfg).await,
                None => Ok(Vec::new()),
            }
        };

        let (results, graph_results) = tokio::join!(vector_future, graph_future);
        let results = results?;
        let graph_results = graph_results.unwrap_or_else(|e| {
            warn!(error = %e, "graph search failed");
            Vec::new()
        });

        let memories: Vec<MemoryItem> = if let (true, Some(reranker)) =
            (should_rerank && !results.is_empty(), self.reranker.as_ref())
        {
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

            match reranker.rerank(query, &docs, opts.limit).await {
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
                        .take(opts.limit)
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

        let mut memories = if let Some(decay_cfg) = decay_cfg {
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
            sort_by_score_desc(&mut decayed, |m| m.score.unwrap_or(0.0));
            decayed
        } else {
            memories
        };

        if let Some(thresh) = opts.threshold {
            memories.retain(|m| m.score.unwrap_or(0.0) >= thresh);
        }

        let relations: Vec<GraphRelation> =
            graph_results.iter().map(graph_result_to_relation).collect();

        if self.config.decay.as_ref().is_some_and(|d| d.enabled)
            && (!memories.is_empty() || !relations.is_empty())
        {
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
            let rehearsal_filter =
                MemoryFilter::from_session(opts.user_id, opts.agent_id, opts.run_id);

            rehearsal::spawn_rehearsal(
                self.vector_index.clone(),
                self.graph_pipeline.as_ref().map(|gp| gp.store().clone()),
                mem_ids,
                rel_triples,
                rehearsal_filter,
            );
        }

        Ok(SearchResult {
            memories,
            relations,
        })
    }
}
