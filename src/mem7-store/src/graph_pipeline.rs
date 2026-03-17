use std::sync::Arc;

use mem7_core::{GraphRelation, MemoryFilter, sort_by_score_desc};
use mem7_datetime::now_iso;
use mem7_embedding::EmbeddingClient;
use mem7_error::Result;
use mem7_graph::GraphStore;
use mem7_llm::LlmClient;
use tracing::{debug, warn};

use crate::constants::*;
use crate::decay;

/// Facade over the multi-step graph extraction and search pipeline.
///
/// Hides the internal details of entity extraction → relation extraction →
/// embedding → storage → conflict detection from the `MemoryEngine`.
pub(crate) struct GraphPipeline {
    graph: Arc<dyn GraphStore>,
    llm: Arc<dyn LlmClient>,
    embedder: Arc<dyn EmbeddingClient>,
    custom_prompt: Option<String>,
}

impl GraphPipeline {
    pub fn new(
        graph: Arc<dyn GraphStore>,
        llm: Arc<dyn LlmClient>,
        embedder: Arc<dyn EmbeddingClient>,
        custom_prompt: Option<String>,
    ) -> Self {
        Self {
            graph,
            llm,
            embedder,
            custom_prompt,
        }
    }

    pub fn store(&self) -> &Arc<dyn GraphStore> {
        &self.graph
    }

    /// Extract entities and relations from conversation text, embed entity names,
    /// store them in the graph, then run conflict detection to soft-delete contradicted relations.
    pub async fn add(
        &self,
        conversation: &str,
        filter: &MemoryFilter,
    ) -> Result<Vec<GraphRelation>> {
        let custom_prompt = self.custom_prompt.as_deref();

        let mut entities = mem7_graph::extraction::extract_entities(
            self.llm.as_ref(),
            conversation,
            custom_prompt,
        )
        .await?;

        debug!(count = entities.len(), "graph: extracted entities");

        if entities.is_empty() {
            return Ok(Vec::new());
        }

        let mut relations = mem7_graph::extraction::extract_relations(
            self.llm.as_ref(),
            conversation,
            &entities,
            None,
        )
        .await?;

        debug!(count = relations.len(), "graph: extracted relations");

        let entity_names: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();
        let embeddings = self.embedder.embed(&entity_names).await?;
        let now = now_iso();

        for (entity, emb) in entities.iter_mut().zip(&embeddings) {
            entity.embedding = Some(emb.clone());
            entity.created_at = Some(now.clone());
        }
        for rel in &mut relations {
            rel.created_at = Some(now.clone());
        }

        self.graph
            .add_relations(&relations, &entities, filter)
            .await?;

        let mut existing_triples = Vec::new();
        let mut seen_keys = std::collections::HashSet::new();
        for emb in &embeddings {
            let hits = self
                .graph
                .search_by_embedding(emb, filter, GRAPH_SEARCH_THRESHOLD, GRAPH_CONFLICT_LIMIT)
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
                self.llm.as_ref(),
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
                self.graph.invalidate_relations(&deletions, filter).await?;
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

    /// Semantic graph search: extract entities from query, embed them,
    /// cosine match graph nodes, 1-hop traversal, BM25 rerank.
    pub async fn search(
        &self,
        query: &str,
        filter: &MemoryFilter,
        limit: usize,
        decay_cfg: Option<&mem7_config::DecayConfig>,
    ) -> Result<Vec<mem7_graph::GraphSearchResult>> {
        let custom_prompt = self.custom_prompt.as_deref();

        let entities =
            mem7_graph::extraction::extract_entities(self.llm.as_ref(), query, custom_prompt)
                .await?;

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
            let hits = self
                .graph
                .search_by_embedding(
                    emb,
                    filter,
                    GRAPH_SEARCH_THRESHOLD,
                    limit * GRAPH_SEARCH_MULTIPLIER,
                )
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

        if let Some(decay_cfg) = decay_cfg {
            for r in &mut reranked {
                let age = decay::age_from_option(
                    r.last_accessed_at.as_deref().or(r.created_at.as_deref()),
                );
                let ac = r.mentions.unwrap_or(0);
                r.score = r.score.map(|s| decay::apply_decay(s, age, ac, decay_cfg));
            }
            sort_by_score_desc(&mut reranked, |r| r.score.unwrap_or(0.0));
            reranked.truncate(limit);
        }

        Ok(reranked)
    }
}
