use std::sync::Arc;

use mem7_config::MemoryEngineConfig;
use mem7_core::GraphRelation;
use mem7_embedding::EmbeddingClient;
use mem7_error::{Mem7Error, Result};
use mem7_history::{HistoryStore, SqliteHistory};
use mem7_llm::LlmClient;
use mem7_reranker::RerankerClient;
use mem7_vector::VectorIndex;
use tracing::info;

use crate::graph_pipeline::GraphPipeline;

pub(crate) fn graph_result_to_relation(r: &mem7_graph::GraphSearchResult) -> GraphRelation {
    GraphRelation {
        source: r.source.clone(),
        relationship: r.relationship.clone(),
        destination: r.destination.clone(),
        score: r.score,
    }
}

/// The core memory engine. Orchestrates the full add/search/get/update/delete/history pipeline.
pub struct MemoryEngine {
    pub(crate) llm: Arc<dyn LlmClient>,
    pub(crate) embedder: Arc<dyn EmbeddingClient>,
    pub(crate) vector_index: Arc<dyn VectorIndex>,
    pub(crate) history: Arc<dyn HistoryStore>,
    pub(crate) reranker: Option<Arc<dyn RerankerClient>>,
    pub(crate) graph_pipeline: Option<GraphPipeline>,
    pub(crate) config: MemoryEngineConfig,
}

impl MemoryEngine {
    pub async fn new(config: MemoryEngineConfig) -> Result<Self> {
        let problems = config.validate();
        if !problems.is_empty() {
            return Err(Mem7Error::Config(problems.join("; ")));
        }

        let llm = mem7_llm::create_llm(&config.llm)?;
        let embedder = mem7_embedding::create_embedding(&config.embedding)?;
        let vector_index = mem7_vector::create_vector_index(&config.vector)?;
        let history: Arc<dyn HistoryStore> =
            Arc::new(SqliteHistory::new(&config.history.db_path).await?);
        let reranker = config
            .reranker
            .as_ref()
            .map(mem7_reranker::create_reranker)
            .transpose()?;

        let graph_pipeline = if let Some(graph_cfg) = &config.graph {
            let store = mem7_graph::create_graph_store(graph_cfg).await?;
            let g_llm = graph_cfg
                .llm
                .as_ref()
                .map(mem7_llm::create_llm)
                .transpose()?
                .unwrap_or_else(|| llm.clone());
            let custom_prompt = graph_cfg.custom_prompt.clone();
            Some(GraphPipeline::new(
                store,
                g_llm,
                embedder.clone(),
                custom_prompt,
            ))
        } else {
            None
        };

        info!(
            reranker = reranker.is_some(),
            graph = graph_pipeline.is_some(),
            "MemoryEngine initialized"
        );

        Ok(Self {
            llm,
            embedder,
            vector_index,
            history,
            reranker,
            graph_pipeline,
            config,
        })
    }
}
