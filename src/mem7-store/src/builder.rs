use std::sync::Arc;

use mem7_config::MemoryEngineConfig;
use mem7_embedding::EmbeddingClient;
use mem7_error::{Mem7Error, Result};
use mem7_history::{HistoryStore, SqliteHistory};
use mem7_llm::LlmClient;
use mem7_reranker::RerankerClient;
use mem7_vector::VectorIndex;
use tracing::info;

use crate::engine::MemoryEngine;
use crate::graph_pipeline::GraphPipeline;

/// Builder for `MemoryEngine` with dependency injection support.
///
/// Use this when you need to supply custom or mock implementations for
/// LLM, embedding, vector, history, reranker, or graph components.
///
/// ```rust,ignore
/// let engine = MemoryEngineBuilder::new(config)
///     .llm(my_llm)
///     .history(my_history)
///     .build()
///     .await?;
/// ```
pub struct MemoryEngineBuilder {
    config: MemoryEngineConfig,
    llm: Option<Arc<dyn LlmClient>>,
    embedder: Option<Arc<dyn EmbeddingClient>>,
    vector_index: Option<Arc<dyn VectorIndex>>,
    history: Option<Arc<dyn HistoryStore>>,
    reranker: Option<Arc<dyn RerankerClient>>,
    graph_pipeline: Option<GraphPipeline>,
}

impl MemoryEngineBuilder {
    pub fn new(config: MemoryEngineConfig) -> Self {
        Self {
            config,
            llm: None,
            embedder: None,
            vector_index: None,
            history: None,
            reranker: None,
            graph_pipeline: None,
        }
    }

    pub fn llm(mut self, llm: Arc<dyn LlmClient>) -> Self {
        self.llm = Some(llm);
        self
    }

    pub fn embedder(mut self, embedder: Arc<dyn EmbeddingClient>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    pub fn vector_index(mut self, vi: Arc<dyn VectorIndex>) -> Self {
        self.vector_index = Some(vi);
        self
    }

    pub fn history(mut self, history: Arc<dyn HistoryStore>) -> Self {
        self.history = Some(history);
        self
    }

    pub fn reranker(mut self, reranker: Arc<dyn RerankerClient>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Build the `MemoryEngine`, creating default implementations for any
    /// components not explicitly provided.
    pub async fn build(self) -> Result<MemoryEngine> {
        let problems = self.config.validate();
        if !problems.is_empty() {
            return Err(Mem7Error::Config(problems.join("; ")));
        }

        let llm = match self.llm {
            Some(l) => l,
            None => mem7_llm::create_llm(&self.config.llm)?,
        };
        let embedder = match self.embedder {
            Some(e) => e,
            None => mem7_embedding::create_embedding(&self.config.embedding)?,
        };
        let vector_index = match self.vector_index {
            Some(v) => v,
            None => mem7_vector::create_vector_index(&self.config.vector)?,
        };
        let history: Arc<dyn HistoryStore> = match self.history {
            Some(h) => h,
            None => Arc::new(SqliteHistory::new(&self.config.history.db_path).await?),
        };
        let reranker = match self.reranker {
            Some(r) => Some(r),
            None => self
                .config
                .reranker
                .as_ref()
                .map(mem7_reranker::create_reranker)
                .transpose()?,
        };

        let graph_pipeline = match self.graph_pipeline {
            Some(gp) => Some(gp),
            None => {
                if let Some(graph_cfg) = &self.config.graph {
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
                }
            }
        };

        info!(
            reranker = reranker.is_some(),
            graph = graph_pipeline.is_some(),
            "MemoryEngine initialized (builder)"
        );

        Ok(MemoryEngine {
            llm,
            embedder,
            vector_index,
            history,
            reranker,
            graph_pipeline,
            config: self.config,
        })
    }
}
