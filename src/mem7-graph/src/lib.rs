pub mod bm25;
pub mod extraction;
mod flat;
#[cfg(feature = "kuzu")]
mod kuzu_store;
mod neo4j;
pub mod prompts;
pub mod types;

pub use flat::FlatGraph;
#[cfg(feature = "kuzu")]
pub use kuzu_store::KuzuGraphStore;
pub use neo4j::Neo4jGraphStore;
pub use types::*;

use std::sync::Arc;

use async_trait::async_trait;
use mem7_config::GraphConfig;
use mem7_core::MemoryFilter;
use mem7_error::{Mem7Error, Result};

/// Trait for graph storage backends.
#[async_trait]
pub trait GraphStore: Send + Sync {
    /// Store extracted entities (provider may use this for node creation).
    async fn add_entities(&self, entities: &[Entity], filter: &MemoryFilter) -> Result<()>;

    /// Store extracted relations along with their entity metadata.
    async fn add_relations(
        &self,
        relations: &[Relation],
        entities: &[Entity],
        filter: &MemoryFilter,
    ) -> Result<()>;

    /// Text-based search for relations (legacy fallback).
    async fn search(
        &self,
        query: &str,
        filter: &MemoryFilter,
        limit: usize,
    ) -> Result<Vec<GraphSearchResult>>;

    /// Semantic search: find entities by embedding cosine similarity, then
    /// return all valid relations touching matched entities (1-hop traversal).
    async fn search_by_embedding(
        &self,
        embedding: &[f32],
        filter: &MemoryFilter,
        threshold: f32,
        limit: usize,
    ) -> Result<Vec<GraphSearchResult>>;

    /// Soft-delete relations by marking them as `valid = false`.
    /// Each tuple is `(source, relationship, destination)`.
    async fn invalidate_relations(
        &self,
        triples: &[(String, String, String)],
        filter: &MemoryFilter,
    ) -> Result<()>;

    /// Delete all relations matching the filter.
    async fn delete_all(&self, filter: &MemoryFilter) -> Result<()>;

    /// Remove all data from the graph store.
    async fn reset(&self) -> Result<()>;
}

/// Create a graph store from configuration.
pub async fn create_graph_store(config: &GraphConfig) -> Result<Arc<dyn GraphStore>> {
    match config.provider.as_str() {
        "flat" => Ok(Arc::new(FlatGraph::new())),
        #[cfg(feature = "kuzu")]
        "kuzu" => {
            let path = config.kuzu_db_path.as_deref().unwrap_or("mem7_graph.kuzu");
            Ok(Arc::new(KuzuGraphStore::new(path)?))
        }
        #[cfg(not(feature = "kuzu"))]
        "kuzu" => Err(Mem7Error::Config(
            "kuzu provider requires the `kuzu` feature: cargo add mem7-graph --features kuzu"
                .into(),
        )),
        "neo4j" => {
            let url = config
                .neo4j_url
                .as_deref()
                .ok_or_else(|| Mem7Error::Config("neo4j_url is required".into()))?;
            let username = config
                .neo4j_username
                .as_deref()
                .ok_or_else(|| Mem7Error::Config("neo4j_username is required".into()))?;
            let password = config
                .neo4j_password
                .as_deref()
                .ok_or_else(|| Mem7Error::Config("neo4j_password is required".into()))?;
            let database = config.neo4j_database.as_deref();
            Ok(Arc::new(
                Neo4jGraphStore::new(url, username, password, database).await?,
            ))
        }
        other => Err(Mem7Error::Config(format!(
            "unknown graph provider: {other}"
        ))),
    }
}
