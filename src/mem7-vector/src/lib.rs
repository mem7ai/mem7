mod distance;
mod filter;
mod flat;
mod upstash;

pub use distance::DistanceMetric;
pub use flat::FlatIndex;
pub use upstash::UpstashVectorIndex;

use std::sync::Arc;

use async_trait::async_trait;
use mem7_config::VectorConfig;
use mem7_core::MemoryFilter;
use mem7_error::{Mem7Error, Result};
use tracing::info;
use uuid::Uuid;

/// A vector search result entry.
#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    pub id: Uuid,
    pub score: f32,
    pub payload: serde_json::Value,
}

/// Trait for vector index implementations.
#[async_trait]
pub trait VectorIndex: Send + Sync {
    async fn insert(&self, id: Uuid, vector: &[f32], payload: serde_json::Value) -> Result<()>;
    async fn search(
        &self,
        query: &[f32],
        limit: usize,
        filters: Option<&MemoryFilter>,
    ) -> Result<Vec<VectorSearchResult>>;
    async fn delete(&self, id: &Uuid) -> Result<()>;
    async fn update(
        &self,
        id: &Uuid,
        vector: Option<&[f32]>,
        payload: Option<serde_json::Value>,
    ) -> Result<()>;
    async fn get(&self, id: &Uuid) -> Result<Option<(Vec<f32>, serde_json::Value)>>;
    async fn list(
        &self,
        filters: Option<&MemoryFilter>,
        limit: Option<usize>,
    ) -> Result<Vec<(Uuid, serde_json::Value)>>;
    async fn reset(&self) -> Result<()>;
}

/// Create a vector index from config.
pub fn create_vector_index(config: &VectorConfig) -> Result<Arc<dyn VectorIndex>> {
    match config.provider.as_str() {
        "upstash" => {
            let url = config
                .upstash_url
                .as_deref()
                .ok_or_else(|| Mem7Error::Config("upstash_url is required".into()))?;
            let token = config
                .upstash_token
                .as_deref()
                .ok_or_else(|| Mem7Error::Config("upstash_token is required".into()))?;
            info!(namespace = %config.collection_name, "using Upstash Vector");
            Ok(Arc::new(UpstashVectorIndex::new(
                url,
                token,
                &config.collection_name,
            )))
        }
        "flat" | "" => {
            info!("using in-memory FlatIndex");
            Ok(Arc::new(FlatIndex::new(DistanceMetric::Cosine)))
        }
        other => Err(Mem7Error::Config(format!(
            "unknown vector store provider: {other}"
        ))),
    }
}
