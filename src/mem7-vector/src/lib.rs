mod distance;
mod filter;
mod index;
mod upstash;

pub use distance::DistanceMetric;
pub use index::FlatIndex;
pub use upstash::UpstashVectorIndex;

use async_trait::async_trait;
use mem7_core::MemoryFilter;
use mem7_error::Result;
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
