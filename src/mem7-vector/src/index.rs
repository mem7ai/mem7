use std::collections::HashMap;
use std::sync::RwLock;

use async_trait::async_trait;
use mem7_error::{Mem7Error, Result};
use mem7_core::MemoryFilter;
use uuid::Uuid;

use crate::distance::DistanceMetric;
use crate::filter::matches_filter;
use crate::{VectorIndex, VectorSearchResult};

struct VectorEntry {
    vector: Vec<f32>,
    payload: serde_json::Value,
}

/// A brute-force flat vector index. Suitable for small-to-medium datasets.
/// Can be replaced with HNSW later without changing the public API.
pub struct FlatIndex {
    entries: RwLock<HashMap<Uuid, VectorEntry>>,
    metric: DistanceMetric,
}

impl FlatIndex {
    pub fn new(metric: DistanceMetric) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            metric,
        }
    }
}

#[async_trait]
impl VectorIndex for FlatIndex {
    async fn insert(&self, id: Uuid, vector: &[f32], payload: serde_json::Value) -> Result<()> {
        let mut entries = self
            .entries
            .write()
            .map_err(|e| Mem7Error::VectorStore(e.to_string()))?;
        entries.insert(
            id,
            VectorEntry {
                vector: vector.to_vec(),
                payload,
            },
        );
        Ok(())
    }

    async fn search(
        &self,
        query: &[f32],
        limit: usize,
        filters: Option<&MemoryFilter>,
    ) -> Result<Vec<VectorSearchResult>> {
        let entries = self
            .entries
            .read()
            .map_err(|e| Mem7Error::VectorStore(e.to_string()))?;

        let mut scored: Vec<VectorSearchResult> = entries
            .iter()
            .filter(|(_, entry)| {
                filters
                    .map(|f| matches_filter(&entry.payload, f))
                    .unwrap_or(true)
            })
            .map(|(id, entry)| VectorSearchResult {
                id: *id,
                score: self.metric.similarity(query, &entry.vector),
                payload: entry.payload.clone(),
            })
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored)
    }

    async fn delete(&self, id: &Uuid) -> Result<()> {
        let mut entries = self
            .entries
            .write()
            .map_err(|e| Mem7Error::VectorStore(e.to_string()))?;
        entries.remove(id);
        Ok(())
    }

    async fn update(
        &self,
        id: &Uuid,
        vector: Option<&[f32]>,
        payload: Option<serde_json::Value>,
    ) -> Result<()> {
        let mut entries = self
            .entries
            .write()
            .map_err(|e| Mem7Error::VectorStore(e.to_string()))?;

        if let Some(entry) = entries.get_mut(id) {
            if let Some(v) = vector {
                entry.vector = v.to_vec();
            }
            if let Some(p) = payload {
                entry.payload = p;
            }
            Ok(())
        } else {
            Err(Mem7Error::NotFound(format!("vector entry {id}")))
        }
    }

    async fn get(&self, id: &Uuid) -> Result<Option<(Vec<f32>, serde_json::Value)>> {
        let entries = self
            .entries
            .read()
            .map_err(|e| Mem7Error::VectorStore(e.to_string()))?;
        Ok(entries
            .get(id)
            .map(|e| (e.vector.clone(), e.payload.clone())))
    }

    async fn list(
        &self,
        filters: Option<&MemoryFilter>,
        limit: Option<usize>,
    ) -> Result<Vec<(Uuid, serde_json::Value)>> {
        let entries = self
            .entries
            .read()
            .map_err(|e| Mem7Error::VectorStore(e.to_string()))?;

        let mut results: Vec<(Uuid, serde_json::Value)> = entries
            .iter()
            .filter(|(_, entry)| {
                filters
                    .map(|f| matches_filter(&entry.payload, f))
                    .unwrap_or(true)
            })
            .map(|(id, entry)| (*id, entry.payload.clone()))
            .collect();

        results.sort_by(|a, b| a.0.cmp(&b.0));

        if let Some(limit) = limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    async fn reset(&self) -> Result<()> {
        let mut entries = self
            .entries
            .write()
            .map_err(|e| Mem7Error::VectorStore(e.to_string()))?;
        entries.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn insert_and_search() {
        let index = FlatIndex::new(DistanceMetric::Cosine);
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();

        index
            .insert(id1, &[1.0, 0.0, 0.0], serde_json::json!({"user_id": "alice"}))
            .await
            .unwrap();
        index
            .insert(id2, &[0.0, 1.0, 0.0], serde_json::json!({"user_id": "bob"}))
            .await
            .unwrap();

        let results = index.search(&[1.0, 0.0, 0.0], 1, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, id1);

        let filter = MemoryFilter {
            user_id: Some("bob".into()),
            ..Default::default()
        };
        let results = index
            .search(&[1.0, 0.0, 0.0], 10, Some(&filter))
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, id2);
    }
}
