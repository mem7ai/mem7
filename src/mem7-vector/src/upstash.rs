use async_trait::async_trait;
use mem7_error::{Mem7Error, Result};
use mem7_core::MemoryFilter;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::debug;
use uuid::Uuid;

use crate::{VectorIndex, VectorSearchResult};

/// Upstash Vector REST API client.
pub struct UpstashVectorIndex {
    client: Client,
    base_url: String,
    token: String,
    namespace: String,
}

impl UpstashVectorIndex {
    pub fn new(base_url: &str, token: &str, namespace: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            token: token.to_string(),
            namespace: namespace.to_string(),
        }
    }

    fn url(&self, endpoint: &str) -> String {
        if self.namespace.is_empty() {
            format!("{}/{endpoint}", self.base_url)
        } else {
            format!("{}/{endpoint}/{}", self.base_url, self.namespace)
        }
    }

    async fn post<T: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &str,
        body: &T,
    ) -> Result<R> {
        let url = self.url(endpoint);
        debug!(url = %url, "upstash request");

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .json(body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Mem7Error::VectorStore(format!(
                "Upstash HTTP {status}: {text}"
            )));
        }

        resp.json().await.map_err(|e| {
            Mem7Error::VectorStore(format!("Upstash response parse error: {e}"))
        })
    }
}

// --- Upstash REST API types ---

#[derive(Serialize)]
struct UpsertEntry {
    id: String,
    vector: Vec<f32>,
    metadata: serde_json::Value,
}

#[derive(Serialize)]
struct QueryRequest {
    vector: Vec<f32>,
    #[serde(rename = "topK")]
    top_k: usize,
    #[serde(rename = "includeMetadata")]
    include_metadata: bool,
    #[serde(rename = "includeVectors")]
    include_vectors: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    filter: Option<String>,
}

#[derive(Serialize)]
struct RangeRequest {
    cursor: String,
    limit: usize,
    #[serde(rename = "includeMetadata")]
    include_metadata: bool,
    #[serde(rename = "includeVectors")]
    include_vectors: bool,
}

#[derive(Deserialize)]
struct UpstashResponse<T> {
    result: T,
}

#[derive(Deserialize)]
struct QueryResultEntry {
    id: String,
    score: f32,
    metadata: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct FetchResultEntry {
    id: String,
    vector: Option<Vec<f32>>,
    metadata: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct RangeResult {
    #[serde(rename = "nextCursor")]
    next_cursor: String,
    vectors: Vec<FetchResultEntry>,
}

fn strip_nulls(val: &serde_json::Value) -> serde_json::Value {
    match val {
        serde_json::Value::Object(map) => {
            let filtered: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            serde_json::Value::Object(filtered)
        }
        other => other.clone(),
    }
}

fn build_filter(filter: &MemoryFilter) -> Option<String> {
    let mut conditions = Vec::new();
    if let Some(ref uid) = filter.user_id {
        conditions.push(format!("user_id = '{uid}'"));
    }
    if let Some(ref aid) = filter.agent_id {
        conditions.push(format!("agent_id = '{aid}'"));
    }
    if let Some(ref rid) = filter.run_id {
        conditions.push(format!("run_id = '{rid}'"));
    }
    if conditions.is_empty() {
        None
    } else {
        Some(conditions.join(" AND "))
    }
}

fn matches_filter_local(metadata: &serde_json::Value, filter: &MemoryFilter) -> bool {
    if let Some(ref uid) = filter.user_id
        && metadata.get("user_id").and_then(|v| v.as_str()) != Some(uid.as_str())
    {
        return false;
    }
    if let Some(ref aid) = filter.agent_id
        && metadata.get("agent_id").and_then(|v| v.as_str()) != Some(aid.as_str())
    {
        return false;
    }
    if let Some(ref rid) = filter.run_id
        && metadata.get("run_id").and_then(|v| v.as_str()) != Some(rid.as_str())
    {
        return false;
    }
    true
}

#[async_trait]
impl VectorIndex for UpstashVectorIndex {
    async fn insert(&self, id: Uuid, vector: &[f32], payload: serde_json::Value) -> Result<()> {
        let entries = vec![UpsertEntry {
            id: id.to_string(),
            vector: vector.to_vec(),
            metadata: strip_nulls(&payload),
        }];
        let _: UpstashResponse<String> = self.post("upsert", &entries).await?;
        Ok(())
    }

    async fn search(
        &self,
        query: &[f32],
        limit: usize,
        filters: Option<&MemoryFilter>,
    ) -> Result<Vec<VectorSearchResult>> {
        let req = QueryRequest {
            vector: query.to_vec(),
            top_k: limit,
            include_metadata: true,
            include_vectors: false,
            filter: filters.and_then(build_filter),
        };
        let resp: UpstashResponse<Vec<QueryResultEntry>> = self.post("query", &req).await?;

        Ok(resp
            .result
            .into_iter()
            .filter_map(|entry| {
                let id = Uuid::parse_str(&entry.id).ok()?;
                Some(VectorSearchResult {
                    id,
                    score: entry.score,
                    payload: entry.metadata.unwrap_or(serde_json::Value::Null),
                })
            })
            .collect())
    }

    async fn delete(&self, id: &Uuid) -> Result<()> {
        let ids = vec![id.to_string()];
        let _: UpstashResponse<serde_json::Value> = self.post("delete", &ids).await?;
        Ok(())
    }

    async fn update(
        &self,
        id: &Uuid,
        vector: Option<&[f32]>,
        payload: Option<serde_json::Value>,
    ) -> Result<()> {
        // Upstash upsert replaces the entry; we need the full vector + metadata.
        let existing = self.get(id).await?;

        let (vec_data, meta_data) = match existing {
            Some((v, m)) => (v, m),
            None => return Err(Mem7Error::NotFound(format!("vector entry {id}"))),
        };

        let final_vec = vector.map(|v| v.to_vec()).unwrap_or(vec_data);
        let final_meta = payload
            .map(|p| strip_nulls(&p))
            .unwrap_or_else(|| strip_nulls(&meta_data));

        let entries = vec![UpsertEntry {
            id: id.to_string(),
            vector: final_vec,
            metadata: final_meta,
        }];
        let _: UpstashResponse<String> = self.post("upsert", &entries).await?;
        Ok(())
    }

    async fn get(&self, id: &Uuid) -> Result<Option<(Vec<f32>, serde_json::Value)>> {
        let url = if self.namespace.is_empty() {
            format!("{}/fetch/{id}", self.base_url)
        } else {
            format!("{}/fetch/{id}?ns={}", self.base_url, self.namespace)
        };

        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Mem7Error::VectorStore(format!(
                "Upstash fetch HTTP {status}: {text}"
            )));
        }

        let data: UpstashResponse<Option<FetchResultEntry>> = resp.json().await.map_err(|e| {
            Mem7Error::VectorStore(format!("Upstash fetch parse error: {e}"))
        })?;

        Ok(data.result.map(|entry| {
            (
                entry.vector.unwrap_or_default(),
                entry.metadata.unwrap_or(serde_json::Value::Null),
            )
        }))
    }

    async fn list(
        &self,
        filters: Option<&MemoryFilter>,
        limit: Option<usize>,
    ) -> Result<Vec<(Uuid, serde_json::Value)>> {
        let mut all_results = Vec::new();
        let mut cursor = "0".to_string();
        let page_size = 100;

        loop {
            let req = RangeRequest {
                cursor,
                limit: page_size,
                include_metadata: true,
                include_vectors: false,
            };
            let resp: UpstashResponse<RangeResult> = self.post("range", &req).await?;

            for entry in resp.result.vectors {
                if let Ok(id) = Uuid::parse_str(&entry.id) {
                    let metadata = entry.metadata.unwrap_or(serde_json::Value::Null);
                    let passes = filters
                        .map(|f| matches_filter_local(&metadata, f))
                        .unwrap_or(true);
                    if passes {
                        all_results.push((id, metadata));
                    }
                }
            }

            if let Some(lim) = limit
                && all_results.len() >= lim
            {
                all_results.truncate(lim);
                break;
            }

            if resp.result.next_cursor.is_empty() || resp.result.next_cursor == "0" {
                break;
            }
            cursor = resp.result.next_cursor;
        }

        all_results.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(all_results)
    }

    async fn reset(&self) -> Result<()> {
        let url = self.url("reset");
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Mem7Error::VectorStore(format!(
                "Upstash reset HTTP {status}: {text}"
            )));
        }
        Ok(())
    }
}
