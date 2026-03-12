use async_trait::async_trait;
use mem7_core::MemoryFilter;
use mem7_error::{Mem7Error, Result};
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

        resp.json()
            .await
            .map_err(|e| Mem7Error::VectorStore(format!("Upstash response parse error: {e}")))
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
struct FetchRequest {
    ids: Vec<String>,
    #[serde(rename = "includeMetadata")]
    include_metadata: bool,
    #[serde(rename = "includeVectors")]
    include_vectors: bool,
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

    if let Some(ref meta) = filter.metadata
        && let Some(obj) = meta.as_object()
    {
        for (key, cond) in obj {
            if let Some(expr) = translate_metadata_condition(key, cond) {
                conditions.push(expr);
            }
        }
    }

    if conditions.is_empty() {
        None
    } else {
        Some(conditions.join(" AND "))
    }
}

/// Translate a single metadata filter entry to an Upstash filter expression.
/// Uses `metadata.{key}` path prefix for nested access.
fn translate_metadata_condition(key: &str, condition: &serde_json::Value) -> Option<String> {
    let field = format!("metadata.{key}");
    match key {
        "AND" => {
            let parts: Vec<String> = condition
                .as_array()?
                .iter()
                .filter_map(|f| {
                    let obj = f.as_object()?;
                    let sub: Vec<String> = obj
                        .iter()
                        .filter_map(|(k, v)| translate_metadata_condition(k, v))
                        .collect();
                    if sub.is_empty() {
                        None
                    } else {
                        Some(sub.join(" AND "))
                    }
                })
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(format!("({})", parts.join(" AND ")))
            }
        }
        "OR" => {
            let parts: Vec<String> = condition
                .as_array()?
                .iter()
                .filter_map(|f| {
                    let obj = f.as_object()?;
                    let sub: Vec<String> = obj
                        .iter()
                        .filter_map(|(k, v)| translate_metadata_condition(k, v))
                        .collect();
                    if sub.is_empty() {
                        None
                    } else {
                        Some(sub.join(" AND "))
                    }
                })
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(format!("({})", parts.join(" OR ")))
            }
        }
        "NOT" => {
            let parts: Vec<String> = condition
                .as_array()?
                .iter()
                .filter_map(|f| {
                    let obj = f.as_object()?;
                    let sub: Vec<String> = obj
                        .iter()
                        .filter_map(|(k, v)| translate_metadata_condition(k, v))
                        .collect();
                    if sub.is_empty() {
                        None
                    } else {
                        Some(format!("NOT ({})", sub.join(" AND ")))
                    }
                })
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join(" AND "))
            }
        }
        _ => match condition {
            serde_json::Value::Object(ops) => {
                let parts: Vec<String> = ops
                    .iter()
                    .filter_map(|(op, val)| translate_operator(&field, op, val))
                    .collect();
                if parts.is_empty() {
                    None
                } else {
                    Some(parts.join(" AND "))
                }
            }
            serde_json::Value::String(s) => Some(format!("{field} = '{s}'")),
            serde_json::Value::Number(n) => Some(format!("{field} = {n}")),
            serde_json::Value::Bool(b) => Some(format!("{field} = {b}")),
            _ => None,
        },
    }
}

fn translate_operator(field: &str, op: &str, val: &serde_json::Value) -> Option<String> {
    match op {
        "eq" => Some(format_equality(field, val)),
        "ne" => Some(format!("{field} != {}", format_value(val))),
        "gt" => Some(format!("{field} > {}", format_value(val))),
        "gte" => Some(format!("{field} >= {}", format_value(val))),
        "lt" => Some(format!("{field} < {}", format_value(val))),
        "lte" => Some(format!("{field} <= {}", format_value(val))),
        "in" => {
            let items: Vec<String> = val.as_array()?.iter().map(format_value).collect();
            Some(format!("{field} IN ({})", items.join(", ")))
        }
        "nin" => {
            let items: Vec<String> = val.as_array()?.iter().map(format_value).collect();
            Some(format!("{field} NOT IN ({})", items.join(", ")))
        }
        "contains" => {
            let s = val.as_str()?;
            Some(format!("{field} GLOB '*{s}*'"))
        }
        "icontains" => {
            let s = val.as_str()?;
            Some(format!("{field} GLOB '*{s}*'"))
        }
        _ => None,
    }
}

fn format_equality(field: &str, val: &serde_json::Value) -> String {
    format!("{field} = {}", format_value(val))
}

fn format_value(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::String(s) => format!("'{s}'"),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        other => other.to_string(),
    }
}

fn matches_filter_local(payload: &serde_json::Value, filter: &MemoryFilter) -> bool {
    crate::filter::matches_filter(payload, filter)
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
        let ids = vec![id.to_string()];

        let resp: UpstashResponse<Vec<FetchResultEntry>> = self
            .post(
                "fetch",
                &FetchRequest {
                    ids,
                    include_metadata: true,
                    include_vectors: true,
                },
            )
            .await?;

        Ok(resp.result.into_iter().next().map(|entry| {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_build_filter_user_id_only() {
        let f = MemoryFilter {
            user_id: Some("alice".into()),
            ..Default::default()
        };
        assert_eq!(build_filter(&f), Some("user_id = 'alice'".into()));
    }

    #[test]
    fn test_build_filter_metadata_simple_eq() {
        let f = MemoryFilter {
            metadata: Some(json!({"status": "active"})),
            ..Default::default()
        };
        assert_eq!(build_filter(&f), Some("metadata.status = 'active'".into()));
    }

    #[test]
    fn test_build_filter_metadata_operators() {
        let f = MemoryFilter {
            metadata: Some(json!({"score": {"gt": 50}})),
            ..Default::default()
        };
        assert_eq!(build_filter(&f), Some("metadata.score > 50".into()));

        let f2 = MemoryFilter {
            metadata: Some(json!({"score": {"lte": 100}})),
            ..Default::default()
        };
        assert_eq!(build_filter(&f2), Some("metadata.score <= 100".into()));
    }

    #[test]
    fn test_build_filter_metadata_in() {
        let f = MemoryFilter {
            metadata: Some(json!({"tag": {"in": ["rust", "python"]}})),
            ..Default::default()
        };
        assert_eq!(
            build_filter(&f),
            Some("metadata.tag IN ('rust', 'python')".into())
        );
    }

    #[test]
    fn test_build_filter_metadata_contains() {
        let f = MemoryFilter {
            metadata: Some(json!({"desc": {"contains": "hello"}})),
            ..Default::default()
        };
        assert_eq!(
            build_filter(&f),
            Some("metadata.desc GLOB '*hello*'".into())
        );
    }

    #[test]
    fn test_build_filter_and_combinator() {
        let f = MemoryFilter {
            metadata: Some(json!({"AND": [
                {"status": "active"},
                {"score": {"gt": 50}}
            ]})),
            ..Default::default()
        };
        let result = build_filter(&f).unwrap();
        assert_eq!(
            result,
            "(metadata.status = 'active' AND metadata.score > 50)"
        );
    }

    #[test]
    fn test_build_filter_or_combinator() {
        let f = MemoryFilter {
            metadata: Some(json!({"OR": [
                {"status": "active"},
                {"status": "pending"}
            ]})),
            ..Default::default()
        };
        let result = build_filter(&f).unwrap();
        assert_eq!(
            result,
            "(metadata.status = 'active' OR metadata.status = 'pending')"
        );
    }

    #[test]
    fn test_build_filter_combined_first_class_and_metadata() {
        let f = MemoryFilter {
            user_id: Some("alice".into()),
            metadata: Some(json!({"status": "active"})),
            ..Default::default()
        };
        let result = build_filter(&f).unwrap();
        assert_eq!(result, "user_id = 'alice' AND metadata.status = 'active'");
    }

    #[test]
    fn test_build_filter_no_conditions() {
        let f = MemoryFilter::default();
        assert_eq!(build_filter(&f), None);
    }

    #[test]
    fn test_build_filter_ne_operator() {
        let f = MemoryFilter {
            metadata: Some(json!({"status": {"ne": "deleted"}})),
            ..Default::default()
        };
        assert_eq!(
            build_filter(&f),
            Some("metadata.status != 'deleted'".into())
        );
    }

    #[test]
    fn test_build_filter_not_combinator() {
        let f = MemoryFilter {
            metadata: Some(json!({"NOT": [{"status": "deleted"}]})),
            ..Default::default()
        };
        let result = build_filter(&f).unwrap();
        assert_eq!(result, "NOT (metadata.status = 'deleted')");
    }
}
