use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use mem7_core::MemoryFilter;
use mem7_error::{Mem7Error, Result};
use tracing::{debug, info};

use crate::GraphStore;
use crate::types::{Entity, GraphSearchResult, Relation};

/// Kuzu-backed graph store (embedded, Cypher-based).
///
/// Kuzu's Rust API is synchronous, so all operations are bridged to async via
/// `tokio::task::spawn_blocking`, following the same pattern as `tokio-rusqlite`.
///
/// Kuzu does not have native vector similarity, so `search_by_embedding` loads
/// entity embeddings and computes cosine similarity in Rust, then 1-hop via Cypher.
pub struct KuzuGraphStore {
    db: Arc<kuzu::Database>,
}

impl KuzuGraphStore {
    /// Open or create a Kuzu database at the given path and ensure the schema exists.
    pub fn new(db_path: &str) -> Result<Self> {
        let path = Path::new(db_path);
        let db = kuzu::Database::new(path, kuzu::SystemConfig::default())
            .map_err(|e| Mem7Error::Graph(format!("failed to open Kuzu DB: {e}")))?;

        {
            let conn = kuzu::Connection::new(&db)
                .map_err(|e| Mem7Error::Graph(format!("failed to create Kuzu connection: {e}")))?;

            conn.query(
                "CREATE NODE TABLE IF NOT EXISTS Entity(\
                    name STRING, \
                    entity_type STRING, \
                    embedding DOUBLE[], \
                    created_at STRING, \
                    mentions INT64, \
                    user_id STRING, \
                    agent_id STRING, \
                    run_id STRING, \
                    PRIMARY KEY(name))",
            )
            .map_err(|e| Mem7Error::Graph(format!("failed to create Entity table: {e}")))?;

            conn.query(
                "CREATE REL TABLE IF NOT EXISTS RELATES(\
                    FROM Entity TO Entity, \
                    relationship STRING, \
                    created_at STRING, \
                    mentions INT64, \
                    valid BOOLEAN, \
                    user_id STRING, \
                    agent_id STRING, \
                    run_id STRING)",
            )
            .map_err(|e| Mem7Error::Graph(format!("failed to create RELATES table: {e}")))?;
        }

        info!(path = db_path, "KuzuGraphStore initialized");
        Ok(Self { db: Arc::new(db) })
    }
}

fn cosine_similarity(a: &[f64], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| *x * (*y as f64)).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b = b
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    let denom = norm_a * norm_b;
    if denom == 0.0 {
        0.0
    } else {
        (dot / denom) as f32
    }
}

#[async_trait]
impl GraphStore for KuzuGraphStore {
    async fn add_entities(&self, entities: &[Entity], filter: &MemoryFilter) -> Result<()> {
        let db = self.db.clone();
        let entities: Vec<Entity> = entities.to_vec();
        let user_id = filter.user_id.clone().unwrap_or_default();
        let agent_id = filter.agent_id.clone().unwrap_or_default();
        let run_id = filter.run_id.clone().unwrap_or_default();

        tokio::task::spawn_blocking(move || {
            let conn = kuzu::Connection::new(&db)
                .map_err(|e| Mem7Error::Graph(format!("connection error: {e}")))?;

            for entity in &entities {
                let embedding_str = entity
                    .embedding
                    .as_ref()
                    .map(|emb| {
                        let vals: Vec<String> = emb.iter().map(|v| v.to_string()).collect();
                        format!("[{}]", vals.join(","))
                    })
                    .unwrap_or_else(|| "[]".to_string());

                let cypher = format!(
                    "MERGE (e:Entity {{name: '{}'}}) \
                     ON CREATE SET e.entity_type = '{}', \
                                   e.embedding = {}, \
                                   e.created_at = '{}', \
                                   e.mentions = 1, \
                                   e.user_id = '{}', e.agent_id = '{}', e.run_id = '{}' \
                     ON MATCH SET e.mentions = e.mentions + 1{}",
                    escape_cypher(&entity.name),
                    escape_cypher(&entity.entity_type),
                    embedding_str,
                    entity.created_at.as_deref().unwrap_or(""),
                    escape_cypher(&user_id),
                    escape_cypher(&agent_id),
                    escape_cypher(&run_id),
                    if entity.embedding.is_some() {
                        format!(", e.embedding = {embedding_str}")
                    } else {
                        String::new()
                    },
                );
                conn.query(&cypher)
                    .map_err(|e| Mem7Error::Graph(format!("add entity error: {e}")))?;
            }

            debug!(count = entities.len(), "kuzu: entities added");
            Ok(())
        })
        .await
        .map_err(|e| Mem7Error::Graph(format!("spawn_blocking join error: {e}")))?
    }

    async fn add_relations(
        &self,
        relations: &[Relation],
        entities: &[Entity],
        filter: &MemoryFilter,
    ) -> Result<()> {
        self.add_entities(entities, filter).await?;

        let db = self.db.clone();
        let relations: Vec<Relation> = relations.to_vec();
        let user_id = filter.user_id.clone().unwrap_or_default();
        let agent_id = filter.agent_id.clone().unwrap_or_default();
        let run_id = filter.run_id.clone().unwrap_or_default();

        tokio::task::spawn_blocking(move || {
            let conn = kuzu::Connection::new(&db)
                .map_err(|e| Mem7Error::Graph(format!("connection error: {e}")))?;

            for rel in &relations {
                let cypher = format!(
                    "MATCH (s:Entity {{name: '{}'}}), (d:Entity {{name: '{}'}}) \
                     MERGE (s)-[r:RELATES {{relationship: '{}'}}]->(d) \
                     ON CREATE SET r.created_at = '{}', \
                                   r.mentions = 1, \
                                   r.valid = true, \
                                   r.user_id = '{}', r.agent_id = '{}', r.run_id = '{}' \
                     ON MATCH SET r.mentions = r.mentions + 1",
                    escape_cypher(&rel.source),
                    escape_cypher(&rel.destination),
                    escape_cypher(&rel.relationship),
                    rel.created_at.as_deref().unwrap_or(""),
                    escape_cypher(&user_id),
                    escape_cypher(&agent_id),
                    escape_cypher(&run_id),
                );
                conn.query(&cypher)
                    .map_err(|e| Mem7Error::Graph(format!("add relation error: {e}")))?;
            }

            debug!(count = relations.len(), "kuzu: relations added");
            Ok(())
        })
        .await
        .map_err(|e| Mem7Error::Graph(format!("spawn_blocking join error: {e}")))?
    }

    async fn search(
        &self,
        query: &str,
        filter: &MemoryFilter,
        limit: usize,
    ) -> Result<Vec<GraphSearchResult>> {
        let db = self.db.clone();
        let query = query.to_string();
        let user_id = filter.user_id.clone();
        let agent_id = filter.agent_id.clone();

        tokio::task::spawn_blocking(move || {
            let conn = kuzu::Connection::new(&db)
                .map_err(|e| Mem7Error::Graph(format!("connection error: {e}")))?;

            let escaped_query = escape_cypher(&query);

            let mut where_clauses = vec![
                format!(
                    "(s.name CONTAINS '{escaped_query}' \
                     OR d.name CONTAINS '{escaped_query}' \
                     OR r.relationship CONTAINS '{escaped_query}')"
                ),
                "r.valid = true".to_string(),
            ];

            if let Some(uid) = &user_id {
                where_clauses.push(format!("r.user_id = '{}'", escape_cypher(uid)));
            }
            if let Some(aid) = &agent_id {
                where_clauses.push(format!("r.agent_id = '{}'", escape_cypher(aid)));
            }

            let where_str = where_clauses.join(" AND ");

            let cypher = format!(
                "MATCH (s:Entity)-[r:RELATES]->(d:Entity) \
                 WHERE {where_str} \
                 RETURN s.name, r.relationship, d.name \
                 LIMIT {limit}"
            );

            let result = conn
                .query(&cypher)
                .map_err(|e| Mem7Error::Graph(format!("search error: {e}")))?;

            let mut results = Vec::new();
            for row in result {
                if row.len() >= 3 {
                    let source = value_to_string(&row[0]).unwrap_or_default();
                    let relationship = value_to_string(&row[1]).unwrap_or_default();
                    let destination = value_to_string(&row[2]).unwrap_or_default();

                    results.push(GraphSearchResult {
                        source,
                        relationship,
                        destination,
                        score: None,
                        created_at: None,
                        mentions: None,
                        last_accessed_at: None,
                    });
                }
            }

            debug!(count = results.len(), "kuzu: search results");
            Ok(results)
        })
        .await
        .map_err(|e| Mem7Error::Graph(format!("spawn_blocking join error: {e}")))?
    }

    async fn search_by_embedding(
        &self,
        embedding: &[f32],
        filter: &MemoryFilter,
        threshold: f32,
        limit: usize,
    ) -> Result<Vec<GraphSearchResult>> {
        let db = self.db.clone();
        let embedding = embedding.to_vec();
        let user_id = filter.user_id.clone();
        let agent_id = filter.agent_id.clone();

        tokio::task::spawn_blocking(move || {
            let conn = kuzu::Connection::new(&db)
                .map_err(|e| Mem7Error::Graph(format!("connection error: {e}")))?;

            // Step 1: Load all entity embeddings and compute cosine similarity in Rust
            let mut entity_where = Vec::new();
            if let Some(uid) = &user_id {
                entity_where.push(format!("e.user_id = '{}'", escape_cypher(uid)));
            }
            if let Some(aid) = &agent_id {
                entity_where.push(format!("e.agent_id = '{}'", escape_cypher(aid)));
            }

            let where_str = if entity_where.is_empty() {
                String::new()
            } else {
                format!(" WHERE {}", entity_where.join(" AND "))
            };

            let cypher = format!("MATCH (e:Entity){where_str} RETURN e.name, e.embedding");

            let result = conn
                .query(&cypher)
                .map_err(|e| Mem7Error::Graph(format!("load entities error: {e}")))?;

            let mut matched_entities: Vec<(String, f32)> = Vec::new();
            for row in result {
                if row.len() < 2 {
                    continue;
                }
                let name = value_to_string(&row[0]).unwrap_or_default();
                if let Some(emb_vec) = value_to_f64_list(&row[1]) {
                    if emb_vec.is_empty() {
                        continue;
                    }
                    let sim = cosine_similarity(&emb_vec, &embedding);
                    if sim >= threshold {
                        matched_entities.push((name, sim));
                    }
                }
            }

            if matched_entities.is_empty() {
                return Ok(Vec::new());
            }

            // Step 2: 1-hop traversal for matched entities
            let mut results = Vec::new();
            let mut seen = std::collections::HashSet::new();

            for (entity_name, sim) in &matched_entities {
                let escaped = escape_cypher(entity_name);
                let mut hop_where = vec!["r.valid = true".to_string()];
                if let Some(uid) = &user_id {
                    hop_where.push(format!("r.user_id = '{}'", escape_cypher(uid)));
                }
                let hop_filter = hop_where.join(" AND ");

                // Outgoing
                let out_cypher = format!(
                    "MATCH (s:Entity {{name: '{escaped}'}})-[r:RELATES]->(d:Entity) \
                     WHERE {hop_filter} \
                     RETURN s.name, r.relationship, d.name"
                );
                if let Ok(out_result) = conn.query(&out_cypher) {
                    for row in out_result {
                        if row.len() >= 3 {
                            let src = value_to_string(&row[0]).unwrap_or_default();
                            let rel = value_to_string(&row[1]).unwrap_or_default();
                            let dst = value_to_string(&row[2]).unwrap_or_default();
                            let key = (src.clone(), rel.clone(), dst.clone());
                            if seen.insert(key) {
                                results.push(GraphSearchResult {
                                    source: src,
                                    relationship: rel,
                                    destination: dst,
                                    score: Some(*sim),
                                    created_at: None,
                                    mentions: None,
                                    last_accessed_at: None,
                                });
                            }
                        }
                    }
                }

                // Incoming
                let in_cypher = format!(
                    "MATCH (s:Entity)-[r:RELATES]->(d:Entity {{name: '{escaped}'}}) \
                     WHERE {hop_filter} \
                     RETURN s.name, r.relationship, d.name"
                );
                if let Ok(in_result) = conn.query(&in_cypher) {
                    for row in in_result {
                        if row.len() >= 3 {
                            let src = value_to_string(&row[0]).unwrap_or_default();
                            let rel = value_to_string(&row[1]).unwrap_or_default();
                            let dst = value_to_string(&row[2]).unwrap_or_default();
                            let key = (src.clone(), rel.clone(), dst.clone());
                            if seen.insert(key) {
                                results.push(GraphSearchResult {
                                    source: src,
                                    relationship: rel,
                                    destination: dst,
                                    score: Some(*sim),
                                    created_at: None,
                                    mentions: None,
                                    last_accessed_at: None,
                                });
                            }
                        }
                    }
                }
            }

            results.sort_by(|a, b| {
                b.score
                    .unwrap_or(0.0)
                    .partial_cmp(&a.score.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(limit);

            debug!(count = results.len(), "kuzu: embedding search results");
            Ok(results)
        })
        .await
        .map_err(|e| Mem7Error::Graph(format!("spawn_blocking join error: {e}")))?
    }

    async fn invalidate_relations(
        &self,
        triples: &[(String, String, String)],
        filter: &MemoryFilter,
    ) -> Result<()> {
        let db = self.db.clone();
        let triples: Vec<(String, String, String)> = triples.to_vec();
        let user_id = filter.user_id.clone();

        tokio::task::spawn_blocking(move || {
            let conn = kuzu::Connection::new(&db)
                .map_err(|e| Mem7Error::Graph(format!("connection error: {e}")))?;

            for (src, rel, dst) in &triples {
                let mut where_clauses = vec![format!("r.relationship = '{}'", escape_cypher(rel))];
                if let Some(uid) = &user_id {
                    where_clauses.push(format!("r.user_id = '{}'", escape_cypher(uid)));
                }
                let where_str = where_clauses.join(" AND ");

                let cypher = format!(
                    "MATCH (s:Entity {{name: '{}'}})-[r:RELATES]->(d:Entity {{name: '{}'}}) \
                     WHERE {where_str} \
                     SET r.valid = false",
                    escape_cypher(src),
                    escape_cypher(dst),
                );
                conn.query(&cypher)
                    .map_err(|e| Mem7Error::Graph(format!("invalidate relation error: {e}")))?;
            }

            debug!(count = triples.len(), "kuzu: relations invalidated");
            Ok(())
        })
        .await
        .map_err(|e| Mem7Error::Graph(format!("spawn_blocking join error: {e}")))?
    }

    async fn rehearse_relations(
        &self,
        triples: &[(String, String, String)],
        filter: &MemoryFilter,
        now: &str,
    ) -> Result<()> {
        let db = self.db.clone();
        let triples: Vec<(String, String, String)> = triples.to_vec();
        let user_id = filter.user_id.clone();
        let now = now.to_string();

        tokio::task::spawn_blocking(move || {
            let conn = kuzu::Connection::new(&db)
                .map_err(|e| Mem7Error::Graph(format!("connection error: {e}")))?;

            for (src, rel, dst) in &triples {
                let mut where_clauses = vec![
                    format!("r.relationship = '{}'", escape_cypher(rel)),
                    "r.valid = true".to_string(),
                ];
                if let Some(uid) = &user_id {
                    where_clauses.push(format!("r.user_id = '{}'", escape_cypher(uid)));
                }
                let where_str = where_clauses.join(" AND ");

                let cypher = format!(
                    "MATCH (s:Entity {{name: '{}'}})-[r:RELATES]->(d:Entity {{name: '{}'}}) \
                     WHERE {where_str} \
                     SET r.mentions = r.mentions + 1, r.last_accessed_at = '{}'",
                    escape_cypher(src),
                    escape_cypher(dst),
                    escape_cypher(&now),
                );
                let _ = conn.query(&cypher);
            }

            debug!(count = triples.len(), "kuzu: relations rehearsed");
            Ok(())
        })
        .await
        .map_err(|e| Mem7Error::Graph(format!("spawn_blocking join error: {e}")))?
    }

    async fn delete_all(&self, filter: &MemoryFilter) -> Result<()> {
        let db = self.db.clone();
        let user_id = filter.user_id.clone();

        tokio::task::spawn_blocking(move || {
            let conn = kuzu::Connection::new(&db)
                .map_err(|e| Mem7Error::Graph(format!("connection error: {e}")))?;

            if let Some(uid) = &user_id {
                let cypher = format!(
                    "MATCH (s:Entity)-[r:RELATES]->(d:Entity) \
                     WHERE r.user_id = '{}' DELETE r",
                    escape_cypher(uid)
                );
                conn.query(&cypher)
                    .map_err(|e| Mem7Error::Graph(format!("delete relations error: {e}")))?;

                let cypher = format!(
                    "MATCH (e:Entity) WHERE e.user_id = '{}' DELETE e",
                    escape_cypher(uid)
                );
                conn.query(&cypher)
                    .map_err(|e| Mem7Error::Graph(format!("delete entities error: {e}")))?;
            }

            Ok(())
        })
        .await
        .map_err(|e| Mem7Error::Graph(format!("spawn_blocking join error: {e}")))?
    }

    async fn reset(&self) -> Result<()> {
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || {
            let conn = kuzu::Connection::new(&db)
                .map_err(|e| Mem7Error::Graph(format!("connection error: {e}")))?;

            conn.query("MATCH ()-[r:RELATES]->() DELETE r")
                .map_err(|e| Mem7Error::Graph(format!("reset relations error: {e}")))?;
            conn.query("MATCH (e:Entity) DELETE e")
                .map_err(|e| Mem7Error::Graph(format!("reset entities error: {e}")))?;

            info!("kuzu: graph reset");
            Ok(())
        })
        .await
        .map_err(|e| Mem7Error::Graph(format!("spawn_blocking join error: {e}")))?
    }
}

fn escape_cypher(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\'', "\\'")
}

fn value_to_string(v: &kuzu::Value) -> Option<String> {
    match v {
        kuzu::Value::String(s) => Some(s.clone()),
        _ => Some(format!("{v:?}")),
    }
}

fn value_to_f64_list(v: &kuzu::Value) -> Option<Vec<f64>> {
    match v {
        kuzu::Value::List(_, values) => {
            let mut result = Vec::new();
            for val in values {
                match val {
                    kuzu::Value::Double(d) => result.push(*d),
                    kuzu::Value::Float(f) => result.push(*f as f64),
                    kuzu::Value::Int64(i) => result.push(*i as f64),
                    _ => {}
                }
            }
            Some(result)
        }
        _ => None,
    }
}
