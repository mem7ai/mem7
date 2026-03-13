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
                let cypher = format!(
                    "MERGE (e:Entity {{name: '{}'}}) \
                     SET e.entity_type = '{}', e.user_id = '{}', e.agent_id = '{}', e.run_id = '{}'",
                    escape_cypher(&entity.name),
                    escape_cypher(&entity.entity_type),
                    escape_cypher(&user_id),
                    escape_cypher(&agent_id),
                    escape_cypher(&run_id),
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
        // Ensure all referenced entities exist first
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
                     SET r.user_id = '{}', r.agent_id = '{}', r.run_id = '{}'",
                    escape_cypher(&rel.source),
                    escape_cypher(&rel.destination),
                    escape_cypher(&rel.relationship),
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

            let mut where_clauses = vec![format!(
                "(s.name CONTAINS '{escaped_query}' \
                 OR d.name CONTAINS '{escaped_query}' \
                 OR r.relationship CONTAINS '{escaped_query}')"
            )];

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
                    });
                }
            }

            debug!(count = results.len(), "kuzu: search results");
            Ok(results)
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
