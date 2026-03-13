use std::sync::Arc;

use async_trait::async_trait;
use mem7_core::MemoryFilter;
use mem7_error::{Mem7Error, Result};
use neo4rs::{Graph, query};
use tracing::{debug, info};

use crate::GraphStore;
use crate::types::{Entity, GraphSearchResult, Relation};

/// Neo4j-backed graph store for production use.
pub struct Neo4jGraphStore {
    graph: Arc<Graph>,
}

impl Neo4jGraphStore {
    /// Connect to a Neo4j instance and ensure schema constraints exist.
    pub async fn new(
        url: &str,
        username: &str,
        password: &str,
        database: Option<&str>,
    ) -> Result<Self> {
        let mut config = neo4rs::ConfigBuilder::default()
            .uri(url)
            .user(username)
            .password(password);

        if let Some(db) = database {
            config = config.db(db);
        }

        let graph = Graph::connect(
            config
                .build()
                .map_err(|e| Mem7Error::Graph(format!("Neo4j config error: {e}")))?,
        )
        .await
        .map_err(|e| Mem7Error::Graph(format!("Neo4j connection error: {e}")))?;

        graph
            .run(query(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            ))
            .await
            .map_err(|e| Mem7Error::Graph(format!("constraint creation error: {e}")))?;

        info!(url, "Neo4jGraphStore connected");
        Ok(Self {
            graph: Arc::new(graph),
        })
    }
}

#[async_trait]
impl GraphStore for Neo4jGraphStore {
    async fn add_entities(&self, entities: &[Entity], filter: &MemoryFilter) -> Result<()> {
        for entity in entities {
            let q = query(
                "MERGE (e:Entity {name: $name}) \
                 SET e.entity_type = $entity_type, \
                     e.user_id = $user_id, \
                     e.agent_id = $agent_id, \
                     e.run_id = $run_id",
            )
            .param("name", entity.name.as_str())
            .param("entity_type", entity.entity_type.as_str())
            .param("user_id", filter.user_id.as_deref().unwrap_or(""))
            .param("agent_id", filter.agent_id.as_deref().unwrap_or(""))
            .param("run_id", filter.run_id.as_deref().unwrap_or(""));

            self.graph
                .run(q)
                .await
                .map_err(|e| Mem7Error::Graph(format!("add entity error: {e}")))?;
        }

        debug!(count = entities.len(), "neo4j: entities added");
        Ok(())
    }

    async fn add_relations(
        &self,
        relations: &[Relation],
        entities: &[Entity],
        filter: &MemoryFilter,
    ) -> Result<()> {
        self.add_entities(entities, filter).await?;

        for rel in relations {
            let q = query(
                "MATCH (s:Entity {name: $src}), (d:Entity {name: $dst}) \
                 MERGE (s)-[r:RELATES {relationship: $rel}]->(d) \
                 SET r.user_id = $user_id, \
                     r.agent_id = $agent_id, \
                     r.run_id = $run_id",
            )
            .param("src", rel.source.as_str())
            .param("dst", rel.destination.as_str())
            .param("rel", rel.relationship.as_str())
            .param("user_id", filter.user_id.as_deref().unwrap_or(""))
            .param("agent_id", filter.agent_id.as_deref().unwrap_or(""))
            .param("run_id", filter.run_id.as_deref().unwrap_or(""));

            self.graph
                .run(q)
                .await
                .map_err(|e| Mem7Error::Graph(format!("add relation error: {e}")))?;
        }

        debug!(count = relations.len(), "neo4j: relations added");
        Ok(())
    }

    async fn search(
        &self,
        query_str: &str,
        filter: &MemoryFilter,
        limit: usize,
    ) -> Result<Vec<GraphSearchResult>> {
        let cypher = "\
            MATCH (s:Entity)-[r:RELATES]->(d:Entity) \
            WHERE (toLower(s.name) CONTAINS toLower($query) \
                   OR toLower(d.name) CONTAINS toLower($query) \
                   OR toLower(r.relationship) CONTAINS toLower($query)) \
                  AND ($user_id = '' OR r.user_id = $user_id) \
                  AND ($agent_id = '' OR r.agent_id = $agent_id) \
            RETURN s.name AS source, r.relationship AS relationship, d.name AS destination \
            LIMIT $limit";

        let q = query(cypher)
            .param("query", query_str)
            .param("user_id", filter.user_id.as_deref().unwrap_or(""))
            .param("agent_id", filter.agent_id.as_deref().unwrap_or(""))
            .param("limit", limit as i64);

        let mut result = self
            .graph
            .execute(q)
            .await
            .map_err(|e| Mem7Error::Graph(format!("search error: {e}")))?;

        let mut results = Vec::new();
        while let Ok(Some(row)) = result.next().await {
            let source: String = row.get("source").unwrap_or_default();
            let relationship: String = row.get("relationship").unwrap_or_default();
            let destination: String = row.get("destination").unwrap_or_default();

            results.push(GraphSearchResult {
                source,
                relationship,
                destination,
            });
        }

        debug!(count = results.len(), "neo4j: search results");
        Ok(results)
    }

    async fn delete_all(&self, filter: &MemoryFilter) -> Result<()> {
        if let Some(uid) = &filter.user_id {
            let q = query(
                "MATCH (s:Entity)-[r:RELATES]->(d:Entity) \
                 WHERE r.user_id = $user_id DELETE r",
            )
            .param("user_id", uid.as_str());

            self.graph
                .run(q)
                .await
                .map_err(|e| Mem7Error::Graph(format!("delete relations error: {e}")))?;

            let q = query("MATCH (e:Entity) WHERE e.user_id = $user_id DELETE e")
                .param("user_id", uid.as_str());

            self.graph
                .run(q)
                .await
                .map_err(|e| Mem7Error::Graph(format!("delete entities error: {e}")))?;
        }

        Ok(())
    }

    async fn reset(&self) -> Result<()> {
        self.graph
            .run(query("MATCH ()-[r:RELATES]->() DELETE r"))
            .await
            .map_err(|e| Mem7Error::Graph(format!("reset relations error: {e}")))?;

        self.graph
            .run(query("MATCH (e:Entity) DELETE e"))
            .await
            .map_err(|e| Mem7Error::Graph(format!("reset entities error: {e}")))?;

        info!("neo4j: graph reset");
        Ok(())
    }
}
