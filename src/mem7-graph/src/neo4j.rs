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
                 ON CREATE SET e.entity_type = $entity_type, \
                               e.user_id = $user_id, \
                               e.agent_id = $agent_id, \
                               e.run_id = $run_id, \
                               e.created_at = timestamp(), \
                               e.mentions = 1 \
                 ON MATCH SET  e.mentions = e.mentions + 1",
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

            if let Some(emb) = &entity.embedding {
                let emb_q = query(
                    "MATCH (e:Entity {name: $name}) \
                     CALL db.create.setNodeVectorProperty(e, 'embedding', $embedding)",
                )
                .param("name", entity.name.as_str())
                .param("embedding", emb.clone());

                self.graph
                    .run(emb_q)
                    .await
                    .map_err(|e| Mem7Error::Graph(format!("set embedding error: {e}")))?;
            }
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
                 ON CREATE SET r.user_id = $user_id, \
                               r.agent_id = $agent_id, \
                               r.run_id = $run_id, \
                               r.created_at = timestamp(), \
                               r.mentions = 1, \
                               r.valid = true \
                 ON MATCH SET  r.mentions = r.mentions + 1",
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
            WHERE r.valid = true \
                  AND (toLower(s.name) CONTAINS toLower($query) \
                       OR toLower(d.name) CONTAINS toLower($query) \
                       OR toLower(r.relationship) CONTAINS toLower($query)) \
                  AND ($user_id = '' OR r.user_id = $user_id) \
                  AND ($agent_id = '' OR r.agent_id = $agent_id) \
                  AND ($run_id = '' OR r.run_id = $run_id) \
            RETURN s.name AS source, r.relationship AS relationship, d.name AS destination \
            LIMIT $limit";

        let q = query(cypher)
            .param("query", query_str)
            .param("user_id", filter.user_id.as_deref().unwrap_or(""))
            .param("agent_id", filter.agent_id.as_deref().unwrap_or(""))
            .param("run_id", filter.run_id.as_deref().unwrap_or(""))
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
                score: None,
                created_at: None,
                mentions: None,
                last_accessed_at: None,
            });
        }

        debug!(count = results.len(), "neo4j: search results");
        Ok(results)
    }

    async fn search_by_embedding(
        &self,
        embedding: &[f32],
        filter: &MemoryFilter,
        threshold: f32,
        limit: usize,
    ) -> Result<Vec<GraphSearchResult>> {
        let cypher = "\
            MATCH (n:Entity) \
            WHERE n.embedding IS NOT NULL \
                  AND ($user_id = '' OR n.user_id = $user_id) \
                  AND ($agent_id = '' OR n.agent_id = $agent_id) \
                  AND ($run_id = '' OR n.run_id = $run_id) \
            WITH n, vector.similarity.cosine(n.embedding, $embedding) AS similarity \
            WHERE similarity >= $threshold \
            CALL { \
                WITH n \
                MATCH (n)-[r:RELATES]->(m:Entity) \
                WHERE r.valid = true \
                  AND ($user_id = '' OR r.user_id = $user_id) \
                  AND ($agent_id = '' OR r.agent_id = $agent_id) \
                  AND ($run_id = '' OR r.run_id = $run_id) \
                RETURN n.name AS source, r.relationship AS relationship, m.name AS destination, similarity, \
                       r.created_at AS rel_created_at, r.mentions AS rel_mentions, r.last_accessed_at AS rel_last_accessed \
                UNION \
                WITH n, similarity \
                MATCH (n)<-[r:RELATES]-(m:Entity) \
                WHERE r.valid = true \
                  AND ($user_id = '' OR r.user_id = $user_id) \
                  AND ($agent_id = '' OR r.agent_id = $agent_id) \
                  AND ($run_id = '' OR r.run_id = $run_id) \
                RETURN m.name AS source, r.relationship AS relationship, n.name AS destination, similarity, \
                       r.created_at AS rel_created_at, r.mentions AS rel_mentions, r.last_accessed_at AS rel_last_accessed \
            } \
            RETURN DISTINCT source, relationship, destination, similarity, rel_created_at, rel_mentions, rel_last_accessed \
            ORDER BY similarity DESC \
            LIMIT $limit";

        let q = query(cypher)
            .param("embedding", embedding.to_vec())
            .param("threshold", threshold as f64)
            .param("user_id", filter.user_id.as_deref().unwrap_or(""))
            .param("agent_id", filter.agent_id.as_deref().unwrap_or(""))
            .param("run_id", filter.run_id.as_deref().unwrap_or(""))
            .param("limit", limit as i64);

        let mut result = self
            .graph
            .execute(q)
            .await
            .map_err(|e| Mem7Error::Graph(format!("embedding search error: {e}")))?;

        let mut results = Vec::new();
        while let Ok(Some(row)) = result.next().await {
            let source: String = row.get("source").unwrap_or_default();
            let relationship: String = row.get("relationship").unwrap_or_default();
            let destination: String = row.get("destination").unwrap_or_default();
            let similarity: f64 = row.get("similarity").unwrap_or_default();
            let rel_created_at: Option<i64> = row.get("rel_created_at").ok();
            let rel_mentions: Option<i64> = row.get("rel_mentions").ok();
            let rel_last_accessed: Option<String> = row.get("rel_last_accessed").ok();

            results.push(GraphSearchResult {
                source,
                relationship,
                destination,
                score: Some(similarity as f32),
                created_at: rel_created_at.map(|ts| format!("{ts}")),
                mentions: rel_mentions.map(|m| m as u32),
                last_accessed_at: rel_last_accessed,
            });
        }

        debug!(count = results.len(), "neo4j: embedding search results");
        Ok(results)
    }

    async fn invalidate_relations(
        &self,
        triples: &[(String, String, String)],
        filter: &MemoryFilter,
    ) -> Result<()> {
        for (src, rel, dst) in triples {
            let q = query(
                "MATCH (s:Entity {name: $src})-[r:RELATES {relationship: $rel}]->(d:Entity {name: $dst}) \
                 WHERE ($user_id = '' OR r.user_id = $user_id) \
                   AND ($agent_id = '' OR r.agent_id = $agent_id) \
                   AND ($run_id = '' OR r.run_id = $run_id) \
                 SET r.valid = false",
            )
            .param("src", src.as_str())
            .param("rel", rel.as_str())
            .param("dst", dst.as_str())
            .param("user_id", filter.user_id.as_deref().unwrap_or(""))
            .param("agent_id", filter.agent_id.as_deref().unwrap_or(""))
            .param("run_id", filter.run_id.as_deref().unwrap_or(""));

            self.graph
                .run(q)
                .await
                .map_err(|e| Mem7Error::Graph(format!("invalidate relation error: {e}")))?;
        }

        debug!(count = triples.len(), "neo4j: relations invalidated");
        Ok(())
    }

    async fn rehearse_relations(
        &self,
        triples: &[(String, String, String)],
        filter: &MemoryFilter,
        now: &str,
    ) -> Result<()> {
        for (src, rel, dst) in triples {
            let q = query(
                "MATCH (s:Entity {name: $src})-[r:RELATES {relationship: $rel}]->(d:Entity {name: $dst}) \
                 WHERE r.valid = true \
                   AND ($user_id = '' OR r.user_id = $user_id) \
                   AND ($agent_id = '' OR r.agent_id = $agent_id) \
                   AND ($run_id = '' OR r.run_id = $run_id) \
                 SET r.mentions = r.mentions + 1, r.last_accessed_at = $now",
            )
            .param("src", src.as_str())
            .param("rel", rel.as_str())
            .param("dst", dst.as_str())
            .param("user_id", filter.user_id.as_deref().unwrap_or(""))
            .param("agent_id", filter.agent_id.as_deref().unwrap_or(""))
            .param("run_id", filter.run_id.as_deref().unwrap_or(""))
            .param("now", now);

            self.graph
                .run(q)
                .await
                .map_err(|e| Mem7Error::Graph(format!("rehearse relation error: {e}")))?;
        }

        debug!(count = triples.len(), "neo4j: relations rehearsed");
        Ok(())
    }

    async fn delete_all(&self, filter: &MemoryFilter) -> Result<()> {
        let q = query(
            "MATCH (s:Entity)-[r:RELATES]->(d:Entity) \
             WHERE ($user_id = '' OR r.user_id = $user_id) \
               AND ($agent_id = '' OR r.agent_id = $agent_id) \
               AND ($run_id = '' OR r.run_id = $run_id) \
             DELETE r",
        )
        .param("user_id", filter.user_id.as_deref().unwrap_or(""))
        .param("agent_id", filter.agent_id.as_deref().unwrap_or(""))
        .param("run_id", filter.run_id.as_deref().unwrap_or(""));

        self.graph
            .run(q)
            .await
            .map_err(|e| Mem7Error::Graph(format!("delete relations error: {e}")))?;

        self.graph
            .run(query("MATCH (e:Entity) WHERE NOT (e)--() DELETE e"))
            .await
            .map_err(|e| Mem7Error::Graph(format!("delete orphan entities error: {e}")))?;

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
