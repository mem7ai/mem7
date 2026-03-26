use std::sync::RwLock;

use async_trait::async_trait;
use mem7_core::MemoryFilter;
use mem7_error::Result;

use crate::GraphStore;
use crate::types::{Entity, GraphSearchResult, Relation};

#[derive(Debug, Clone)]
struct StoredEntity {
    name: String,
    entity_type: String,
    embedding: Option<Vec<f32>>,
    #[allow(dead_code)]
    created_at: Option<String>,
    mentions: u32,
    #[allow(dead_code)]
    last_accessed_at: Option<String>,
    user_id: Option<String>,
    agent_id: Option<String>,
    run_id: Option<String>,
}

#[derive(Debug, Clone)]
struct StoredRelation {
    source: String,
    relationship: String,
    destination: String,
    created_at: Option<String>,
    mentions: u32,
    valid: bool,
    last_accessed_at: Option<String>,
    user_id: Option<String>,
    agent_id: Option<String>,
    run_id: Option<String>,
}

/// In-memory graph store for development and testing.
pub struct FlatGraph {
    entities: RwLock<Vec<StoredEntity>>,
    relations: RwLock<Vec<StoredRelation>>,
}

impl FlatGraph {
    pub fn new() -> Self {
        Self {
            entities: RwLock::new(Vec::new()),
            relations: RwLock::new(Vec::new()),
        }
    }
}

impl Default for FlatGraph {
    fn default() -> Self {
        Self::new()
    }
}

fn matches_filter(
    user_id: &Option<String>,
    agent_id: &Option<String>,
    run_id: &Option<String>,
    filter: &MemoryFilter,
) -> bool {
    if let Some(uid) = &filter.user_id {
        if user_id.as_deref() != Some(uid.as_str()) {
            return false;
        }
    }
    if let Some(aid) = &filter.agent_id {
        if agent_id.as_deref() != Some(aid.as_str()) {
            return false;
        }
    }
    if let Some(rid) = &filter.run_id {
        if run_id.as_deref() != Some(rid.as_str()) {
            return false;
        }
    }
    true
}

#[async_trait]
impl GraphStore for FlatGraph {
    async fn add_entities(&self, entities: &[Entity], filter: &MemoryFilter) -> Result<()> {
        let mut store = self.entities.write().expect("entity lock poisoned");
        for entity in entities {
            if let Some(existing) = store.iter_mut().find(|e| {
                e.name == entity.name && matches_filter(&e.user_id, &e.agent_id, &e.run_id, filter)
            }) {
                existing.mentions += 1;
                if entity.embedding.is_some() {
                    existing.embedding.clone_from(&entity.embedding);
                }
                if entity.entity_type != existing.entity_type {
                    existing.entity_type.clone_from(&entity.entity_type);
                }
            } else {
                store.push(StoredEntity {
                    name: entity.name.clone(),
                    entity_type: entity.entity_type.clone(),
                    embedding: entity.embedding.clone(),
                    created_at: entity.created_at.clone(),
                    mentions: 1,
                    last_accessed_at: entity.created_at.clone(),
                    user_id: filter.user_id.clone(),
                    agent_id: filter.agent_id.clone(),
                    run_id: filter.run_id.clone(),
                });
            }
        }
        Ok(())
    }

    async fn add_relations(
        &self,
        relations: &[Relation],
        entities: &[Entity],
        filter: &MemoryFilter,
    ) -> Result<()> {
        self.add_entities(entities, filter).await?;

        let mut store = self.relations.write().expect("relation lock poisoned");
        for r in relations {
            if let Some(existing) = store.iter_mut().find(|e| {
                e.source == r.source
                    && e.relationship == r.relationship
                    && e.destination == r.destination
                    && e.valid
                    && matches_filter(&e.user_id, &e.agent_id, &e.run_id, filter)
            }) {
                existing.mentions += 1;
            } else {
                store.push(StoredRelation {
                    source: r.source.clone(),
                    relationship: r.relationship.clone(),
                    destination: r.destination.clone(),
                    created_at: r.created_at.clone(),
                    mentions: 1,
                    valid: true,
                    last_accessed_at: r.created_at.clone(),
                    user_id: filter.user_id.clone(),
                    agent_id: filter.agent_id.clone(),
                    run_id: filter.run_id.clone(),
                });
            }
        }
        Ok(())
    }

    async fn search(
        &self,
        query: &str,
        filter: &MemoryFilter,
        limit: usize,
    ) -> Result<Vec<GraphSearchResult>> {
        let store = self.relations.read().expect("relation lock poisoned");
        let query_lower = query.to_lowercase();

        let results: Vec<GraphSearchResult> = store
            .iter()
            .filter(|r| {
                r.valid
                    && matches_filter(&r.user_id, &r.agent_id, &r.run_id, filter)
                    && (r.source.to_lowercase().contains(&query_lower)
                        || r.destination.to_lowercase().contains(&query_lower)
                        || r.relationship.to_lowercase().contains(&query_lower))
            })
            .take(limit)
            .map(|r| GraphSearchResult {
                source: r.source.clone(),
                relationship: r.relationship.clone(),
                destination: r.destination.clone(),
                score: None,
                created_at: r.created_at.clone(),
                mentions: Some(r.mentions),
                last_accessed_at: r.last_accessed_at.clone(),
            })
            .collect();

        Ok(results)
    }

    async fn search_by_embedding(
        &self,
        embedding: &[f32],
        filter: &MemoryFilter,
        threshold: f32,
        limit: usize,
    ) -> Result<Vec<GraphSearchResult>> {
        let entities = self.entities.read().expect("entity lock poisoned");

        // Find entities whose embedding is above the similarity threshold
        let matched_names: Vec<(&str, f32)> = entities
            .iter()
            .filter(|e| matches_filter(&e.user_id, &e.agent_id, &e.run_id, filter))
            .filter_map(|e| {
                e.embedding.as_ref().map(|emb| {
                    let sim = mem7_vector::cosine_similarity(emb, embedding);
                    (e.name.as_str(), sim)
                })
            })
            .filter(|(_, sim)| *sim >= threshold)
            .collect();

        if matched_names.is_empty() {
            return Ok(Vec::new());
        }

        // 1-hop: collect all valid relations touching matched entities
        let relations = self.relations.read().expect("relation lock poisoned");
        let mut results: Vec<GraphSearchResult> = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for (name, sim) in &matched_names {
            for r in relations.iter() {
                if !r.valid || !matches_filter(&r.user_id, &r.agent_id, &r.run_id, filter) {
                    continue;
                }
                if r.source.as_str() == *name || r.destination.as_str() == *name {
                    let key = (
                        r.source.clone(),
                        r.relationship.clone(),
                        r.destination.clone(),
                    );
                    if seen.insert(key) {
                        results.push(GraphSearchResult {
                            source: r.source.clone(),
                            relationship: r.relationship.clone(),
                            destination: r.destination.clone(),
                            score: Some(*sim),
                            created_at: r.created_at.clone(),
                            mentions: Some(r.mentions),
                            last_accessed_at: r.last_accessed_at.clone(),
                        });
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

        Ok(results)
    }

    async fn invalidate_relations(
        &self,
        triples: &[(String, String, String)],
        filter: &MemoryFilter,
    ) -> Result<()> {
        let mut store = self.relations.write().expect("relation lock poisoned");
        for r in store.iter_mut() {
            if !matches_filter(&r.user_id, &r.agent_id, &r.run_id, filter) {
                continue;
            }
            for (src, rel, dst) in triples {
                if r.source == *src && r.relationship == *rel && r.destination == *dst && r.valid {
                    r.valid = false;
                }
            }
        }
        Ok(())
    }

    async fn rehearse_relations(
        &self,
        triples: &[(String, String, String)],
        filter: &MemoryFilter,
        now: &str,
    ) -> Result<()> {
        let mut store = self.relations.write().expect("relation lock poisoned");
        for r in store.iter_mut() {
            if !r.valid || !matches_filter(&r.user_id, &r.agent_id, &r.run_id, filter) {
                continue;
            }
            for (src, rel, dst) in triples {
                if r.source == *src && r.relationship == *rel && r.destination == *dst {
                    r.mentions += 1;
                    r.last_accessed_at = Some(now.to_string());
                }
            }
        }
        Ok(())
    }

    async fn delete_all(&self, filter: &MemoryFilter) -> Result<()> {
        let mut rel_store = self.relations.write().expect("relation lock poisoned");
        rel_store.retain(|r| !matches_filter(&r.user_id, &r.agent_id, &r.run_id, filter));

        let referenced_entities: std::collections::HashSet<String> = rel_store
            .iter()
            .flat_map(|r| [r.source.clone(), r.destination.clone()])
            .collect();

        let mut ent_store = self.entities.write().expect("entity lock poisoned");
        ent_store.retain(|e| referenced_entities.contains(&e.name));

        Ok(())
    }

    async fn reset(&self) -> Result<()> {
        self.relations
            .write()
            .expect("relation lock poisoned")
            .clear();
        self.entities.write().expect("entity lock poisoned").clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_filter(user_id: &str) -> MemoryFilter {
        MemoryFilter {
            user_id: Some(user_id.to_string()),
            agent_id: None,
            run_id: None,
            metadata: None,
        }
    }

    fn scoped_filter(user_id: &str, agent_id: &str, run_id: &str) -> MemoryFilter {
        MemoryFilter {
            user_id: Some(user_id.to_string()),
            agent_id: Some(agent_id.to_string()),
            run_id: Some(run_id.to_string()),
            metadata: None,
        }
    }

    fn make_entity(name: &str, etype: &str, embedding: Option<Vec<f32>>) -> Entity {
        Entity {
            name: name.into(),
            entity_type: etype.into(),
            embedding,
            created_at: None,
            mentions: 0,
        }
    }

    fn make_relation(src: &str, rel: &str, dst: &str) -> Relation {
        Relation {
            source: src.into(),
            relationship: rel.into(),
            destination: dst.into(),
            created_at: None,
            mentions: 0,
            valid: true,
        }
    }

    #[tokio::test]
    async fn add_and_search_relations() {
        let graph = FlatGraph::new();
        let filter = test_filter("user1");

        let entities = vec![
            make_entity("Alice", "Person", None),
            make_entity("tennis", "Activity", None),
        ];
        let relations = vec![make_relation("Alice", "loves_playing", "tennis")];

        graph
            .add_relations(&relations, &entities, &filter)
            .await
            .unwrap();

        let results = graph.search("Alice", &filter, 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source, "Alice");
        assert_eq!(results[0].relationship, "loves_playing");
        assert_eq!(results[0].destination, "tennis");
    }

    #[tokio::test]
    async fn add_entities_stores_and_upserts() {
        let graph = FlatGraph::new();
        let filter = test_filter("user1");

        let entities = vec![make_entity("Alice", "Person", Some(vec![1.0, 0.0]))];
        graph.add_entities(&entities, &filter).await.unwrap();

        // Adding again should increment mentions
        graph.add_entities(&entities, &filter).await.unwrap();

        let store = graph.entities.read().unwrap();
        assert_eq!(store.len(), 1);
        assert_eq!(store[0].mentions, 2);
        assert!(store[0].embedding.is_some());
    }

    #[tokio::test]
    async fn search_by_embedding_finds_related() {
        let graph = FlatGraph::new();
        let filter = test_filter("user1");

        let entities = vec![
            make_entity("Alice", "Person", Some(vec![1.0, 0.0, 0.0])),
            make_entity("Bob", "Person", Some(vec![0.0, 1.0, 0.0])),
        ];
        let relations = vec![
            make_relation("Alice", "friend_of", "Bob"),
            make_relation("Alice", "likes", "tennis"),
        ];

        graph
            .add_relations(&relations, &entities, &filter)
            .await
            .unwrap();

        // Query embedding very similar to Alice's
        let query_emb = vec![0.99, 0.01, 0.0];
        let results = graph
            .search_by_embedding(&query_emb, &filter, 0.7, 10)
            .await
            .unwrap();

        // Should find relations touching Alice (2 relations)
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn search_by_embedding_respects_threshold() {
        let graph = FlatGraph::new();
        let filter = test_filter("user1");

        let entities = vec![make_entity("Alice", "Person", Some(vec![1.0, 0.0]))];
        let relations = vec![make_relation("Alice", "likes", "coffee")];

        graph
            .add_relations(&relations, &entities, &filter)
            .await
            .unwrap();

        // Orthogonal embedding — should not match
        let query_emb = vec![0.0, 1.0];
        let results = graph
            .search_by_embedding(&query_emb, &filter, 0.7, 10)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn invalidate_relations_soft_deletes() {
        let graph = FlatGraph::new();
        let filter = test_filter("user1");

        let entities = vec![make_entity("USER", "Person", None)];
        let relations = vec![
            make_relation("USER", "works_at", "Google"),
            make_relation("USER", "lives_in", "NYC"),
        ];

        graph
            .add_relations(&relations, &entities, &filter)
            .await
            .unwrap();

        // Invalidate only works_at
        graph
            .invalidate_relations(
                &[("USER".into(), "works_at".into(), "Google".into())],
                &filter,
            )
            .await
            .unwrap();

        // Text search should only find lives_in (valid=true)
        let results = graph.search("USER", &filter, 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].relationship, "lives_in");
    }

    #[tokio::test]
    async fn relation_dedup_increments_mentions() {
        let graph = FlatGraph::new();
        let filter = test_filter("user1");

        let entities = vec![make_entity("Alice", "Person", None)];
        let relations = vec![make_relation("Alice", "likes", "coffee")];

        graph
            .add_relations(&relations, &entities, &filter)
            .await
            .unwrap();
        graph
            .add_relations(&relations, &entities, &filter)
            .await
            .unwrap();

        let store = graph.relations.read().unwrap();
        assert_eq!(store.len(), 1);
        assert_eq!(store[0].mentions, 2);
    }

    #[tokio::test]
    async fn search_by_relationship() {
        let graph = FlatGraph::new();
        let filter = test_filter("user1");

        let entities = vec![
            make_entity("Bob", "Person", None),
            make_entity("Google", "Organization", None),
        ];
        let relations = vec![make_relation("Bob", "works_at", "Google")];

        graph
            .add_relations(&relations, &entities, &filter)
            .await
            .unwrap();

        let results = graph.search("works", &filter, 10).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn search_respects_user_scope() {
        let graph = FlatGraph::new();
        let filter1 = test_filter("user1");
        let filter2 = test_filter("user2");

        let entities = vec![make_entity("X", "Other", None)];
        let rels = vec![make_relation("X", "rel", "Y")];

        graph
            .add_relations(&rels, &entities, &filter1)
            .await
            .unwrap();

        let r1 = graph.search("X", &filter1, 10).await.unwrap();
        assert_eq!(r1.len(), 1);

        let r2 = graph.search("X", &filter2, 10).await.unwrap();
        assert_eq!(r2.len(), 0);
    }

    #[tokio::test]
    async fn search_case_insensitive() {
        let graph = FlatGraph::new();
        let filter = test_filter("u");

        let entities = vec![make_entity("Alice", "Person", None)];
        let rels = vec![make_relation("Alice", "likes", "Coffee")];

        graph
            .add_relations(&rels, &entities, &filter)
            .await
            .unwrap();

        assert_eq!(graph.search("alice", &filter, 10).await.unwrap().len(), 1);
        assert_eq!(graph.search("COFFEE", &filter, 10).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn search_limit() {
        let graph = FlatGraph::new();
        let filter = test_filter("u");

        let entities = vec![make_entity("A", "Other", None)];

        for i in 0..10 {
            let rels = vec![make_relation("A", &format!("rel_{i}"), &format!("B{i}"))];
            graph
                .add_relations(&rels, &entities, &filter)
                .await
                .unwrap();
        }

        let r = graph.search("A", &filter, 3).await.unwrap();
        assert_eq!(r.len(), 3);
    }

    #[tokio::test]
    async fn delete_all_by_user() {
        let graph = FlatGraph::new();
        let filter1 = test_filter("user1");
        let filter2 = test_filter("user2");

        let entities = vec![make_entity("X", "Other", None)];
        let rels = vec![make_relation("X", "r", "Y")];

        graph
            .add_relations(&rels, &entities, &filter1)
            .await
            .unwrap();
        graph
            .add_relations(&rels, &entities, &filter2)
            .await
            .unwrap();

        graph.delete_all(&filter1).await.unwrap();

        let empty_filter = MemoryFilter::default();
        let r = graph.search("X", &empty_filter, 10).await.unwrap();
        assert_eq!(r.len(), 1);
    }

    #[tokio::test]
    async fn delete_all_respects_agent_and_run_scope() {
        let graph = FlatGraph::new();
        let scoped_a = scoped_filter("user1", "agent-a", "run-a");
        let scoped_b = scoped_filter("user1", "agent-b", "run-b");

        let entities = vec![make_entity("Shared", "Other", None)];
        let rels_a = vec![make_relation("Shared", "likes", "Rust")];
        let rels_b = vec![make_relation("Shared", "likes", "Python")];

        graph
            .add_relations(&rels_a, &entities, &scoped_a)
            .await
            .unwrap();
        graph
            .add_relations(&rels_b, &entities, &scoped_b)
            .await
            .unwrap();

        graph.delete_all(&scoped_a).await.unwrap();

        let remaining_a = graph.search("Shared", &scoped_a, 10).await.unwrap();
        let remaining_b = graph.search("Shared", &scoped_b, 10).await.unwrap();
        assert!(remaining_a.is_empty());
        assert_eq!(remaining_b.len(), 1);
        assert_eq!(remaining_b[0].destination, "Python");
    }

    #[tokio::test]
    async fn reset_clears_all() {
        let graph = FlatGraph::new();
        let filter = test_filter("u");

        let entities = vec![make_entity("X", "Other", None)];
        let rels = vec![make_relation("X", "r", "Y")];

        graph
            .add_relations(&rels, &entities, &filter)
            .await
            .unwrap();

        graph.reset().await.unwrap();

        let empty_filter = MemoryFilter::default();
        assert!(
            graph
                .search("X", &empty_filter, 10)
                .await
                .unwrap()
                .is_empty()
        );
        assert!(graph.entities.read().unwrap().is_empty());
    }
}
