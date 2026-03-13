use std::sync::RwLock;

use async_trait::async_trait;
use mem7_core::MemoryFilter;
use mem7_error::Result;

use crate::GraphStore;
use crate::types::{Entity, GraphSearchResult, Relation};

#[derive(Debug, Clone)]
struct StoredRelation {
    source: String,
    relationship: String,
    destination: String,
    user_id: Option<String>,
    agent_id: Option<String>,
    run_id: Option<String>,
}

/// In-memory graph store for development and testing.
pub struct FlatGraph {
    relations: RwLock<Vec<StoredRelation>>,
}

impl FlatGraph {
    pub fn new() -> Self {
        Self {
            relations: RwLock::new(Vec::new()),
        }
    }
}

impl Default for FlatGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl GraphStore for FlatGraph {
    async fn add_entities(&self, _entities: &[Entity], _filter: &MemoryFilter) -> Result<()> {
        Ok(())
    }

    async fn add_relations(
        &self,
        relations: &[Relation],
        _entities: &[Entity],
        filter: &MemoryFilter,
    ) -> Result<()> {
        let mut store = self.relations.write().unwrap();
        for r in relations {
            store.push(StoredRelation {
                source: r.source.clone(),
                relationship: r.relationship.clone(),
                destination: r.destination.clone(),
                user_id: filter.user_id.clone(),
                agent_id: filter.agent_id.clone(),
                run_id: filter.run_id.clone(),
            });
        }
        Ok(())
    }

    async fn search(
        &self,
        query: &str,
        filter: &MemoryFilter,
        limit: usize,
    ) -> Result<Vec<GraphSearchResult>> {
        let store = self.relations.read().unwrap();
        let query_lower = query.to_lowercase();

        let results: Vec<GraphSearchResult> = store
            .iter()
            .filter(|r| {
                if let Some(uid) = &filter.user_id {
                    if r.user_id.as_deref() != Some(uid.as_str()) {
                        return false;
                    }
                }
                if let Some(aid) = &filter.agent_id {
                    if r.agent_id.as_deref() != Some(aid.as_str()) {
                        return false;
                    }
                }
                if let Some(rid) = &filter.run_id {
                    if r.run_id.as_deref() != Some(rid.as_str()) {
                        return false;
                    }
                }

                r.source.to_lowercase().contains(&query_lower)
                    || r.destination.to_lowercase().contains(&query_lower)
                    || r.relationship.to_lowercase().contains(&query_lower)
            })
            .take(limit)
            .map(|r| GraphSearchResult {
                source: r.source.clone(),
                relationship: r.relationship.clone(),
                destination: r.destination.clone(),
            })
            .collect();

        Ok(results)
    }

    async fn delete_all(&self, filter: &MemoryFilter) -> Result<()> {
        let mut store = self.relations.write().unwrap();
        store.retain(|r| {
            if let Some(uid) = &filter.user_id {
                if r.user_id.as_deref() == Some(uid.as_str()) {
                    return false;
                }
            }
            true
        });
        Ok(())
    }

    async fn reset(&self) -> Result<()> {
        let mut store = self.relations.write().unwrap();
        store.clear();
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

    #[tokio::test]
    async fn add_and_search_relations() {
        let graph = FlatGraph::new();
        let filter = test_filter("user1");

        let entities = vec![
            Entity {
                name: "Alice".into(),
                entity_type: "Person".into(),
            },
            Entity {
                name: "tennis".into(),
                entity_type: "Activity".into(),
            },
        ];

        let relations = vec![Relation {
            source: "Alice".into(),
            relationship: "loves_playing".into(),
            destination: "tennis".into(),
        }];

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
    async fn search_by_relationship() {
        let graph = FlatGraph::new();
        let filter = test_filter("user1");

        let entities = vec![
            Entity {
                name: "Bob".into(),
                entity_type: "Person".into(),
            },
            Entity {
                name: "Google".into(),
                entity_type: "Organization".into(),
            },
        ];

        let relations = vec![Relation {
            source: "Bob".into(),
            relationship: "works_at".into(),
            destination: "Google".into(),
        }];

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

        let entities = vec![Entity {
            name: "X".into(),
            entity_type: "Other".into(),
        }];

        let rels = vec![Relation {
            source: "X".into(),
            relationship: "rel".into(),
            destination: "Y".into(),
        }];

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

        let entities = vec![Entity {
            name: "Alice".into(),
            entity_type: "Person".into(),
        }];

        let rels = vec![Relation {
            source: "Alice".into(),
            relationship: "likes".into(),
            destination: "Coffee".into(),
        }];

        graph
            .add_relations(&rels, &entities, &filter)
            .await
            .unwrap();

        let r = graph.search("alice", &filter, 10).await.unwrap();
        assert_eq!(r.len(), 1);

        let r = graph.search("COFFEE", &filter, 10).await.unwrap();
        assert_eq!(r.len(), 1);
    }

    #[tokio::test]
    async fn search_limit() {
        let graph = FlatGraph::new();
        let filter = test_filter("u");

        let entities = vec![Entity {
            name: "A".into(),
            entity_type: "Other".into(),
        }];

        for i in 0..10 {
            let rels = vec![Relation {
                source: "A".into(),
                relationship: format!("rel_{i}"),
                destination: format!("B{i}"),
            }];
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

        let entities = vec![Entity {
            name: "X".into(),
            entity_type: "Other".into(),
        }];

        let rels = vec![Relation {
            source: "X".into(),
            relationship: "r".into(),
            destination: "Y".into(),
        }];

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
    async fn reset_clears_all() {
        let graph = FlatGraph::new();
        let filter = test_filter("u");

        let entities = vec![Entity {
            name: "X".into(),
            entity_type: "Other".into(),
        }];

        let rels = vec![Relation {
            source: "X".into(),
            relationship: "r".into(),
            destination: "Y".into(),
        }];

        graph
            .add_relations(&rels, &entities, &filter)
            .await
            .unwrap();

        graph.reset().await.unwrap();

        let empty_filter = MemoryFilter::default();
        let r = graph.search("X", &empty_filter, 10).await.unwrap();
        assert_eq!(r.len(), 0);
    }
}
