use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use mem7_core::{
    AddOptions, ChatMessage, MemoryAction, MemoryEvent, MemoryFilter, SearchOptions, new_memory_id,
};
use mem7_embedding::EmbeddingClient;
use mem7_error::Result;
use mem7_history::HistoryStore;
use mem7_llm::{LlmClient, LlmMessage, LlmResponse, ResponseFormat};
use mem7_vector::{VectorIndex, VectorSearchResult};
use uuid::Uuid;

// ── Mock LLM ─────────────────────────────────────────────────────────

struct MockLlm {
    response: String,
}

#[async_trait]
impl LlmClient for MockLlm {
    async fn chat_completion(
        &self,
        _messages: &[LlmMessage],
        _response_format: Option<&ResponseFormat>,
    ) -> Result<LlmResponse> {
        Ok(LlmResponse {
            content: self.response.clone(),
        })
    }
}

// ── Mock Embedding ───────────────────────────────────────────────────

struct MockEmbedder {
    dims: usize,
}

#[async_trait]
impl EmbeddingClient for MockEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let mut v = vec![0.0_f32; self.dims];
                if !v.is_empty() {
                    v[i % self.dims] = 1.0;
                }
                v
            })
            .collect())
    }
}

// ── Mock Vector Index ────────────────────────────────────────────────

struct MockVectorIndex {
    data: RwLock<HashMap<Uuid, (Vec<f32>, serde_json::Value)>>,
}

impl MockVectorIndex {
    fn new() -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
        }
    }

    fn matches_filter(payload: &serde_json::Value, filter: &MemoryFilter) -> bool {
        if let Some(uid) = &filter.user_id
            && payload.get("user_id").and_then(|v| v.as_str()) != Some(uid.as_str())
        {
            return false;
        }
        if let Some(aid) = &filter.agent_id
            && payload.get("agent_id").and_then(|v| v.as_str()) != Some(aid.as_str())
        {
            return false;
        }
        if let Some(rid) = &filter.run_id
            && payload.get("run_id").and_then(|v| v.as_str()) != Some(rid.as_str())
        {
            return false;
        }
        true
    }
}

#[async_trait]
impl VectorIndex for MockVectorIndex {
    async fn insert(&self, id: Uuid, vector: &[f32], payload: serde_json::Value) -> Result<()> {
        self.data
            .write()
            .expect("lock poisoned")
            .insert(id, (vector.to_vec(), payload));
        Ok(())
    }

    async fn search(
        &self,
        _query: &[f32],
        limit: usize,
        filters: Option<&MemoryFilter>,
    ) -> Result<Vec<VectorSearchResult>> {
        let data = self.data.read().expect("lock poisoned");
        Ok(data
            .iter()
            .filter(|(_, (_, payload))| {
                filters
                    .map(|filter| Self::matches_filter(payload, filter))
                    .unwrap_or(true)
            })
            .take(limit)
            .map(|(id, (_, payload))| VectorSearchResult {
                id: *id,
                score: 0.9,
                payload: payload.clone(),
            })
            .collect())
    }

    async fn delete(&self, id: &Uuid) -> Result<()> {
        self.data.write().expect("lock poisoned").remove(id);
        Ok(())
    }

    async fn update(
        &self,
        id: &Uuid,
        vector: Option<&[f32]>,
        payload: Option<serde_json::Value>,
    ) -> Result<()> {
        let mut data = self.data.write().expect("lock poisoned");
        if let Some(entry) = data.get_mut(id) {
            if let Some(v) = vector {
                entry.0 = v.to_vec();
            }
            if let Some(p) = payload {
                entry.1 = p;
            }
        }
        Ok(())
    }

    async fn get(&self, id: &Uuid) -> Result<Option<(Vec<f32>, serde_json::Value)>> {
        Ok(self.data.read().expect("lock poisoned").get(id).cloned())
    }

    async fn list(
        &self,
        filters: Option<&MemoryFilter>,
        limit: Option<usize>,
    ) -> Result<Vec<(Uuid, serde_json::Value)>> {
        let data = self.data.read().expect("lock poisoned");
        let iter = data
            .iter()
            .filter(|(_, (_, payload))| {
                filters
                    .map(|filter| Self::matches_filter(payload, filter))
                    .unwrap_or(true)
            })
            .map(|(id, (_, p))| (*id, p.clone()));
        match limit {
            Some(n) => Ok(iter.take(n).collect()),
            None => Ok(iter.collect()),
        }
    }

    async fn reset(&self) -> Result<()> {
        self.data.write().expect("lock poisoned").clear();
        Ok(())
    }
}

// ── Mock History ─────────────────────────────────────────────────────

struct MockHistory {
    events: RwLock<Vec<MemoryEvent>>,
}

impl MockHistory {
    fn new() -> Self {
        Self {
            events: RwLock::new(Vec::new()),
        }
    }
}

#[async_trait]
impl HistoryStore for MockHistory {
    async fn add_event(
        &self,
        memory_id: Uuid,
        old_value: Option<&str>,
        new_value: Option<&str>,
        action: MemoryAction,
    ) -> Result<MemoryEvent> {
        let event = MemoryEvent {
            id: new_memory_id(),
            memory_id,
            old_value: old_value.map(String::from),
            new_value: new_value.map(String::from),
            action,
            created_at: "2025-01-01T00:00:00Z".into(),
        };
        self.events
            .write()
            .expect("lock poisoned")
            .push(event.clone());
        Ok(event)
    }

    async fn get_history(&self, memory_id: Uuid) -> Result<Vec<MemoryEvent>> {
        Ok(self
            .events
            .read()
            .expect("lock poisoned")
            .iter()
            .filter(|e| e.memory_id == memory_id)
            .cloned()
            .collect())
    }

    async fn reset(&self) -> Result<()> {
        self.events.write().expect("lock poisoned").clear();
        Ok(())
    }
}

// ── Helper ───────────────────────────────────────────────────────────

fn default_config() -> mem7_config::MemoryEngineConfig {
    mem7_config::MemoryEngineConfig::default()
}

async fn build_engine() -> mem7_store::MemoryEngine {
    mem7_store::MemoryEngineBuilder::new(default_config())
        .llm(Arc::new(MockLlm {
            response: r#"{"facts": ["user likes Rust"]}"#.into(),
        }))
        .embedder(Arc::new(MockEmbedder { dims: 4 }))
        .vector_index(Arc::new(MockVectorIndex::new()))
        .history(Arc::new(MockHistory::new()))
        .build()
        .await
        .unwrap()
}

// ── Tests ────────────────────────────────────────────────────────────

#[tokio::test]
async fn raw_add_stores_messages() {
    let engine = build_engine().await;
    let messages = vec![
        ChatMessage {
            role: "user".into(),
            content: "Hello world".into(),
            images: vec![],
        },
        ChatMessage {
            role: "assistant".into(),
            content: "Hi there".into(),
            images: vec![],
        },
    ];

    let opts = AddOptions {
        user_id: Some("u1"),
        infer: false,
        ..Default::default()
    };
    let result = engine.add_with_options(&messages, &opts).await.unwrap();
    assert_eq!(result.results.len(), 2);
    assert!(result.results.iter().all(|r| r.action == MemoryAction::Add));
}

#[tokio::test]
async fn raw_add_skips_system_messages() {
    let engine = build_engine().await;
    let messages = vec![
        ChatMessage {
            role: "system".into(),
            content: "You are helpful".into(),
            images: vec![],
        },
        ChatMessage {
            role: "user".into(),
            content: "Hello".into(),
            images: vec![],
        },
    ];

    let result = engine
        .add(&messages, None, None, None, None, false)
        .await
        .unwrap();
    assert_eq!(result.results.len(), 1);
    assert_eq!(result.results[0].new_value.as_deref(), Some("Hello"));
}

#[tokio::test]
async fn get_returns_stored_memory() {
    let engine = build_engine().await;
    let messages = vec![ChatMessage {
        role: "user".into(),
        content: "Remember this".into(),
        images: vec![],
    }];

    let result = engine
        .add(&messages, Some("u1"), None, None, None, false)
        .await
        .unwrap();
    let id = result.results[0].id;

    let item = engine.get(id).await.unwrap();
    assert_eq!(item.text, "Remember this");
    assert_eq!(item.user_id.as_deref(), Some("u1"));
}

#[tokio::test]
async fn update_changes_text() {
    let engine = build_engine().await;
    let messages = vec![ChatMessage {
        role: "user".into(),
        content: "original".into(),
        images: vec![],
    }];

    let result = engine
        .add(&messages, None, None, None, None, false)
        .await
        .unwrap();
    let id = result.results[0].id;

    engine.update(id, "updated text").await.unwrap();

    let item = engine.get(id).await.unwrap();
    assert_eq!(item.text, "updated text");
}

#[tokio::test]
async fn delete_removes_memory() {
    let engine = build_engine().await;
    let messages = vec![ChatMessage {
        role: "user".into(),
        content: "to delete".into(),
        images: vec![],
    }];

    let result = engine
        .add(&messages, None, None, None, None, false)
        .await
        .unwrap();
    let id = result.results[0].id;

    engine.delete(id).await.unwrap();

    let err = engine.get(id).await;
    assert!(err.is_err());
}

#[tokio::test]
async fn search_returns_results() {
    let engine = build_engine().await;
    let messages = vec![ChatMessage {
        role: "user".into(),
        content: "I love Rust".into(),
        images: vec![],
    }];

    engine
        .add(&messages, Some("u1"), None, None, None, false)
        .await
        .unwrap();

    let opts = SearchOptions {
        user_id: Some("u1"),
        limit: 5,
        ..Default::default()
    };
    let result = engine.search_with_options("Rust", &opts).await.unwrap();
    assert!(!result.memories.is_empty());
    assert_eq!(result.memories[0].text, "I love Rust");
}

#[tokio::test]
async fn reset_clears_everything() {
    let engine = build_engine().await;
    let messages = vec![ChatMessage {
        role: "user".into(),
        content: "data".into(),
        images: vec![],
    }];

    engine
        .add(&messages, None, None, None, None, false)
        .await
        .unwrap();

    engine.reset().await.unwrap();

    let all = engine.get_all(None, None, None, None, None).await.unwrap();
    assert!(all.is_empty());
}

#[tokio::test]
async fn history_tracks_operations() {
    let engine = build_engine().await;
    let messages = vec![ChatMessage {
        role: "user".into(),
        content: "track me".into(),
        images: vec![],
    }];

    let result = engine
        .add(&messages, None, None, None, None, false)
        .await
        .unwrap();
    let id = result.results[0].id;

    engine.update(id, "updated").await.unwrap();
    engine.delete(id).await.unwrap();

    let history = engine.history(id).await.unwrap();
    assert_eq!(history.len(), 3);
    assert_eq!(history[0].action, MemoryAction::Add);
    assert_eq!(history[1].action, MemoryAction::Update);
    assert_eq!(history[2].action, MemoryAction::Delete);
}

#[tokio::test]
async fn delete_all_requires_scope() {
    let engine = build_engine().await;
    let err = engine.delete_all(None, None, None).await.unwrap_err();
    assert!(err.to_string().contains("delete_all requires at least one"));
}

#[tokio::test]
async fn delete_all_only_removes_matching_scope_and_writes_history() {
    let engine = build_engine().await;
    let messages = vec![ChatMessage {
        role: "user".into(),
        content: "scoped memory".into(),
        images: vec![],
    }];

    let scoped = engine
        .add(&messages, Some("u1"), Some("a1"), Some("r1"), None, false)
        .await
        .unwrap();
    let retained = engine
        .add(&messages, Some("u1"), Some("a2"), Some("r2"), None, false)
        .await
        .unwrap();

    engine
        .delete_all(Some("u1"), Some("a1"), Some("r1"))
        .await
        .unwrap();

    let deleted_id = scoped.results[0].id;
    assert!(engine.get(deleted_id).await.is_err());
    let remaining = engine.get(retained.results[0].id).await.unwrap();
    assert_eq!(remaining.agent_id.as_deref(), Some("a2"));

    let history = engine.history(deleted_id).await.unwrap();
    assert_eq!(history.len(), 2);
    assert_eq!(history[0].action, MemoryAction::Add);
    assert_eq!(history[1].action, MemoryAction::Delete);
}
