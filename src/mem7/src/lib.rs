//! # mem7
//!
//! LLM-powered long-term memory engine with a Rust core and Python bindings.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use mem7::{MemoryEngine, MemoryEngineConfig, ChatMessage, MemoryFilter};
//!
//! #[tokio::main]
//! async fn main() -> mem7::Result<()> {
//!     let config = MemoryEngineConfig::default();
//!     let engine = MemoryEngine::new(config).await?;
//!
//!     let messages = vec![
//!         ChatMessage { role: "user".into(), content: "I love Rust".into() },
//!     ];
//!     let result = engine.add(&messages, None, None, None, None, true).await?;
//!     println!("{result:?}");
//!
//!     Ok(())
//! }
//! ```

// ── Core types ───────────────────────────────────────────────────────
pub use mem7_core::{
    AddResult, ChatMessage, Fact, GraphRelation, MemoryAction, MemoryActionResult, MemoryEvent,
    MemoryFilter, MemoryItem, SearchResult, new_memory_id,
};

// ── Configuration ────────────────────────────────────────────────────
pub use mem7_config::{
    EmbeddingConfig, GraphConfig, HistoryConfig, LlmConfig, MemoryEngineConfig, RerankerConfig,
    TelemetryConfig, VectorConfig,
};

// ── Error handling ───────────────────────────────────────────────────
pub use mem7_error::{Mem7Error, Result};

// ── Engine ───────────────────────────────────────────────────────────
pub use mem7_store::MemoryEngine;

// ── Provider traits & built-in implementations ───────────────────────

/// LLM client trait and the built-in OpenAI-compatible implementation.
pub mod llm {
    pub use mem7_llm::{LlmClient, LlmMessage, LlmResponse, OpenAICompatibleLlm, ResponseFormat};
}

/// Embedding client trait and the built-in OpenAI-compatible implementation.
pub mod embedding {
    pub use mem7_embedding::{EmbeddingClient, OpenAICompatibleEmbedding};
}

/// Vector index trait and built-in implementations (in-process flat index, Upstash).
pub mod vector {
    pub use mem7_vector::{
        DistanceMetric, FlatIndex, UpstashVectorIndex, VectorIndex, VectorSearchResult,
    };
}

/// SQLite-backed audit history.
pub mod history {
    pub use mem7_history::SqliteHistory;
}

/// Reranker trait and built-in implementations (Cohere, LLM-based).
pub mod reranker {
    pub use mem7_reranker::{
        CohereReranker, LlmReranker, RerankDocument, RerankResult, RerankerClient,
    };
}

/// Graph store trait and built-in implementations (FlatGraph, Kuzu, Neo4j).
pub mod graph {
    #[cfg(feature = "kuzu")]
    pub use mem7_graph::KuzuGraphStore;
    pub use mem7_graph::{
        FlatGraph, GraphStore, Neo4jGraphStore,
        types::{Entity, GraphSearchResult, Relation},
    };
}

/// Memory deduplication utilities.
pub mod dedup {
    pub use mem7_dedup::*;
}

/// OpenTelemetry integration (requires `otel` feature).
#[cfg(feature = "otel")]
pub mod telemetry {
    pub use mem7_telemetry::{init, shutdown};
}
