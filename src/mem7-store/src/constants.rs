/// Maximum number of existing memories to retrieve per fact during deduplication.
pub const DEDUP_CANDIDATE_LIMIT: usize = 5;

/// Minimum cosine similarity threshold for graph embedding search.
pub const GRAPH_SEARCH_THRESHOLD: f32 = 0.5;

/// Maximum graph triples to scan during conflict detection.
pub const GRAPH_CONFLICT_LIMIT: usize = 50;

/// Over-fetch multiplier for graph search results before BM25 reranking.
pub const GRAPH_SEARCH_MULTIPLIER: usize = 3;

/// Default reranker top-k multiplier when not specified in config.
pub const DEFAULT_RERANK_MULTIPLIER: usize = 3;
