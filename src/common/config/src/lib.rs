use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    #[serde(default = "default_llm_provider")]
    pub provider: String,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// When true, image messages are sent to the LLM to produce text
    /// descriptions that are then stored as memory content.
    #[serde(default)]
    pub enable_vision: bool,
}

fn default_llm_provider() -> String {
    "openai".into()
}

fn default_temperature() -> f32 {
    0.0
}

fn default_max_tokens() -> u32 {
    1000
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: default_llm_provider(),
            base_url: "https://api.openai.com/v1".into(),
            api_key: String::new(),
            model: "gpt-4.1-nano".into(),
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            enable_vision: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    #[serde(default = "default_embedding_provider")]
    pub provider: String,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    #[serde(default = "default_embedding_dims")]
    pub dims: usize,
    pub cache_dir: Option<String>,
}

fn default_embedding_provider() -> String {
    "openai".into()
}

fn default_embedding_dims() -> usize {
    1536
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: default_embedding_provider(),
            base_url: "https://api.openai.com/v1".into(),
            api_key: String::new(),
            model: "text-embedding-3-small".into(),
            dims: default_embedding_dims(),
            cache_dir: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorConfig {
    #[serde(default = "default_vector_provider")]
    pub provider: String,
    #[serde(default = "default_collection")]
    pub collection_name: String,
    #[serde(default = "default_embedding_dims")]
    pub dims: usize,
    pub upstash_url: Option<String>,
    pub upstash_token: Option<String>,
}

fn default_vector_provider() -> String {
    "flat".into()
}

fn default_collection() -> String {
    "mem7_memories".into()
}

impl Default for VectorConfig {
    fn default() -> Self {
        Self {
            provider: default_vector_provider(),
            collection_name: default_collection(),
            dims: default_embedding_dims(),
            upstash_url: None,
            upstash_token: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryConfig {
    #[serde(default = "default_history_path")]
    pub db_path: String,
}

fn default_history_path() -> String {
    "mem7_history.db".into()
}

impl Default for HistoryConfig {
    fn default() -> Self {
        Self {
            db_path: default_history_path(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankerConfig {
    pub provider: String,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    #[serde(default = "default_top_k_multiplier")]
    pub top_k_multiplier: usize,
}

fn default_top_k_multiplier() -> usize {
    3
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            provider: "cohere".into(),
            model: None,
            api_key: None,
            base_url: None,
            top_k_multiplier: default_top_k_multiplier(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    #[serde(default = "default_graph_provider")]
    pub provider: String,
    pub kuzu_db_path: Option<String>,
    pub neo4j_url: Option<String>,
    pub neo4j_username: Option<String>,
    pub neo4j_password: Option<String>,
    pub neo4j_database: Option<String>,
    pub custom_prompt: Option<String>,
    pub llm: Option<LlmConfig>,
}

fn default_graph_provider() -> String {
    "flat".into()
}

fn default_kuzu_db_path() -> String {
    "mem7_graph.kuzu".into()
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            provider: default_graph_provider(),
            kuzu_db_path: Some(default_kuzu_db_path()),
            neo4j_url: None,
            neo4j_username: None,
            neo4j_password: None,
            neo4j_database: None,
            custom_prompt: None,
            llm: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    #[serde(default = "default_otlp_endpoint")]
    pub otlp_endpoint: String,
    #[serde(default = "default_service_name")]
    pub service_name: String,
}

fn default_otlp_endpoint() -> String {
    "http://localhost:4317".into()
}

fn default_service_name() -> String {
    "mem7".into()
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            otlp_endpoint: default_otlp_endpoint(),
            service_name: default_service_name(),
        }
    }
}

/// Configuration for the Ebbinghaus-inspired memory decay / forgetting curve.
///
/// When enabled, older memories are deprioritized during search and dedup via a
/// stretched-exponential retention factor. Memories that are accessed frequently
/// decay more slowly (spaced-repetition effect).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayConfig {
    /// Master switch. When `false` (the default), all scoring is unmodified.
    #[serde(default)]
    pub enabled: bool,
    /// Base half-life in seconds before any rehearsal bonus. Default: 7 days.
    #[serde(default = "default_base_half_life")]
    pub base_half_life_secs: f64,
    /// Stretched-exponential shape parameter (0 < gamma <= 1).
    /// Lower values produce slower initial decay with a steeper tail.
    #[serde(default = "default_decay_shape")]
    pub decay_shape: f64,
    /// Minimum retention floor so no memory ever fully vanishes.
    #[serde(default = "default_min_retention")]
    pub min_retention: f64,
    /// How much each access (rehearsal) increases memory stability.
    #[serde(default = "default_rehearsal_factor")]
    pub rehearsal_factor: f64,
}

fn default_base_half_life() -> f64 {
    604800.0 // 7 days
}

fn default_decay_shape() -> f64 {
    0.8
}

fn default_min_retention() -> f64 {
    0.1
}

fn default_rehearsal_factor() -> f64 {
    0.5
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            base_half_life_secs: default_base_half_life(),
            decay_shape: default_decay_shape(),
            min_retention: default_min_retention(),
            rehearsal_factor: default_rehearsal_factor(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryEngineConfig {
    #[serde(default)]
    pub llm: LlmConfig,
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub vector: VectorConfig,
    #[serde(default)]
    pub history: HistoryConfig,
    pub reranker: Option<RerankerConfig>,
    pub graph: Option<GraphConfig>,
    pub telemetry: Option<TelemetryConfig>,
    pub decay: Option<DecayConfig>,
    pub custom_fact_extraction_prompt: Option<String>,
    pub custom_update_memory_prompt: Option<String>,
}
