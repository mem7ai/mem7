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
    pub custom_fact_extraction_prompt: Option<String>,
    pub custom_update_memory_prompt: Option<String>,
}
