use thiserror::Error;

#[derive(Debug, Error)]
pub enum Mem7Error {
    #[error("LLM error: {0}")]
    Llm(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Vector store error: {0}")]
    VectorStore(String),

    #[error("History DB error: {0}")]
    History(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("HTTP error: {source}")]
    Http {
        #[from]
        source: reqwest::Error,
    },

    #[error("JSON error: {source}")]
    Json {
        #[from]
        source: serde_json::Error,
    },

    #[error("SQLite error: {0}")]
    Sqlite(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("{0}")]
    Other(String),
}

impl From<tokio_rusqlite::Error> for Mem7Error {
    fn from(e: tokio_rusqlite::Error) -> Self {
        Mem7Error::Sqlite(e.to_string())
    }
}

impl From<rusqlite::Error> for Mem7Error {
    fn from(e: rusqlite::Error) -> Self {
        Mem7Error::Sqlite(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Mem7Error>;
