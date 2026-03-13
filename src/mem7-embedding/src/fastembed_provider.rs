use async_trait::async_trait;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use mem7_error::{Mem7Error, Result};
use std::sync::{Arc, Mutex};
use tracing::debug;

use crate::EmbeddingClient;

pub struct FastEmbedClient {
    model: Arc<Mutex<TextEmbedding>>,
}

impl FastEmbedClient {
    pub fn new(model_name: &str, cache_dir: Option<&str>) -> Result<Self> {
        let embedding_model = parse_model(model_name)?;

        let mut opts = InitOptions::new(embedding_model).with_show_download_progress(true);
        if let Some(dir) = cache_dir {
            opts = opts.with_cache_dir(dir.into());
        }

        let model = TextEmbedding::try_new(opts)
            .map_err(|e| Mem7Error::Embedding(format!("fastembed init: {e}")))?;

        debug!(model = model_name, "FastEmbed model loaded");

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
        })
    }
}

#[async_trait]
impl EmbeddingClient for FastEmbedClient {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let model = self.model.clone();
        let docs: Vec<String> = texts.to_vec();

        tokio::task::spawn_blocking(move || {
            let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
            let mut guard = model
                .lock()
                .map_err(|e| Mem7Error::Embedding(format!("fastembed lock: {e}")))?;
            guard
                .embed(refs, None)
                .map_err(|e| Mem7Error::Embedding(format!("fastembed embed: {e}")))
        })
        .await
        .map_err(|e| Mem7Error::Embedding(format!("fastembed task join: {e}")))?
    }
}

fn parse_model(name: &str) -> Result<EmbeddingModel> {
    let model = match name {
        "AllMiniLML6V2" | "all-MiniLM-L6-v2" => EmbeddingModel::AllMiniLML6V2,
        "AllMiniLML6V2Q" | "all-MiniLM-L6-v2-q" => EmbeddingModel::AllMiniLML6V2Q,
        "AllMiniLML12V2" | "all-MiniLM-L12-v2" => EmbeddingModel::AllMiniLML12V2,
        "AllMiniLML12V2Q" | "all-MiniLM-L12-v2-q" => EmbeddingModel::AllMiniLML12V2Q,
        "BGEBaseENV15" | "BAAI/bge-base-en-v1.5" => EmbeddingModel::BGEBaseENV15,
        "BGEBaseENV15Q" | "BAAI/bge-base-en-v1.5-q" => EmbeddingModel::BGEBaseENV15Q,
        "BGESmallENV15" | "BAAI/bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
        "BGESmallENV15Q" | "BAAI/bge-small-en-v1.5-q" => EmbeddingModel::BGESmallENV15Q,
        "BGELargeENV15" | "BAAI/bge-large-en-v1.5" => EmbeddingModel::BGELargeENV15,
        "BGELargeENV15Q" | "BAAI/bge-large-en-v1.5-q" => EmbeddingModel::BGELargeENV15Q,
        "NomicEmbedTextV1" | "nomic-embed-text-v1" => EmbeddingModel::NomicEmbedTextV1,
        "NomicEmbedTextV15" | "nomic-embed-text-v1.5" => EmbeddingModel::NomicEmbedTextV15,
        "NomicEmbedTextV15Q" | "nomic-embed-text-v1.5-q" => EmbeddingModel::NomicEmbedTextV15Q,
        "MultilingualE5Small" | "multilingual-e5-small" => EmbeddingModel::MultilingualE5Small,
        "MultilingualE5Base" | "multilingual-e5-base" => EmbeddingModel::MultilingualE5Base,
        "MultilingualE5Large" | "multilingual-e5-large" => EmbeddingModel::MultilingualE5Large,
        "MxbaiEmbedLargeV1" | "mxbai-embed-large-v1" => EmbeddingModel::MxbaiEmbedLargeV1,
        "MxbaiEmbedLargeV1Q" | "mxbai-embed-large-v1-q" => EmbeddingModel::MxbaiEmbedLargeV1Q,
        "GTEBaseENV15" | "gte-base-en-v1.5" => EmbeddingModel::GTEBaseENV15,
        "GTEBaseENV15Q" | "gte-base-en-v1.5-q" => EmbeddingModel::GTEBaseENV15Q,
        "GTELargeENV15" | "gte-large-en-v1.5" => EmbeddingModel::GTELargeENV15,
        "GTELargeENV15Q" | "gte-large-en-v1.5-q" => EmbeddingModel::GTELargeENV15Q,
        _ => {
            return Err(Mem7Error::Config(format!(
                "unknown fastembed model: {name}. Use TextEmbedding::list_supported_models() for available models."
            )));
        }
    };
    Ok(model)
}
