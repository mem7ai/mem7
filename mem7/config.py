"""Configuration for the mem7 memory engine."""

import os
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

OPENAI_COMPATIBLE_PROVIDER_ALIASES = {
    "openai": "openai",
    "openai-compatible": "openai",
    "openai_compatible": "openai",
    "ollama": "ollama",
    "vllm": "vllm",
    "lmstudio": "lmstudio",
    "lm-studio": "lmstudio",
    "deepseek": "deepseek",
    "groq": "openai",
    "together": "openai",
    "xai": "openai",
    "grok": "openai",
    "azure_openai": "openai",
    "azure-openai": "openai",
}

EMBEDDING_PROVIDER_ALIASES = {
    **OPENAI_COMPATIBLE_PROVIDER_ALIASES,
    "fastembed": "fastembed",
}

VECTOR_PROVIDER_ALIASES = {
    "flat": "flat",
    "memory": "flat",
    "in-memory": "flat",
    "upstash": "upstash",
}

GRAPH_PROVIDER_ALIASES = {
    "flat": "flat",
    "memory": "flat",
    "neo4j": "neo4j",
    "kuzu": "kuzu",
}

RERANKER_PROVIDER_ALIASES = {
    "cohere": "cohere",
    "llm": "llm",
}


def _normalize_provider_name(provider: Any) -> str:
    return str(provider or "").strip().lower().replace(" ", "-")


def _coerce_supported_provider(
    kind: str,
    provider: Any,
    aliases: dict[str, str],
) -> str:
    normalized = _normalize_provider_name(provider)
    mapped = aliases.get(normalized)
    if mapped is not None:
        return mapped

    supported = ", ".join(sorted(set(aliases.values())))
    raise ValueError(
        f"Unsupported {kind} provider '{provider}'. mem7 currently supports: {supported}."
    )


def _maybe_provider_config(config: Any) -> tuple[dict[str, Any], bool]:
    if isinstance(config, dict) and "config" in config:
        return config, True
    return config or {}, False


def _coerce_llm_like_config(config: Any) -> dict[str, Any]:
    raw, provider_style = _maybe_provider_config(config)
    if not isinstance(raw, dict):
        return raw
    if not provider_style:
        return raw

    cfg = raw.get("config") or {}
    return {
        "provider": raw.get("provider", "openai"),
        "base_url": cfg.get("base_url") or cfg.get("baseURL") or "https://api.openai.com/v1",
        "api_key": cfg.get("api_key") or cfg.get("apiKey") or os.environ.get("OPENAI_API_KEY", ""),
        "model": cfg.get("model") or "gpt-4.1-nano",
        "temperature": cfg.get("temperature", 0.0),
        "max_tokens": cfg.get("max_tokens", 1000),
    }


def _coerce_embedding_config(config: Any) -> dict[str, Any]:
    raw, provider_style = _maybe_provider_config(config)
    if not isinstance(raw, dict):
        return raw
    if not provider_style:
        return raw

    cfg = raw.get("config") or {}
    dims = cfg.get("dims")
    if dims is None:
        dims = cfg.get("embedding_dims")
    if dims is None:
        dims = cfg.get("embedding_model_dims")
    if dims is None:
        dims = 1536

    return {
        "provider": raw.get("provider", "openai"),
        "base_url": cfg.get("base_url") or cfg.get("baseURL") or "https://api.openai.com/v1",
        "api_key": cfg.get("api_key") or cfg.get("apiKey") or os.environ.get("OPENAI_API_KEY", ""),
        "model": cfg.get("model") or "text-embedding-3-small",
        "dims": dims,
        "cache_dir": cfg.get("cache_dir") or cfg.get("cacheDir"),
    }


def _coerce_vector_config(config: Any) -> dict[str, Any]:
    raw, provider_style = _maybe_provider_config(config)
    if not isinstance(raw, dict):
        return raw
    if not provider_style:
        return raw

    provider = raw.get("provider", "flat")
    if provider == "memory":
        provider = "flat"

    cfg = raw.get("config") or {}
    dims = cfg.get("dims")
    if dims is None:
        dims = cfg.get("dimension")
    if dims is None:
        dims = cfg.get("embedding_model_dims")
    if dims is None:
        dims = 1536

    return {
        "provider": provider,
        "collection_name": cfg.get("collection_name")
        or cfg.get("collectionName")
        or "mem7_memories",
        "dims": dims,
        "upstash_url": cfg.get("upstash_url") or cfg.get("upstashUrl"),
        "upstash_token": cfg.get("upstash_token") or cfg.get("upstashToken"),
    }


def _coerce_graph_config(config: Any) -> dict[str, Any]:
    raw, provider_style = _maybe_provider_config(config)
    if not isinstance(raw, dict):
        return raw
    if not provider_style:
        return raw

    cfg = raw.get("config") or {}
    llm = raw.get("llm")
    if llm is None and "llm" in cfg:
        llm = cfg["llm"]

    return {
        "provider": raw.get("provider", "flat"),
        "kuzu_db_path": cfg.get("kuzu_db_path") or cfg.get("kuzuDbPath"),
        "neo4j_url": cfg.get("neo4j_url") or cfg.get("url"),
        "neo4j_username": cfg.get("neo4j_username") or cfg.get("username"),
        "neo4j_password": cfg.get("neo4j_password") or cfg.get("password"),
        "neo4j_database": cfg.get("neo4j_database") or cfg.get("database"),
        "custom_prompt": raw.get("custom_prompt") or cfg.get("custom_prompt"),
        "llm": _coerce_llm_like_config(llm) if llm else None,
    }


class LlmConfig(BaseModel):
    provider: str = "openai"
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model: str = "gpt-4.1-nano"
    temperature: float = 0.0
    max_tokens: int = 1000
    enable_vision: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalize_provider(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        adapted = dict(data)
        adapted["provider"] = _coerce_supported_provider(
            "LLM",
            adapted.get("provider", "openai"),
            OPENAI_COMPATIBLE_PROVIDER_ALIASES,
        )
        return adapted


class EmbeddingConfig(BaseModel):
    provider: str = "openai"
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model: str = "text-embedding-3-small"
    dims: int = 1536
    cache_dir: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_provider(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        adapted = dict(data)
        adapted["provider"] = _coerce_supported_provider(
            "embedding",
            adapted.get("provider", "openai"),
            EMBEDDING_PROVIDER_ALIASES,
        )
        return adapted


class VectorConfig(BaseModel):
    provider: str = "flat"
    collection_name: str = "mem7_memories"
    dims: int = 1536
    upstash_url: Optional[str] = None
    upstash_token: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_provider(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        adapted = dict(data)
        adapted["provider"] = _coerce_supported_provider(
            "vector store",
            adapted.get("provider", "flat"),
            VECTOR_PROVIDER_ALIASES,
        )
        return adapted


class HistoryConfig(BaseModel):
    db_path: str = "mem7_history.db"


class RerankerConfig(BaseModel):
    provider: str = "cohere"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    top_k_multiplier: int = 3

    @model_validator(mode="before")
    @classmethod
    def _normalize_provider(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        adapted = dict(data)
        adapted["provider"] = _coerce_supported_provider(
            "reranker",
            adapted.get("provider", "cohere"),
            RERANKER_PROVIDER_ALIASES,
        )
        return adapted


class GraphConfig(BaseModel):
    provider: str = "flat"
    kuzu_db_path: Optional[str] = "mem7_graph.kuzu"
    neo4j_url: Optional[str] = None
    neo4j_username: Optional[str] = None
    neo4j_password: Optional[str] = None
    neo4j_database: Optional[str] = None
    custom_prompt: Optional[str] = None
    llm: Optional[LlmConfig] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_provider(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        adapted = dict(data)
        adapted["provider"] = _coerce_supported_provider(
            "graph",
            adapted.get("provider", "flat"),
            GRAPH_PROVIDER_ALIASES,
        )
        return adapted


class TelemetryConfig(BaseModel):
    otlp_endpoint: str = "http://localhost:4317"
    service_name: str = "mem7"


class DecayConfig(BaseModel):
    """Ebbinghaus-inspired forgetting curve configuration."""

    enabled: bool = False
    base_half_life_secs: float = 604800.0  # 7 days
    decay_shape: float = 0.8
    min_retention: float = 0.1
    rehearsal_factor: float = 0.5


class ContextConfig(BaseModel):
    """Context-aware scoring configuration.

    When enabled, each memory's score is multiplied by a coefficient from a
    (memory_type, task_type) weight matrix, demoting contextually irrelevant
    memories.
    """

    enabled: bool = False
    weights: Optional[dict[str, dict[str, float]]] = None


class MemoryConfig(BaseModel):
    llm: LlmConfig = Field(default_factory=LlmConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector: VectorConfig = Field(default_factory=VectorConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    reranker: Optional[RerankerConfig] = None
    graph: Optional[GraphConfig] = None
    telemetry: Optional[TelemetryConfig] = None
    decay: Optional[DecayConfig] = None
    context: Optional[ContextConfig] = None
    custom_fact_extraction_prompt: Optional[str] = None
    custom_update_memory_prompt: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _adapt_mem0_style_config(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        adapted = dict(data)

        if "embedder" in adapted and "embedding" not in adapted:
            adapted["embedding"] = _coerce_embedding_config(adapted.pop("embedder"))

        if "vector_store" in adapted and "vector" not in adapted:
            adapted["vector"] = _coerce_vector_config(adapted.pop("vector_store"))

        if "graph_store" in adapted and "graph" not in adapted:
            adapted["graph"] = _coerce_graph_config(adapted.pop("graph_store"))

        if "history_db_path" in adapted and "history" not in adapted:
            adapted["history"] = {"db_path": adapted.pop("history_db_path")}

        if "llm" in adapted:
            adapted["llm"] = _coerce_llm_like_config(adapted["llm"])

        if "embedding" in adapted:
            adapted["embedding"] = _coerce_embedding_config(adapted["embedding"])

        if "vector" in adapted:
            adapted["vector"] = _coerce_vector_config(adapted["vector"])

        if "graph" in adapted:
            adapted["graph"] = _coerce_graph_config(adapted["graph"])

        return adapted

    def to_json(self) -> str:
        return self.model_dump_json()
