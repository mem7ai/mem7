"""Configuration for the mem7 memory engine."""

import os
from typing import Optional

from pydantic import BaseModel, Field


class LlmConfig(BaseModel):
    provider: str = "openai"
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model: str = "gpt-4.1-nano"
    temperature: float = 0.0
    max_tokens: int = 1000


class EmbeddingConfig(BaseModel):
    provider: str = "openai"
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model: str = "text-embedding-3-small"
    dims: int = 1536


class VectorConfig(BaseModel):
    provider: str = "flat"
    collection_name: str = "mem7_memories"
    dims: int = 1536
    upstash_url: Optional[str] = None
    upstash_token: Optional[str] = None


class HistoryConfig(BaseModel):
    db_path: str = "mem7_history.db"


class DecayConfig(BaseModel):
    """Ebbinghaus-inspired forgetting curve configuration."""

    enabled: bool = False
    base_half_life_secs: float = 604800.0  # 7 days
    decay_shape: float = 0.8
    min_retention: float = 0.1
    rehearsal_factor: float = 0.5


class MemoryConfig(BaseModel):
    llm: LlmConfig = Field(default_factory=LlmConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector: VectorConfig = Field(default_factory=VectorConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    decay: Optional[DecayConfig] = None
    custom_fact_extraction_prompt: Optional[str] = None
    custom_update_memory_prompt: Optional[str] = None

    def to_json(self) -> str:
        return self.model_dump_json()
