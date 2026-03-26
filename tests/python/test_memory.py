"""Integration tests for the mem7 Memory interface.

These tests require OPENAI_API_KEY to be set for the LLM/embedding calls.
For CI, use a mock server or set a vLLM endpoint.
"""

import os

import pytest

from mem7 import Memory
from mem7.config import GraphConfig, HistoryConfig, MemoryConfig
from mem7.memory import _event_to_dict, _memory_item_to_dict, _require_scope


@pytest.fixture
def memory():
    """Create a Memory instance with in-memory history DB."""
    config = MemoryConfig(
        history=HistoryConfig(db_path=":memory:"),
    )
    return Memory(config=config)


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestMemoryIntegration:
    def test_add_and_search(self, memory):
        result = memory.add("My name is Alice and I love pizza.", user_id="alice")
        assert "results" in result
        assert len(result["results"]) > 0

        search = memory.search("What food does Alice like?", user_id="alice")
        assert "results" in search
        assert len(search["results"]) > 0
        assert search["memories"] == search["results"]

    def test_add_and_get_all(self, memory):
        memory.add("I work at Google as a software engineer.", user_id="bob")
        items = memory.get_all(user_id="bob")
        assert "results" in items
        assert len(items["results"]) > 0
        assert any(
            "Google" in item["memory"] or "software" in item["memory"].lower()
            for item in items["results"]
        )

    def test_update_memory(self, memory):
        result = memory.add("I prefer Python for coding.", user_id="carol")
        assert len(result["results"]) > 0

        memory_id = result["results"][0]["id"]
        response = memory.update(memory_id, "I prefer Rust for coding.")
        assert response == {"message": "Memory updated successfully!"}
        item = memory.get(memory_id)
        assert "Rust" in item["memory"]

    def test_delete_memory(self, memory):
        result = memory.add("I have a cat named Whiskers.", user_id="dave")
        assert len(result["results"]) > 0

        memory_id = result["results"][0]["id"]
        response = memory.delete(memory_id)
        assert response == {"message": "Memory deleted successfully!"}
        assert memory.get(memory_id) is None

    def test_history(self, memory):
        result = memory.add("I like hiking.", user_id="eve")
        assert len(result["results"]) > 0

        memory_id = result["results"][0]["id"]
        events = memory.history(memory_id)
        assert len(events) > 0
        assert events[0]["event"] == "ADD"
        assert events[0]["new_memory"] is not None

    def test_reset(self, memory):
        memory.add("Some fact.", user_id="frank")
        memory.reset()
        items = memory.get_all(user_id="frank")
        assert items == {"results": [], "memories": []}


class TestMemoryUnit:
    """Unit tests that don't need an API key (test config/init only)."""

    def test_config_defaults(self):
        config = MemoryConfig()
        assert config.llm.model == "gpt-4.1-nano"
        assert config.embedding.model == "text-embedding-3-small"
        assert config.vector.collection_name == "mem7_memories"

    def test_config_to_json(self):
        config = MemoryConfig()
        json_str = config.to_json()
        assert "gpt-4.1-nano" in json_str
        assert "text-embedding-3-small" in json_str

    def test_graph_config_serializes(self):
        config = MemoryConfig(
            graph=GraphConfig(provider="flat"),
            history=HistoryConfig(db_path=":memory:"),
        )
        json_str = config.to_json()
        assert '"graph"' in json_str
        assert '"provider":"flat"' in json_str

    def test_mem0_style_config_maps_to_mem7_fields(self):
        config = MemoryConfig(
            llm={
                "provider": "groq",
                "config": {
                    "model": "gpt-4.1-mini",
                    "temperature": 0.2,
                },
            },
            embedder={
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-large",
                    "embedding_dims": 3072,
                },
            },
            vector_store={
                "provider": "memory",
                "config": {
                    "collection_name": "mem0_memories",
                },
            },
            graph_store={
                "provider": "neo4j",
                "config": {
                    "url": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": "password",
                },
            },
            history_db_path="compat.db",
        )
        assert config.llm.model == "gpt-4.1-mini"
        assert config.llm.provider == "openai"
        assert config.embedding.model == "text-embedding-3-large"
        assert config.embedding.dims == 3072
        assert config.vector.provider == "flat"
        assert config.vector.collection_name == "mem0_memories"
        assert config.graph is not None
        assert config.graph.provider == "neo4j"
        assert config.graph.neo4j_url == "bolt://localhost:7687"
        assert config.history.db_path == "compat.db"

    def test_provider_aliases_normalize_to_supported_backends(self):
        config = MemoryConfig(
            llm={"provider": "azure_openai"},
            embedding={"provider": "together"},
            vector={"provider": "memory"},
            graph={"provider": "memory"},
            reranker={"provider": "cohere", "api_key": "test"},
        )
        assert config.llm.provider == "openai"
        assert config.embedding.provider == "openai"
        assert config.vector.provider == "flat"
        assert config.graph is not None
        assert config.graph.provider == "flat"

    def test_unsupported_vector_provider_fails_fast(self):
        with pytest.raises(ValueError, match="Unsupported vector store provider 'qdrant'"):
            MemoryConfig(vector={"provider": "qdrant"})

    def test_require_scope_accepts_scope_from_filters(self):
        user_id, agent_id, run_id, filters = _require_scope(filters={"user_id": "u1", "tag": "vip"})
        assert user_id == "u1"
        assert agent_id is None
        assert run_id is None
        assert filters == {"tag": "vip"}

    def test_require_scope_allows_empty_scope(self):
        user_id, agent_id, run_id, filters = _require_scope()
        assert user_id is None
        assert agent_id is None
        assert run_id is None
        assert filters is None

    def test_memory_item_compat_shape(self):
        class DummyItem:
            id = "mem-1"
            text = "likes pizza"
            hash = "0c3e8262220d84c7754ce7db13c7ce95"
            user_id = "u1"
            agent_id = None
            run_id = None
            actor_id = "actor-1"
            role = None
            metadata = '{"topic": "food"}'
            created_at = "2026-01-01T00:00:00Z"
            updated_at = "2026-01-01T00:00:00Z"
            score = 0.9
            last_accessed_at = "2026-01-01T00:00:00Z"
            access_count = 2
            memory_type = "preference"

        result = _memory_item_to_dict(DummyItem())
        assert result["memory"] == "likes pizza"
        assert result["text"] == "likes pizza"
        assert result["user_id"] == "u1"
        assert result["actor_id"] == "actor-1"
        assert result["metadata"] == {"topic": "food"}
        assert result["memory_type"] == "preference"
        assert result["hash"] == "0c3e8262220d84c7754ce7db13c7ce95"

    def test_history_event_compat_shape(self):
        class DummyEvent:
            id = "evt-1"
            memory_id = "mem-1"
            old_value = None
            new_value = "likes pizza"
            action = "ADD"
            created_at = "2026-01-01T00:00:00Z"
            updated_at = None
            is_deleted = False
            actor_id = "actor-1"
            role = "assistant"

        result = _event_to_dict(DummyEvent())
        assert result["event"] == "ADD"
        assert result["new_memory"] == "likes pizza"
        assert result["old_memory"] is None
        assert result["is_deleted"] is False
        assert result["actor_id"] == "actor-1"
        assert result["role"] == "assistant"
