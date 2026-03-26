"""Integration tests for the mem7 Memory interface.

These tests require OPENAI_API_KEY to be set for the LLM/embedding calls.
For CI, use a mock server or set a vLLM endpoint.
"""

import os

import pytest

from mem7 import Memory
from mem7.config import GraphConfig, HistoryConfig, MemoryConfig


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
        assert "memories" in search
        assert len(search["memories"]) > 0

    def test_add_and_get_all(self, memory):
        memory.add("I work at Google as a software engineer.", user_id="bob")
        items = memory.get_all(user_id="bob")
        assert len(items) > 0
        assert any("Google" in item["text"] or "software" in item["text"].lower() for item in items)

    def test_update_memory(self, memory):
        result = memory.add("I prefer Python for coding.", user_id="carol")
        assert len(result["results"]) > 0

        memory_id = result["results"][0]["id"]
        memory.update(memory_id, "I prefer Rust for coding.")
        item = memory.get(memory_id)
        assert "Rust" in item["text"]

    def test_delete_memory(self, memory):
        result = memory.add("I have a cat named Whiskers.", user_id="dave")
        assert len(result["results"]) > 0

        memory_id = result["results"][0]["id"]
        memory.delete(memory_id)

        with pytest.raises(Exception):
            memory.get(memory_id)

    def test_history(self, memory):
        result = memory.add("I like hiking.", user_id="eve")
        assert len(result["results"]) > 0

        memory_id = result["results"][0]["id"]
        events = memory.history(memory_id)
        assert len(events) > 0
        assert events[0]["action"] == "ADD"

    def test_reset(self, memory):
        memory.add("Some fact.", user_id="frank")
        memory.reset()
        items = memory.get_all(user_id="frank")
        assert len(items) == 0


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
