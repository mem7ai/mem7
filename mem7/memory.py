"""Thin Python wrapper around the Rust MemoryEngine."""

from __future__ import annotations

import json as _json
from typing import Any, Dict, Optional, Union

from mem7._mem7 import PyAsyncMemoryEngine, PyMemoryEngine
from mem7.config import MemoryConfig


def _memory_item_to_dict(item) -> dict:
    return {
        "id": item.id,
        "text": item.text,
        "user_id": item.user_id,
        "agent_id": item.agent_id,
        "run_id": item.run_id,
        "metadata": item.metadata,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
        "score": item.score,
    }


def _action_result_to_dict(r) -> dict:
    return {
        "id": r.id,
        "action": r.action,
        "old_value": r.old_value,
        "new_value": r.new_value,
    }


def _add_result_to_dict(result) -> dict:
    return {"results": [_action_result_to_dict(r) for r in result.results]}


def _search_result_to_dict(result) -> dict:
    return {"memories": [_memory_item_to_dict(m) for m in result.memories]}


def _event_to_dict(e) -> dict:
    return {
        "id": e.id,
        "memory_id": e.memory_id,
        "old_value": e.old_value,
        "new_value": e.new_value,
        "action": e.action,
        "created_at": e.created_at,
    }


class Memory:
    """Synchronous memory interface. Wraps the Rust MemoryEngine."""

    def __init__(self, config: Optional[MemoryConfig] = None):
        cfg = config or MemoryConfig()
        self._engine = PyMemoryEngine(cfg.to_json())

    @classmethod
    def from_config(cls, config_dict: dict) -> Memory:
        cfg = MemoryConfig(**config_dict)
        return cls(config=cfg)

    def add(
        self,
        messages: Union[str, list],
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
    ) -> dict:
        msgs = _normalize_messages(messages)
        meta_json = _json.dumps(metadata) if metadata is not None else None
        result = self._engine.add(
            msgs, user_id=user_id, agent_id=agent_id, run_id=run_id,
            metadata=meta_json, infer=infer,
        )
        return _add_result_to_dict(result)

    def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
    ) -> dict:
        filters_json = _json.dumps(filters) if filters is not None else None
        result = self._engine.search(
            query, user_id=user_id, agent_id=agent_id, run_id=run_id, limit=limit,
            filters=filters_json, rerank=rerank,
        )
        return _search_result_to_dict(result)

    def get(self, memory_id: str) -> dict:
        return _memory_item_to_dict(self._engine.get(memory_id))

    def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> list:
        filters_json = _json.dumps(filters) if filters is not None else None
        items = self._engine.get_all(
            user_id=user_id, agent_id=agent_id, run_id=run_id, filters=filters_json
        )
        return [_memory_item_to_dict(item) for item in items]

    def update(self, memory_id: str, new_text: str) -> None:
        self._engine.update(memory_id, new_text)

    def delete(self, memory_id: str) -> None:
        self._engine.delete(memory_id)

    def delete_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self._engine.delete_all(user_id=user_id, agent_id=agent_id, run_id=run_id)

    def history(self, memory_id: str) -> list:
        events = self._engine.history(memory_id)
        return [_event_to_dict(e) for e in events]

    def reset(self) -> None:
        self._engine.reset()


class AsyncMemory:
    """Async memory interface backed by native Rust coroutines via pyo3-async-runtimes."""

    def __init__(self) -> None:
        self._engine: Optional[PyAsyncMemoryEngine] = None

    @classmethod
    async def create(cls, config: Optional[MemoryConfig] = None) -> AsyncMemory:
        """Async factory -- use this instead of ``__init__``."""
        cfg = config or MemoryConfig()
        obj = cls()
        obj._engine = await PyAsyncMemoryEngine.create(cfg.to_json())
        return obj

    @classmethod
    async def from_config(cls, config_dict: dict) -> AsyncMemory:
        cfg = MemoryConfig(**config_dict)
        return await cls.create(config=cfg)

    def _check_engine(self) -> PyAsyncMemoryEngine:
        if self._engine is None:
            raise RuntimeError(
                "AsyncMemory is not initialized. "
                "Use 'await AsyncMemory.create(config)' instead of 'AsyncMemory()'."
            )
        return self._engine

    async def add(
        self,
        messages: Union[str, list],
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
    ) -> dict:
        engine = self._check_engine()
        msgs = _normalize_messages(messages)
        meta_json = _json.dumps(metadata) if metadata is not None else None
        result = await engine.add(
            msgs, user_id=user_id, agent_id=agent_id, run_id=run_id,
            metadata=meta_json, infer=infer,
        )
        return _add_result_to_dict(result)

    async def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
    ) -> dict:
        engine = self._check_engine()
        filters_json = _json.dumps(filters) if filters is not None else None
        result = await engine.search(
            query, user_id=user_id, agent_id=agent_id, run_id=run_id, limit=limit,
            filters=filters_json, rerank=rerank,
        )
        return _search_result_to_dict(result)

    async def get(self, memory_id: str) -> dict:
        engine = self._check_engine()
        item = await engine.get(memory_id)
        return _memory_item_to_dict(item)

    async def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> list:
        engine = self._check_engine()
        filters_json = _json.dumps(filters) if filters is not None else None
        items = await engine.get_all(
            user_id=user_id, agent_id=agent_id, run_id=run_id, filters=filters_json
        )
        return [_memory_item_to_dict(item) for item in items]

    async def update(self, memory_id: str, new_text: str) -> None:
        engine = self._check_engine()
        await engine.update(memory_id, new_text)

    async def delete(self, memory_id: str) -> None:
        engine = self._check_engine()
        await engine.delete(memory_id)

    async def delete_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        engine = self._check_engine()
        await engine.delete_all(user_id=user_id, agent_id=agent_id, run_id=run_id)

    async def history(self, memory_id: str) -> list:
        engine = self._check_engine()
        events = await engine.history(memory_id)
        return [_event_to_dict(e) for e in events]

    async def reset(self) -> None:
        engine = self._check_engine()
        await engine.reset()


def _normalize_messages(messages: Union[str, list]) -> list[tuple[str, str]]:
    """Convert various input formats to list of (role, content) tuples."""
    if isinstance(messages, str):
        return [("user", messages)]
    if isinstance(messages, list):
        result = []
        for msg in messages:
            if isinstance(msg, dict):
                result.append((msg.get("role", "user"), msg.get("content", "")))
            elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                result.append((str(msg[0]), str(msg[1])))
            elif isinstance(msg, str):
                result.append(("user", msg))
            else:
                result.append(("user", str(msg)))
        return result
    return [("user", str(messages))]
