"""Thin Python wrapper around the Rust MemoryEngine."""

from __future__ import annotations

import json as _json
from typing import Any, Dict, Optional, Union

from mem7._mem7 import PyAsyncMemoryEngine, PyMemoryEngine
from mem7.config import MemoryConfig


def _parse_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return _json.loads(value)
    except _json.JSONDecodeError:
        return value


def _require_scope(
    *,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    effective_filters = dict(filters) if filters else {}

    if user_id is None and "user_id" in effective_filters:
        user_id = effective_filters.pop("user_id")
    if agent_id is None and "agent_id" in effective_filters:
        agent_id = effective_filters.pop("agent_id")
    if run_id is None and "run_id" in effective_filters:
        run_id = effective_filters.pop("run_id")

    return user_id, agent_id, run_id, (effective_filters or None)


def _memory_item_to_dict(item) -> dict:
    metadata = _parse_json_value(item.metadata)
    memory_text = item.text

    d = {
        "id": item.id,
        "memory": memory_text,
        "text": item.text,
        "hash": item.hash,
        "user_id": item.user_id,
        "agent_id": item.agent_id,
        "run_id": item.run_id,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
        "last_accessed_at": item.last_accessed_at,
        "access_count": item.access_count,
    }
    if metadata is not None:
        d["metadata"] = metadata
    if item.score is not None:
        d["score"] = item.score
    if item.actor_id is not None:
        d["actor_id"] = item.actor_id
    if item.role is not None:
        d["role"] = item.role
    if item.memory_type is not None:
        d["memory_type"] = item.memory_type
    return d


def _action_result_to_dict(r) -> dict:
    return {
        "id": r.id,
        "memory": r.new_value or r.old_value or "",
        "event": r.action,
        "action": r.action,
        "old_value": r.old_value,
        "new_value": r.new_value,
    }


def _relation_to_dict(r) -> dict:
    relation = {
        "source": r.source,
        "relationship": r.relationship,
        "destination": r.destination,
    }
    if r.score is not None:
        relation["score"] = r.score
    return relation


def _add_result_to_dict(result) -> dict:
    return {
        "results": [_action_result_to_dict(r) for r in result.results],
        "relations": [_relation_to_dict(r) for r in result.relations],
    }


def _search_result_to_dict(result) -> dict:
    results = [_memory_item_to_dict(m) for m in result.memories]
    return {
        "results": results,
        "memories": results,
        "relations": [_relation_to_dict(r) for r in result.relations],
    }


def _event_to_dict(e) -> dict:
    return {
        "id": e.id,
        "memory_id": e.memory_id,
        "old_memory": e.old_value,
        "new_memory": e.new_value,
        "event": e.action,
        "old_value": e.old_value,
        "new_value": e.new_value,
        "action": e.action,
        "created_at": e.created_at,
        "updated_at": e.updated_at,
        "is_deleted": e.is_deleted,
        "actor_id": e.actor_id,
        "role": e.role,
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
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> dict:
        user_id, agent_id, run_id, _ = _require_scope(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
        )
        msgs = _normalize_messages(messages)
        effective_metadata = dict(metadata or {})
        if memory_type is not None:
            effective_metadata.setdefault("memory_type", memory_type)
        if prompt is not None:
            effective_metadata.setdefault("prompt", prompt)
        meta_json = _json.dumps(effective_metadata) if effective_metadata else None
        result = self._engine.add(
            msgs,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=meta_json,
            infer=infer,
        )
        return _add_result_to_dict(result)

    def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
        threshold: Optional[float] = None,
        task_type: Optional[str] = None,
    ) -> dict:
        user_id, agent_id, run_id, effective_filters = _require_scope(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            filters=filters,
        )
        filters_json = _json.dumps(effective_filters) if effective_filters is not None else None
        result = self._engine.search(
            query,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            limit=limit,
            filters=filters_json,
            rerank=rerank,
            threshold=threshold,
            task_type=task_type,
        )
        return _search_result_to_dict(result)

    def get(self, memory_id: str) -> Optional[dict]:
        try:
            return _memory_item_to_dict(self._engine.get(memory_id))
        except Exception as exc:  # pragma: no cover - native exception type
            if "Not found" in str(exc) or "not found" in str(exc):
                return None
            raise

    def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = 100,
    ) -> dict:
        user_id, agent_id, run_id, effective_filters = _require_scope(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            filters=filters,
        )
        filters_json = _json.dumps(effective_filters) if effective_filters is not None else None
        items = self._engine.get_all(
            user_id=user_id, agent_id=agent_id, run_id=run_id, filters=filters_json, limit=limit
        )
        results = [_memory_item_to_dict(item) for item in items]
        return {"results": results, "memories": results}

    def update(self, memory_id: str, new_text: Optional[str] = None, **kwargs: Any) -> dict:
        if new_text is None:
            new_text = kwargs.pop("text", None)
        if new_text is None:
            raise TypeError("update() missing required argument: 'text'")
        self._engine.update(memory_id, new_text)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id: str) -> dict:
        self._engine.delete(memory_id)
        return {"message": "Memory deleted successfully!"}

    def delete_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> dict:
        _require_scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        self._engine.delete_all(user_id=user_id, agent_id=agent_id, run_id=run_id)
        return {"message": "Memories deleted successfully!"}

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
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> dict:
        engine = self._check_engine()
        user_id, agent_id, run_id, _ = _require_scope(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
        )
        msgs = _normalize_messages(messages)
        effective_metadata = dict(metadata or {})
        if memory_type is not None:
            effective_metadata.setdefault("memory_type", memory_type)
        if prompt is not None:
            effective_metadata.setdefault("prompt", prompt)
        meta_json = _json.dumps(effective_metadata) if effective_metadata else None
        result = await engine.add(
            msgs,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=meta_json,
            infer=infer,
        )
        return _add_result_to_dict(result)

    async def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
        threshold: Optional[float] = None,
        task_type: Optional[str] = None,
    ) -> dict:
        engine = self._check_engine()
        user_id, agent_id, run_id, effective_filters = _require_scope(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            filters=filters,
        )
        filters_json = _json.dumps(effective_filters) if effective_filters is not None else None
        result = await engine.search(
            query,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            limit=limit,
            filters=filters_json,
            rerank=rerank,
            threshold=threshold,
            task_type=task_type,
        )
        return _search_result_to_dict(result)

    async def get(self, memory_id: str) -> Optional[dict]:
        engine = self._check_engine()
        try:
            item = await engine.get(memory_id)
        except Exception as exc:  # pragma: no cover - native exception type
            if "Not found" in str(exc) or "not found" in str(exc):
                return None
            raise
        return _memory_item_to_dict(item)

    async def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = 100,
    ) -> dict:
        engine = self._check_engine()
        user_id, agent_id, run_id, effective_filters = _require_scope(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            filters=filters,
        )
        filters_json = _json.dumps(effective_filters) if effective_filters is not None else None
        items = await engine.get_all(
            user_id=user_id, agent_id=agent_id, run_id=run_id, filters=filters_json, limit=limit
        )
        results = [_memory_item_to_dict(item) for item in items]
        return {"results": results, "memories": results}

    async def update(self, memory_id: str, new_text: Optional[str] = None, **kwargs: Any) -> dict:
        engine = self._check_engine()
        if new_text is None:
            new_text = kwargs.pop("text", None)
        if new_text is None:
            raise TypeError("update() missing required argument: 'text'")
        await engine.update(memory_id, new_text)
        return {"message": "Memory updated successfully!"}

    async def delete(self, memory_id: str) -> dict:
        engine = self._check_engine()
        await engine.delete(memory_id)
        return {"message": "Memory deleted successfully!"}

    async def delete_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> dict:
        engine = self._check_engine()
        _require_scope(user_id=user_id, agent_id=agent_id, run_id=run_id)
        await engine.delete_all(user_id=user_id, agent_id=agent_id, run_id=run_id)
        return {"message": "Memories deleted successfully!"}

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
    if isinstance(messages, dict):
        return [(messages.get("role", "user"), messages.get("content", ""))]
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
