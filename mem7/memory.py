"""Thin Python wrapper around the Rust MemoryEngine."""

from __future__ import annotations

import asyncio
import json
from typing import Optional, Union

from mem7._mem7 import PyMemoryEngine
from mem7.config import MemoryConfig


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
    ) -> dict:
        msgs = _normalize_messages(messages)
        raw = self._engine.add(msgs, user_id=user_id, agent_id=agent_id, run_id=run_id)
        return json.loads(raw)

    def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 5,
    ) -> dict:
        raw = self._engine.search(
            query, user_id=user_id, agent_id=agent_id, run_id=run_id, limit=limit
        )
        return json.loads(raw)

    def get(self, memory_id: str) -> dict:
        return json.loads(self._engine.get(memory_id))

    def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> list:
        raw = self._engine.get_all(user_id=user_id, agent_id=agent_id, run_id=run_id)
        return json.loads(raw)

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
        return json.loads(self._engine.history(memory_id))

    def reset(self) -> None:
        self._engine.reset()


class AsyncMemory:
    """Async memory interface. Runs sync Rust calls in a thread executor."""

    def __init__(self, config: Optional[MemoryConfig] = None):
        cfg = config or MemoryConfig()
        self._engine = PyMemoryEngine(cfg.to_json())

    @classmethod
    def from_config(cls, config_dict: dict) -> AsyncMemory:
        cfg = MemoryConfig(**config_dict)
        return cls(config=cfg)

    async def add(
        self,
        messages: Union[str, list],
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> dict:
        msgs = _normalize_messages(messages)
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None,
            lambda: self._engine.add(msgs, user_id=user_id, agent_id=agent_id, run_id=run_id),
        )
        return json.loads(raw)

    async def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 5,
    ) -> dict:
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None,
            lambda: self._engine.search(
                query, user_id=user_id, agent_id=agent_id, run_id=run_id, limit=limit
            ),
        )
        return json.loads(raw)

    async def get(self, memory_id: str) -> dict:
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, lambda: self._engine.get(memory_id))
        return json.loads(raw)

    async def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> list:
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None,
            lambda: self._engine.get_all(
                user_id=user_id, agent_id=agent_id, run_id=run_id
            ),
        )
        return json.loads(raw)

    async def update(self, memory_id: str, new_text: str) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: self._engine.update(memory_id, new_text)
        )

    async def delete(self, memory_id: str) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._engine.delete(memory_id))

    async def delete_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._engine.delete_all(
                user_id=user_id, agent_id=agent_id, run_id=run_id
            ),
        )

    async def history(self, memory_id: str) -> list:
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, lambda: self._engine.history(memory_id))
        return json.loads(raw)

    async def reset(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._engine.reset())


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
