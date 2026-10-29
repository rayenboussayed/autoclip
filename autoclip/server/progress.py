from __future__ import annotations

import asyncio
import contextlib
import json
from collections import defaultdict
from typing import Any

from fastapi import WebSocket


class ProgressManager:
    """Fan-out progress events to every browser tab watching a job."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._sockets: dict[str, set[WebSocket]] = defaultdict(set)
        self._queues: dict[str, list[asyncio.Queue[dict[str, Any]]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def register(self, job_id: str, ws: Any) -> None:
        await ws.accept()
        async with self._lock:
            self._sockets[job_id].add(ws)

    async def unregister(self, job_id: str, ws: Any) -> None:
        async with self._lock:
            self._sockets[job_id].discard(ws)

    async def broadcast(self, job_id: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._sockets.get(job_id, ()))
        for ws in targets:
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                await self.unregister(job_id, ws)

    def publish(self, job_id: str, payload: dict[str, Any]) -> None:
        """Thread-safe publish from background workers."""
        if self._loop is None:
            return
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._loop.create_task, self.broadcast(job_id, payload))

    def subscribe(self, job_id: str) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._queues[job_id].append(q)
        return q

    async def feed(self, job_id: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            queues = list(self._queues.get(job_id, ()))
        for q in queues:
            await q.put(payload)


progress_manager = ProgressManager()
