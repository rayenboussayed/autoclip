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
        self._tasks: set[asyncio.Task[None]] = set()
        self._lock = asyncio.Lock()

    def bind_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
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
        loop = self._loop
        if loop is None:
            return
        # Build the coroutine only once the callback runs: if the loop is already closed,
        # call_soon_threadsafe raises first and no coroutine is left un-awaited.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(self._start_broadcast, job_id, payload)

    def _start_broadcast(self, job_id: str, payload: dict[str, Any]) -> None:
        task = asyncio.ensure_future(self.broadcast(job_id, payload))
        # Keep a strong reference: the event loop only holds weak ones, so an unreferenced
        # task can be garbage-collected mid-flight.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


progress_manager = ProgressManager()
