from __future__ import annotations

import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schemas import JobState


@dataclass
class Job:
    job_id: str
    filename: str
    input_path: Path
    output_dir: Path
    config: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    state: JobState = JobState.queued
    step: str = "queued"
    message: str = "Queued"
    percent: float = 0.0
    error: str | None = None
    finished_at: float | None = None
    outputs: list[Path] = field(default_factory=list)
    work_dir: Path | None = None

    def update(
        self,
        *,
        step: str | None = None,
        message: str | None = None,
        percent: float | None = None,
        state: JobState | None = None,
        error: str | None = None,
    ) -> None:
        if step is not None:
            self.step = step
        if message is not None:
            self.message = message
        if percent is not None:
            self.percent = percent
        if state is not None:
            self.state = state
        if error is not None:
            self.error = error
        if state in (JobState.completed, JobState.failed, JobState.cancelled):
            self.finished_at = time.time()

    def to_status(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": self.state.value,
            "step": self.step,
            "message": self.message,
            "percent": self.percent,
            "error": self.error,
            "outputs": [{"name": p.name, "size_bytes": p.stat().st_size if p.is_file() else 0} for p in self.outputs],
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()

    def create(self, **kwargs: Any) -> Job:
        job = Job(job_id=str(uuid.uuid4()), **kwargs)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def remove(self, job_id: str) -> None:
        """Drop a job from the registry (the caller has already deleted its files)."""
        with self._lock:
            self._jobs.pop(job_id, None)

    def evict_expired(self, ttl_seconds: int) -> None:
        now = time.time()
        with self._lock:
            expired = [jid for jid, j in self._jobs.items() if j.finished_at and now - j.finished_at > ttl_seconds]
            for jid in expired:
                j = self._jobs.pop(jid)
                if j.output_dir.exists():
                    shutil.rmtree(j.output_dir, ignore_errors=True)


registry = JobRegistry()
