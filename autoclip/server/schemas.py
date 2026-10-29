from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class JobState(StrEnum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class ProgressEvent(BaseModel):
    job_id: str
    step: str
    message: str
    percent: float = 0.0
    state: JobState = JobState.running
    at: datetime = Field(default_factory=datetime.utcnow)


class JobCreated(BaseModel):
    job_id: str
    status_url: str
    ws_url: str
    download_url: str


class OutputFile(BaseModel):
    name: str
    size_bytes: int


class JobStatus(BaseModel):
    job_id: str
    state: JobState
    step: str
    message: str
    percent: float
    error: str | None = None
    outputs: list[OutputFile] = []
    created_at: datetime
    finished_at: datetime | None = None


class ConfigValidation(BaseModel):
    accepted: bool
    errors: list[str] = []
    warnings: list[str] = []
    config: dict[str, Any] | None = None
