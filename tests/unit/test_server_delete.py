"""Deleting a job must not block the server loop and must really kill its pipeline process."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from autoclip.server import config as server_config  # noqa: E402
from autoclip.server.main import create_app  # noqa: E402
from autoclip.server.state import Job, registry  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setenv("AUTH_USERNAME", "")
    monkeypatch.setenv("AUTH_PASSWORD", "")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    server_config.get_settings.cache_clear()
    yield
    server_config.get_settings.cache_clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _job(tmp_path: Path, job_id: str) -> Job:
    output = tmp_path / f"{job_id}-out"
    output.mkdir()
    (output / "clip.mp4").write_text("x", encoding="utf-8")
    input_path = tmp_path / f"{job_id}.mp4"
    input_path.write_text("x", encoding="utf-8")
    job = Job(job_id=job_id, filename=input_path.name, input_path=input_path, output_dir=output, config={})
    with registry._lock:
        registry._jobs[job.job_id] = job
    return job


def test_delete_job_does_not_block_the_event_loop(tmp_path: Path) -> None:
    """The 200 ms grace period must be an await: a synchronous sleep freezes every websocket and request."""
    from autoclip.server.routers.api import delete_job

    job = _job(tmp_path, "heartbeat-delete")

    async def scenario() -> int:
        ticks = 0
        done = False

        async def heartbeat() -> None:
            nonlocal ticks
            while not done:
                await asyncio.sleep(0.01)
                ticks += 1

        beat = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)  # let the heartbeat start before the delete
        await delete_job(job.job_id)
        done = True
        await beat
        return ticks

    assert asyncio.run(scenario()) >= 5
    assert registry.get(job.job_id) is None


def test_delete_job_kills_a_running_pipeline_process(client: TestClient, tmp_path: Path) -> None:
    job = _job(tmp_path, "running-delete")
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    job.process = process
    try:
        response = client.delete(f"/api/jobs/{job.job_id}")
        assert response.status_code == 200
        assert response.json() == {"deleted": "running-delete"}
        assert process.poll() is not None, "terminate_tree must kill the pipeline"
        assert job.process is None
        assert not job.output_dir.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
