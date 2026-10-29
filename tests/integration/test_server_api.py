"""Integration tests for the FastAPI front end using the generated media fixture."""

from __future__ import annotations

import dataclasses
import json
import zipfile
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from autoclip.server import config as server_config  # noqa: E402
from autoclip.server.main import create_app  # noqa: E402
from tests.support import install_stub_whisper, make_cfg  # noqa: E402


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    monkeypatch.setenv("AUTH_USERNAME", "")
    monkeypatch.setenv("AUTH_PASSWORD", "")
    monkeypatch.setenv("MAX_FILE_SIZE", str(1024 * 1024 * 1024))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "2")
    server_config.get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as c:
        yield c


def _wait_for_job(client: TestClient, job_id: str, timeout: int = 120) -> dict[str, Any]:
    import time

    deadline = time.monotonic() + timeout
    payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["state"] in ("completed", "failed"):
            return payload
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not finish: {payload}")


def test_create_job_uploads_and_renders_a_video(client: TestClient, media: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    video_bytes = media.video.read_bytes()
    custom = dataclasses.asdict(
        make_cfg(media, media.video.parent / "out", clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    )
    custom.pop("input", None)
    custom.pop("output_dir", None)

    response = client.post(
        "/api/jobs",
        files={"file": (media.video.name, video_bytes, "video/mp4")},
        data={"config_json": json.dumps(custom)},
    )
    assert response.status_code == 201, response.text
    job = response.json()
    assert {"job_id", "status_url", "ws_url", "download_url"} <= set(job)

    status = _wait_for_job(client, job["job_id"])
    assert status["state"] == "completed", status
    assert status["outputs"], status
    assert status["percent"] == 100.0

    download = client.get(job["download_url"])
    assert download.status_code == 200
    assert download.headers["content-type"].startswith("application/zip")
    with zipfile.ZipFile(BytesIO(download.content)) as archive:
        names = archive.namelist()
        assert any(name.endswith(".mp4") for name in names)
        assert any(name.endswith("clips.json") for name in names)


def test_create_job_accepts_a_config_file(client: TestClient, media: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    import io

    custom = {"clips": {"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}}}
    config_file = io.BytesIO(json.dumps(custom).encode("utf-8"))
    response = client.post(
        "/api/jobs",
        files={
            "file": (media.video.name, media.video.read_bytes(), "video/mp4"),
            "config": ("config.json", config_file, "application/json"),
        },
    )
    assert response.status_code == 201, response.text
    status = _wait_for_job(client, response.json()["job_id"])
    assert status["state"] == "completed", status


def test_create_job_fails_on_invalid_media(client: TestClient) -> None:
    response = client.post("/api/jobs", files={"file": ("bad.mp4", b"not a video", "video/mp4")})
    assert response.status_code == 201, response.text
    job_id = response.json()["job_id"]
    status = _wait_for_job(client, job_id)
    assert status["state"] == "failed"
    assert status["error"]
    download = client.get(f"/api/jobs/{job_id}/download")
    assert download.status_code == 409


def test_job_websocket_reports_terminal_state(client: TestClient, media: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    custom = {"clips": {"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}}}
    response = client.post(
        "/api/jobs",
        files={"file": (media.video.name, media.video.read_bytes(), "video/mp4")},
        data={"config_json": json.dumps(custom)},
    )
    assert response.status_code == 201
    job_id = response.json()["job_id"]
    messages: list[dict[str, Any]] = []
    with client.websocket_connect(f"/api/jobs/{job_id}/ws") as websocket:
        import time

        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                messages.append(websocket.receive_json())
            except Exception:
                break
            if messages and messages[-1].get("state") in ("completed", "failed"):
                break
    assert messages
    assert any(m.get("state") in ("running", "completed", "failed") for m in messages)
    assert messages[-1].get("state") in ("completed", "failed")

    finished = _wait_for_job(client, job_id)
    assert finished["state"] == "completed", finished
