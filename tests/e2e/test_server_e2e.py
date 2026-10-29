"""End-to-end checks of the HTTP/WebSocket server with a real upload and a real render.

Opt in with ``AUTOCLIP_E2E_TESTS=1 pytest tests/e2e/test_server_e2e.py -m e2e --no-cov``. The server
runs as a real subprocess (``python -m autoclip.server``); the payload is the generated 40 s fixture
video, the config asks for one short clip, and the test drives the whole surface: the HTML page, the
multipart upload, the WebSocket progress stream, the JSON status, the ZIP download and the delete.

This is the only check that proves the server wiring against a real render - the unit suite uses
``TestClient`` and never leaves the process.
"""

from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
ws_client = pytest.importorskip("websockets.sync.client")

from autoclip.core.media import probe  # noqa: E402

from ..support import Media  # noqa: E402

pytestmark = [pytest.mark.e2e]

ROOT = Path(__file__).resolve().parents[2]
STARTUP_TIMEOUT = 60.0
JOB_TIMEOUT = 900.0


def _free_port() -> int:
    """A port the OS says is free right now (the server binds it a moment later)."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    """Encode a multipart/form-data body without a client library."""
    boundary = uuid.uuid4().hex
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, (filename, data, content_type) in files.items():
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n".encode()
            + data
            + b"\r\n"
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _request(
    url: str, *, data: bytes | None = None, headers: dict[str, str] | None = None, method: str | None = None
) -> tuple[int, bytes, dict[str, str]]:
    """One HTTP request; an error status is returned, not raised."""
    request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed localhost URL
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers)


def _wait_until_ready(process: subprocess.Popen[str], base_url: str) -> None:
    """Poll ``/view/`` until the server answers (or fail with its output when it died)."""
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout is not None else ""
            raise AssertionError(f"the server exited with {process.returncode}:\n{output[-4000:]}")
        try:
            status, _, _ = _request(f"{base_url}/view/")
        except OSError:
            time.sleep(0.2)
            continue
        if status == 200:
            return
        time.sleep(0.2)
    raise AssertionError(f"the server did not answer on {base_url} within {STARTUP_TIMEOUT:.0f}s")


@dataclass(frozen=True)
class LiveServer:
    """A server subprocess and the folders it writes."""

    port: int
    output_dir: Path
    cache_dir: Path
    process: subprocess.Popen[str]

    @property
    def base_url(self) -> str:
        """The HTTP origin of the server."""
        return f"http://127.0.0.1:{self.port}"

    @property
    def ws_origin(self) -> str:
        """The WebSocket origin of the server."""
        return f"ws://127.0.0.1:{self.port}"


@pytest.fixture(scope="module")
def live_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LiveServer]:
    """Start the real server on a free port with isolated output/cache folders."""
    root = tmp_path_factory.mktemp("server")
    output_dir, cache_dir = root / "outputs", root / "cache"
    port = _free_port()
    env = os.environ.copy()
    env.update(
        {
            "HOST": "127.0.0.1",
            "PORT": str(port),
            "OUTPUT_DIR": str(output_dir),
            "CACHE_DIR": str(cache_dir),
            "AUTH_USERNAME": "",
            "AUTH_PASSWORD": "",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
    )
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    process = subprocess.Popen(
        [sys.executable, "-m", "autoclip.server"],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    server = LiveServer(port, output_dir, cache_dir, process)
    try:
        _wait_until_ready(process, server.base_url)
        yield server
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
        if process.stdout is not None:
            process.stdout.close()


def _upload_config(media: Media) -> dict[str, Any]:
    """A fast, deterministic config for the fixture video (the server overrides input/output)."""
    return {
        "clips": {
            "kinds": ["short"],
            "scenes": ["best"],
            "short": {"count": 1, "min_seconds": 5, "max_seconds": 10, "aspect": "9:16"},
        },
        "transcript": {"mode": "file", "file": str(media.transcript)},
        "video": {"short_side": 96, "preset": "ultrafast", "gpu": "off", "thumbnail": True},
        "audio": {"denoise": "none"},
        "compute": {"device": "cpu"},
        "logging": {"enabled": False},
        "resources": {"max_workers": 1},
    }


def _wait_for_completion(base_url: str, job_id: str) -> dict[str, Any]:
    """Poll the status endpoint until the job finishes (or the deadline passes)."""
    deadline = time.monotonic() + JOB_TIMEOUT
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status, body, _ = _request(f"{base_url}/api/jobs/{job_id}")
        assert status == 200, (status, body[:2000])
        last = json.loads(body)
        if last["state"] in {"completed", "failed", "cancelled"}:
            return last
        time.sleep(0.25)
    raise AssertionError(f"job {job_id} did not finish within {JOB_TIMEOUT:.0f}s; last status: {last}")


def test_server_renders_an_uploaded_video(live_server: LiveServer, media: Media, tmp_path: Path) -> None:
    """Upload -> WebSocket progress -> JSON status -> ZIP download -> probe -> delete."""
    # The HTML front end is served next to the JSON API (no auth configured).
    status, page, _ = _request(f"{live_server.base_url}/view/")
    assert status == 200
    assert b"autoclip" in page.lower()

    config = _upload_config(media)
    body, content_type = _multipart(
        {"config_json": json.dumps(config)},
        {"file": (media.video.name, media.video.read_bytes(), "video/mp4")},
    )
    status, payload, _ = _request(f"{live_server.base_url}/api/jobs", data=body, headers={"Content-Type": content_type})
    assert status == 201, payload[:2000]
    created = json.loads(payload)
    job_id = created["job_id"]
    assert created["status_url"] == f"/api/jobs/{job_id}"
    assert created["ws_url"] == f"/api/jobs/{job_id}/ws"

    # The websocket sends the current state at once and then every progress event.
    steps: list[str] = []
    states: list[str] = []
    ws_url = f"{live_server.ws_origin}{created['ws_url']}"
    with ws_client.connect(ws_url, proxy=None, open_timeout=30, close_timeout=5) as websocket:
        while True:
            message = json.loads(websocket.recv(timeout=JOB_TIMEOUT))
            if message.get("keepalive"):
                continue
            assert message["job_id"] == job_id
            steps.append(str(message["step"]))
            states.append(str(message["state"]))
            if message["state"] in {"completed", "failed", "cancelled"}:
                break

    final = _wait_for_completion(live_server.base_url, job_id)
    assert final["state"] == "completed", final
    assert final["error"] is None, final
    assert final["percent"] == 100.0, final
    assert states[-1] == "completed", states
    # A client that connected while the job still ran must have seen intermediate progress; one that
    # connected after the render only gets the final snapshot (the API replays the current state).
    assert all(step for step in steps), steps
    if len(states) > 1:
        assert any(step not in {"queued", "starting", "completed"} for step in steps[1:-1]), steps
    assert final["outputs"] and final["outputs"][0]["name"].endswith(".mp4"), final

    status, detail, _ = _request(f"{live_server.base_url}/view/jobs/{job_id}")
    assert status == 200
    assert job_id.encode() in detail

    # The archive mirrors the output folder: the clip, its sidecars and the clips.json index.
    status, archive, _ = _request(f"{live_server.base_url}/api/jobs/{job_id}/download")
    assert status == 200, archive[:2000]
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        names = zf.namelist()
        clips = [name for name in names if name.endswith(".mp4")]
        assert len(clips) == 1, names
        assert clips[0].startswith("short/"), names
        assert any(name.endswith(".srt") for name in names), names
        assert any(name.endswith(".jpg") for name in names), names
        assert any(name.endswith("clips.json") for name in names), names
        assert all(".work" not in Path(name).parts for name in names), names
        for name in names:
            target = tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))
    extracted = tmp_path / clips[0]
    rendered = next(
        path
        for path in (live_server.output_dir / "jobs" / job_id / "out").rglob("*.mp4")
        if ".work" not in path.relative_to(live_server.output_dir).parts
    )
    info = probe(extracted)
    assert info.width == 96, (info.width, info.height)
    assert 4.5 <= info.duration <= 10.5, info.duration
    rows = json.loads(next(rendered.parent.glob("clips.json")).read_text(encoding="utf-8"))
    assert len(rows) == 1 and rows[0]["scene"] == "best", rows

    # Deleting removes the job from the registry and its files from the disk.
    status, deleted, _ = _request(f"{live_server.base_url}/api/jobs/{job_id}", method="DELETE")
    assert status == 200, deleted[:2000]
    assert json.loads(deleted) == {"deleted": job_id}
    status, _, _ = _request(f"{live_server.base_url}/api/jobs/{job_id}")
    assert status == 404
    status, _, _ = _request(f"{live_server.base_url}/view/jobs/{job_id}")
    assert status == 404
    assert not (live_server.output_dir / "jobs" / job_id / "out").exists()
    leftovers = live_server.output_dir / "jobs" / job_id
    assert not leftovers.exists() or not any(leftovers.iterdir())
