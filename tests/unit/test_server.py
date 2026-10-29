"""Smoke tests for the optional HTTP/WebSocket front end (skipped when FastAPI is not installed)."""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from autoclip import __version__  # noqa: E402
from autoclip.server import config as server_config  # noqa: E402
from autoclip.server.main import create_app  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep ``get_settings`` out of the way: it is cached for the whole process.

    A test that enables auth would otherwise leave auth-enabled settings behind for every later
    test - the websocket endpoint would close at its auth check and never reach the keepalive loop.
    """
    monkeypatch.setenv("AUTH_USERNAME", "")
    monkeypatch.setenv("AUTH_PASSWORD", "")
    server_config.get_settings.cache_clear()
    yield
    server_config.get_settings.cache_clear()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setenv("AUTH_USERNAME", "")
    monkeypatch.setenv("AUTH_PASSWORD", "")
    monkeypatch.setenv("MAX_FILE_SIZE", "1024")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    server_config.get_settings.cache_clear()
    app = create_app()
    return TestClient(app)


def test_index_page_renders(client: TestClient) -> None:
    response = client.get("/view/")
    assert response.status_code == 200
    assert "autoclip" in response.text
    # The form posts to the JSON API; the page follows the created job to its progress page.
    assert 'hx-post="/api/jobs"' in response.text
    assert "htmx:afterRequest" in response.text
    assert "/view/jobs/" in response.text
    assert create_app().version == __version__


def test_upload_rejects_an_empty_or_oversized_file(client: TestClient) -> None:
    response = client.post("/api/jobs", files={"file": ("empty.bin", b"", "application/octet-stream")})
    assert response.status_code == 400

    response = client.post("/api/jobs", files={"file": ("big.bin", b"x" * 2048, "application/octet-stream")})
    assert response.status_code == 413


def test_upload_rejects_invalid_config(client: TestClient) -> None:
    response = client.post(
        "/api/jobs",
        files={"file": ("clip.mp4", b"not-a-video", "video/mp4")},
        data={"config_json": "{not json"},
    )
    assert response.status_code == 400


def test_job_status_404(client: TestClient) -> None:
    response = client.get("/api/jobs/does-not-exist")
    assert response.status_code == 404


def test_run_job_processes_a_small_video(monkeypatch: pytest.MonkeyPatch, media: Any, tmp_path: Path) -> None:

    from autoclip.server.schemas import JobState
    from autoclip.server.service import run_job, zip_outputs
    from autoclip.server.state import Job
    from tests.support import install_stub_whisper, make_cfg

    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    import dataclasses
    import threading

    custom = dataclasses.asdict(
        make_cfg(
            media,
            tmp_path / "out",
            audio={"denoise": "none"},
            compute={"device": "cpu"},
            clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}},
            video={"thumbnail": True},
        )
    )
    custom.pop("input", None)
    custom.pop("output_dir", None)
    custom.setdefault("logging", {})["file"] = "logs/autoclip.log"

    upload = media.video
    out = tmp_path / "jobout"
    out.mkdir()
    job = Job(
        job_id="test-job",
        filename=upload.name,
        input_path=upload,
        output_dir=out,
        config={"_custom": custom},
        work_dir=out / ".work" / upload.stem,
    )
    run_job(job)
    assert job.state == JobState.completed, job.error
    assert job.outputs, "no clips rendered"
    assert any(p.suffix == ".mp4" for p in job.outputs)
    archive = zip_outputs(job)
    assert archive.is_file()
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    clips = [name for name in names if name.endswith(".mp4")]
    # The archive mirrors the output folder: clips, their sidecars and the index, never the work folder.
    assert len(clips) == len(job.outputs) == 1, names
    assert clips[0].startswith("short/"), names
    assert any(name.endswith(".srt") for name in names), names
    assert any(name.endswith(".jpg") for name in names), names
    assert any(name.endswith("clips.json") for name in names), names
    assert all(".work" not in Path(name).parts for name in names), names

    # Split mode renders the source into several consecutive parts.
    split_cfg = dataclasses.asdict(make_cfg(media, tmp_path / "out2", clips={"mode": "split", "split_seconds": 10}))
    split_cfg.pop("input", None)
    split_cfg.pop("output_dir", None)
    split_out = tmp_path / "splitout"
    split_out.mkdir()
    split_job = Job(
        job_id="split-job",
        filename=upload.name,
        input_path=upload,
        output_dir=split_out,
        config={"_custom": split_cfg},
        work_dir=split_out / ".work" / upload.stem,
    )
    run_job(split_job)
    assert split_job.state == JobState.completed, split_job.error
    assert len(split_job.outputs) >= 2

    # A cancellation request aborts the job and marks it failed.
    cancel_event = threading.Event()
    cancel_event.set()
    cancel_cfg = dataclasses.asdict(
        make_cfg(
            media,
            tmp_path / "out3",
            clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}},
        )
    )
    cancel_cfg.pop("input", None)
    cancel_cfg.pop("output_dir", None)
    cancel_out = tmp_path / "cancelout"
    cancel_out.mkdir()
    cancel_job = Job(
        job_id="cancel-job",
        filename=upload.name,
        input_path=upload,
        output_dir=cancel_out,
        config={"_custom": cancel_cfg},
        work_dir=cancel_out / ".work" / upload.stem,
    )
    run_job(cancel_job, cancel_event=cancel_event)
    assert cancel_job.state == JobState.failed

    # An impossible length range produces no clips and a failed job.
    empty_cfg = dataclasses.asdict(
        make_cfg(
            media,
            tmp_path / "out4",
            clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 1000, "max_seconds": 1001}},
        )
    )
    empty_cfg.pop("input", None)
    empty_cfg.pop("output_dir", None)
    empty_out = tmp_path / "emptyout"
    empty_out.mkdir()
    empty_job = Job(
        job_id="empty-job",
        filename=upload.name,
        input_path=upload,
        output_dir=empty_out,
        config={"_custom": empty_cfg},
        work_dir=empty_out / ".work" / upload.stem,
    )
    run_job(empty_job)
    assert empty_job.state == JobState.failed
    assert empty_job.error

    # A missing source is reported as a failed job instead of raising.
    bad_out = tmp_path / "badout"
    bad_out.mkdir()
    bad_job = Job(
        job_id="bad-job",
        filename="missing.mp4",
        input_path=tmp_path / "no-such.mp4",
        output_dir=bad_out,
        config={},
        work_dir=bad_out / ".work" / "no-such",
    )
    run_job(bad_job)
    assert bad_job.state == JobState.failed
    assert bad_job.error

    return None


def test_state_evicts_expired_jobs_and_retains_active_ones(tmp_path: Path) -> None:

    from autoclip.server.schemas import JobState
    from autoclip.server.state import Job, registry

    expired = Job(
        job_id="expired",
        filename="x.mp4",
        input_path=tmp_path / "x.mp4",
        output_dir=tmp_path / "expiredout",
        config={},
    )
    expired.output_dir.mkdir()
    (expired.output_dir / "clip.mp4").write_text("x", encoding="utf-8")
    with registry._lock:
        registry._jobs[expired.job_id] = expired
    expired.state = JobState.completed
    expired.finished_at = 1.0
    registry.evict_expired(0)
    assert registry.get("expired") is None
    assert not (tmp_path / "expiredout").exists()

    active = Job(
        job_id="active",
        filename="y.mp4",
        input_path=tmp_path / "y.mp4",
        output_dir=tmp_path / "activeout",
        config={},
    )
    active.output_dir.mkdir(exist_ok=True)
    with registry._lock:
        registry._jobs[active.job_id] = active
    active.state = JobState.running
    active.finished_at = None
    registry.evict_expired(0)
    assert registry.get("active") is not None
    assert registry.list()  # covers the list() accessor
    with registry._lock:
        registry._jobs.pop("active", None)

    # Evicting a job whose output directory is already gone skips the rmtree branch.
    missing = Job(
        job_id="missing",
        filename="z.mp4",
        input_path=tmp_path / "z.mp4",
        output_dir=tmp_path / "does-not-exist",
        config={},
    )
    with registry._lock:
        registry._jobs[missing.job_id] = missing
    missing.state = JobState.completed
    missing.finished_at = 1.0
    registry.evict_expired(0)
    assert registry.get("missing") is None


class _FakeSocket:
    def __init__(self, fail: bool = False) -> None:
        self.accepted = False
        self.sent: list[str] = []
        self.fail = fail

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, text: str) -> None:
        if self.fail:
            raise RuntimeError("boom")
        self.sent.append(text)


def test_progress_manager_fanout() -> None:
    import asyncio

    from autoclip.server.progress import ProgressManager

    manager = ProgressManager()
    loop = asyncio.new_event_loop()
    loop2 = asyncio.new_event_loop()
    try:
        manager.bind_loop(loop)
        socket = _FakeSocket()
        bad_socket = _FakeSocket(fail=True)
        asyncio.set_event_loop(loop)
        loop.run_until_complete(manager.register("job", socket))
        loop.run_until_complete(manager.register("job", bad_socket))
        assert socket.accepted and bad_socket.accepted
        loop.run_until_complete(manager.broadcast("job", {"hello": True}))
        assert socket.sent and bad_socket not in manager._sockets["job"]
        # A publish with no loop or on a closed loop is ignored without leaving a pending coroutine.
        manager.bind_loop(None)
        manager.publish("job", {"ignored": True})
        loop3 = asyncio.new_event_loop()
        loop3.close()
        manager.bind_loop(loop3)
        manager.publish("job", {"ignored": True})
        # A live loop schedules the broadcast and the socket really receives it.
        manager.bind_loop(loop2)
        manager.publish("job", {"published": True})
        loop2.run_until_complete(asyncio.sleep(0.01))
        assert "published" in socket.sent[-1]
        manager.bind_loop(None)
        loop.run_until_complete(manager.unregister("job", socket))
    finally:
        pending = asyncio.all_tasks(loop2)
        for task in pending:
            task.cancel()
        if pending:
            loop2.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
        loop2.close()


def test_create_job_rejects_empty_and_invalid_documents(client: TestClient) -> None:
    assert client.post("/api/jobs", files={"file": ("empty.mp4", b"", "video/mp4")}).status_code == 400
    response = client.post("/api/jobs", files={"file": ("bad.mp4", b"x", "video/mp4")}, data={"config_json": "123"})
    assert response.status_code == 400
    assert "config must be a JSON object" in response.text


def test_delete_job_removes_an_existing_job(client: TestClient, tmp_path: Path) -> None:
    from autoclip.server.state import Job, registry

    target = tmp_path / "delete-target"
    target.mkdir()
    input_path = tmp_path / "clip.mp4"
    input_path.write_text("x", encoding="utf-8")
    (target / "clip.mp4").write_text("x", encoding="utf-8")
    job = Job(
        job_id="deletable",
        filename="clip.mp4",
        input_path=input_path,
        output_dir=target,
        config={},
    )
    with registry._lock:
        registry._jobs[job.job_id] = job
    response = client.delete(f"/api/jobs/{job.job_id}")
    assert response.status_code == 200
    assert response.json() == {"deleted": "deletable"}
    assert not target.exists()
    assert client.delete(f"/api/jobs/{job.job_id}").status_code == 404


def test_websocket_rejects_unknown_jobs_and_reports_known_state(client: TestClient, tmp_path: Path) -> None:
    from fastapi.websockets import WebSocketDisconnect

    from autoclip.server.schemas import JobState
    from autoclip.server.state import Job, registry

    job = Job(
        job_id="ws-job",
        filename="clip.mp4",
        input_path=tmp_path / "clip.mp4",
        output_dir=tmp_path / "wsout",
        config={},
    )
    job.output_dir.mkdir(exist_ok=True)
    job.state = JobState.running
    with registry._lock:
        registry._jobs[job.job_id] = job

    try:
        with client.websocket_connect(f"/api/jobs/{job.job_id}/ws") as websocket:
            payload = websocket.receive_json()
            assert payload["job_id"] == job.job_id
            assert payload["state"] == "running"
        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/api/jobs/missing/ws"):
            pass
    finally:
        with registry._lock:
            registry._jobs.pop(job.job_id, None)


def test_view_job_404(client: TestClient) -> None:
    response = client.get("/view/jobs/does-not-exist")
    assert response.status_code == 404
    assert "not found" in response.text


def test_progress_manager_ignores_publish_without_a_loop() -> None:
    from autoclip.server.progress import ProgressManager

    manager = ProgressManager()
    manager.bind_loop(None)
    manager.publish("job", {"ignored": True})


def test_view_job_renders_an_existing_job(client: TestClient, tmp_path: Path) -> None:
    from autoclip.server.state import Job, registry

    job = Job(
        job_id="existing-job",
        filename="clip.mp4",
        input_path=tmp_path / "clip.mp4",
        output_dir=tmp_path / "existingout",
        config={},
    )
    job.output_dir.mkdir(exist_ok=True)
    with registry._lock:
        registry._jobs[job.job_id] = job
    response = client.get(f"/view/jobs/{job.job_id}")
    assert response.status_code == 200
    assert job.job_id in response.text
    with registry._lock:
        registry._jobs.pop(job.job_id, None)


def test_main_entrypoint_runs_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy
    import sys
    import types

    from autoclip.server import config as server_config

    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda *a, **k: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://example/otlp")
    server_config.get_settings.cache_clear()

    import autoclip.server.main as main_mod

    monkeypatch.setattr(main_mod, "uvicorn", fake_uvicorn, raising=False)
    import importlib

    importlib.reload(main_mod)
    assert main_mod._OTEL_AVAILABLE is True

    import asyncio

    # Cover the lifespan evictor without waiting 60 s: shrink its tick and let one tick run while the
    # app is alive. Never patch asyncio.sleep itself - a no-op sleep turns this loop into a spin.
    monkeypatch.setattr(main_mod, "EVICT_SECONDS", 0.01)

    async def _consume_lifespan() -> None:
        async with main_mod.lifespan(main_mod.app):
            await asyncio.sleep(0.1)

    asyncio.run(_consume_lifespan())

    # force the FastAPI instrumentation to fail to cover its except branch
    fastapi_instr = sys.modules["opentelemetry.instrumentation.fastapi"]

    class _BrokenInstrumentor:
        @staticmethod
        def instrument_app(_app: object) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(fastapi_instr, "FastAPIInstrumentor", _BrokenInstrumentor)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://example/otlp")
    server_config.get_settings.cache_clear()
    importlib.reload(main_mod)
    # reloading creates app again and hits the except branch at lines 73-74
    app = main_mod.create_app()
    assert app is not None

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    server_config.get_settings.cache_clear()
    # break the OpenTelemetry trace import to cover the outer except branch
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace", None)
    importlib.reload(main_mod)
    assert main_mod._OTEL_AVAILABLE is False

    monkeypatch.setattr("builtins.__import__", __import__("builtins").__import__)
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    server_config.get_settings.cache_clear()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    importlib.reload(main_mod)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    assert main_mod.main() == 0

    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    try:
        runpy.run_module("autoclip.server.__main__", run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        assert exc.code == 0


def test_run_job_reports_pending_clips_as_failed(monkeypatch: pytest.MonkeyPatch, media: Any, tmp_path: Path) -> None:
    import dataclasses
    from unittest.mock import patch

    from autoclip.server.schemas import JobState
    from autoclip.server.service import run_job
    from autoclip.server.state import Job
    from tests.support import install_stub_whisper, make_cfg

    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    custom = dataclasses.asdict(
        make_cfg(
            media,
            tmp_path / "out",
            clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}},
        )
    )
    custom.pop("input", None)
    custom.pop("output_dir", None)

    out = tmp_path / "pendingout"
    out.mkdir()
    job = Job(
        job_id="pending-job",
        filename=media.video.name,
        input_path=media.video,
        output_dir=out,
        config={"_custom": custom},
        work_dir=out / ".work" / media.video.stem,
    )
    with patch("autoclip.server.service.render_all"):
        run_job(job)
    assert job.state == JobState.completed, job.error
    assert "failed" in (job.message or "").lower() or job.error is not None


def test_auth_is_required_when_credentials_are_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AUTH_USERNAME", "user")
    monkeypatch.setenv("AUTH_PASSWORD", "pass")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    server_config.get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as c:
        assert c.get("/view/").status_code == 401
        assert c.get("/view/", auth=("user", "pass")).status_code == 200
        bad = c.get("/api/jobs/whatever", auth=("user", "wrong"))
        assert bad.status_code == 401


def test_api_websocket_rejects_bad_tokens_and_unknown_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_USERNAME", "user")
    monkeypatch.setenv("AUTH_PASSWORD", "pass")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MAX_FILE_SIZE", "1024")
    from autoclip.server import config as server_config

    server_config.get_settings.cache_clear()

    from fastapi.testclient import TestClient
    from fastapi.websockets import WebSocketDisconnect

    from autoclip.server.main import create_app

    app = create_app()
    with TestClient(app) as client:
        # Bad token: the endpoint closes with 1008 before accepting.
        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/api/jobs/missing/ws?token=bad"):
            pass
        # Valid token on a job that does not exist is also refused.
        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/api/jobs/missing/ws?token=user:pass") as websocket:
            websocket.receive_json()


def test_run_job_task_ignores_missing_jobs() -> None:
    import asyncio

    from autoclip.server.routers.api import _run_job_task

    asyncio.run(_run_job_task("does-not-exist"))


def test_get_semaphore_is_reused_per_loop_and_rebuilt_on_a_new_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from autoclip.server.routers import api

    monkeypatch.setattr(api, "_sem", None)
    monkeypatch.setattr(api, "_sem_loop", None)

    async def _twice() -> tuple[asyncio.Semaphore, asyncio.Semaphore]:
        return api._get_semaphore(), api._get_semaphore()

    first, second = asyncio.run(_twice())
    assert first is second

    async def _again() -> asyncio.Semaphore:
        return api._get_semaphore()

    # A semaphore is bound to the loop that created it; a new loop must get a new one.
    assert asyncio.run(_again()) is not first


def test_job_websocket_handles_disconnect_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    from pathlib import Path
    from typing import cast

    from fastapi.websockets import WebSocket, WebSocketDisconnect

    from autoclip.server import progress as progress_mod
    from autoclip.server.routers import api
    from autoclip.server.state import Job, registry

    class _Socket:
        query_params: dict[str, str] = {}

        async def accept(self) -> None:
            return None

        async def close(self, code: int = 1000) -> None:
            return None

        async def send_text(self, text: str) -> None:
            count = getattr(self, "_sends", 0) + 1
            self._sends = count
            if count >= 2:
                raise WebSocketDisconnect()

        async def receive_text(self) -> str:
            return ""

    async def _run() -> None:
        job = Job(
            job_id="ws-disconnect",
            filename="clip.mp4",
            input_path=Path("/tmp/clip.mp4"),
            output_dir=Path("/tmp/wsout"),
            config={},
        )
        job.output_dir.mkdir(exist_ok=True)
        with registry._lock:
            registry._jobs[job.job_id] = job
        manager = progress_mod.ProgressManager()
        manager.bind_loop(asyncio.get_running_loop())
        # Patch the module constants/managers (restored by monkeypatch), never asyncio.sleep itself:
        # a global no-op sleep turns the server's eviction loop into an unyielding spin.
        monkeypatch.setattr(api, "KEEPALIVE_SECONDS", 0.0)
        monkeypatch.setattr(progress_mod, "progress_manager", manager)
        try:
            await api.job_websocket(cast(WebSocket, _Socket()), job.job_id)
        finally:
            with registry._lock:
                registry._jobs.pop(job.job_id, None)

    asyncio.run(_run())


def test_api_download_missing_job_returns_404(client: TestClient) -> None:
    response = client.get("/api/jobs/does-not-exist/download")
    assert response.status_code == 404


def test_api_delete_missing_job_returns_404(client: TestClient) -> None:
    response = client.delete("/api/jobs/does-not-exist")
    assert response.status_code == 404


def test_api_upload_invalid_config_file_returns_400(client: TestClient) -> None:
    import io

    response = client.post(
        "/api/jobs",
        files={
            "file": ("x.mp4", b"x", "video/mp4"),
            "config": ("bad.json", io.BytesIO(b"{bad"), "application/json"),
        },
    )
    assert response.status_code == 400
    assert "invalid config JSON" in response.text


def test_api_missing_zip_file_returns_500(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from autoclip.server.schemas import JobState
    from autoclip.server.state import Job, registry

    out = tmp_path / "missingzip"
    out.mkdir()
    job = Job(
        job_id="missingzip",
        filename="clip.mp4",
        input_path=out / "clip.mp4",
        output_dir=out,
        config={},
    )
    (out / "clip.mp4").write_text("x", encoding="utf-8")
    job.state = JobState.completed
    with registry._lock:
        registry._jobs[job.job_id] = job
    monkeypatch.setattr("autoclip.server.routers.api.zip_outputs", lambda _job: tmp_path / "definitely-missing.zip")
    response = client.get(f"/api/jobs/{job.job_id}/download")
    assert response.status_code == 500
    with registry._lock:
        registry._jobs.pop(job.job_id, None)


def test_api_websocket_keepalive_path_is_covered(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.server.routers import api
    from autoclip.server.state import Job, registry

    job = Job(
        job_id="keepalive-job",
        filename="clip.mp4",
        input_path=tmp_path / "clip.mp4",
        output_dir=tmp_path / "keepaliveout",
        config={},
    )
    job.output_dir.mkdir(exist_ok=True)
    with registry._lock:
        registry._jobs[job.job_id] = job

    # Shrink the keepalive tick instead of replacing asyncio.sleep globally: a global no-op sleep
    # would also turn the server's eviction loop into an unyielding spin.
    monkeypatch.setattr(api, "KEEPALIVE_SECONDS", 0.01)
    with client.websocket_connect(f"/api/jobs/{job.job_id}/ws") as websocket:
        assert websocket.receive_json()["job_id"] == job.job_id
        assert websocket.receive_json()["keepalive"] is True
    with registry._lock:
        registry._jobs.pop(job.job_id, None)


def test_delete_job_without_existing_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_USERNAME", "")
    monkeypatch.setenv("AUTH_PASSWORD", "")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MAX_FILE_SIZE", "1024")
    from autoclip.server import config as server_config

    server_config.get_settings.cache_clear()
    from fastapi.testclient import TestClient

    from autoclip.server.main import create_app
    from autoclip.server.state import Job, registry

    app = create_app()
    job = Job(
        job_id="missing-input-job",
        filename="clip.mp4",
        input_path=tmp_path / "definitely-missing.mp4",
        output_dir=tmp_path / "missinginputout",
        config={},
    )
    job.output_dir.mkdir(exist_ok=True)
    with registry._lock:
        registry._jobs[job.job_id] = job
    with TestClient(app) as client:
        response = client.delete(f"/api/jobs/{job.job_id}")
        assert response.status_code == 200
    with registry._lock:
        registry._jobs.pop(job.job_id, None)
