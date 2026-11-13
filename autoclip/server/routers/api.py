from __future__ import annotations

import asyncio
import json
import logging
import secrets
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse, JSONResponse

from ...core.errors import AutoclipError, ConfigError
from ...core.preview import preview_image
from ...core.process import terminate_tree
from ..auth import require_auth
from ..config import get_settings
from ..schemas import JobCreated, JobState, JobStatus, OutputFile
from ..service import build_config, run_job, zip_outputs
from ..state import registry

log = logging.getLogger("autoclip.server")

router = APIRouter(prefix="/api", tags=["api"], dependencies=[])

_TASKS: set[asyncio.Task[Any]] = set()

# How long an idle websocket waits before a keepalive frame. A module constant (not a literal in the
# loop) so tests can shrink it instead of patching asyncio.sleep globally.
KEEPALIVE_SECONDS = 30.0


async def _read_custom_config(config: UploadFile | None, config_json: str | None) -> dict[str, Any] | None:
    """The optional user config from the multipart form (a JSON file or a raw JSON string).

    ``config`` (the file) is read first, ``config_json`` (the string) overrides it — the same precedence
    for every endpoint that accepts one. Never-trusted keys (``input``/``output_dir``) are dropped.
    """
    custom: dict[str, Any] | None = None
    if config is not None:
        try:
            custom = json.loads((await config.read()).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=400, detail=f"invalid config JSON: {e}") from e
    if config_json:
        try:
            custom = json.loads(config_json)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"invalid config_json: {e}") from e
    if custom is not None and not isinstance(custom, dict):
        raise HTTPException(status_code=400, detail="config must be a JSON object")
    if custom is not None:
        custom.pop("input", None)
        custom.pop("output_dir", None)
    return custom


def _safe_filename(name: str | None) -> str:
    """The client's filename as a bare basename; never trusted as a path component."""
    filename = Path(name or "upload.bin").name
    return filename if filename not in ("", ".", "..") else "upload.bin"


@router.post("/jobs", response_model=JobCreated, status_code=status.HTTP_201_CREATED)
async def create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),  # noqa: B008
    config_json: str | None = Form(default=None),
    config: UploadFile | None = File(default=None),  # noqa: B008
    _user: str = Depends(require_auth),
) -> JobCreated:
    settings = get_settings()
    if registry.active_count() >= settings.max_pending_jobs:
        raise HTTPException(status_code=429, detail="too many pending jobs; wait for one to finish")
    if file.size is not None and file.size > settings.max_file_size:
        raise HTTPException(status_code=413, detail=f"file exceeds MAX_FILE_SIZE ({settings.max_file_size} bytes)")

    custom = await _read_custom_config(config, config_json)

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    job_dir = settings.output_dir / "jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    filename = _safe_filename(file.filename)
    job = registry.create(
        filename=filename,
        input_path=Path(),  # set below
        output_dir=Path(),  # set below
        config={"_custom": custom} if custom else {},
    )
    job.input_path = job_dir / job.job_id / filename
    job.output_dir = job_dir / job.job_id / "out"
    job.input_path.parent.mkdir(parents=True, exist_ok=True)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        size = await _spool_upload(file, job.input_path, settings.max_file_size)
    except _UploadTooLarge as e:
        shutil.rmtree(job_dir / job.job_id, ignore_errors=True)
        registry.remove(job.job_id)
        raise HTTPException(status_code=413, detail=f"file exceeds MAX_FILE_SIZE ({settings.max_file_size} bytes)") from e
    if size == 0:
        shutil.rmtree(job_dir / job.job_id, ignore_errors=True)
        registry.remove(job.job_id)
        raise HTTPException(status_code=400, detail="empty upload")

    background_tasks.add_task(_run_job_task, job.job_id)
    return JobCreated(
        job_id=job.job_id,
        status_url=f"/api/jobs/{job.job_id}",
        ws_url=f"/api/jobs/{job.job_id}/ws",
        download_url=f"/api/jobs/{job.job_id}/download",
    )


@router.post("/preview")
async def preview(
    file: UploadFile = File(...),  # noqa: B008
    kind: str = Form(default="short"),
    config_json: str | None = Form(default=None),
    config: UploadFile | None = File(default=None),  # noqa: B008
    _user: str = Depends(require_auth),
) -> Response:
    """A PNG of one representative frame with the configured layout and captions (mock caption text).

    Nothing is persisted and no job is created: the upload is written to a throwaway folder under the
    cache dir, one frame is rendered and the folder is deleted before the response is returned.
    """
    settings = get_settings()
    if kind not in ("short", "long"):
        raise HTTPException(status_code=400, detail="kind must be 'short' or 'long'")
    data = await file.read()
    if len(data) > settings.max_file_size:
        raise HTTPException(status_code=413, detail=f"file exceeds MAX_FILE_SIZE ({settings.max_file_size} bytes)")
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="empty upload")

    custom = await _read_custom_config(config, config_json)
    root = settings.cache_dir / "preview" / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    try:
        src = root / _safe_filename(file.filename)
        src.write_bytes(data)
        try:
            cfg = build_config(custom, src, root / "out")
        except ConfigError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        try:
            png = await asyncio.to_thread(preview_image, src, cfg, kind)
        except AutoclipError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            log.warning("preview failed: %s: %s", type(e).__name__, e)
            raise HTTPException(status_code=500, detail="preview failed") from e
        return Response(content=png, media_type="image/png", headers={"Cache-Control": "no-store"})
    finally:
        shutil.rmtree(root, ignore_errors=True)


async def _run_job_task(job_id: str) -> None:
    settings = get_settings()
    job = registry.get(job_id)
    if job is None:
        return
    sem = _get_semaphore()
    async with sem:
        if job.cancel_requested:
            job.update(state=JobState.cancelled, error="cancelled")
            return
        job.state = JobState.running
        job.step = "queued"
        job.message = "Starting"
        loop = asyncio.get_event_loop()
        from ..progress import progress_manager

        progress_manager.bind_loop(loop)
        await asyncio.to_thread(run_job, job, None)
        await asyncio.to_thread(registry.evict_expired, settings.job_ttl_seconds)


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str, _user: str = Depends(require_auth)) -> JobStatus:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    return JobStatus(
        job_id=job.job_id,
        state=job.state,
        step=job.step,
        message=job.message,
        percent=job.percent,
        error=job.error,
        outputs=[OutputFile(name=p.name, size_bytes=p.stat().st_size if p.is_file() else 0) for p in job.outputs],
        created_at=datetime.fromtimestamp(job.created_at),
        finished_at=datetime.fromtimestamp(job.finished_at) if job.finished_at else None,
    )


@router.get("/jobs/{job_id}/download")
async def download_job(job_id: str, _user: str = Depends(require_auth)) -> FileResponse:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.state != JobState.completed:
        raise HTTPException(status_code=409, detail="job not completed")
    zip_path = zip_outputs(job)
    if not zip_path.is_file():
        raise HTTPException(status_code=500, detail="output archive missing")
    return FileResponse(zip_path, media_type="application/zip", filename=f"{job_id}.zip")


@router.websocket("/jobs/{job_id}/ws")
async def job_websocket(websocket: WebSocket, job_id: str) -> None:
    # Auth for WS is best-effort via query param token (?token=) when enabled.
    settings = get_settings()
    if settings.auth_enabled:
        token = websocket.query_params.get("token", "")
        expected = f"{settings.auth_username}:{settings.auth_password}"
        # Browsers cannot set an Authorization header on a WebSocket, so the same credentials arrive
        # as a query token; compare it the same constant-time way as the Basic header.
        if not secrets.compare_digest(token, expected):
            await websocket.close(code=1008)
            return
    job = registry.get(job_id)
    if job is None:
        await websocket.close(code=1008)
        return
    from ..progress import progress_manager

    await progress_manager.register(job_id, websocket)
    try:
        # send current state immediately
        await websocket.send_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "step": job.step,
                    "message": job.message,
                    "percent": job.percent,
                    "state": job.state.value,
                }
            )
        )
        while True:
            await asyncio.sleep(KEEPALIVE_SECONDS)
            await websocket.send_text(json.dumps({"job_id": job_id, "keepalive": True}))
    except WebSocketDisconnect:
        pass
    finally:
        await progress_manager.unregister(job_id, websocket)


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str, _user: str = Depends(require_auth)) -> JSONResponse:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.process is not None:
        job.cancel_requested = True
        await asyncio.to_thread(terminate_tree, job.process)
        job.process = None
    # give the worker a moment to release handles before removing the folder
    await asyncio.sleep(0.2)
    shutil.rmtree(job.output_dir, ignore_errors=True)
    if job.input_path.exists():
        job.input_path.unlink(missing_ok=True)
    registry.remove(job_id)
    return JSONResponse({"deleted": job_id})


_sem: asyncio.Semaphore | None = None
_sem_loop: asyncio.AbstractEventLoop | None = None


def _get_semaphore() -> asyncio.Semaphore:
    # A Semaphore belongs to the loop that first waits on it. Tests (and a restarted loop) would
    # otherwise reuse one bound to a closed loop, so recreate it whenever the running loop changes.
    global _sem, _sem_loop  # noqa: PLW0603
    loop = asyncio.get_running_loop()
    if _sem is None or _sem_loop is not loop:
        settings = get_settings()
        _sem = asyncio.Semaphore(settings.max_concurrent_jobs)
        _sem_loop = loop
    return _sem
