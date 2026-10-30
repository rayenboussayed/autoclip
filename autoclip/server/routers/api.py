from __future__ import annotations

import asyncio
import json
import secrets
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse, JSONResponse

from ...core.process import terminate_tree
from ..auth import require_auth
from ..config import get_settings
from ..schemas import JobCreated, JobState, JobStatus
from ..service import run_job, zip_outputs
from ..state import registry

router = APIRouter(prefix="/api", tags=["api"], dependencies=[])

_TASKS: set[asyncio.Task[Any]] = set()

# How long an idle websocket waits before a keepalive frame. A module constant (not a literal in the
# loop) so tests can shrink it instead of patching asyncio.sleep globally.
KEEPALIVE_SECONDS = 30.0


@router.post("/jobs", response_model=JobCreated, status_code=status.HTTP_201_CREATED)
async def create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),  # noqa: B008
    config_json: str | None = Form(default=None),
    config: UploadFile | None = File(default=None),  # noqa: B008
    user: str = Depends(require_auth),
) -> JobCreated:
    settings = get_settings()
    data = await file.read()
    if len(data) > settings.max_file_size:
        raise HTTPException(status_code=413, detail=f"file exceeds MAX_FILE_SIZE ({settings.max_file_size} bytes)")
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="empty upload")

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

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    job_dir = settings.output_dir / "jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    job = registry.create(
        filename=file.filename or "upload",
        input_path=Path(),  # set below
        output_dir=Path(),  # set below
        config={"_custom": custom} if custom else {},
    )
    job.input_path = job_dir / job.job_id / (file.filename or "upload.bin")
    job.output_dir = job_dir / job.job_id / "out"
    job.input_path.parent.mkdir(parents=True, exist_ok=True)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.input_path.write_bytes(data)

    settings = get_settings()
    background_tasks.add_task(_run_job_task, job.job_id)
    return JobCreated(
        job_id=job.job_id,
        status_url=f"/api/jobs/{job.job_id}",
        ws_url=f"/api/jobs/{job.job_id}/ws",
        download_url=f"/api/jobs/{job.job_id}/download",
    )


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
async def get_job(job_id: str, user: str = Depends(require_auth)) -> JobStatus:
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
        outputs=job.to_status()["outputs"],
        created_at=datetime.fromtimestamp(job.created_at),
        finished_at=datetime.fromtimestamp(job.finished_at) if job.finished_at else None,
    )


@router.get("/jobs/{job_id}/download")
async def download_job(job_id: str, user: str = Depends(require_auth)) -> FileResponse:
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
async def delete_job(job_id: str, user: str = Depends(require_auth)) -> JSONResponse:
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
