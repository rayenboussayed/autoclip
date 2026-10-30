from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

from ..core.config import Config, parse
from ..core.process import pipeline_command
from .config import get_settings
from .progress import progress_manager
from .schemas import JobState
from .state import Job

log = logging.getLogger("autoclip.server")


def _publish(job: Job, step: str, message: str, percent: float, state: str | None = None) -> None:
    job.update(step=step, message=message, percent=percent, state=JobState(state) if state else None)
    progress_manager.publish(
        job.job_id,
        {
            "job_id": job.job_id,
            "step": step,
            "message": message,
            "percent": percent,
            "state": job.state.value,
        },
    )


def prepare_config(job: Job, custom: dict[str, Any] | None) -> Config:
    get_settings()
    base: dict[str, Any] = {}
    if custom:
        base.update(custom)
    # server jobs are always about the uploaded file
    base.pop("input", None)
    base.pop("output_dir", None)
    cfg = parse(Config, base) if base else Config()
    cfg.input = [str(job.input_path)]
    cfg.output_dir = str(job.output_dir)
    return cfg


def _run_job_inprocess(job: Job, cfg: Config) -> None:
    from ..core.batch import run_batch
    from ..core.events import Reporter
    from ..core.hardware import pick_encoder

    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.work_dir = job.output_dir / ".work" / (job.input_path.stem)
    job.work_dir.mkdir(parents=True, exist_ok=True)
    job.process = None

    def _sink(p: Any) -> None:
        _publish(job, p.stage, p.message, p.percent)

    reporter = Reporter(_sink)
    _publish(job, "probing", "Probing media", 5.0)
    encoder = pick_encoder(cfg.video, allow_gpu=cfg.compute.device != "cpu")
    _publish(job, "transcribing", "Transcription ready", 12.0)
    result = run_batch(cfg, downloader=None, encoder=encoder, log_file=None, report=reporter)
    # collect outputs
    _publish(job, "packaging", "Collecting outputs", 90.0)
    outs: list[Path] = []
    for cj in sorted(job.output_dir.rglob("clips.json")):
        if ".work" in cj.relative_to(job.output_dir).parts:
            continue
        try:
            rows = json.loads(cj.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(rows, list):
            for r in rows:
                name = r if isinstance(r, str) else (r.get("file") if isinstance(r, dict) else None)
                if name:
                    p = cj.parent / str(name)
                    if p.is_file():
                        outs.append(p)
    job.outputs = outs
    any_index = any(p.name == "clips.json" for p in job.output_dir.rglob("*"))
    if result.clips_failed and result.clips_done == 0 and not any_index:
        job.update(state=JobState.failed, error=f"{result.clips_failed} clip(s) failed")
        _publish(job, "failed", f"{result.clips_failed} clip(s) failed", 100.0, state="failed")
    elif result.clips_failed:
        job.update(state=JobState.completed, error=f"{result.clips_failed} clip(s) failed")
        _publish(job, "completed", f"Done with {result.clips_failed} failure(s)", 100.0, state="completed")
    else:
        job.update(state=JobState.completed)
        _publish(job, "completed", "Done", 100.0, state="completed")


def run_job(job: Job, cancel_event=None) -> None:  # type: ignore[no-untyped-def]
    try:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        job.work_dir = job.output_dir / ".work" / (job.input_path.stem)
        job.work_dir.mkdir(parents=True, exist_ok=True)
        _publish(job, "validating", "Validating configuration", 2.0)
        if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
            job.cancel_requested = True
        cfg = prepare_config(job, job.config.get("_custom"))
        if job.cancel_requested:
            job.update(state=JobState.cancelled, error="cancelled")
            _publish(job, "cancelled", "Cancelled", 100.0, state="cancelled")
            return
        if os.environ.get("AUTOCLIP_SERVER_INPROCESS") == "1":
            return _run_job_inprocess(job, cfg)
        config_path = job.work_dir / "config.json"
        try:
            from dataclasses import asdict

            config_path.write_text(json.dumps(asdict(cfg), default=str, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            if hasattr(cfg, "model_dump"):
                config_path.write_text(json.dumps(cfg.model_dump(), default=str, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                raise

        job.output_dir.mkdir(parents=True, exist_ok=True)
        # launch subprocess with NDJSON events
        cmd = [*pipeline_command(config_path), "--events", "jsonl"]
        env = dict(os.environ)
        if not getattr(sys, "frozen", False):
            from pathlib import Path

            project_root = Path(__file__).resolve().parents[2]
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(project_root), env.get("PYTHONPATH", "")]))
        proc = subprocess.Popen(
            cmd,
            cwd=str(config_path.parent),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=os.name != "nt",
        )
        job.process = proc
        cancelled = False
        result_failed: int | None = None
        stderr_tail: list[str] = []
        for raw_line in proc.stdout or []:
            line = raw_line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                if len(stderr_tail) >= 20:
                    del stderr_tail[0]
                stderr_tail.append(line)
                continue
            if not isinstance(ev, dict):
                continue
            if ev.get("event") == "progress":
                step = str(ev.get("stage", job.step))
                pct = float(ev.get("percent", job.percent))
                msg = str(ev.get("message", ""))
                _publish(job, step, msg, pct)
            elif ev.get("event") == "result":
                if isinstance(ev.get("failed"), int):
                    result_failed = ev["failed"]
        rc = proc.wait()
        job.process = None
        if job.cancel_requested:
            cancelled = True
        crash_msg: str | None = None
        if rc is not None and rc < 0:
            if job.cancel_requested or cancelled:
                cancelled = True
            else:
                crash_msg = f"worker killed by signal {-rc}"
        _publish(job, "packaging", "Collecting outputs", 90.0)
        # collect outputs from clips.json rows
        outs: list[Path] = []
        for cj in sorted(job.output_dir.rglob("clips.json")):
            if ".work" in cj.relative_to(job.output_dir).parts:
                continue
            try:
                rows = json.loads(cj.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(rows, list):
                for r in rows:
                    name = r if isinstance(r, str) else (r.get("file") if isinstance(r, dict) else None)
                    if not name:
                        continue
                    p = cj.parent / str(name)
                    if p.is_file():
                        outs.append(p)
        job.outputs = outs
        failed_count = result_failed if result_failed is not None else 0
        if cancelled:
            job.update(state=JobState.cancelled, error="cancelled")
            _publish(job, "cancelled", "Cancelled", 100.0, state="cancelled")
        elif crash_msg is not None:
            detail = crash_msg
            if stderr_tail:
                detail = crash_msg + ": " + " | ".join(stderr_tail[-5:])
            job.update(state=JobState.failed, error=detail)
            _publish(job, "failed", detail, 100.0, state="failed")
        elif rc not in (0, None):
            detail = " | ".join(stderr_tail[-20:]) if stderr_tail else f"worker exited with code {rc}"
            job.update(state=JobState.failed, error=detail)
            _publish(job, "failed", detail, 100.0, state="failed")
        elif failed_count > 0 and not outs:
            detail = f"{failed_count} clip(s) failed"
            job.update(state=JobState.failed, error=detail)
            _publish(job, "failed", detail, 100.0, state="failed")
        else:
            job.update(error=f"{failed_count} clip(s) failed" if failed_count else None)
            if failed_count:
                _publish(job, "completed", f"Done with {failed_count} failure(s)", 100.0, state="completed")
            else:
                _publish(job, "completed", "Done", 100.0, state="completed")
    except Exception as e:
        job.process = None
        job.update(state=JobState.failed, error=str(e))
        _publish(job, "failed", str(e), 100.0, state="failed")
        log.exception("server job %s failed", job.job_id)


def zip_outputs(job: Job) -> Path:
    """Archive a finished job, mirroring its output folder (clips, sidecars and every ``clips.json``).

    The work folder is left out: it holds the database and the transcript cache, which a downloader
    does not need. Keeping the relative paths avoids two ``clips.json`` files (one per kind) colliding
    when the same source produced both a short and a long clip.
    """
    settings = get_settings()
    zip_path = settings.cache_dir / f"{job.job_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(job.output_dir.rglob("*")):
            relative = path.relative_to(job.output_dir)
            if path.is_file() and ".work" not in relative.parts:
                zf.write(path, arcname=str(relative))
    return zip_path
