from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from ..core.config import Config, FileConfig, JsonValue, RuntimeOverrides, parse
from ..core.events import Progress
from ..core.process import pipeline_command
from .config import get_settings
from .progress import progress_manager
from .schemas import JobState
from .state import Job, job_zip_path

log = logging.getLogger("autoclip.server")


class _CancelEvent(Protocol):
    """The only cancellation surface ``run_job`` needs from an event object."""

    def is_set(self) -> bool: ...


def _custom_config(job: Job) -> dict[str, JsonValue] | None:
    """The user's file keys, if the job supplied any."""
    custom = job.config.get("_custom")
    return custom if isinstance(custom, dict) else None


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


def build_config(custom: dict[str, JsonValue] | None, input_path: Path, output_dir: Path) -> Config:
    """The runtime configuration of a job: the uploaded file config plus the server's own input/output.

    The server owns the input and the output folder, so a user config can never point the job at
    another file or write outside its own folder.
    """
    file_cfg = parse(FileConfig, dict(custom or {}))
    return file_cfg.to_config(RuntimeOverrides(input=[str(input_path)], output_dir=str(output_dir)))


def prepare_config(job: Job, custom: dict[str, JsonValue] | None) -> Config:
    """Validate settings, then build the job config; the server owns its input and output paths."""
    get_settings()
    return build_config(custom, job.input_path, job.output_dir)


def _file_config(job: Job, custom: dict[str, JsonValue] | None) -> dict[str, JsonValue]:
    """The nine-key config the child process reads (the input is the uploaded file)."""
    data = asdict(parse(FileConfig, dict(custom or {})))
    data["input"] = [str(job.input_path)]
    return data


def _collect_outputs(output_dir: Path) -> list[Path]:
    """Clip paths from every ``clips.json`` plus all uncaptioned derivatives under ``output_dir``."""
    outputs: list[Path] = []
    for index_path in sorted(output_dir.rglob("clips.json")):
        if ".work" in index_path.relative_to(output_dir).parts:
            continue
        try:
            rows = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            file_name = row if isinstance(row, str) else (row.get("file") if isinstance(row, dict) else None)
            if file_name:
                clip_path = index_path.parent / str(file_name)
                if clip_path.is_file():
                    outputs.append(clip_path)
    for derivative in sorted(output_dir.rglob("*_uncaptioned.mp4")):
        if ".work" not in derivative.relative_to(output_dir).parts:
            outputs.append(derivative)
    return outputs


def _run_job_inprocess(job: Job, cfg: Config) -> None:
    """Run one job in this process (the test/debug path) and publish its terminal state."""
    from ..core.batch import run_batch
    from ..core.events import Reporter
    from ..core.hardware import pick_encoder

    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.work_dir = job.output_dir / ".work" / job.input_path.stem
    job.work_dir.mkdir(parents=True, exist_ok=True)
    job.process = None

    def _sink(p: Progress) -> None:
        _publish(job, p.stage, p.message, p.percent)

    reporter = Reporter(_sink)
    _publish(job, "probing", "Probing media", 5.0)
    encoder = pick_encoder(cfg.video, cfg.quality, allow_gpu=True)
    _publish(job, "transcribing", "Transcription ready", 12.0)
    result = run_batch(cfg, downloader=None, encoder=encoder, log_file=None, report=reporter, config_dir=job.output_dir)
    # collect outputs
    _publish(job, "packaging", "Collecting outputs", 90.0)
    job.outputs = _collect_outputs(job.output_dir)
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


def run_job(job: Job, cancel_event: _CancelEvent | None = None) -> None:
    """Run one job (in a subprocess unless ``AUTOCLIP_SERVER_INPROCESS=1``) and publish its terminal state."""
    from ..core import ai

    ai.reset_run_breaker()  # the server is one process for many jobs; the breaker is per job, not forever
    try:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        job.work_dir = job.output_dir / ".work" / job.input_path.stem
        job.work_dir.mkdir(parents=True, exist_ok=True)
        _publish(job, "validating", "Validating configuration", 2.0)
        if cancel_event is not None and cancel_event.is_set():
            job.cancel_requested = True
        custom = _custom_config(job)
        cfg = prepare_config(job, custom)
        if job.cancel_requested:
            job.update(state=JobState.cancelled, error="cancelled")
            _publish(job, "cancelled", "Cancelled", 100.0, state="cancelled")
            return
        if os.environ.get("AUTOCLIP_SERVER_INPROCESS") == "1":
            return _run_job_inprocess(job, cfg)
        config_path = job.work_dir / "config.json"
        config_path.write_text(
            json.dumps(_file_config(job, custom), default=str, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # launch subprocess with NDJSON events; the output folder is a runtime override, not a file key
        cmd = [*pipeline_command(config_path), "--output-dir", str(job.output_dir), "--events", "jsonl"]
        env = dict(os.environ)
        if not getattr(sys, "frozen", False):
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
            elif ev.get("event") == "result" and isinstance(ev.get("failed"), int):
                result_failed = ev["failed"]
        rc = proc.wait()
        job.process = None
        if job.cancel_requested:
            cancelled = True
        crash_msg: str | None = None
        if rc < 0:
            if job.cancel_requested or cancelled:
                cancelled = True
            else:
                crash_msg = f"worker killed by signal {-rc}"
        _publish(job, "packaging", "Collecting outputs", 90.0)
        # collect outputs from clips.json rows, plus the always-exported raw derivatives
        job.outputs = _collect_outputs(job.output_dir)
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
        elif rc != 0:
            detail = " | ".join(stderr_tail[-20:]) if stderr_tail else f"worker exited with code {rc}"
            job.update(state=JobState.failed, error=detail)
            _publish(job, "failed", detail, 100.0, state="failed")
        elif failed_count > 0 and not job.outputs:
            detail = f"{failed_count} clip(s) failed"
            job.update(state=JobState.failed, error=detail)
            _publish(job, "failed", detail, 100.0, state="failed")
        else:
            job.update(error=f"{failed_count} clip(s) failed" if failed_count else None)
            if failed_count:
                _publish(job, "completed", f"Done with {failed_count} failure(s)", 100.0, state="completed")
            else:
                _publish(job, "completed", "Done", 100.0, state="completed")
    except Exception:
        log.exception("job %s failed", job.job_id)
        job.process = None
        job.update(state=JobState.failed, error="job failed")  # client sees no traceback; operator reads the log
        _publish(job, "failed", "job failed", 100.0, state="failed")


def zip_outputs(job: Job) -> Path:
    """Archive a finished job, mirroring its output folder (clips, sidecars and every ``clips.json``).

    The work folder is left out: it holds the database and the transcript cache, which a downloader
    does not need. Keeping the relative paths avoids two ``clips.json`` files (one per kind) colliding
    when the same source produced both a short and a long clip.
    """
    zip_path = job_zip_path(job.job_id)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(job.output_dir.rglob("*")):
            relative = path.relative_to(job.output_dir)
            if path.is_file() and ".work" not in relative.parts:
                try:
                    zf.write(path, arcname=str(relative))
                except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                    log.warning("could not add %s to job %s zip: %s", path, job.job_id, exc)
    return zip_path
