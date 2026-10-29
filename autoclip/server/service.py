from __future__ import annotations

import contextlib
import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Any

from ..core.config import Config, parse
from ..core.errors import AutoclipError
from ..core.hardware import pick_encoder
from ..core.media import probe
from ..core.models import RenderJob
from ..core.moments import energy_track, select_moments
from ..core.pipeline import needs_text, ranges_moments, split_video
from ..core.render import canvas_size, estimate_render_bytes
from ..core.resources import RESOURCES
from ..core.runner import render_all
from ..core.store import Cache, Store
from ..core.transcript import get_transcript
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
    if cfg.logging.file:
        job.output_dir.mkdir(parents=True, exist_ok=True)
    RESOURCES.configure(cfg)
    return cfg


def run_job(job: Job, cancel_event=None) -> None:  # type: ignore[no-untyped-def]
    try:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        job.work_dir = job.output_dir / ".work" / (job.input_path.stem)
        job.work_dir.mkdir(parents=True, exist_ok=True)
        _publish(job, "validating", "Validating configuration", 2.0)
        cfg = prepare_config(job, job.config.get("_custom"))
        store = Store(job.work_dir / "autoclip.db", fresh=True)
        cache = Cache(store, RESOURCES)
        try:
            _publish(job, "probing", "Probing media", 5.0)
            info = probe(job.input_path)
            log.info("server job %s: input %s (%.1fs)", job.job_id, job.input_path.name, info.duration)
            cc = cfg.clips
            if cc.mode == "split":
                out = job.output_dir / "long" / job.input_path.stem
                out.mkdir(parents=True, exist_ok=True)
                _publish(job, "rendering", "Splitting video", 50.0)
                parts = split_video(job.input_path, out, cc.split_seconds, info.duration, job.input_path.stem)
                (out / "clips.json").write_text(json.dumps([p.name for p in parts], indent=2), encoding="utf-8")
                job.outputs = parts
                _publish(job, "completed", "Done", 100.0, state="completed")
                return
            jobs_list = []
            needs = 0
            encoder = pick_encoder(cfg.video, allow_gpu=cfg.compute.device != "cpu")
            _publish(job, "transcribing", "Building energy track", 12.0)
            energy = energy_track(job.input_path, info, cache)
            have_text = get_transcript(
                job.input_path,
                cfg,
                job.work_dir,
                store,
                info,
                energy,
                needs_text(cfg),
                name=job.input_path.stem,
            )
            _publish(job, "transcribing", "Transcription ready", 35.0)
            moments_by_kind = {}
            for kind in cc.kinds:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("cancelled")
                kcfg = cc.kind(kind)
                _publish(job, "selecting", f"Selecting {kind} moments", 45.0)
                moments = (
                    ranges_moments(store, kcfg, info, cfg, have_text)
                    if cc.mode == "ranges"
                    else select_moments(job.input_path, info, store, cache, cfg, kind, energy, have_text)
                )
                if not moments:
                    continue
                out = job.output_dir / kind / job.input_path.stem
                out.mkdir(parents=True, exist_ok=True)
                size = canvas_size(
                    kcfg.aspect,
                    cfg.video.short_side,
                    info.width,
                    info.height,
                    upscale=cfg.video.upscale,
                    fill=cfg.video.layout == "fill",
                )
                numbers: dict[str, int] = {}
                render_jobs = []
                for m in moments:
                    n = numbers[m.scene] = numbers.get(m.scene, 0) + 1
                    dst = out / f"{job.input_path.stem}_{m.scene}_{n:02d}_{int(m.start)}s.{cfg.video.container}"
                    clip_id = store.save_moment(kind, m.scene, n, m)
                    render_jobs.append(
                        RenderJob(
                            clip_id, m, job.input_path, dst, kind, kcfg.aspect, encoder, info.has_audio, cfg, store.path, None, os.getpid()
                        )
                    )
                    needs = max(needs, estimate_render_bytes(size, m.end - m.start))
                moments_by_kind[kind] = (moments, render_jobs)
                jobs_list.extend(render_jobs)
            if not jobs_list:
                raise AutoclipError("no clips found for any requested kind")
            _publish(job, "rendering", "Rendering clips", 60.0)
            render_all(jobs_list, needs)
            _publish(job, "packaging", "Writing clip index", 90.0)
            rows = store.clips()
            keys = ("file", "scene", "start", "end", "title", "text", "score", "llm_score", "width", "height", "trimmed", "wall")
            for kind in dict.fromkeys(r["kind"] for r in rows):
                done = [r for r in rows if r["kind"] == kind and r["status"] == "done"]
                summary = [{k: (round(r[k], 2) if isinstance(r[k], float) else r[k]) for k in keys} for r in done]
                (job.output_dir / kind / job.input_path.stem / "clips.json").write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            job.outputs = [p for p in job.output_dir.rglob("*.mp4") if ".work" not in p.relative_to(job.output_dir).parts]
            failed = sum(r["status"] != "done" for r in rows)
            if failed:
                job.update(error=f"{failed} clip(s) failed")
                _publish(job, "completed", f"Done with {failed} failure(s)", 100.0, state="completed")
            else:
                _publish(job, "completed", "Done", 100.0, state="completed")
        finally:
            with contextlib.suppress(Exception):
                cache.close()
            with contextlib.suppress(Exception):
                store.close()
    except Exception as e:
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
