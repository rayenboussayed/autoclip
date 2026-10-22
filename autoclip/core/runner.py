"""Running render jobs: one clip per job, in this process or in a pool whose size follows free CPU and RAM."""

from __future__ import annotations

import gc
import logging
import multiprocessing
import os
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace

from .hardware import HW_MAX_WORKERS
from .log import setup_logging, traced
from .models import RenderJob
from .render import render_clip
from .resources import RESOURCES
from .store import Cache, Store

log = logging.getLogger("autoclip")


def render_job(job: RenderJob) -> bool:
    """Render one clip and record the outcome in SQLite. Never raises (a failed clip must not stop the batch)."""
    if os.getpid() != job.parent_pid:  # fresh interpreter: rebuild logging and the resource monitor
        setup_logging(job.log_file, job.cfg.logging.level, None)
        RESOURCES.configure(job.cfg)
    t0 = time.perf_counter()
    store = Store(job.db_path)
    cache = Cache(store, RESOURCES)
    vclip = None
    m = job.moment
    try:
        from moviepy import VideoFileClip

        log.info("[%d] rendering %s (%.1f-%.1fs, %d threads)", job.clip_id, job.dst.name, m.start, m.end, job.threads)
        vclip = VideoFileClip(str(job.src), audio=False)
        size, lines = render_clip(vclip, job, store, cache)
        if lines:
            blocks = [f"{k}\n{_srt_time(c.start)} --> {_srt_time(c.end)}\n{c.text}\n" for k, c in enumerate(lines, 1)]
            job.dst.with_suffix(".srt").write_text("\n".join(blocks), encoding="utf-8")
        wall = time.perf_counter() - t0
        store.finish_clip(job.clip_id, ok=True, file=job.dst.name, size=size, trimmed=m.trimmed, wall=wall)
        log.info("[%d] done in %.1fs: %.1fs -> %.1fs after trimming", job.clip_id, wall, m.end - m.start, m.trimmed)
        return True
    except Exception as e:
        log.exception("[%d] render failed", job.clip_id)
        store.finish_clip(job.clip_id, ok=False, file=job.dst.name, wall=time.perf_counter() - t0, error=f"{type(e).__name__}: {e}")
        return False
    finally:
        if vclip is not None:
            vclip.close()
        cache.close()
        store.close()
        gc.collect()


def _srt_time(t: float) -> str:
    """Seconds -> SubRip ``HH:MM:SS,mmm``."""
    ms = round(max(t, 0.0) * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


@traced("render_all")
def render_all(jobs: list[RenderJob], est_bytes: int) -> None:
    """Run the jobs with adaptive parallelism; outcomes end up in the ``clips`` table.

    Before *every* submission the plan is recomputed from the CPU budget and the RAM free at that moment, so
    concurrency grows when memory is plentiful and shrinks when it gets tight. With one usable core (or one
    job) everything runs in-process. A worker killed by the OS (usually out of memory) is retried alone.
    """
    if not jobs:
        return
    pending = deque(jobs)
    limit = HW_MAX_WORKERS if jobs[0].encoder.hardware else 1 << 30  # GPUs allow only a few encode sessions
    plan = RESOURCES.plan_workers(est_bytes, len(jobs), limit)
    capacity = min(RESOURCES.cpu_budget(), RESOURCES.res.max_workers or RESOURCES.cpu_budget(), len(jobs), limit)
    log.info("parallel plan: up to %d worker(s) x %d thread(s) for %d clip(s)", plan.workers, plan.threads, len(jobs))
    if capacity <= 1:
        for job in jobs:
            render_job(replace(job, threads=RESOURCES.cpu_budget()))
        return
    running: dict[Future[bool], RenderJob] = {}
    retry: list[RenderJob] = []
    with ProcessPoolExecutor(max_workers=capacity, mp_context=multiprocessing.get_context("spawn")) as pool:
        while pending or running:
            plan = RESOURCES.plan_workers(est_bytes, len(pending) + len(running), limit)
            while pending and len(running) < plan.workers:
                job = replace(pending.popleft(), threads=plan.threads)
                log.info(
                    "submit clip %d (running=%d allowed=%d threads=%d | %s)",
                    job.clip_id,
                    len(running) + 1,
                    plan.workers,
                    job.threads,
                    RESOURCES.brief(),
                )
                running[pool.submit(render_job, job)] = job
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                job = running.pop(future)
                try:
                    future.result()
                except BrokenProcessPool:
                    log.warning("worker for clip %d died (probably out of memory): retrying it alone", job.clip_id)
                    retry.append(job)
            if retry and not running:
                break
    for job in retry + list(pending):
        gc.collect()
        render_job(replace(job, threads=RESOURCES.cpu_budget()))
