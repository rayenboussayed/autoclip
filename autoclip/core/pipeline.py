"""The pipeline for one video: probe, transcribe, pick moments, render the clips.

Both front ends (the command line and the desktop app) call :func:`process_video`; it owns the
whole per-video flow and never raises for a single broken clip.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
from pathlib import Path

from .config import Config, KindCfg, parse_hms
from .errors import AutoclipError
from .media import ffmpeg_exe, probe
from .models import Encoder, MediaInfo, Moment, RenderJob
from .moments import energy_track, make_title, select_moments
from .render import canvas_size, estimate_render_bytes
from .resources import RESOURCES
from .runner import render_all
from .store import Cache, Store
from .transcript import get_transcript

log = logging.getLogger("autoclip")


def split_video(src: Path, out_dir: Path, seconds: float, duration: float, name: str | None = None) -> list[Path]:
    """Cut ``src`` into consecutive parts of ``seconds`` with ffmpeg stream copy (instant, no re-encode)."""
    parts = []
    for k in range(math.ceil(duration / seconds)):
        dst = out_dir / f"{name or src.stem}_part{k + 1:03d}.mp4"
        cmd = [
            ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-ss",
            f"{k * seconds:.3f}",
            "-i",
            str(src),
            "-t",
            f"{seconds:.3f}",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(dst),
        ]
        subprocess.run(cmd, check=True)
        parts.append(dst)
    return parts


def ranges_moments(store: Store, kcfg: KindCfg, info: MediaInfo, cfg: Config, have_text: bool) -> list[Moment]:
    """Moments for the explicit ``ranges`` of a clip kind (scene type "custom"), clamped to the video."""
    out = []
    for span in kcfg.ranges:
        lo, hi = (min(parse_hms(x), info.duration) for x in span.split("-"))
        text = store.window_text(lo, hi) if have_text else ""
        out.append(Moment(lo, hi, text, scene="custom", title=make_title(text, cfg)))
    return out


def needs_text(cfg: Config) -> bool:
    """Whether the run needs a transcript: highlights select on text, captions and titles read it."""
    return cfg.clips.mode == "highlights" or (cfg.captions.enabled and not cfg.skip.caption) or (cfg.title.enabled and not cfg.skip.title)


def process_video(
    src: Path,
    cfg: Config,
    log_file: Path | None,
    encoder: Encoder,
    name: str | None = None,
    subtitle: Path | None = None,
    subtitle_manual: bool = False,
) -> tuple[int, int]:
    """Run the pipeline for one video. Returns ``(clips_done, clips_failed)``.

    Clips go to ``<output_dir>/<short|long>/<video name>/``; the database and transcript cache live in
    ``<output_dir>/.work/<video name>/`` (the database is recreated empty each time). ``name`` renames a
    downloaded original (``<title> [<id>]``) without touching the file; ``subtitle`` is the SRT the
    downloader picked for it.
    """
    root, cc, stem = Path(cfg.output_dir), cfg.clips, name or src.stem
    info = probe(src)
    log.info("input: %s (%.1fs) %dx%d @ %.2ffps, audio: %s", src.name, info.duration, info.width, info.height, info.fps, info.has_audio)
    if cc.mode == "split":
        out = root / "long" / stem
        out.mkdir(parents=True, exist_ok=True)
        parts = split_video(src, out, cc.split_seconds, info.duration, stem)
        (out / "clips.json").write_text(json.dumps([p.name for p in parts], indent=2), encoding="utf-8")
        return len(parts), 0

    work = root / ".work" / stem
    work.mkdir(parents=True, exist_ok=True)
    store = Store(work / "autoclip.db", fresh=True)
    cache = Cache(store, RESOURCES)
    try:
        db = energy_track(src, info, cache)
        have_text = get_transcript(
            src,
            cfg,
            work,
            store,
            info,
            db,
            needs_text(cfg),
            subtitle=subtitle,
            subtitle_manual=subtitle_manual,
            name=stem,
        )
        jobs: list[RenderJob] = []
        needs = 0
        for kind in cc.kinds:
            kcfg = cc.kind(kind)
            moments = (
                ranges_moments(store, kcfg, info, cfg, have_text)
                if cc.mode == "ranges"
                else select_moments(src, info, store, cache, cfg, kind, db, have_text)
            )
            if not moments:
                log.warning("no %s clips for %s (video shorter than the length range, or no ranges given)", kind, name or src.name)
                continue
            out = root / kind / stem
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
            for m in moments:
                n = numbers[m.scene] = numbers.get(m.scene, 0) + 1
                dst = out / f"{stem}_{m.scene}_{n:02d}_{int(m.start)}s.{cfg.video.container}"
                clip_id = store.save_moment(kind, m.scene, n, m)
                jobs.append(
                    RenderJob(clip_id, m, src, dst, kind, kcfg.aspect, encoder, info.has_audio, cfg, store.path, log_file, os.getpid())
                )
                needs = max(needs, estimate_render_bytes(size, m.end - m.start))
        if not jobs:
            raise AutoclipError("no clips found for any requested kind (is the video shorter than the length ranges?)")
        render_all(jobs, needs)
        rows = store.clips()
    finally:
        cache.close()
        store.close()
    keys = ("file", "scene", "start", "end", "title", "text", "score", "llm_score", "width", "height", "trimmed", "wall")
    for kind in dict.fromkeys(r["kind"] for r in rows):
        done = [r for r in rows if r["kind"] == kind and r["status"] == "done"]
        summary = [{k: round(r[k], 2) if isinstance(r[k], float) else r[k] for k in keys} for r in done]
        (root / kind / stem / "clips.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in rows:
        if r["status"] != "done":
            log.warning("%s clip %s #%d failed: %s", r["kind"], r["scene"], r["n"], r["error"] or "did not finish")
    return sum(r["status"] == "done" for r in rows), sum(r["status"] != "done" for r in rows)
