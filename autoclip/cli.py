"""Command line: ``python -m autoclip config.json`` - one pipeline run per video listed in the config."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .config import Config, KindCfg, load_config
from .errors import AutoclipError, ConfigError
from .hardware import pick_encoder
from .log import setup_logging
from .media import ffmpeg_exe, probe
from .models import Encoder, MediaInfo, Moment, RenderJob
from .moments import energy_track, make_title, select_moments
from .render import canvas_size, estimate_render_bytes
from .resources import MB, RESOURCES
from .runner import render_all
from .store import Cache, Store
from .transcript import get_transcript

log = logging.getLogger("autoclip")


def parse_hms(text: str) -> float:
    """``[[HH:]MM:]SS[.ms]`` -> seconds."""
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def split_video(src: Path, out_dir: Path, seconds: float, duration: float) -> list[Path]:
    """Cut ``src`` into consecutive parts of ``seconds`` with ffmpeg stream copy (instant, no re-encode)."""
    parts = []
    for k in range(math.ceil(duration / seconds)):
        dst = out_dir / f"{src.stem}_part{k + 1:03d}.mp4"
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


def process_video(src: Path, cfg: Config, log_file: Path | None, encoder: Encoder) -> tuple[int, int]:
    """Run the pipeline for one video. Returns ``(clips_done, clips_failed)``.

    Clips go to ``<output_dir>/<short|long>/<video name>/``; the database and transcript cache live in
    ``<output_dir>/.work/<video name>/`` (the database is recreated empty each time).
    """
    root, cc = Path(cfg.output_dir), cfg.clips
    info = probe(src)
    log.info("input: %s (%.1fs) %dx%d @ %.2ffps, audio: %s", src.name, info.duration, info.width, info.height, info.fps, info.has_audio)
    if cc.mode == "split":
        out = root / "long" / src.stem
        out.mkdir(parents=True, exist_ok=True)
        parts = split_video(src, out, cc.split_seconds, info.duration)
        (out / "clips.json").write_text(json.dumps([p.name for p in parts], indent=2), encoding="utf-8")
        return len(parts), 0

    work = root / ".work" / src.stem
    work.mkdir(parents=True, exist_ok=True)
    store = Store(work / "autoclip.db", fresh=True)
    cache = Cache(store, RESOURCES)
    try:
        db = energy_track(src, info, cache)
        have_text = get_transcript(src, cfg, work, store, info, db, cfg.captions.enabled or cc.mode == "highlights" or cfg.title.enabled)
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
                log.warning("no %s clips for %s (video shorter than the length range, or no ranges given)", kind, src.name)
                continue
            out = root / kind / src.stem
            out.mkdir(parents=True, exist_ok=True)
            size = canvas_size(kcfg.aspect, cfg.video.short_side, info.width, info.height)
            numbers: dict[str, int] = {}
            for m in moments:
                n = numbers[m.scene] = numbers.get(m.scene, 0) + 1
                dst = out / f"{src.stem}_{m.scene}_{n:02d}_{int(m.start)}s.mp4"
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
        (root / kind / src.stem / "clips.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in rows:
        if r["status"] != "done":
            log.warning("%s clip %s #%d failed: %s", r["kind"], r["scene"], r["n"], r["error"] or "did not finish")
    return sum(r["status"] == "done" for r in rows), sum(r["status"] != "done" for r in rows)


def init_config(path: Path, full: bool) -> None:
    """Write a starter (or complete) config file; never overwrites an existing one."""
    if path.exists():
        raise ConfigError(f"{path} already exists (not overwriting)")
    starter = {
        "input": ["video.mp4"],
        "clips": {
            "kinds": ["short", "long"],
            "scenes": ["best"],
            "short": {"count": 3, "min_seconds": 20, "max_seconds": 60, "aspect": "9:16"},
            "long": {"count": 1, "min_seconds": 180, "max_seconds": 600, "aspect": "16:9"},
        },
    }
    data = asdict(Config()) if full else starter
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Exit status: 0 all clips rendered, 1 some video or clip failed, 2 configuration error."""
    ap = argparse.ArgumentParser(
        prog="autoclip", description="Cut long videos into the best short clips (free, local). All options live in the config file."
    )
    ap.add_argument("config", nargs="?", default="config.json", type=Path, help="path to the JSON config (default: ./config.json)")
    ap.add_argument("--init", action="store_true", help="write a tiny starter config and exit")
    ap.add_argument("--init-full", action="store_true", help="write a config with every option and its default, and exit")
    args = ap.parse_args(argv)
    try:
        if args.init or args.init_full:
            init_config(args.config, args.init_full)
            return 0
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    lg = cfg.logging
    log_file = (
        (Path(lg.file) if lg.file else Path(cfg.output_dir) / "logs" / f"autoclip_{datetime.now():%Y%m%d_%H%M%S}.log")
        if lg.enabled
        else None
    )
    setup_logging(log_file, lg.level, lg.console_level)
    RESOURCES.configure(cfg)
    snap = RESOURCES.snapshot()
    log.info(
        "autoclip start | python %s on %s | cpus=%d ram=%.0fMB free=%.0fMB | %d video(s) | log: %s",
        platform.python_version(),
        platform.system(),
        RESOURCES.cpus,
        snap.mem_total / MB,
        snap.mem_available / MB,
        len(cfg.input),
        log_file,
    )
    encoder = pick_encoder(cfg.video, allow_gpu=cfg.compute.device != "cpu")
    t0, done, failed = time.perf_counter(), 0, 0
    for n, name in enumerate(cfg.input, 1):
        log.info("=== video %d/%d: %s ===", n, len(cfg.input), Path(name).name)
        try:
            ok, bad = process_video(Path(name), cfg, log_file, encoder)
        except Exception as e:  # one broken video must not stop the batch
            log.error("%s failed: %s: %s", Path(name).name, type(e).__name__, e, exc_info=not isinstance(e, AutoclipError))
            ok, bad = 0, 1
        done, failed = done + ok, failed + bad
    log.info(
        "finished: %d clip(s) done, %d failed in %.1fs (peak rss %.0fMB)", done, failed, time.perf_counter() - t0, RESOURCES.peak_rss / MB
    )
    return 1 if failed else 0
