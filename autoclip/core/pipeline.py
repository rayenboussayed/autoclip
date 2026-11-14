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
from collections.abc import Sequence
from pathlib import Path

from .config import Config, KindCfg, parse_hms
from .errors import AutoclipError
from .events import Cancelled, CancelToken, Reporter
from .internal.automation import SPLIT_SECONDS
from .internal.quality import quality_short_side
from .log import health_check, safe_path_text, traced
from .media import external_env, ffmpeg_exe, probe
from .models import Encoder, MediaInfo, Moment, RenderJob
from .moments import energy_track, make_title, select_moments
from .render import canvas_size, estimate_render_bytes
from .resources import RESOURCES
from .runner import render_all
from .store import Cache, ClipRow, Store
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
        subprocess.run(cmd, check=True, env=external_env())
        parts.append(dst)
    return parts


def ranges_moments(
    store: Store, kcfg: KindCfg, info: MediaInfo, cfg: Config, have_text: bool, fillers: frozenset[str] | None = None, kind: str = "short"
) -> list[Moment]:
    """Moments for the explicit ``ranges`` of a clip kind (scene type "custom"), clamped to the video."""
    if fillers is None:
        fillers = frozenset()
    from . import ai

    use_llm = not cfg.skip.llm and ai.llm_available(cfg.llm.base_url, cfg.llm.model)
    out = []
    for span in kcfg.ranges:
        lo, hi = (min(parse_hms(x), info.duration) for x in span.split("-"))
        text = store.window_text(lo, hi) if have_text else ""
        title = make_title(text, fillers) if kind == "long" else ""
        caption = ""
        if use_llm and text:
            try:
                from . import ai

                title = ai.generate_title(cfg, text) or title
                caption = ai.generate_caption(cfg, text)
            except Exception as e:
                log.warning("AI generation failed (%s: %s); keeping the heuristic title", type(e).__name__, e)
        out.append(Moment(lo, hi, text, scene="custom", title=title, caption=caption))
    return out


def needs_text(cfg: Config) -> bool:
    """Whether the run needs a transcript: highlights select on text, captions and titles read it."""
    return cfg.clips.mode == "highlights" or cfg.captions.enabled or not cfg.skip.title


def _number(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None


def _write_clip_summaries(root: Path, stem: str, rows: Sequence[ClipRow]) -> None:
    """Write ``clips.json`` for every kind present, from the rows the database holds right now (done only).

    Also used on the cancellation path: a partial run still leaves an index for what finished.
    """
    for clip_kind in dict.fromkeys(r["kind"] for r in rows):
        done = [r for r in rows if r["kind"] == clip_kind and r["status"] == "done"]
        summary = [
            {
                "file": r["file"],
                "scene": r["scene"],
                "start": round(r["start"], 2),
                "end": round(r["end"], 2),
                "title": r["title"],
                "caption": r["caption"],
                "text": r["text"],
                "score": _number(r["score"]),
                "llm_score": _number(r["llm_score"]),
                "width": r["width"],
                "height": r["height"],
                "trimmed": round(r["trimmed"], 2),
                "wall": round(r["wall"], 2),
            }
            for r in done
        ]
        (root / clip_kind / stem / "clips.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


@traced
def process_video(
    src: Path,
    cfg: Config,
    log_file: Path | None,
    encoder: Encoder,
    name: str | None = None,
    subtitle: Path | None = None,
    subtitle_manual: bool = False,
    config_dir: Path | None = None,
    report: Reporter | None = None,
    cancel: CancelToken | None = None,
) -> tuple[int, int]:
    """Run the pipeline for one video. Returns ``(clips_done, clips_failed)``.

    Clips go to ``<output_dir>/<short|long>/<video name>/``; the database and transcript cache live in
    ``<output_dir>/.work/<video name>/`` (the database is recreated empty each time). ``name`` renames a
    downloaded original (``<title> [<id>]``) without touching the file; ``subtitle`` is the SRT the
    downloader picked for it. ``report`` (and the underlying ``cancel`` token) provide shared progress
    events and cancellation; both are optional and default to no-op.
    """
    reporter = report if report is not None else Reporter(cancel=cancel)
    root, cc, stem = Path(cfg.output_dir), cfg.clips, name or src.stem
    from .internal.content import fillers_from, load_lexicon

    fillers = fillers_from(load_lexicon(config_dir))
    log.info("system health before preprocess/unsilence: %s", health_check("unsilence"))
    destination = root / ".work" / stem / f"unsilence_{stem}.mp4"
    reporter("validate", 0.02, "cleaning silence")
    try:
        from .unsilence import remove_silence

        src = remove_silence(src, cfg, destination, reporter=reporter, cancel=cancel)
        if src == destination:
            subtitle = None  # the new timeline is no longer aligned with platform subtitles
    except Exception as exc:
        log.warning("preprocessing: cannot clean %s (%s); continuing with the original", stem, exc)
        pass
    info = probe(src)
    reporter("validate", 0.2, f"probed {src.name}")
    log.info("system health before input: %s", health_check("input"))
    log.info(
        "input metadata: path=%s | size=%d | duration=%.2fs | %dx%d @ %.2ffps | audio=%s",
        safe_path_text(src),
        src.stat().st_size if src.exists() else 0,
        info.duration,
        info.width,
        info.height,
        info.fps,
        info.has_audio,
    )
    if cc.mode == "split":
        out = root / "long" / stem
        out.mkdir(parents=True, exist_ok=True)
        parts = split_video(src, out, SPLIT_SECONDS, info.duration, stem)
        (out / "clips.json").write_text(json.dumps([p.name for p in parts], indent=2), encoding="utf-8")
        reporter("package", 1.0, "split complete")
        return len(parts), 0

    work = root / ".work" / stem
    work.mkdir(parents=True, exist_ok=True)
    store = Store(work / "autoclip.db", fresh=True)
    cache = Cache(store, RESOURCES)
    try:
        db = energy_track(src, info, cache)
        if cancel is not None:
            cancel.check()
        log.info("system health before transcript: %s", health_check("transcript"))
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
        if needs_text(cfg):
            reporter("transcribe", 1.0, "transcript ready" if have_text else "transcription skipped - selecting on audio only")
        else:
            reporter("transcribe", 1.0, "no transcript needed")
        jobs: list[RenderJob] = []
        needs = 0
        for kind in cc.kinds:
            kcfg = cc.kind(kind)
            moments = (
                ranges_moments(store, kcfg, info, cfg, have_text, kind=kind)
                if cc.mode == "ranges"
                else select_moments(src, info, store, cache, cfg, kind, db, have_text, config_dir=config_dir)
            )
            if not moments:
                log.warning("no %s clips for %s (video shorter than the length range, or no ranges given)", kind, name or src.name)
                continue
            out = root / kind / stem
            out.mkdir(parents=True, exist_ok=True)
            size = canvas_size(
                kcfg.aspect,
                quality_short_side(cfg.quality),
                info.width,
                info.height,
                fill=cfg.video.layout == "fill",
            )
            numbers: dict[str, int] = {}
            for m in moments:
                n = numbers[m.scene] = numbers.get(m.scene, 0) + 1
                dst = out / f"{stem}_{m.scene}_{n:02d}_{int(m.start)}s.mp4"
                clip_id = store.save_moment(kind, m.scene, n, m)
                jobs.append(
                    RenderJob(
                        clip_id,
                        m,
                        src,
                        dst,
                        kind,
                        kcfg.aspect,
                        encoder,
                        info.has_audio,
                        cfg,
                        store.path,
                        log_file,
                        os.getpid(),
                        fillers=fillers,
                    )
                )
                needs = max(needs, estimate_render_bytes(size, m.end - m.start))
        if not jobs:
            raise AutoclipError("no clips found for any requested kind (is the video shorter than the length ranges?)")
        reporter("select", 1.0, f"{len(jobs)} clips planned")
        log.info("system health before render: %s", health_check("render"))
        reporter("render", 0.0, f"0/{len(jobs)} clips to process")
        try:
            render_all(jobs, needs, report=reporter, cancel=cancel)
        except Cancelled:
            _write_clip_summaries(root, stem, store.clips())  # keep what finished: a half-written run is still useful
            raise
        reporter("render", 1.0, f"{len(jobs)}/{len(jobs)} clips processed")
        rows = store.clips()
        log.info(
            "output base: %s | clips=%d (failed=%d)",
            safe_path_text(root),
            sum(r["status"] == "done" for r in rows),
            sum(r["status"] != "done" for r in rows),
        )
    finally:
        cache.close()
        store.close()
    # Optional Stable Diffusion thumbnails (serialized in the AI queue; skipped when unavailable or offline).
    if not cfg.skip.thumbnail and not cfg.skip.llm:
        try:
            from . import ai as _ai

            if _ai.sd_available():
                import cv2

                for r in rows:
                    if r["status"] != "done":
                        continue
                    clip_path = root / r["kind"] / stem / r["file"]
                    out_jpg = clip_path.with_suffix(".jpg")
                    frame_path = clip_path.with_suffix(".frame.jpg")
                    try:
                        cap = cv2.VideoCapture(str(clip_path))
                        ok, frame = cap.read()
                        cap.release()
                        if ok:
                            cv2.imwrite(str(frame_path), frame)
                            _ai.generate_sd_thumbnail(frame_path, out_jpg, cfg.image.prompt)
                    finally:
                        frame_path.unlink(missing_ok=True)
        except Exception as e:
            log.warning("Stable Diffusion thumbnail pass failed (%s: %s)", type(e).__name__, e)

    _write_clip_summaries(root, stem, rows)
    for clip_kind in dict.fromkeys(r["kind"] for r in rows):
        log.info("output folder: %s", safe_path_text(root / clip_kind / stem))
        for r in rows:
            if r["kind"] != clip_kind:
                continue
            sidecar_stem = r["file"].rpartition(".")[0]
            if r["caption"]:
                (root / clip_kind / stem / (sidecar_stem + ".txt")).write_text(r["caption"], encoding="utf-8")
            log.info(
                "output: %s | %s/%s | %dx%d | %.2fs trimmed=%.2f wall=%.1fs srt=%s thumb=%s.jpg",
                safe_path_text(root / clip_kind / stem / r["file"]),
                clip_kind,
                r["scene"],
                int(r["width"]),
                int(r["height"]),
                float(r["end"] - r["start"]),
                float(r["trimmed"] or 0.0),
                float(r["wall"] or 0.0),
                safe_path_text(root / clip_kind / stem / (sidecar_stem + ".srt")),
                sidecar_stem,
            )

    for r in rows:
        if r["status"] != "done":
            log.warning("%s clip %s #%d failed: %s", r["kind"], r["scene"], r["n"], r["error"] or "did not finish")
    reporter("package", 1.0, "index written")
    return sum(r["status"] == "done" for r in rows), sum(r["status"] != "done" for r in rows)
