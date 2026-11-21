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
from .cutmap import CutMap, MeasuredTimeline
from .derive.constants import SPLIT_SECONDS, quality_short_side
from .errors import AutoclipError
from .events import Cancelled, CancelToken, Reporter
from .log import health_check, safe_path_text, traced
from .media import external_env, ffmpeg_exe, probe
from .models import Encoder, MediaInfo, Moment, RenderJob
from .moments import make_title, select_moments
from .optimizer import prepare_source
from .render import canvas_size, estimate_render_bytes
from .resources import RESOURCES
from .runner import render_all
from .scenes import detect_scenes
from .store import Cache, ClipRow, Store
from .transcript import FusedStt, get_transcript, transcript_model

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
    store: Store,
    kcfg: KindCfg,
    info: MediaInfo,
    cfg: Config,
    have_text: bool,
    fillers: frozenset[str] | None = None,
    kind: str = "short",
    cutmap: CutMap | None = None,
    timeline: MeasuredTimeline | None = None,
) -> list[Moment]:
    """Moments for the explicit ``ranges`` of a clip kind (scene type "custom"), clamped to the video."""
    if fillers is None:
        fillers = frozenset()
    cutmap = cutmap or CutMap.identity(info.duration)
    timeline = timeline or MeasuredTimeline.from_cutmap(cutmap)
    from . import ai

    use_llm = not cfg.skip.llm and ai.llm_available(cfg.llm.base_url, cfg.llm.model)
    out = []
    for span in kcfg.ranges:
        source_lo, source_hi = (min(parse_hms(x), cutmap.duration) for x in span.split("-"))
        lo, hi = timeline.remap_window(*cutmap.source_span_to_optimized(source_lo, source_hi))
        if hi - lo <= 0:
            log.warning("clips.ranges entry %s was fully removed by silence; no custom clip for that span", span)
            continue
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
        out.append(Moment(lo, hi, text, scene="custom", title=title, caption=caption, source_start=source_lo, source_end=source_hi))
    return out


def raw_moment(
    store: Store,
    cutmap: CutMap,
    timeline: MeasuredTimeline,
    have_text: bool,
) -> Moment:
    """One moment spanning the whole silence-removed source, on its measured timeline.

    Unlike selection, this is unconditional: a raw-only run always has exactly one job, even without a
    transcript or when the video is shorter than any of the ordinary length ranges. Source bounds refer to
    the original video, while playback bounds refer to the optimized file captions are burned into.
    """
    text = store.window_text(0.0, timeline.last) if have_text else ""
    return Moment(
        0.0,
        timeline.last,
        text,
        scene="raw",
        source_start=0.0,
        source_end=cutmap.duration,
    )


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
                "source_start": _number(r["source_start"]),
                "source_end": _number(r["source_end"]),
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
    original_source = src
    from .derive.policy import fillers_from, load_lexicon

    fillers = fillers_from(load_lexicon(config_dir))
    info = probe(src)
    original_info = info
    reporter("validate", 0.02, "probed source")
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
    log.info("system health before the optimizer: %s", health_check("optimizer"))
    want_text = needs_text(cfg) and not cfg.skip.stt and original_info.has_audio
    # Choose the Whisper model once: both the fused pass and get_transcript must key the same cache.
    selected_model = transcript_model() if want_text else ""
    fused = FusedStt(original_source, cfg, work, stem, want_text, selected_model, subtitle=subtitle) if want_text else None
    prepared = prepare_source(
        original_source,
        cfg,
        original_info,
        work,
        cancel=cancel,
        stt_factory=fused.plan if fused is not None else None,
    )
    src, info, cutmap, timeline = prepared.path, prepared.info, prepared.cutmap, prepared.timeline
    db = prepared.energy_db
    if prepared.path != original_source:
        log.info("using the optimized source %s (%.2fs)", safe_path_text(prepared.path), prepared.info.duration)
    store = Store(work / "autoclip.db", fresh=True)
    cache = Cache(store, RESOURCES)
    try:
        cuts = detect_scenes(src, info, cancel=cancel) if cc.mode != "ranges" and "raw" not in cc.kinds else []
        if cancel is not None:
            cancel.check()
        log.info("system health before transcript: %s", health_check("transcript"))
        have_text = get_transcript(
            original_source,
            cfg,
            work,
            store,
            original_info,
            needs_text(cfg),
            subtitle=subtitle,
            subtitle_manual=subtitle_manual,
            name=stem,
            cutmap=cutmap,
            cache_source=original_source,
            model_name_override=selected_model or None,
            stt_failed=prepared.stt_failed,
            timeline=timeline,
            cancel=cancel,
        )
        if needs_text(cfg):
            reporter("transcribe", 1.0, "transcript ready" if have_text else "transcription skipped - selecting on audio only")
        else:
            reporter("transcribe", 1.0, "no transcript needed")
        jobs: list[RenderJob] = []
        raw_jobs: list[RenderJob] = []
        needs = 0
        raw_needs = 0
        for kind in cc.kinds:
            kcfg = cc.kind(kind)
            moments = (
                [raw_moment(store, cutmap, timeline, have_text)]
                if kind == "raw"
                else (
                    ranges_moments(store, kcfg, info, cfg, have_text, kind=kind, cutmap=cutmap, timeline=timeline)
                    if cc.mode == "ranges"
                    else select_moments(
                        src,
                        info,
                        store,
                        cache,
                        cfg,
                        kind,
                        db,
                        have_text,
                        config_dir=config_dir,
                        cutmap=cutmap,
                        timeline=timeline,
                        shots=cuts,
                    )
                )
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
                job = RenderJob(
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
                    audio_ready=prepared.audio_ready,
                )
                estimate = estimate_render_bytes(size, m.end - m.start)
                if kind == "raw":
                    raw_jobs.append(job)  # a whole-source estimate would serialise every other clip
                    raw_needs = max(raw_needs, estimate)
                else:
                    jobs.append(job)
                    needs = max(needs, estimate)
        if not jobs and not raw_jobs:
            raise AutoclipError("no clips found for any requested kind (is the video shorter than the length ranges?)")
        total = len(jobs) + len(raw_jobs)
        reporter("select", 1.0, f"{total} clips planned")
        log.info("system health before render: %s", health_check("render"))
        reporter("render", 0.0, f"0/{total} clips to process")
        try:
            if jobs:
                render_all(jobs, needs, report=reporter, cancel=cancel)
            if raw_jobs:
                reporter("render", len(jobs) / total, f"{len(jobs)}/{total} clips processed")
                # One job: in-process, no pool. Keeps its duration-scaled estimate away from other clips.
                render_all(raw_jobs, raw_needs, cancel=cancel)
        except Cancelled:
            _write_clip_summaries(root, stem, store.clips())  # keep what finished: a half-written run is still useful
            raise
        reporter("render", 1.0, f"{total}/{total} clips processed")
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
    _write_clip_summaries(root, stem, rows)
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
