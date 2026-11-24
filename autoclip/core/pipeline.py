"""The pipeline for one video: probe, transcribe, pick moments, render the clips.

Both front ends (the command line and the desktop app) call :func:`process_video`; it owns the
whole per-video flow and never raises for a single broken clip.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from .config import Config, KindCfg, parse_hms
from .cutmap import CutMap, MeasuredTimeline
from .derive.constants import SPLIT_SECONDS, quality_short_side
from .derive.policy import fillers_from, load_lexicon
from .events import Cancelled, CancelToken, Reporter
from .io import srt_time
from .log import health_check, safe_path_text, traced
from .media import external_env, ffmpeg_exe, probe
from .models import Encoder, MediaInfo, Moment, RenderJob
from .moments import make_title, select_moments
from .optimizer import prepare_source
from .render import canvas_size, estimate_render_bytes
from .resources import RESOURCES
from .runner import render_all
from .scenes import detect_scenes
from .store import CHECKPOINT_STAGES, Cache, ClipRow, Store
from .transcript import FusedStt, get_transcript, transcript_model

log = logging.getLogger("autoclip")


def split_video(src: Path, out_dir: Path, seconds: float, duration: float, name: str | None = None) -> list[Path]:
    """Cut ``src`` into consecutive parts of ``seconds`` with ffmpeg stream copy (instant, no re-encode)."""
    parts: list[Path] = []
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
        source_lo, source_hi = (min(parse_hms(bound), cutmap.duration) for bound in span.split("-"))
        lo, hi = timeline.remap_window(*cutmap.source_span_to_optimized(source_lo, source_hi))
        if hi - lo <= 0:
            log.warning("clips.ranges entry %s was fully removed by silence; no custom clip for that span", span)
            continue
        text = store.window_text(lo, hi) if have_text else ""
        title = make_title(text, fillers) if kind == "long" else ""
        caption = ""
        if use_llm and text:
            try:
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
    """Whether selection, captions or titles actually read the transcript."""
    return cfg.clips.mode == "highlights" or cfg.captions.enabled or not cfg.skip.title


def wants_transcript(cfg: Config) -> bool:
    """Whether to produce STT at all: only ``skip.stt`` forbids it, never ``skip.caption``."""
    return not cfg.skip.stt


def _clip_filename(stem: str, scene: str, number: int, start: float) -> str:
    """The deterministic name of a clip, shared by fresh selection and checkpoint reconstruction."""
    return f"{stem}_{scene}_{number:02d}_{int(start)}s.mp4"


def _run_key(cfg: Config, source: Path, subtitle: Path | None = None) -> str:
    """Hash the source/subtitle identity and run settings; a change invalidates the checkpoint."""
    stat = source.stat()
    identity: list[object] = [source.name, stat.st_size, stat.st_mtime_ns]
    if subtitle is not None:
        sub_stat = subtitle.stat()
        identity += [subtitle.name, sub_stat.st_size, sub_stat.st_mtime_ns]
    settings = {key: value for key, value in asdict(cfg).items() if key not in ("input", "output_dir", "fresh")}
    return hashlib.sha256(json.dumps({"source": identity, "settings": settings}, sort_keys=True, default=str).encode()).hexdigest()


def _moment_from_row(row: ClipRow) -> Moment:
    """Rebuild a selected moment from its stored clip row (no re-selection on resume)."""
    return Moment(
        row["start"],
        row["end"],
        row["text"],
        score=row["score"],
        llm_score=row["llm_score"],
        title=row["title"],
        caption=row["caption"],
        scene=row["scene"],
        source_start=row["source_start"],
        source_end=row["source_end"],
    )


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


def _export_uncaptioned_raw(root: Path, stem: str, optimized: Path, store: Store, *, write_subtitle: bool) -> Path:
    """Copy the silence-removed source into ``raw/<stem>/`` with its full-transcript SRT beside it.

    The video is a plain copy: ``optimized`` is the already-built source in ``.work``, so no second
    encode is needed. ``store`` segments are already on that file's measured clock, so each becomes
    one SubRip cue without any time remapping here.
    """
    raw_folder = root / "raw" / stem
    raw_folder.mkdir(parents=True, exist_ok=True)
    # Exactly one raw moment starts at 0, so its render name is always `_raw_01_0s`.
    uncaptioned_video = raw_folder / f"{stem}_raw_01_0s_uncaptioned.mp4"
    if not (
        uncaptioned_video.is_file()
        and uncaptioned_video.stat().st_size == optimized.stat().st_size
        and uncaptioned_video.stat().st_mtime_ns >= optimized.stat().st_mtime_ns
    ):
        shutil.copyfile(optimized, uncaptioned_video)
    subtitle_path = uncaptioned_video.with_suffix(".srt")
    segments = store.light_segments()
    if write_subtitle and segments:
        subtitle_text = "\n".join(
            f"{index}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n" for index, (start, end, text, _) in enumerate(segments, 1)
        )
        subtitle_path.write_text(subtitle_text, encoding="utf-8")
        log.info("raw export: %s (%d transcript cues)", safe_path_text(uncaptioned_video), len(segments))
    else:
        subtitle_path.unlink(missing_ok=True)  # skip.stt (or no transcript): no sidecar at all
        log.info("raw export: %s (no transcript sidecar)", safe_path_text(uncaptioned_video))
    return uncaptioned_video


def _generate_sd_thumbnails(root: Path, stem: str, rows: Sequence[ClipRow], cfg: Config) -> None:
    """Replace finished clips' frame covers with Stable Diffusion ones when the backend is available.

    An unavailable or failing backend is logged and skipped; it must never fail the video.
    """
    try:
        from . import ai

        if not ai.sd_available():
            return
        import cv2

        for row in rows:
            if row["status"] != "done":
                continue
            clip_path = root / row["kind"] / stem / row["file"]
            thumbnail = clip_path.with_suffix(".jpg")
            source_frame = clip_path.with_suffix(".frame.jpg")
            try:
                capture = cv2.VideoCapture(str(clip_path))
                ok, frame = capture.read()
                capture.release()
                if ok:
                    cv2.imwrite(str(source_frame), frame)
                    ai.generate_sd_thumbnail(source_frame, thumbnail, cfg.image.prompt)
            finally:
                source_frame.unlink(missing_ok=True)
    except Exception as e:
        log.warning("Stable Diffusion thumbnail pass failed (%s: %s)", type(e).__name__, e)


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
    db_path = work / "autoclip.db"
    if cfg.fresh:
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(f"{db_path}{suffix}").unlink(missing_ok=True)
    store = Store(db_path)
    cache = Cache(store, RESOURCES)
    run_key = _run_key(cfg, original_source, subtitle)
    saved_key, saved_stage = store.checkpoint()
    if saved_key != run_key:
        store.reset_pipeline()
        saved_stage = ""

    def reached(target: str) -> bool:
        return saved_stage in CHECKPOINT_STAGES[CHECKPOINT_STAGES.index(target) :]

    log.info("checkpoint: run=%s stage=%s", run_key[:8], saved_stage or "none")
    try:
        log.info("system health before the optimizer: %s", health_check("optimizer"))
        want_text = wants_transcript(cfg) and original_info.has_audio
        # Fused STT belongs to the first assembly; a resumed one already has its transcript.
        build_transcript = want_text and not reached("prepared")
        selected_model = transcript_model() if build_transcript else ""
        fused = FusedStt(original_source, cfg, work, stem, want_text, selected_model, subtitle=subtitle) if build_transcript else None
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
        store.set_checkpoint(run_key, "prepared")

        cuts = detect_scenes(src, info, cancel=cancel) if cc.mode != "ranges" and "raw" not in cc.kinds else []
        if cancel is not None:
            cancel.check()
        log.info("system health before transcript: %s", health_check("transcript"))
        # A sidecar/cached transcript is still read under skip.stt (selection and titles may use it),
        # but a fresh STT pass only happens when wants_transcript allows it.
        load_transcript = needs_text(cfg) or wants_transcript(cfg)
        if reached("transcribed") and store.light_segments():
            have_text = True
            log.info("checkpoint: reusing the stored transcript")
        else:
            have_text = get_transcript(
                original_source,
                cfg,
                work,
                store,
                original_info,
                load_transcript,
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
        if load_transcript:
            reporter("transcribe", 1.0, "transcript ready" if have_text else "transcription skipped - selecting on audio only")
        else:
            reporter("transcribe", 1.0, "no transcript needed")
        store.set_checkpoint(run_key, "transcribed")
        _export_uncaptioned_raw(root, stem, src, store, write_subtitle=not cfg.skip.stt)

        jobs: list[RenderJob] = []
        raw_jobs: list[RenderJob] = []
        needs = 0
        raw_needs = 0
        resume_selected = reached("selected")
        stored_rows = store.clips() if resume_selected else []
        resume_selection = resume_selected and bool(stored_rows)
        if resume_selection:
            log.info("checkpoint: reusing %d planned clips", len(stored_rows))
        # De-duplicate locally: cc.kinds still drives scene detection and preset validation.
        kinds = list(dict.fromkeys((*cc.kinds, "raw")))
        for kind in kinds:
            kcfg = cc.kind(kind)
            if resume_selection:
                planned = [
                    (
                        row["id"],
                        _moment_from_row(row),
                        row["status"],
                        row["file"] or _clip_filename(stem, row["scene"], row["n"], row["start"]),
                    )
                    for row in stored_rows
                    if row["kind"] == kind
                ]
            else:
                numbers: dict[str, int] = {}
                if kind == "raw":
                    moments = [raw_moment(store, cutmap, timeline, have_text)]
                elif cc.mode == "ranges":
                    moments = ranges_moments(store, kcfg, info, cfg, have_text, kind=kind, cutmap=cutmap, timeline=timeline)
                else:
                    moments = select_moments(
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
                if not moments:
                    log.warning("no %s clips for %s (video shorter than the length range, or no ranges given)", kind, name or src.name)
                    continue
                planned = []
                for moment in moments:
                    number = numbers[moment.scene] = numbers.get(moment.scene, 0) + 1
                    clip_id = store.save_moment(kind, moment.scene, number, moment)
                    filename = _clip_filename(stem, moment.scene, number, moment.start)
                    planned.append((clip_id, moment, "pending", filename))
            clip_dir = root / kind / stem
            clip_dir.mkdir(parents=True, exist_ok=True)
            size = canvas_size(
                kcfg.aspect,
                quality_short_side(cfg.quality),
                info.width,
                info.height,
                fill=cfg.video.layout == "fill",
            )
            for clip_id, moment, status, filename in planned:
                clip_path = clip_dir / filename
                if status == "done" and clip_path.is_file() and clip_path.stat().st_size > 0:
                    log.info("checkpoint: keeping finished %s", safe_path_text(clip_path))
                    continue
                if status != "pending":
                    store.requeue_clip(clip_id)
                job = RenderJob(
                    clip_id,
                    moment,
                    src,
                    clip_path,
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
                estimate = estimate_render_bytes(size, moment.end - moment.start)
                if kind == "raw":
                    raw_jobs.append(job)  # a whole-source estimate would serialise every other clip
                    raw_needs = max(raw_needs, estimate)
                else:
                    jobs.append(job)
                    needs = max(needs, estimate)
        total = len(jobs) + len(raw_jobs)
        store.set_checkpoint(run_key, "selected")
        reporter("select", 1.0, f"{total} clips planned")
        log.info("system health before render: %s", health_check("render"))
        reporter("render", 0.0, f"0/{total} clips to process")
        try:
            # raw renders last, in its own pass: its whole-source estimate would serialise the other
            # clips. Keep the final progress event inside the try so a cancel still writes the
            # partial index for whatever finished.
            if jobs:
                render_all(jobs, needs, cancel=cancel)
            render_all(raw_jobs, raw_needs, report=reporter, cancel=cancel)
            reporter("render", 1.0, f"{total}/{total} clips processed")
            rows = store.clips()
            store.set_checkpoint(run_key, "rendered")
        except Cancelled:
            _write_clip_summaries(root, stem, store.clips())  # keep what finished: a half-written run is still useful
            raise
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
    if not cfg.skip.thumbnail and not cfg.skip.llm:
        _generate_sd_thumbnails(root, stem, rows, cfg)

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
