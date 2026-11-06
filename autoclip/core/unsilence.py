"""Pre-processing that removes silent spans from a video."""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .errors import AutoclipError
from .events import CancelToken, Reporter
from .internal import CHUNK_MINUTES as _CHUNK_MINUTES
from .internal.quality import quality_profile
from .log import LOG as log
from .log import health_check, traced
from .media import SR, decode_audio, probe
from .models import MediaInfo


def _pydub_available() -> bool:
    try:
        import importlib

        importlib.import_module("pydub")
    except Exception:
        return False
    try:
        importlib.import_module("audioop")
        return True
    except ImportError:
        pass
    try:
        fallback = importlib.import_module("audioop_lts")
    except Exception:
        return False
    sys.modules["audioop"] = fallback
    return True


def _merge_intervals(raw: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sort and merge (touching) intervals into one ascending list."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(raw):
        if end - start < 1e-3:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _window_seconds(cfg: Config) -> float:  # cfg retained for call-site compatibility
    """The seconds loaded at once when cleaning silence, with a memory-safe chunk size."""
    return _CHUNK_MINUTES * 60.0


def _split_for_assembly(intervals: list[tuple[float, float]], cfg: Config) -> list[tuple[float, float]]:
    """Split kept intervals longer than 5 minutes * (1 + 0.3) into ~5-minute pieces, in order.

    Output content is unchanged (the same source ranges concatenated identically); splitting only
    bounds how big any single concat/encode segment is, which matters for several-hour livestreams.
    """
    chunk = _window_seconds(cfg)
    limit = chunk * 1.3  # 30 % more than a chunk is tolerated; beyond it, cut at the chunk boundary
    out: list[tuple[float, float]] = []
    for start, end in intervals:
        if end - start <= limit:
            out.append((start, end))
            continue
        pos = start
        while pos < end:
            end_next = min(pos + chunk, end)
            out.append((pos, end_next))
            pos = end_next
    return out


def _non_silent_intervals(cfg: Config, src: Path, info: MediaInfo) -> list[tuple[float, float]]:
    """Non-silent intervals of ``src`` (audio), found in 5-minute windows so a several-hour file never fits in RAM."""
    from pydub.silence import detect_nonsilent

    duration = info.duration
    keep = cfg.unsilence.keep_silence_ms / 1000.0
    window_seconds = _window_seconds(cfg)
    raw: list[tuple[float, float]] = []
    cursor = 0.0
    while cursor < duration:
        window = min(window_seconds, duration - cursor)
        samples = decode_audio(src, cursor, window, SR)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        from pydub import AudioSegment

        audio = AudioSegment(data=pcm.tobytes(), sample_width=2, frame_rate=SR, channels=1)
        del samples, pcm
        for start_ms, end_ms in detect_nonsilent(
            audio,
            min_silence_len=cfg.unsilence.min_silence_ms,
            silence_thresh=cfg.unsilence.silence_threshold_db,
        ):
            lo = max(0.0, cursor + start_ms / 1000.0 - keep)
            hi = min(duration, cursor + end_ms / 1000.0 + keep)
            if hi - lo >= max(cfg.unsilence.min_segment_seconds, 1e-4):
                raw.append((lo, hi))
        del audio
        cursor += window
    merged = _merge_intervals(raw)
    if not merged:
        return []
    if len(merged) == 1 and merged[0][0] <= 1e-3 and merged[0][1] >= duration - 1e-3:
        return [(0.0, duration)]
    return merged


@traced("unsilence", logging.DEBUG)
def remove_silence(
    src: Path,
    cfg: Config,
    dst: Path,
    reporter: Reporter | None = None,
    cancel: CancelToken | None = None,
) -> Path:
    if not cfg.unsilence.enabled:
        return src
    if not src.is_file():
        raise AutoclipError(f"unsilence: source file does not exist ({src.name})")
    dst.parent.mkdir(parents=True, exist_ok=True)
    info = probe(src)
    if not info.has_audio:
        try:
            shutil.copyfile(src, dst)
            return dst
        except OSError as exc:
            log.warning("unsilence: cannot copy %s to %s: %s", src.name, dst.name, exc)
            return src
    if not _pydub_available():
        log.warning("unsilence is enabled but pydub/audioop is unavailable; continuing with the original")
        shutil.copyfile(src, dst)
        return dst
    temporary = dst.with_suffix(".tmp.mp4")
    log.info("system health before unsilence detection: %s", health_check("unsilence"))
    try:
        from moviepy import VideoFileClip, concatenate_videoclips

        # Silence is detected window-by-window so the whole several-hour file is never decoded at once.
        try:
            intervals = _non_silent_intervals(cfg, src, info)
        except Exception as exc:
            log.warning("unsilence: cannot read audio of %s: %s", src.name, exc)
            return src
        if not intervals:
            log.info("unsilence: no removable silence found in %s", src.name)
            shutil.copyfile(src, dst)
            return dst
        if cancel is not None:
            cancel.check()
        clip = VideoFileClip(str(src))
        parts: list[Any] = []  # moviepy clips are untyped
        duration = float(clip.duration or 0.0)
        intervals = [(max(0.0, start), min(duration, end)) for start, end in intervals]
        intervals = [(start, end) for start, end in intervals if end - start >= cfg.unsilence.min_segment_seconds]
        intervals = _split_for_assembly(intervals, cfg)
        try:
            for start, end in intervals:
                parts.append(clip.subclipped(start, end))
                if cancel is not None:
                    cancel.check()
            assembled = concatenate_videoclips(parts, method="compose")
            _profile = quality_profile(cfg.quality)
            try:
                assembled.write_videofile(
                    str(temporary),
                    fps=clip.fps or None,
                    codec="libx264",
                    preset=_profile.preset,
                    audio_codec="aac",
                    audio_bitrate=f"{_profile.aac_bitrate_kbps}k",
                    audio_fps=clip.audio.fps if getattr(clip, "audio", None) is not None and getattr(clip.audio, "fps", None) else 44100,
                    ffmpeg_params=["-crf", str(_profile.crf)],
                    threads=1,
                    logger=None,
                )
            finally:
                assembled.close()
        finally:
            for part in parts:
                with contextlib.suppress(Exception):
                    part.close()
            clip.close()
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise AutoclipError(f"unsilence produced no output for {src.name}")
        os.replace(temporary, dst)
        return dst
    except Exception as exc:
        log.warning("unsilence: could not process %s (%s); continuing with the original", src.name, exc)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError as unlink_exc:
            log.debug("unsilence: cannot remove temporary file %s: %s", temporary, unlink_exc)
        if src != dst:
            try:
                shutil.copyfile(src, dst)
                return dst
            except OSError:
                pass
        return src
