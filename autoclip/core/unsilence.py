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
from .log import LOG as log
from .log import traced
from .media import SR, decode_audio, probe


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


def _non_silent_intervals(cfg: Config, audio: Any) -> list[tuple[float, float]]:
    from pydub.silence import detect_nonsilent

    raw = detect_nonsilent(
        audio,
        min_silence_len=cfg.unsilence.min_silence_ms,
        silence_thresh=cfg.unsilence.silence_threshold_db,
    )
    duration_s = len(audio) / 1000.0
    keep = cfg.unsilence.keep_silence_ms / 1000.0
    merged: list[tuple[float, float]] = []
    for start_ms, end_ms in raw:
        lo = max(0.0, start_ms / 1000.0 - keep)
        hi = min(duration_s, end_ms / 1000.0 + keep)
        if hi - lo < max(cfg.unsilence.min_segment_seconds, 1e-4):
            continue
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    if not merged:
        return []
    if len(merged) == 1 and merged[0][0] <= 1e-3 and merged[0][1] >= duration_s - 1e-3:
        return [(0.0, duration_s)]
    return merged


@traced("unsilence", logging.DEBUG)
def remove_silence(
    src: Path,
    cfg: Config,
    dst: Path,
    reporter: object | None = None,
    cancel: object | None = None,
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
    try:
        from moviepy import VideoFileClip, concatenate_videoclips
        from pydub import AudioSegment

        # Decode through our own ffmpeg wrapper: pydub's AudioSegment.from_file calls ffprobe,
        # which the bundled imageio-ffmpeg does not ship (so unsilence silently fell back on
        # machines without a system ffmpeg).
        try:
            samples = decode_audio(src, 0.0, info.duration, SR)
        except Exception as exc:
            log.warning("unsilence: cannot read audio of %s: %s", src.name, exc)
            return src
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        audio = AudioSegment(data=pcm.tobytes(), sample_width=2, frame_rate=SR, channels=1)
        intervals = _non_silent_intervals(cfg, audio)
        if not intervals:
            log.info("unsilence: no removable silence found in %s", src.name)
            shutil.copyfile(src, dst)
            return dst
        if cancel is not None:
            cancel.check()  # type: ignore[attr-defined]
        clip = VideoFileClip(str(src))
        parts: list[Any] = []
        duration = float(clip.duration or 0.0)
        intervals = [(max(0.0, start), min(duration, end)) for start, end in intervals]
        intervals = [(start, end) for start, end in intervals if end - start >= cfg.unsilence.min_segment_seconds]
        try:
            for start, end in intervals:
                parts.append(clip.subclipped(start, end))
                if cancel is not None:
                    cancel.check()  # type: ignore[attr-defined]
            assembled = concatenate_videoclips(parts, method="compose")
            try:
                assembled.write_videofile(
                    str(temporary),
                    fps=clip.fps or None,
                    codec="libx264",
                    preset=cfg.unsilence.quality_preset,
                    audio_codec="aac",
                    audio_bitrate=f"{cfg.unsilence.audio_bitrate_kbps}k",
                    audio_fps=clip.audio.fps if getattr(clip, "audio", None) is not None and getattr(clip.audio, "fps", None) else 44100,
                    ffmpeg_params=["-crf", str(cfg.unsilence.quality_crf)],
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
