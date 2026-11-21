"""One streaming pass over an audio track: 0.5 s loudness and non-silent spans."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .derive.constants import (
    UNSILENCE_KEEP_SILENCE_MS,
    UNSILENCE_MIN_SEGMENT_SECONDS,
    UNSILENCE_MIN_SILENCE_MS,
    UNSILENCE_SILENCE_THRESHOLD_DB,
)
from .events import CancelToken
from .log import traced
from .media import FRAME, SR, external_env, ffmpeg_exe
from .models import MediaInfo

F32 = np.ndarray[Any, np.dtype[np.float32]]


@dataclass(slots=True)
class SpeechScan:
    """The two views of one streaming decode that the pipeline shares."""

    energy_db: F32
    intervals: list[tuple[float, float]]


def _merge(raw: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(raw):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
            continue
        merged[-1][1] = max(merged[-1][1], end)
    return [(a, b) for a, b in merged if b - a >= UNSILENCE_MIN_SEGMENT_SECONDS]


def _stop_process(proc: subprocess.Popen[bytes]) -> None:
    """Stop an ffmpeg analysis child quickly during cancellation, then escalate."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=1.0)


@traced("scan_audio")
def scan_audio(
    src: Path,
    info: MediaInfo,
    cancel: CancelToken | None = None,
) -> SpeechScan:
    """Stream ``src`` once, returning loudness frames and kept spans."""
    if not info.has_audio or info.duration <= 0:
        return SpeechScan(np.zeros(1, dtype=np.float32), [])
    command = [ffmpeg_exe(), "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    frame = int(SR * FRAME)
    small = int(SR * 0.01)
    half_second = frame // small
    hangover = UNSILENCE_MIN_SILENCE_MS / 1000.0
    keep = UNSILENCE_KEEP_SILENCE_MS / 1000.0
    energy: list[F32] = []
    raw: list[tuple[float, float]] = []
    state = "silence"
    opened = 0.0
    quiet_at: float | None = None
    index = 0
    left = b""
    energy_squares = 0.0
    energy_count = 0
    with tempfile.TemporaryFile() as stderr:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr, env=external_env(), bufsize=0)
        try:
            assert proc.stdout is not None
            while data := proc.stdout.read(64 * 1024):
                if cancel is not None:
                    cancel.check()
                buf = left + data
                usable = len(buf) // (2 * small) * (2 * small)
                usable_samples = usable // 2
                x = np.frombuffer(buf[:usable], dtype=np.int16).astype(np.float32) / 32768.0
                nf = usable_samples // small
                mean_squares = np.mean(x[: nf * small].reshape(nf, small) ** 2, axis=1)
                for index_in_block, mean_square in enumerate(mean_squares):
                    start = (index + index_in_block) * 0.01
                    db = 20 * np.log10(float(np.sqrt(mean_square + 1e-12)) + 1e-9)
                    energy_squares += float(mean_square) * small
                    energy_count += 1
                    if state == "silence":
                        if db >= UNSILENCE_SILENCE_THRESHOLD_DB:
                            state, opened, quiet_at = "speech", start, None
                    elif db >= UNSILENCE_SILENCE_THRESHOLD_DB:
                        quiet_at = None
                    elif db < UNSILENCE_SILENCE_THRESHOLD_DB - 6.0:
                        if quiet_at is None:
                            quiet_at = start
                        if start - quiet_at >= hangover:
                            interval = (max(0.0, opened - keep), min(info.duration, quiet_at + keep))
                            raw.append(interval)
                            state, opened, quiet_at = "silence", 0.0, None
                    if energy_count == half_second:
                        rms = float(np.sqrt(energy_squares / (half_second * small) + 1e-12))
                        energy.append(np.array([20 * np.log10(rms + 1e-9)], dtype=np.float32))
                        energy_squares = 0.0
                        energy_count = 0
                index += nf
                left = buf[usable:]
            rc = proc.wait()
        except BaseException:
            _stop_process(proc)
            raise
        if rc != 0:
            stderr.seek(0)
            last = stderr.read().decode("utf-8", "replace").strip().splitlines()
            detail = last[-1] if last else "ffmpeg failed"
            raise RuntimeError(f"cannot analyse '{src.name}': {detail}")
    if state == "speech":
        raw.append((max(0.0, opened - keep), info.duration))
    energy_array = np.concatenate(energy) if energy else np.zeros(1, dtype=np.float32)
    merged = _merge(raw)
    return SpeechScan(energy_array, merged)
