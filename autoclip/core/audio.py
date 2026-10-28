"""Clip audio: decoding a window, denoise + loudness, pause / filler removal and the kept-ranges timeline."""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .config import AudioCfg, SilenceCfg
from .log import traced
from .media import AUDIO_SR, decode_audio
from .models import Word

F32 = NDArray[np.float32]
Keep = list[tuple[float, float]]


#: Noise reduction runs on pieces of this length so a long clip never needs one huge spectrogram in RAM.
DENOISE_CHUNK_SECONDS = 60


def load_window_audio(src: Path, start: float, dur: float, highpass: float = 0, fft_denoise: bool = False) -> F32:
    """``[start, start+dur]`` as mono float32 @ 44.1 kHz, optionally high-passed / FFT-denoised by ffmpeg."""
    filters = ([f"highpass=f={highpass}"] if highpass else []) + (["afftdn=nf=-30"] if fft_denoise else [])
    return decode_audio(src, start, dur, AUDIO_SR, filters)


def piece_bounds(n: int, step: int, minimum: int) -> list[tuple[int, int]]:
    """Split ``n`` samples into pieces of about ``step``; a tail shorter than ``minimum`` joins the previous piece."""
    edges = [*range(0, n, step), n]
    if len(edges) > 2 and n - edges[-2] < minimum:
        del edges[-2]
    return list(pairwise(edges))


def frame_db(x: F32, win: float = 0.02) -> NDArray[np.float64]:
    """RMS level in dB per ``win`` seconds (always at least one frame)."""
    n = int(AUDIO_SR * win)
    k = max(len(x) // n, 1)
    rms = np.sqrt(np.mean(np.resize(x, k * n).reshape(k, n) ** 2, axis=1) + 1e-12)
    return np.asarray(20 * np.log10(rms + 1e-9), dtype=np.float64)


@traced("enhance_audio")
def enhance_audio(src: Path, start: float, dur: float, cfg: AudioCfg, threads: int, *, enhance: bool = True, denoise: bool = True) -> F32:
    """Window audio, high-passed, denoised, loudness-normalised (speech frames only) and soft-limited.

    ``cfg.enhance`` is the master off switch; ``enhance`` and ``denoise`` let the caller skip one
    half independently (``skip.audio_enhance`` / ``skip.denoise``).  Denoising without enhancement
    still removes noise but keeps the original level and high-pass.
    """
    if not cfg.enhance or (not enhance and not denoise):
        return load_window_audio(src, start, dur)
    use_nr = denoise and (cfg.denoise == "noisereduce" or (cfg.denoise == "auto" and importlib.util.find_spec("noisereduce") is not None))
    fft = denoise and (cfg.denoise == "ffmpeg" or (cfg.denoise == "auto" and not use_nr))
    x: F32 = load_window_audio(src, start, dur, cfg.highpass_hz if enhance else 0, fft_denoise=fft)
    if use_nr and len(x) >= AUDIO_SR:  # clips under a second are too short to estimate noise from
        import noisereduce

        pieces = [
            noisereduce.reduce_noise(y=x[a:b], sr=AUDIO_SR, stationary=False, n_jobs=threads, prop_decrease=cfg.denoise_strength)
            for a, b in piece_bounds(len(x), DENOISE_CHUNK_SECONDS * AUDIO_SR, AUDIO_SR)
        ]
        x = np.concatenate(pieces).astype(np.float32)
    if not enhance:
        return x
    db = frame_db(x)
    active = db > np.percentile(db, 95) - 25  # speech frames only
    n = int(AUDIO_SR * 0.02)
    frames = np.resize(x, len(db) * n).reshape(len(db), n)[active]
    rms_db = 20 * np.log10(np.sqrt(np.mean(frames**2)) + 1e-9)
    x = x * np.float32(10 ** (float(np.clip(cfg.target_dbfs - rms_db, -12, 24)) / 20))
    knee, ceiling = 0.8, 0.95  # soft limiter: stays under -0.45 dBFS so AAC encoding cannot clip
    over = np.abs(x) > knee
    x[over] = np.sign(x[over]) * (knee + (ceiling - knee) * np.tanh((np.abs(x[over]) - knee) / (ceiling - knee)))
    return x


def is_filler(text: str, fillers: set[str]) -> bool:
    """Is ``text`` (ignoring case and punctuation) a filler word?"""
    return "".join(c for c in text.lower() if c.isalnum() or c == "_") in fillers


def _within_minimum(cut: list[tuple[float, float]], dur: float, min_out: float) -> list[tuple[float, float]]:
    """Give cuts back until what is left is at least ``min_out`` seconds long.

    The smallest cuts are given back first - the fillers (a few hundred milliseconds) before the
    long pauses - so the clip keeps the silence removal that matters most.
    """
    if min_out <= 0:
        return cut
    remaining = list(cut)
    while remaining and dur - sum(b - a for a, b in remaining) < min_out:
        remaining.remove(min(remaining, key=lambda c: c[1] - c[0]))
    return remaining


@traced("compute_keep", logging.DEBUG)
def compute_keep(x: F32, words: Sequence[Word], dur: float, cfg: SilenceCfg, min_out: float = 0.0) -> Keep:
    """Ranges ``(a, b)`` of the window (seconds) to KEEP after shortening long pauses and cutting fillers.

    ``min_out`` is the length the clip must keep: when the trimming would go below it, the least
    important cuts are given back, so a clip never ends up shorter than the user asked for.
    """
    whole = [(0.0, dur)]
    if not cfg.remove:
        return whole
    db, step = frame_db(x), 0.02
    if cfg.threshold_db == "auto":
        # 8 dB over the quiet tenth, at least 18 dB under the loud part, never below -70 dB
        quiet = float(min(np.percentile(db, 10) + 8, np.percentile(db, 95) - 18))
        thr = max(quiet, -70.0)
    else:
        thr = float(cfg.threshold_db)
    edges = np.diff(np.concatenate([[0], (db < thr).astype(np.int8), [0]]))  # run-length encode the silent frames
    cut: list[tuple[float, float]] = [
        (float(i) * step + cfg.keep_padding, float(j) * step - cfg.keep_padding)
        for i, j in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
        if (j - i) * step >= cfg.min_silence
    ]
    cut = [(a, b) for a, b in cut if b - a > 0.05]
    if cfg.remove_fillers:
        fillers = {f.lower() for f in cfg.fillers}
        cut += [(max(0.0, w.start - 0.03), min(dur, w.end + 0.03)) for w in words if not w.pseudo and is_filler(w.text, fillers)]
    cut = _within_minimum(sorted(cut), dur, min_out)
    keep: Keep = []
    t = 0.0
    for a, b in cut:  # sweep: the gaps between removed ranges are what we keep
        if a - t > 0.15:
            keep.append((t, a))
        t = max(t, b)
    if dur - t > 0.15:
        keep.append((t, dur))
    kept = sum(b - a for a, b in keep)
    return keep if kept >= max(3.0, 0.4 * dur) else whole  # never gut a clip


class Timeline:
    """Maps source-window time to trimmed-clip time for a list of kept ranges."""

    def __init__(self, keep: Keep) -> None:
        self.keep = keep
        self.offsets = np.concatenate([[0.0], np.cumsum([b - a for a, b in keep])])
        self.total = float(self.offsets[-1])

    def map_range(self, s: float, e: float) -> tuple[float, float] | None:
        """``[s, e]`` on the trimmed timeline, or ``None`` if it lies entirely in removed parts."""
        hits = [
            (max(s, a) - a + o, min(e, b) - a + o)
            for (a, b), o in zip(self.keep, self.offsets, strict=False)
            if min(e, b) - max(s, a) > 0.02
        ]
        return (hits[0][0], hits[-1][1]) if hits else None


def build_audio(x: F32, keep: Keep, total_out: float) -> F32:
    """Concatenate the kept ranges with 4 ms de-click ramps, padded / cut to exactly ``total_out`` seconds."""
    fade = int(0.004 * AUDIO_SR)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    pieces = []
    for a, b in keep:
        piece = x[int(a * AUDIO_SR) : int(b * AUDIO_SR)].copy()
        if len(keep) > 1 and len(piece) > 2 * fade:
            piece[:fade] *= ramp
            piece[-fade:] *= ramp[::-1]
        pieces.append(piece)
    n = round(total_out * AUDIO_SR)
    return np.pad(np.concatenate(pieces), (0, n))[:n].astype(np.float32)
