"""Values that autoclip chooses for the user and never asks them about."""

from __future__ import annotations

import os

__all__ = [
    "CAPTION_SAFE_BOTTOM",
    "CAPTION_SAFE_TOP",
    "caption_chars",
    "denoiser_threads",
    "transcription_beam_size",
]


CAPTION_SAFE_TOP = 0.10
"""Fraction of a vertical frame kept free at the top for captions."""

CAPTION_SAFE_BOTTOM = 0.20
"""Fraction of a vertical frame kept free at the bottom for captions."""


def transcription_beam_size(quality: str) -> int:
    """Accuracy/detail level for Whisper: 5 when quality matters, 1 when it should be fast."""
    return {"highest": 5, "high": 5, "medium": 4, "low": 3, "lowest": 1}.get(quality, 5)


def denoiser_threads(mode: str = "balance") -> int:
    """Conservative thread count: the ONNX denoiser is small and slows down with too many workers."""
    cpus = os.cpu_count() or 4
    if mode == "performance":
        return min(4, max(1, cpus // 2 or 1))
    if mode == "resources":
        return 1
    return min(2, max(1, cpus // 2 or 1))


def caption_chars(width: int = 1080) -> int:
    """Recommended number of caption characters for one screen at the rendered width."""
    return max(16, min(36, round(width / 45)))
