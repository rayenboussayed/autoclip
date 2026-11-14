"""Runtime-side default policies for autoclip.

These values are derived rather than requested: they are low-risk defaults based on memory, affinity, image
sizes and queue load.  Keeping them here prevents them from becoming user-facing YAML/JSON options.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from .automation import DEFAULT_TRANSCRIPT_MODELS, denoiser_threads, transcription_beam_size
from .quality import download_quality_for

__all__ = [
    "CACHE_DISK_MB",
    "CHUNK_MINUTES",
    "DEFAULT_CPU_PERCENT",
    "DEFAULT_ENCODING_THREADS",
    "DEFAULT_MEMORY_RESERVE_MB",
    "DEFAULT_MEM_FRACTION",
    "RuntimeDecision",
    "default_transcript_model",
    "detect_cpu_count",
    "transcript_model_for",
]

CACHE_DISK_MB = 2048
CHUNK_MINUTES = 5.0
DEFAULT_CPU_PERCENT = 50
DEFAULT_ENCODING_THREADS = 4
DEFAULT_MEM_FRACTION = 0.5
DEFAULT_MEMORY_RESERVE_MB = 512


def default_transcript_model() -> str:
    """Backwards-compatible alias used by tests: quality ``high`` with the locally available device."""
    try:
        import ctranslate2

        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        device = "cpu"
    return transcript_model_for("high", device)


def transcript_model_for(quality: str, device: str = "cpu") -> str:
    """The Whisper model for a quality tier: ``high`` keeps large-v3 only when CUDA runs it."""
    if quality == "high" and device != "cuda":
        return "small"
    return DEFAULT_TRANSCRIPT_MODELS.get(quality, "small")


@dataclass(frozen=True)
class RuntimeDecision:
    """One computed value plus why it was chosen and what it affects."""

    name: str
    value: object
    source: str


def detect_cpu_count() -> int:
    """Cores available to this process, including container and affinity limits."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    from ..resources import cgroup_cpu_quota as _quota

    quota = _quota()
    if quota is not None and quota > 0:  # 0/negative/unreadable means "unlimited"
        n = min(n, max(1, math.ceil(quota)))
    return max(1, n)


def runtime_decisions(quality: str = "high", mode: str = "balance", device: str = "cpu") -> list[RuntimeDecision]:
    """The complete auto-decided set used by this run (suitable for diagnostics)."""
    return [
        RuntimeDecision("chunk_minutes", CHUNK_MINUTES, "fixed memory-safe chunk"),
        RuntimeDecision("cache_disk_mb", CACHE_DISK_MB, "fixed SQLite cache ceiling"),
        RuntimeDecision("encoding_threads", DEFAULT_ENCODING_THREADS, "bounded by encoder pressure"),
        RuntimeDecision("max_cpu_percent", DEFAULT_CPU_PERCENT, "half the available CPU budget"),
        RuntimeDecision("memory_reserve_mb", DEFAULT_MEMORY_RESERVE_MB, "safe OS reserve"),
        RuntimeDecision("memory_fraction", DEFAULT_MEM_FRACTION, "fraction of currently free RAM"),
        RuntimeDecision("download_quality", download_quality_for(quality), f"derived from quality={quality}"),
        RuntimeDecision("transcript_model", transcript_model_for(quality, device), f"derived from quality={quality}, device={device}"),
        RuntimeDecision("transcription_beam_size", transcription_beam_size(quality), f"derived from quality={quality}"),
        RuntimeDecision("denoiser_threads", denoiser_threads(mode), f"derived from mode={mode}"),
        RuntimeDecision("silence_threshold", "auto", "per-input percentile for pause removal (the unsilence pre-pass uses a fixed -40 dB)"),
    ]
