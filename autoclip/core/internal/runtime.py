"""Runtime-side default policies for autoclip (see ``internal.auto`` for the machine-measured engine rules)."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from . import modes
from .automation import denoiser_threads, transcription_beam_size
from .quality import download_quality_for

__all__ = [
    "CACHE_DISK_MB",
    "CHUNK_MINUTES",
    "RuntimeDecision",
    "detect_cpu_count",
    "detect_device",
    "free_ram_mb",
    "runtime_decisions",
    "whisper_model_for",
]

CACHE_DISK_MB = 2048
CHUNK_MINUTES = 5.0


def free_ram_mb() -> float:
    """Free RAM in MB, honouring container limits (cgroup) when they are visible; the host value otherwise."""
    import psutil

    from ..resources import cgroup_memory_available

    available = psutil.virtual_memory().available
    container = cgroup_memory_available()
    if container is not None:
        available = min(available, container)
    return float(available) / float(1 << 20)


def whisper_model_for(device: str = "cpu") -> str:
    """The Whisper model for a machine: ``large-v3`` on CUDA, else ``small`` (the CPU pick can shrink in ``fit_whisper_model``)."""
    return "large-v3" if device == "cuda" else "small"


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
    if quota is not None and quota > 0:  # 0/negative/unreadable means "unlimited"; floor: never over-state cores
        n = min(n, max(1, math.floor(quota)))
    return max(1, n)


def detect_device() -> str:
    """``cuda`` when ctranslate2 sees a GPU, else ``cpu`` (best effort; never raises — diagnostics only).

    The authoritative device resolution (failures reported to the user, the CUDA->CPU retry) is
    ``transcript.asr_device``; this tolerant probe exists for the decision log.
    """
    try:
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def runtime_decisions(quality: str = "high", mode: str = "balance", device: str = "cpu") -> list[RuntimeDecision]:
    """The derived values logged for this run (diagnostics, not a contract)."""
    profile = modes.profile_for(mode)
    return [
        RuntimeDecision("quality", quality, "derived from cpus/free-ram"),
        RuntimeDecision("mode", mode, "derived from cpus/free-ram"),
        RuntimeDecision("chunk_minutes", CHUNK_MINUTES, "fixed memory-safe chunk"),
        RuntimeDecision("cache_disk_mb", CACHE_DISK_MB, "fixed SQLite cache ceiling"),
        RuntimeDecision("encoding_threads", profile.encoding_threads_cap, f"derived from mode={profile.name}"),
        RuntimeDecision("max_cpu_percent", profile.cpu_percent_default, f"derived from mode={profile.name}"),
        RuntimeDecision("memory_reserve_mb", profile.memory_reserve_mb, f"derived from mode={profile.name}"),
        RuntimeDecision("memory_fraction", profile.mem_fraction, f"derived from mode={profile.name}"),
        RuntimeDecision("download_quality", download_quality_for(quality), f"derived from quality={quality}"),
        RuntimeDecision("transcript_model", whisper_model_for(device), f"derived from device={device} (pre free-RAM fit)"),
        RuntimeDecision("transcription_beam_size", transcription_beam_size(device), f"derived from device={device}"),
        RuntimeDecision("denoiser_threads", denoiser_threads(mode), f"derived from mode={mode}"),
        RuntimeDecision("silence_threshold", "auto", "per-input percentile for pause removal (the unsilence pre-pass uses a fixed -40 dB)"),
    ]
