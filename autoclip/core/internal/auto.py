"""Decisions the engine takes from measured resources - never from user configuration.

Every function here is a pure rule: it takes the measurements as arguments (cores, free RAM in MB)
and returns the value (or ``None``) the rest of autoclip acts on. Callers pass live readings
(:func:`detect_cpu_count`, :func:`free_ram_mb`); tests pass exact values. Every chosen value is
logged as ``automation ... | source=...`` by the caller so a run can explain itself.

The quality and concurrency rules only ever pick the *conservative* option (they drop work on weak
machines; they are never raised). :func:`auto_ollama_model` is the one deliberate
capacity-maximising rule: on an ample machine the largest usable model wins.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

__all__ = [
    "JOB_RAM_MB",
    "OLLAMA_RAM_FRACTION",
    "auto_max_concurrent_jobs",
    "auto_mode",
    "auto_ollama_model",
    "auto_quality",
    "auto_whisper_model",
]

#: RAM one server job needs at peak (Whisper + render + Python overhead), in MB.
JOB_RAM_MB = 1536.0
#: An installed model may use at most this fraction of free RAM (the rest is KV cache, the pipeline, the OS).
OLLAMA_RAM_FRACTION = 0.5
_MIB = 1 << 20


def _free_mb(free_ram_mb: float) -> float:
    """A finite free-RAM reading in MB, or ``0.0`` for NaN/infinity/nonsense (the safe direction)."""
    return free_ram_mb if math.isfinite(free_ram_mb) and free_ram_mb > 0.0 else 0.0


def auto_quality(*, cpus: int, free_ram_mb: float) -> Literal["high", "medium", "low"]:
    """The encode tier the machine sustains: ``high`` on any normal desktop, downgraded when it cannot cope.

    ``high`` (the pre-4.x default) needs 4 cores and 3 GB of free RAM; ``medium`` (2 cores, 2 GB) still renders
    comfortably; ``low`` works everywhere else. The tier is never above ``high``: a GPU speeds the encode up,
    it is not asked to make the picture better than what a machine without one would produce.
    """
    if cpus >= 4 and _free_mb(free_ram_mb) >= 3072:
        return "high"
    if cpus >= 2 and _free_mb(free_ram_mb) >= 2048:
        return "medium"
    return "low"


def auto_mode(*, cpus: int, free_ram_mb: float) -> Literal["performance", "resources", "balance"]:
    """The resource profile the machine is given: ``resources`` when it is tight, ``performance`` when ample.

    The adaptive tuner keeps refining the CPU budget inside the chosen profile during the run, exactly as
    when ``balance`` was the user-visible default.
    """
    free = _free_mb(free_ram_mb)
    if cpus <= 2 or free < 1536:
        return "resources"
    if cpus >= 8 and free >= 8192:
        return "performance"
    return "balance"


def auto_whisper_model(*, device: str, cpus: int, free_ram_mb: float) -> str:
    """CUDA: ``large-v3``. CPU: ``medium`` only on a performance-class machine, else ``small``.

    CPU ``medium`` is about five times slower than ``small`` and ``large-v3`` takes hours on a long
    video, so the larger pick is gated by the same ample-machine rule as the resource profile. The
    RAM fit in ``transcript.fit_whisper_model`` still shrinks the result when memory is tight.
    """
    if device == "cuda":
        return "large-v3"
    return "medium" if auto_mode(cpus=cpus, free_ram_mb=free_ram_mb) == "performance" else "small"


def auto_max_concurrent_jobs(*, cpus: int, free_ram_mb: float) -> int:
    """How many pipeline jobs the HTTP server may run at once (1-4).

    Memory-bound first (each job peaks around ``JOB_RAM_MB``), then core-bound (a job is not free on
    one core). Measured once per server process, at the first job; restart to re-evaluate.
    """
    free = _free_mb(free_ram_mb)
    jobs = min(int(cpus) // 2, int(free // JOB_RAM_MB), 4)
    return max(1, jobs)


def auto_ollama_model(installed: Mapping[str, int], free_ram_mb: float) -> str | None:
    """Choose which installed model the pipeline should use for ranking, titles and captions.

    Args:
        installed: model name -> size on disk in bytes (Ollama's ``/api/tags`` ``size`` field); sizes
            of ``0`` or less mean "unknown".
        free_ram_mb: free RAM in MB at probe time.

    Returns:
        The largest model whose size fits inside :data:`OLLAMA_RAM_FRACTION` of free RAM; when none
        fits, the smallest known model; when every size is unknown, the first name alphabetically
        (``None`` for an empty selection, i.e. the AI helpers stay off). With zero usable free RAM the
        smallest known model is returned; whether it fits in practice is the runner's business.
    """
    if not installed:
        return None
    free = _free_mb(free_ram_mb)
    known = {name: size for name, size in installed.items() if size > 0}
    if not known:
        return sorted(installed)[0]
    budget = int(free * _MIB * OLLAMA_RAM_FRACTION)
    fits = {name: size for name, size in known.items() if size <= budget}
    if fits:
        # largest that fits; ties broken alphabetically so the choice is deterministic
        return max(sorted(fits), key=fits.__getitem__)
    return min(sorted(known), key=known.__getitem__)
