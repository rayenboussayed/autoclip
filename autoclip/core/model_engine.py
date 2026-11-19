"""Automatic lifecycle for the models autoclip holds itself.

The engine chooses the cheap/safe path from the resources available at this moment; it is not user-configurable at
present. It always chooses lazily and releases at phase end. In-process unloading is best effort; the strongest
guarantee is a bounded worker process, used only when memory/VRAM is tight.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import subprocess
import sys
from dataclasses import dataclass

from .resources import MB, RESOURCES

log = logging.getLogger("autoclip")


@dataclass(frozen=True, slots=True)
class ModelDecision:
    """Where the next model runs: in-process (fast) or a bounded, one-shot worker (strongest release)."""

    subprocess: bool


def vram_free_mb() -> int | None:
    """Free CUDA VRAM in MB, or None when the machine does not expose it."""
    command = ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=False, timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        used, total = (int(part.strip()) for part in proc.stdout.strip().splitlines()[0].split(","))
    except (ValueError, IndexError):
        return None
    return max(0, total - used)


def choose_model(name: str, min_free_mb: float) -> ModelDecision:
    """Pick a bounded worker for a tight machine, or in-process reuse for a roomy one; no configuration required."""
    snapshot = RESOURCES.snapshot()
    free_mb = snapshot.mem_available / MB
    vram_mb = vram_free_mb()
    if snapshot.pressure == "high":
        log.info("model %s chooses the bounded worker (memory pressure is high)", name)
        return ModelDecision(True)
    if vram_mb is not None and vram_mb < min_free_mb:
        log.info("model %s chooses the bounded worker (free VRAM %.0fMB, needs %.0fMB)", name, vram_mb, min_free_mb)
        return ModelDecision(True)
    if free_mb < min_free_mb * 1.25:
        log.info("model %s chooses the bounded worker (free RAM %.0fMB, needs %.0fMB)", name, free_mb, min_free_mb)
        return ModelDecision(True)
    return ModelDecision(False)


def release_allocations() -> None:
    """Best-effort release after a phase; a subprocess exit is the strong guarantee for a tight worker."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None or not hasattr(torch, "cuda") or not torch.cuda.is_available():
        return
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()
