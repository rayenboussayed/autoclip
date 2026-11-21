"""Measurements taken from the machine, plus persistence for adaptive samples.

Pure I/O and state helpers: nothing here decides what a value means. The policy layer imports
these functions by name, so tests patch *this* module to inject exact measurements.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import deque
from pathlib import Path
from typing import Any

__all__ = [
    "SAMPLE_CAP",
    "clamp",
    "detect_cpu_count",
    "detect_device",
    "free_ram_mb",
    "load_state",
    "save_state",
    "trend",
]

#: Most recent adaptive samples kept for a trend decision.
SAMPLE_CAP = 8


def free_ram_mb() -> float:
    """Free RAM in MB, honouring container limits (cgroup) when they are visible; the host value otherwise."""
    import psutil

    from ..resources import cgroup_memory_available

    available = psutil.virtual_memory().available
    container = cgroup_memory_available()
    if container is not None:
        available = min(available, container)
    return float(available) / float(1 << 20)


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


def load_state(path: Path) -> dict[str, Any] | None:
    """Read adaptive tuner state written by :func:`save_state` (``None`` when absent or invalid)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def save_state(path: Path, state: dict[str, Any]) -> None:
    """Persist adaptive tuner state atomically; a write failure is a debug-level event, never fatal."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except (OSError, UnboundLocalError):  # pragma: no cover
            logging.getLogger("autoclip").debug("could not remove temporary adaptive state", exc_info=True)  # pragma: no cover


def trend(samples: deque[float]) -> int:
    """-1 falling, 0 flat, +1 rising over the recent samples."""
    if len(samples) < 3:
        return 0
    first = sum(list(samples)[: len(samples) // 2]) / (len(samples) // 2)
    second = sum(list(samples)[len(samples) // 2 :]) / (len(samples) - len(samples) // 2)
    if second - first > 0.05:
        return 1
    if first - second > 0.05:
        return -1
    return 0


def clamp(value: float, lo: float, hi: float) -> float:
    """Bound ``value`` to ``[lo, hi]``."""
    return max(lo, min(hi, value))
