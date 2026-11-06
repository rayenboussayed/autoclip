"""Runtime-side default policies for autoclip.

These values are derived rather than requested: they are low-risk defaults based on memory, affinity, image
sizes and queue load.  Keeping them here prevents them from becoming user-facing YAML/JSON options.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CACHE_DISK_MB",
    "CHUNK_MINUTES",
    "DEFAULT_CPU_PERCENT",
    "DEFAULT_ENCODING_THREADS",
    "DEFAULT_MEMORY_RESERVE_MB",
    "DEFAULT_MEM_FRACTION",
    "RuntimeDecision",
    "detect_cpu_count",
]

CACHE_DISK_MB = 2048
CHUNK_MINUTES = 5.0
DEFAULT_CPU_PERCENT = 50
DEFAULT_ENCODING_THREADS = 4
DEFAULT_MEM_FRACTION = 0.5
DEFAULT_MEMORY_RESERVE_MB = 512


@dataclass(frozen=True)
class RuntimeDecision:
    """One computed value plus why it was chosen and what it affects."""

    name: str
    value: object
    source: str


def _container_cpu_quota() -> float | None:
    try:
        quota_text, period_text = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        quota, period = int(quota_text), int(period_text)
        return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    root = Path("/sys/fs/cgroup")
    quota_file = root / "cpu/cpu.cfs_quota_us"
    period_file = root / "cpu/cpu.cfs_period_us"
    try:
        q = int(quota_file.read_text().strip())
        period = int(period_file.read_text().strip())
        return q / period if q and period else None
    except (OSError, ValueError):
        return None


def detect_cpu_count() -> int:
    """Cores available to this process, including container and affinity limits."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    # delegate quota probes to the resources module so the existing monkeypatch-based tests still work
    try:
        from ..resources import cgroup_cpu_quota as _quota

        quota = _quota()
    except Exception:
        quota = _container_cpu_quota()
    if quota:
        n = min(n, math.ceil(quota))
    return max(1, n)


def runtime_decisions() -> list[RuntimeDecision]:
    """The complete auto-decided set used by this run (suitable for diagnostics)."""
    return [
        RuntimeDecision("chunk_minutes", CHUNK_MINUTES, "fixed memory-safe chunk"),
        RuntimeDecision("cache_disk_mb", CACHE_DISK_MB, "fixed SQLite cache ceiling"),
        RuntimeDecision("encoding_threads", DEFAULT_ENCODING_THREADS, "bounded by encoder pressure"),
        RuntimeDecision("max_cpu_percent", DEFAULT_CPU_PERCENT, "half the available CPU budget"),
        RuntimeDecision("memory_reserve_mb", DEFAULT_MEMORY_RESERVE_MB, "safe OS reserve"),
        RuntimeDecision("memory_fraction", DEFAULT_MEM_FRACTION, "fraction of currently free RAM"),
    ]
