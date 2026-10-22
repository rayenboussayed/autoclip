"""CPU / RAM awareness: measures the machine and turns it into worker counts and cache budgets.

Works on Linux, macOS and Windows. Container limits (cgroup v1/v2) are honoured when present; on other
platforms those probes simply find nothing. The plan is recomputed on demand - before every render job and
after every traced operation - so it follows the machine's *current* state, not just its start-up state.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import psutil

from .config import Config

log = logging.getLogger("autoclip")
MB = 1 << 20
CGROUP_ROOT = Path("/sys/fs/cgroup")


def _read_int(path: Path) -> int | None:
    """An integer stored in a (cgroup) file, or ``None`` if missing, unreadable or not a number (e.g. ``max``)."""
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def cgroup_memory_available(root: Path = CGROUP_ROOT) -> int | None:
    """Bytes still available under a container memory limit (cgroup v2, then v1); ``None`` if unlimited."""
    for limit_file, used_file in (
        ("memory.max", "memory.current"),
        ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes"),
    ):
        limit, used = _read_int(root / limit_file), _read_int(root / used_file)
        if limit is not None and used is not None and limit < 1 << 60:  # v1 reports ~2**63 when unlimited
            return max(0, limit - used)
    return None


def cgroup_cpu_quota(root: Path = CGROUP_ROOT) -> float | None:
    """CPU quota in cores imposed by a container runtime (cgroup v2, then v1); ``None`` if unlimited."""
    try:
        quota, period = (root / "cpu.max").read_text().split()
        return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    quota_v1, period_v1 = _read_int(root / "cpu/cpu.cfs_quota_us"), _read_int(root / "cpu/cpu.cfs_period_us")
    return quota_v1 / period_v1 if quota_v1 and period_v1 else None


@dataclass(frozen=True)
class Snapshot:
    """A point-in-time view of the machine."""

    cpu_percent: float
    mem_total: int
    mem_available: int
    rss: int

    @property
    def pressure(self) -> str:
        """``low`` / ``medium`` / ``high`` memory pressure from the share of RAM still available."""
        if self.mem_available < 0.10 * self.mem_total or self.mem_available < 600 * MB:
            return "high"
        return "medium" if self.mem_available < 0.25 * self.mem_total else "low"


@dataclass(frozen=True)
class WorkerPlan:
    """How many jobs to run in parallel and how many CPU threads each one gets."""

    workers: int
    threads: int


class ResourceMonitor:
    """Measures CPU/RAM and derives concurrency and cache budgets from the current state."""

    def __init__(self) -> None:
        self._proc = psutil.Process()
        self._hooks: list[Callable[[], None]] = []
        self._cpu_last = 0.0
        self._cpu_time = 0.0
        self.peak_rss = 0
        self.configure(Config())

    def configure(self, cfg: Config) -> None:
        """Apply the ``resources`` / ``storage`` settings and (re-)detect the usable CPU count."""
        self.res, self.storage = cfg.resources, cfg.storage
        if self.res.cpus > 0:
            self.cpus = self.res.cpus
        else:
            n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
            quota = cgroup_cpu_quota()
            self.cpus = max(1, min(n, math.ceil(quota)) if quota else n)
        psutil.cpu_percent(None)  # prime the counter so later reads are meaningful

    def snapshot(self) -> Snapshot:
        """Measure now (a few cheap syscalls)."""
        memory = psutil.virtual_memory()
        available = memory.available
        in_container = cgroup_memory_available()
        if in_container is not None:
            available = min(available, in_container)
        rss = self._proc.memory_info().rss
        self.peak_rss = max(self.peak_rss, rss)
        now = time.monotonic()
        if now - self._cpu_time >= 0.5:  # psutil reports "since the last call": avoid noisy 0% reads
            self._cpu_last, self._cpu_time = psutil.cpu_percent(None), now
        return Snapshot(self._cpu_last, memory.total, available, rss)

    def brief(self) -> str:
        """One-line state for log messages."""
        s = self.snapshot()
        return f"rss={s.rss / MB:.0f}MB free={s.mem_available / MB:.0f}MB cpu={s.cpu_percent:.0f}% mem={s.pressure}"

    def cpu_budget(self) -> int:
        """Cores the script may use (all of them by default)."""
        return max(1, round(self.cpus * self.res.max_cpu_percent / 100))

    def usable_memory(self) -> int:
        """Bytes the script may still claim right now, keeping the configured reserve free."""
        free = self.snapshot().mem_available - self.res.memory_reserve_mb * MB
        return max(0, int(free * self.res.memory_fraction))

    def plan_workers(self, per_job_bytes: int, jobs: int, limit: int = 1 << 30) -> WorkerPlan:
        """Parallelism for ``jobs`` pending jobs of roughly ``per_job_bytes`` RAM each.

        All cores are used when memory allows; otherwise as many jobs as fit (at least one). Cores that extra
        workers do not occupy are handed to each job as encoder threads. ``limit`` caps the workers (GPU sessions).
        """
        cores = self.cpu_budget()
        fits = max(1, self.usable_memory() // max(per_job_bytes, 1))
        workers = max(1, min(jobs, cores, self.res.max_workers or cores, fits, limit))
        return WorkerPlan(workers, max(1, cores // workers))

    def cache_budget(self) -> int:
        """Bytes the in-memory cache tier may hold now (shrinks under memory pressure)."""
        if self.storage.cache_memory_mb > 0:
            return self.storage.cache_memory_mb * MB
        if self.snapshot().pressure == "high":
            return 8 * MB
        return int(min(max(self.usable_memory() * 0.20, 16 * MB), 768 * MB))

    def register_trim_hook(self, hook: Callable[[], None]) -> None:
        """Run ``hook`` (e.g. ``Cache.trim``) after every traced operation."""
        self._hooks.append(hook)

    def unregister_trim_hook(self, hook: Callable[[], None]) -> None:
        """Stop running ``hook``."""
        self._hooks.remove(hook)

    def after_operation(self, op: str) -> None:
        """Rebalance: re-size caches, and collect garbage when memory is not plentiful."""
        for hook in list(self._hooks):
            try:
                hook()
            except Exception as e:  # rebalancing is best-effort: it must never break the operation it follows
                log.debug("trim hook failed after %s: %s: %s", op, type(e).__name__, e)
        if self.snapshot().pressure != "low":
            log.debug("rebalance after %s: gc freed %d objects", op, gc.collect())


#: The process-wide monitor; ``main`` and every worker call ``RESOURCES.configure(cfg)`` once.
RESOURCES = ResourceMonitor()
