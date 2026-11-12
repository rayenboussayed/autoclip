"""CPU / RAM awareness: measures the machine and turns it into worker counts and cache budgets.

Works on Linux, macOS and Windows. Container limits (cgroup v1/v2) are honoured when present; on other
platforms those probes simply find nothing. The plan is recomputed on demand - before every render job and
after every traced operation - so it follows the machine's *current* state, not just its start-up state.
"""

from __future__ import annotations

import gc
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import psutil

from .config import Config
from .internal import (
    adaptive,
    detect_cpu_count,
    modes,
)

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
        self.mode = "balance"
        self.profile = modes.profile_for(self.mode)
        self._cpu_samples: deque[float] = deque(maxlen=adaptive.SAMPLE_CAP)
        self._adapt_ts = 0.0
        self._oom = False
        self._state_path: Path | None = None
        self.configure(Config())

    def configure(self, cfg: Config | None = None) -> None:
        """(Re-)detect the usable CPU count; apply the active mode profile."""
        mode = "balance"
        if cfg is not None:
            mode = cfg.mode
        self.mode = mode
        self.profile = modes.profile_for(mode)
        self.max_cpu_percent = float(self.profile.cpu_percent_default)
        self.encoding_threads = self.profile.encoding_threads_cap
        self.memory_fraction = self.profile.mem_fraction
        self.memory_reserve_mb = self.profile.memory_reserve_mb
        self.cpus = detect_cpu_count()
        self._cpu_samples = deque(maxlen=adaptive.SAMPLE_CAP)
        if cfg is not None and getattr(cfg, "output_dir", None):
            self._state_path = Path(cfg.output_dir) / ".work" / "adaptive.json"
        else:
            self._state_path = None
        if self._state_path is not None:
            state = adaptive.load_state(self._state_path)
            if isinstance(state, dict) and state.get("mode") == self.mode:
                persisted = state.get("cpu_percent")
                if isinstance(persisted, (int, float)):
                    self.max_cpu_percent = adaptive.clamp(float(persisted), self.profile.cpu_percent_min, self.profile.cpu_percent_max)
                persisted_samples = state.get("samples")
                if isinstance(persisted_samples, list):
                    restored = [float(x) for x in persisted_samples[-adaptive.SAMPLE_CAP :] if isinstance(x, (int, float))]
                    self._cpu_samples = deque(restored, maxlen=adaptive.SAMPLE_CAP)

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
        return max(1, round(self.cpus * self.max_cpu_percent / 100))

    def encoding_thread_budget(self) -> int:
        """Threads to use for a single encode (always capped - # threads == # jobs * threads summed)."""
        requested = self.encoding_threads or self.profile.encoding_threads_cap
        return max(1, min(requested, self.profile.encoding_threads_cap, 4, self.cpu_budget()))

    def usable_memory(self) -> int:
        """Bytes the script may still claim right now, keeping the configured reserve free."""
        free = self.snapshot().mem_available - self.memory_reserve_mb * MB
        return max(0, int(free * self.memory_fraction))

    def plan_workers(self, per_job_bytes: int, jobs: int, limit: int = 1 << 30) -> WorkerPlan:
        """Parallelism for ``jobs`` pending jobs of roughly ``per_job_bytes`` RAM each.

        All cores are used when memory allows; otherwise as many jobs as fit (at least one). Cores that extra
        workers do not occupy are handed to each job as encoder threads. ``limit`` caps the workers (GPU sessions).
        """
        cores = self.cpu_budget()
        fits = max(1, self.usable_memory() // max(per_job_bytes, 1))
        if not getattr(self.profile, "parallel", True):
            workers = 1 if not (jobs <= 4 and per_job_bytes <= 8 * 1024 * 1024) else max(1, min(jobs, cores, fits, limit, 2))
        else:
            workers = max(1, min(jobs, cores, fits, limit))
        return WorkerPlan(workers, max(1, cores // max(workers, 1)))

    def cache_budget(self) -> int:
        """Bytes the in-memory cache tier may hold now (shrinks under memory pressure)."""
        if self.snapshot().pressure == "high":
            return 4 * MB
        return int(min(max(self.usable_memory() * 0.15, 8 * MB), 256 * MB))

    def record_job(self, wall: float, peak_rss: int = 0, threads: int = 1, workers: int = 1) -> None:
        """Feed one finished job's outcome into the adaptive history (best-effort).

        There is no trustworthy per-job CPU signal in the parent process, so the utilization
        history is a coarse thread/core upper bound, not a measurement. The RSS history is honest:
        the caller may pass the job's own peak (a spawned worker knows it); the parent always
        folds in its own current RSS, which bounds in-process runs and retry loops.
        """
        try:
            observed = min(100.0, (max(threads, 1) * max(workers, 1) / max(self.cpus, 1)) * 100.0)
            self._cpu_samples.append(observed)
            own = self._proc.memory_info().rss
            for sample in (own, peak_rss):
                if sample and sample > self.peak_rss:
                    self.peak_rss = sample
        except Exception as e:  # best-effort  # pragma: no cover
            log.debug("record_job failed: %s", e)  # pragma: no cover

    def notify_oom(self) -> None:
        self._oom = True

    def adapt(self) -> None:
        """Nudge ``max_cpu_percent`` based on recent utilisation (throttled, mode-aware)."""
        import time

        now = time.monotonic()
        if now - self._adapt_ts < 5.0:
            return
        self._adapt_ts = now
        profile = self.profile
        samples = list(self._cpu_samples)
        avg = (sum(samples) / len(samples)) if samples else float(profile.cpu_percent_default)
        prev = float(self.max_cpu_percent)
        pressure = "low"
        try:
            pressure = self.snapshot().pressure
        except Exception:  # pragma: no cover
            pressure = "low"  # pragma: no cover
        samples_deque = deque(self._cpu_samples, maxlen=adaptive.SAMPLE_CAP)
        trend_dir = adaptive.trend(samples_deque) if len(samples) >= 3 else 0
        if self._oom:
            new = prev - profile.step_down
            self._oom = False
        elif pressure == "high":
            new = prev - profile.step_down
        elif self.mode == "performance":
            if avg < profile.target_cpu_low:
                new = prev + profile.step_up
            elif avg > profile.target_cpu_high and trend_dir <= 0:
                new = prev - profile.step_down
            elif trend_dir > 0 and avg < profile.target_cpu_high:
                new = prev + profile.step_up * 0.5
            else:
                new = prev
        elif self.mode == "resources":
            if avg > profile.target_cpu_high:
                new = prev - profile.step_down
            elif avg < profile.target_cpu_low and trend_dir >= 0 and pressure == "low":
                new = prev + profile.step_up * 0.5
            else:
                new = prev
        elif avg < profile.target_cpu_low:
            new = prev + profile.step_up
        elif avg > profile.target_cpu_high:
            new = prev - profile.step_down
        elif trend_dir > 0 and avg < profile.target_cpu_high:
            new = prev + profile.step_up * 0.5
        elif trend_dir < 0 and avg > profile.target_cpu_low:
            new = prev - profile.step_down * 0.5
        else:
            new = prev
        new = adaptive.clamp(new, profile.cpu_percent_min, profile.cpu_percent_max)
        if abs(new - prev) >= 0.5:
            self.max_cpu_percent = new
            log.info(
                "automation mode=%s cpu_percent=%.1f mem_fraction=%.2f",
                self.mode,
                self.max_cpu_percent,
                self.memory_fraction,
            )
            if self._state_path is not None:
                adaptive.save_state(
                    self._state_path,
                    {"mode": self.mode, "cpu_percent": new, "samples": samples},
                )

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
        try:
            self.adapt()
        except Exception as e:  # best-effort  # pragma: no cover
            log.debug("adapt failed: %s", e)  # pragma: no cover
        if self.snapshot().pressure != "low":
            log.debug("rebalance after %s: gc freed %d objects", op, gc.collect())


CACHE_DISK_MB = 2048


#: The process-wide monitor; ``main`` and every worker call ``RESOURCES.configure(cfg)`` once.
RESOURCES = ResourceMonitor()
