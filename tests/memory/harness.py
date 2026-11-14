"""Pure harness logic of the opt-in memory-regression suite (fixtures live in ``conftest.py``).

Memory tests are opt-in like the e2e suite: nothing here is used by the offline suite and the
whole module is unit-tested offline in ``tests/unit/test_memory_harness.py``. The engine keeps a
POSIX/Linux measurement contract (``/proc`` and recursive ``psutil.children``) for the subprocess
cases; the in-process cases run on any platform with a best-effort RSS signal.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import subprocess
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import psutil

RSS_POLL_SECONDS = 0.01  # 10 ms: a short ffmpeg child cannot hide its peak between polls


class ProcessLike(Protocol):
    """The tiny slice of ``psutil.Process`` the meter needs (so tests can stand in with a fake)."""

    pid: int

    def memory_info(self) -> Any: ...

    def children(self, recursive: bool = False) -> Iterable[Any]:
        """Return the children (the parameter only mirrors psutil's signature)."""
        _ = recursive
        return ()


def rss_tree(proc: ProcessLike) -> tuple[int, int]:
    """Parent and parent+recursive-children RSS in bytes (children that already exited are ignored)."""
    try:
        own = int(proc.memory_info().rss)
    except Exception:  # best-effort memory probes must not blow up a test; a dead process reads as 0
        own = 0
    tree = own
    try:
        for child in proc.children(recursive=True):
            try:
                tree = max(tree, tree + int(child.memory_info().rss))
            except Exception:
                continue  # a child that just died contributes nothing
    except Exception:
        pass  # the top-level walk failed too; report the parent at least
    return own, tree


def poll_peak(pid: int, stop: threading.Event, acc: list[int]) -> None:
    """Poller thread: fill ``acc`` with the tree-RSS of ``pid`` until ``stop`` is set."""
    proc = psutil.Process(pid)
    while not stop.is_set():
        try:
            _, tree = rss_tree(proc)
            acc.append(tree)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return
        time.sleep(RSS_POLL_SECONDS)


@contextlib.contextmanager
def peak_meter(proc: ProcessLike | None = None) -> Iterator[list[int]]:
    """Measure (parent, parent+children) RSS peaks of ``proc`` around the body.

    Returns a mutable list; read it after the ``with``-block exits. ``proc`` is optional for call
    sites that only need their own process measured.
    """
    samples: list[int] = []
    stop = threading.Event()
    thread = threading.Thread(target=poll_peak, args=(proc.pid if proc is not None else os.getpid(), stop, samples), daemon=True)
    thread.start()
    try:
        yield samples
    finally:
        stop.set()
        thread.join(timeout=2.0)
        if proc is not None:
            own, tree = rss_tree(proc)
            samples.append(tree)  # one last sample catches the final peak right before exit


def peak_of(samples: Sequence[int]) -> int:
    """The peak of a non-empty sample series (0 when the series is empty)."""
    return max(samples) if samples else 0


def parse_vm_hwm(text: str) -> int | None:
    """The VmHWM value (bytes) from a ``/proc/<pid>/status`` dump, or ``None`` when absent."""
    for line in text.splitlines():
        if line.startswith("VmHWM:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024  # kB -> bytes
    return None


def children_rusage_kb() -> int | None:
    """Largest child resident (kB) from ``getrusage(RUSAGE_CHILDREN)`` where the OS supports it."""
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    except (ImportError, ValueError, OSError):
        return None


def compare_budget(measured_mb: float, budget_mb: float) -> bool:
    """``measured_mb`` must not exceed ``budget_mb`` (both in megabytes)."""
    return measured_mb <= budget_mb


def median(values: Sequence[float]) -> float:
    """The median of a non-empty sequence."""
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2)


def run_command(
    argv: list[str],
    *,
    out_dir: Path,
    env_extra: dict[str, str] | None = None,
    poll_interval: float = RSS_POLL_SECONDS,
    cwd: Path | None = None,
) -> tuple[int, list[int]]:
    """Run ``argv`` in a subprocess, polling the parent+children tree RSS.

    Returns ``(returncode, samples)`` with ``samples`` in bytes. ``out_dir`` is deleted and recreated; the child is
    responsible for creating subfolders it needs. ``cwd`` anchors a relative ``out_dir``-independent working directory.
    """
    import shutil
    import threading

    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    repo_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=cwd)
    samples: list[int] = []
    stop = threading.Event()

    def poll() -> None:
        import psutil

        while not stop.is_set():
            try:
                ps = psutil.Process(proc.pid)
                tree = ps.memory_info().rss
                for child in ps.children(recursive=True):
                    with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                        tree = max(tree, tree + child.memory_info().rss)
                samples.append(tree)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return
            time.sleep(poll_interval)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    _, _ = proc.communicate()
    stop.set()
    thread.join(timeout=2)
    return proc.returncode, samples


def run_python(
    snippet: str, args: list[str], *, out_dir: Path, env_extra: dict[str, str] | None = None, poll_interval: float = RSS_POLL_SECONDS
) -> tuple[int, list[int]]:
    """Run ``python -c <snippet>`` in a subprocess (see :func:`run_command`)."""
    import sys

    return run_command([sys.executable, "-c", snippet, *args], out_dir=out_dir, env_extra=env_extra, poll_interval=poll_interval)


@dataclass(frozen=True, slots=True)
class CaseMetrics:
    name: str
    wall_seconds: float
    peak_parent_mb: float
    peak_tree_mb: float
    budget_parent_mb: float
    budget_tree_mb: float
    passed: bool


def write_metrics(directory: Path, cases: Sequence[CaseMetrics]) -> Path:
    """Write ``metrics.json`` next to the case logs; returns its path."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys_version(),
        "cases": [
            {
                "name": case.name,
                "wall_seconds": round(case.wall_seconds, 3),
                "peak_parent_mb": round(case.peak_parent_mb, 1),
                "peak_tree_mb": round(case.peak_tree_mb, 1),
                "budget_parent_mb": case.budget_parent_mb,
                "budget_tree_mb": case.budget_tree_mb,
                "passed": case.passed,
            }
            for case in cases
        ],
    }
    path = directory / "metrics.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def sys_version() -> str:
    import sys as _sys

    return _sys.version.split()[0]


def new_run_directory(root: Path) -> Path:
    """Create a timestamped results folder under ``root``, reusing the convention of the e2e runner."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / stamp
    suffix = 2
    while candidate.exists():
        candidate = root / f"{stamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _mb(bytes_: int) -> float:
    return bytes_ / (1024 * 1024)


__all__ = [
    "RSS_POLL_SECONDS",
    "CaseMetrics",
    "children_rusage_kb",
    "compare_budget",
    "median",
    "ProcessLike",
    "new_run_directory",
    "parse_vm_hwm",
    "peak_meter",
    "peak_of",
    "poll_peak",
    "rss_tree",
    "run_command",
    "run_python",
    "write_metrics",
    "_mb",
]
