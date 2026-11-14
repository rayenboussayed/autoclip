"""Unit tests of the memory-suite harness (offline)."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tests.memory import harness


@dataclass
class FakeMemory:
    rss: int


@dataclass
class FakeChild:
    memory_info: Callable[[], FakeMemory] = field(default_factory=lambda: lambda: FakeMemory(500))


class FakeProc:
    pid = 4242

    def __init__(self, rss: int = 100, children: list[object] | None = None, *, failing: bool = False) -> None:
        self._rss = rss
        self._children: list[object] = children if children is not None else []
        self._failing = failing

    def memory_info(self) -> FakeMemory:
        if self._failing:
            raise RuntimeError("unreadable")
        return FakeMemory(self._rss)

    def children(self, recursive: bool = True) -> list[object]:
        _ = recursive
        if self._failing:
            raise RuntimeError("unreadable")
        return list(self._children)


def test_parse_vm_hwm_reads_bytes() -> None:
    assert harness.parse_vm_hwm("Name:\t\tpython\nVmHWM:\t 12345 kB\n") == 12345 * 1024
    assert harness.parse_vm_hwm("Threads:\t3\n") is None
    assert harness.parse_vm_hwm("VmHWM:\tNaN\n") is None


def test_compare_budget() -> None:
    assert harness.compare_budget(10.0, 10.0) is True
    assert harness.compare_budget(10.1, 10.0) is False


def test_median_odd_and_even() -> None:
    assert harness.median([1, 2, 3]) == 2.0
    assert harness.median([1, 2, 3, 4]) == 2.5
    assert harness.median([]) == 0.0


def test_rss_tree_parent_only() -> None:
    own, tree = harness.rss_tree(FakeProc(rss=1000))
    assert tree == 1000
    assert own == 1000


def test_rss_tree_includes_children() -> None:
    own, tree = harness.rss_tree(FakeProc(rss=1000, children=[FakeChild()]))
    assert tree == 1500
    assert own == 1000


def test_rss_tree_tolerates_dead_children() -> None:
    class _Ghost:
        def memory_info(self) -> FakeMemory:
            return FakeMemory(0)

    own, tree = harness.rss_tree(FakeProc(rss=1000, children=[FakeChild(), _Ghost()]))
    assert tree >= 1500


def test_rss_tree_tolerates_dead_parent() -> None:
    own, tree = harness.rss_tree(FakeProc(rss=1000, failing=True))
    assert own == 0
    assert tree == 0


def test_peak_meter_collects_samples_and_final_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness, "poll_peak", lambda pid, stop, acc: acc.extend([100, 200, 300]))

    class _QuickProc:
        pid = 4242

        def memory_info(self) -> FakeMemory:
            return FakeMemory(5000)

        def children(self, recursive: bool = True) -> list[FakeChild]:
            _ = recursive
            return []

    with harness.peak_meter(_QuickProc()) as samples:
        pass
    assert harness.peak_of(samples) == 5000
    assert samples[:3] == [100, 200, 300]
    assert harness.compare_budget(harness.peak_of(samples) / (1024 * 1024), 900) is True


def test_write_metrics_round_trips(tmp_path: Path) -> None:
    case = harness.CaseMetrics("fixture", 1.5, 123.0, 456.0, 512.0, 900.0, True)
    path = harness.write_metrics(tmp_path, [case])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["cases"][0]["name"] == "fixture"
    assert payload["cases"][0]["passed"] is True
    assert payload["cases"][0]["budget_tree_mb"] == 900.0


def test_write_metrics_creates_the_directory(tmp_path: Path) -> None:
    target = tmp_path / "new" / "sub"
    harness.write_metrics(target, [])
    assert (target / "metrics.json").is_file()


def test_new_run_directory_is_unique(tmp_path: Path) -> None:
    first = harness.new_run_directory(tmp_path)
    second = harness.new_run_directory(tmp_path)
    assert first != second
    assert first.is_dir() and second.is_dir()


def test_mb() -> None:
    assert harness._mb(2 * 1024 * 1024) == 2.0
