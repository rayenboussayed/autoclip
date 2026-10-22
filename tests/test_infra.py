"""Resource monitor, logging and the SQLite store / cache."""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path

import numpy as np
import pytest

from autoclip import resources
from autoclip.config import Config, parse
from autoclip.log import LOG, setup_logging, speed_text, traced
from autoclip.models import Moment, Segment, Word
from autoclip.resources import MB, ResourceMonitor, Snapshot, cgroup_cpu_quota, cgroup_memory_available
from autoclip.store import Cache, MemoryLRU, Store, is_boundary


def cfg_with(**sections: dict[str, object]) -> Config:
    return parse(Config, sections)


# ---- resources -----------------------------------------------------------------------------------
def test_cgroup_memory_v2_v1_unlimited_missing(tmp_path: Path) -> None:
    assert cgroup_memory_available(tmp_path) is None
    (tmp_path / "memory.max").write_text("max")
    (tmp_path / "memory.current").write_text("5")
    assert cgroup_memory_available(tmp_path) is None
    (tmp_path / "memory.max").write_text("1000")
    assert cgroup_memory_available(tmp_path) == 995
    v1 = tmp_path / "v1"
    (v1 / "memory").mkdir(parents=True)
    (v1 / "memory" / "memory.limit_in_bytes").write_text(str(2**63 - 4096))
    (v1 / "memory" / "memory.usage_in_bytes").write_text("10")
    assert cgroup_memory_available(v1) is None  # "unlimited" sentinel
    (v1 / "memory" / "memory.limit_in_bytes").write_text("100")
    assert cgroup_memory_available(v1) == 90


def test_cgroup_cpu_quota(tmp_path: Path) -> None:
    assert cgroup_cpu_quota(tmp_path) is None
    (tmp_path / "cpu.max").write_text("max 100000")
    assert cgroup_cpu_quota(tmp_path) is None
    (tmp_path / "cpu.max").write_text("150000 100000")
    assert cgroup_cpu_quota(tmp_path) == 1.5
    v1 = tmp_path / "v1" / "cpu"
    v1.mkdir(parents=True)
    (v1 / "cpu.cfs_quota_us").write_text("200000")
    (v1 / "cpu.cfs_period_us").write_text("100000")
    assert cgroup_cpu_quota(tmp_path / "v1") == 2.0


def test_snapshot_pressure() -> None:
    assert Snapshot(0, 1000 * MB, 50 * MB, 0).pressure == "high"
    assert Snapshot(0, 10_000 * MB, 1500 * MB, 0).pressure == "medium"
    assert Snapshot(0, 10_000 * MB, 5000 * MB, 0).pressure == "low"


def test_cpu_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    mon = ResourceMonitor()
    mon.configure(cfg_with(resources={"cpus": 6}))
    assert mon.cpus == 6
    monkeypatch.setattr(resources, "cgroup_cpu_quota", lambda: 1.5)
    mon.configure(Config())
    assert mon.cpus <= 2
    monkeypatch.setattr(resources, "cgroup_cpu_quota", lambda: None)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)  # macOS / Windows have no affinity API
    monkeypatch.setattr(os, "cpu_count", lambda: 3)
    mon.configure(Config())
    assert mon.cpus == 3
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    mon.configure(Config())
    assert mon.cpus == 1


def make_monitor(monkeypatch: pytest.MonkeyPatch, free_mb: int, total_mb: int = 16000, **resources_cfg: object) -> ResourceMonitor:
    mon = ResourceMonitor()
    mon.configure(cfg_with(resources={"cpus": 8, **resources_cfg}))
    monkeypatch.setattr(mon, "snapshot", lambda: Snapshot(10.0, total_mb * MB, free_mb * MB, 100 * MB))
    return mon


def test_worker_plan_scales_with_free_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    per_job = 400 * MB
    plenty = make_monitor(monkeypatch, 12000).plan_workers(per_job, 6)
    assert (plenty.workers, plenty.threads) == (6, 1)
    two = make_monitor(monkeypatch, 12000).plan_workers(per_job, 2)
    assert (two.workers, two.threads) == (2, 4)  # leftover cores become encoder threads
    tight = make_monitor(monkeypatch, 1000).plan_workers(per_job, 6)
    assert (tight.workers, tight.threads) == (1, 8)  # one job, but every core still used
    assert make_monitor(monkeypatch, 12000, max_workers=3).plan_workers(per_job, 6).workers == 3
    gpu = make_monitor(monkeypatch, 12000).plan_workers(per_job, 6, limit=3)  # a GPU allows only a few encode sessions
    assert (gpu.workers, gpu.threads) == (3, 2)
    assert make_monitor(monkeypatch, 12000, max_cpu_percent=50).cpu_budget() == 4


def test_cache_budget_follows_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    assert make_monitor(monkeypatch, 100).cache_budget() == 8 * MB
    assert 16 * MB <= make_monitor(monkeypatch, 12000).cache_budget() <= 768 * MB
    fixed = ResourceMonitor()
    fixed.configure(cfg_with(storage={"cache_memory_mb": 5}))
    assert fixed.cache_budget() == 5 * MB


def test_real_snapshot_and_brief() -> None:
    mon = ResourceMonitor()
    snap = mon.snapshot()
    assert snap.mem_total > 0 and snap.rss > 0 and mon.peak_rss >= snap.rss
    assert "rss=" in mon.brief() and mon.usable_memory() >= 0


def test_container_limit_caps_available_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resources, "cgroup_memory_available", lambda: 123 * MB)
    assert ResourceMonitor().snapshot().mem_available == 123 * MB


def test_after_operation_runs_hooks_and_gc(monkeypatch: pytest.MonkeyPatch) -> None:
    mon = make_monitor(monkeypatch, 100)
    calls: list[str] = []
    hook = lambda: calls.append("hook")  # noqa: E731
    mon.register_trim_hook(hook)
    collected: list[int] = []

    def fake_collect() -> int:
        collected.append(1)
        return 0

    monkeypatch.setattr(gc, "collect", fake_collect)
    mon.after_operation("op")
    mon.unregister_trim_hook(hook)
    mon.after_operation("op")
    assert calls == ["hook"] and len(collected) == 2
    relaxed = make_monitor(monkeypatch, 15000)
    relaxed.after_operation("op")
    assert len(collected) == 2  # no gc when memory is plentiful


def test_a_failing_trim_hook_never_breaks_the_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    mon = make_monitor(monkeypatch, 15000)
    ran: list[str] = []

    def bad() -> None:
        raise RuntimeError("boom")

    mon.register_trim_hook(bad)
    mon.register_trim_hook(lambda: ran.append("second"))
    mon.after_operation("op")  # does not raise, later hooks still run
    assert ran == ["second"]


# ---- logging -------------------------------------------------------------------------------------
def test_logging_file_console_and_levels(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log_file = tmp_path / "logs" / "run.log"
    setup_logging(log_file, "INFO", "WARNING")
    LOG.info("to-file-only")
    LOG.warning("to-both")
    LOG.handlers[0].flush()
    text = log_file.read_text(encoding="utf-8")
    assert "to-file-only" in text and "to-both" in text and "pid=" in text
    assert "to-both" in capsys.readouterr().out
    setup_logging(None, "INFO", None)  # no handlers requested -> NullHandler, nothing raises
    LOG.error("silent")
    assert isinstance(LOG.handlers[0], logging.NullHandler)


def test_traced_logs_start_end_and_failure(tmp_path: Path) -> None:
    log_file = tmp_path / "t.log"
    setup_logging(log_file, "DEBUG", None)

    @traced("op-ok")
    def ok(x: int) -> int:
        return x + 1

    @traced()
    def boom() -> None:
        raise ValueError("bad")

    assert ok(1) == 2
    with pytest.raises(ValueError, match="bad"):
        boom()
    text = log_file.read_text(encoding="utf-8")
    assert "START op-ok" in text and "END   op-ok" in text and "FAIL" in text and "ValueError: bad" in text


def test_speed_text() -> None:
    assert speed_text(10, 5) == "2.00x realtime" and speed_text(0, 5) == "n/a" and speed_text(5, 0) == "n/a"


# ---- store ---------------------------------------------------------------------------------------
def segs() -> list[Segment]:
    return [
        Segment(0, 4, "Hello world.", [Word(0, 2, "Hello"), Word(2, 4, "world.")]),
        Segment(4.1, 8, "and then", [Word(4.1, 6, "and", pseudo=True), Word(6, 8, "then", pseudo=True)], pseudo=True),
        Segment(20, 24, "Far away.", [Word(20, 24, "Far away.")]),
    ]


def test_store_fresh_wipes_previous_database(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "a.db"
    s = Store(path)
    s.put_segments(segs())
    s.close()
    s = Store(path)
    assert len(s.light_segments()) == 3
    s.close()
    s = Store(path, fresh=True)
    assert s.light_segments() == []
    s.close()


def test_store_transcript_queries(tmp_path: Path) -> None:
    s = Store(tmp_path / "a.db")
    s.put_segments(segs())
    assert [x[3] for x in s.light_segments()] == [True, True, True]  # sentence end, 12 s gap, last
    assert s.segment_text(0, 1) == "Hello world. and then"
    assert s.window_text(19, 30) == "Far away." and "Far" not in s.window_text(0, 10)
    words = s.window_words(3.0, 5.0)
    assert [(w.text, w.start, w.end) for w in words] == [("world.", 3.0, 4.0), ("and", 4.1, 5.0)]
    assert words[1].pseudo
    s.put_segments(segs()[:1])  # replaces, never appends
    assert len(s.light_segments()) == 1
    s.close()


def test_store_clips_are_tracked_per_kind_and_scene(tmp_path: Path) -> None:
    s = Store(tmp_path / "a.db")
    first = s.save_moment("short", "funny", 1, Moment(0, 5, "text", score=1.5, title="T", scene="funny"))
    second = s.save_moment("long", "sad", 1, Moment(5, 99, "more"))
    third = s.save_moment("short", "sad", 1, Moment(9, 12, "x"))
    assert (first, second, third) == (1, 2, 3)
    s.finish_clip(first, ok=True, file="a.mp4", size=(10, 20), trimmed=4.5, wall=1.0)
    s.finish_clip(second, ok=False, error="boom")
    rows = s.clips()
    assert [(r["kind"], r["scene"], r["status"]) for r in rows] == [
        ("short", "funny", "done"),
        ("long", "sad", "failed"),
        ("short", "sad", "pending"),
    ]
    assert rows[0]["width"] == 10 and rows[1]["error"] == "boom" and rows[0]["title"] == "T" and rows[0]["n"] == 1
    assert [r["id"] for r in s.clips("short")] == [1, 3] and [r["id"] for r in s.clips("long")] == [2]
    s.close()


def test_store_transaction_rolls_back(tmp_path: Path) -> None:
    s = Store(tmp_path / "a.db")
    with pytest.raises(RuntimeError), s.transaction():
        s.conn.execute("INSERT INTO words VALUES (1, 2, 'x', 0)")
        raise RuntimeError
    assert s.window_words(0, 10) == []
    s.close()


def test_kv_roundtrip_and_lru_eviction(tmp_path: Path) -> None:
    s = Store(tmp_path / "a.db")
    assert s.kv_get("k") is None
    s.kv_put("old", b"a" * 100)
    s.kv_put("new", b"b" * 100)
    s.conn.execute("UPDATE kv SET atime = 1 WHERE k = 'old'")
    s.kv_evict(150)
    assert s.kv_get("old") is None and s.kv_get("new") == b"b" * 100
    s.kv_evict(10_000)  # under the cap: nothing happens
    assert s.kv_get("new") is not None
    s.close()


@pytest.mark.parametrize(
    "text, gap, expected",
    [
        ("Done.", 0.0, True),
        ('He said "go."', 0.0, True),
        ("and then", 0.1, False),
        ("no end", 0.7, True),
        ("last", None, True),
        ("\u4eca\u5929\u3002", 0, True),
    ],
)
def test_is_boundary(text: str, gap: float | None, expected: bool) -> None:
    assert is_boundary(text, gap) is expected


def test_memory_lru() -> None:
    m = MemoryLRU(10)
    m.put("a", b"123456")
    m.put("b", b"123456")  # evicts a
    assert m.get("a") is None and m.get("b") == b"123456"
    m.put("b", b"12")  # replacing frees the old size
    assert m.size == 2
    m.put("big", b"x" * 50)  # larger than the whole budget: not kept
    assert m.get("big") is None
    m.put("c", b"123456789")
    assert m.get("b") is None  # b was least recently used
    m.resize(0)
    assert m.size == 0


def test_two_tier_cache(tmp_path: Path) -> None:
    mon = ResourceMonitor()
    mon.configure(cfg_with(storage={"cache_memory_mb": 1, "cache_disk_mb": 1}))
    store = Store(tmp_path / "a.db")
    cache = Cache(store, mon)
    arr = np.arange(1000, dtype=np.float32)
    calls: list[int] = []

    def produce() -> np.ndarray:
        calls.append(1)
        return arr

    np.testing.assert_array_equal(cache.memoize_array("k", produce), arr)  # miss -> compute
    np.testing.assert_array_equal(cache.memoize_array("k", produce), arr)  # RAM hit
    cache.mem.resize(0)
    np.testing.assert_array_equal(cache.memoize_array("k", produce), arr)  # SQLite hit
    assert len(calls) == 1 and store.kv_get("k") is not None
    mon.after_operation("x")  # hook trims: budget restored, blobs under the disk cap
    cache.close()
    store.close()


def test_logging_with_a_non_standard_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import sys

    stream = io.StringIO()  # not a TextIOWrapper (some IDEs / service managers replace stdout like this)
    monkeypatch.setattr(sys, "stdout", stream)
    setup_logging(None, "INFO", "INFO")
    LOG.info("hello")
    assert "hello" in stream.getvalue()
