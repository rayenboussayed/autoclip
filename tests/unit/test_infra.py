"""Resource monitor, logging and the SQLite store / cache."""

from __future__ import annotations

import gc
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from autoclip.core import log as log_module
from autoclip.core import resources
from autoclip.core.config import Config, JsonValue, parse
from autoclip.core.log import LOG, setup_logging, speed_text, traced
from autoclip.core.models import Moment, Segment, Word
from autoclip.core.resources import MB, ResourceMonitor, Snapshot, cgroup_cpu_quota, cgroup_memory_available
from autoclip.core.store import Cache, MemoryLRU, Store, is_boundary


def cfg_with(**sections: object) -> Config:
    return parse(Config, cast(dict[str, JsonValue], sections))


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
    mon.configure()
    mon.cpus = 6
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
    mon.configure()
    mon.cpus = 8
    mon.max_cpu_percent = float(resources_cfg.pop("max_cpu_percent", 100.0))  # type: ignore[arg-type]
    monkeypatch.setattr(mon, "snapshot", lambda: Snapshot(10.0, total_mb * MB, free_mb * MB, 100 * MB))
    return mon


def test_mode_profiles_are_applied(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    perf = parse(Config, {"mode": "performance", "output_dir": str(tmp_path)})
    res = parse(Config, {"mode": "resources", "output_dir": str(tmp_path)})
    bal = parse(Config, {"mode": "balance", "output_dir": str(tmp_path)})
    pmon = ResourceMonitor()
    rmon = ResourceMonitor()
    bmon = ResourceMonitor()
    pmon.configure(perf)
    rmon.configure(res)
    bmon.configure(bal)
    assert pmon.mode == "performance" and pmon.max_cpu_percent == 90.0
    assert rmon.mode == "resources" and rmon.max_cpu_percent == 25.0
    assert bmon.mode == "balance" and bmon.max_cpu_percent == 50.0


def test_mode_resources_is_in_process_and_lean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = parse(Config, {"mode": "resources", "output_dir": str(tmp_path)})
    mon = ResourceMonitor()
    mon.configure(cfg)
    mon.cpus = 8
    monkeypatch.setattr(mon, "usable_memory", lambda: 8_000 * MB)
    plan = mon.plan_workers(100 * MB, 4)
    assert plan.workers == 1


def test_configure_ignores_state_from_another_mode(tmp_path: Path) -> None:
    import json

    state_dir = tmp_path / ".work"
    state_dir.mkdir()
    (state_dir / "adaptive.json").write_text(json.dumps({"mode": "resources", "cpu_percent": 20.0}), encoding="utf-8")
    cfg = parse(Config, {"mode": "performance", "output_dir": str(tmp_path)})
    mon = ResourceMonitor()
    mon.configure(cfg)
    assert mon.max_cpu_percent == mon.profile.cpu_percent_default  # the other mode's state is ignored


def test_configure_ignores_a_non_numeric_persisted_cpu_percent(tmp_path: Path) -> None:
    import json

    state_dir = tmp_path / ".work"
    state_dir.mkdir()
    (state_dir / "adaptive.json").write_text(json.dumps({"mode": "performance", "cpu_percent": "high"}), encoding="utf-8")
    cfg = parse(Config, {"mode": "performance", "output_dir": str(tmp_path)})
    mon = ResourceMonitor()
    mon.configure(cfg)
    assert mon.max_cpu_percent == mon.profile.cpu_percent_default  # unreadable state is ignored, not a crash


def test_mode_adapt_raises_on_low_utilisation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = parse(Config, {"mode": "performance", "output_dir": str(tmp_path)})
    mon = ResourceMonitor()
    mon.configure(cfg)
    mon.cpus = 8
    mon._cpu_samples.clear()
    monkeypatch.setattr(mon, "snapshot", lambda: Snapshot(10.0, 32_000 * MB, 30_000 * MB, 100 * MB))
    old = mon.max_cpu_percent
    for _ in range(3):
        mon.record_job(1.0, peak_rss=100 * MB, threads=1, workers=1)
    mon.adapt()
    assert mon.max_cpu_percent > old


def test_mode_adapt_backs_off_under_high_pressure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = parse(Config, {"mode": "performance", "output_dir": str(tmp_path)})
    mon = ResourceMonitor()
    mon.configure(cfg)
    mon.cpus = 8
    monkeypatch.setattr(mon, "snapshot", lambda: Snapshot(80.0, 16_000 * MB, 500 * MB, 10_000 * MB))
    old = mon.max_cpu_percent
    mon.record_job(1.0, peak_rss=10_000 * MB, threads=8, workers=8)
    mon._adapt_ts = 0.0
    mon.adapt()
    assert mon.max_cpu_percent < old


def test_notify_oom_backs_off_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = parse(Config, {"mode": "performance", "output_dir": str(tmp_path)})
    mon = ResourceMonitor()
    mon.configure(cfg)
    mon.cpus = 8
    monkeypatch.setattr(mon, "snapshot", lambda: Snapshot(10.0, 32_000 * MB, 30_000 * MB, 100 * MB))
    old = mon.max_cpu_percent
    mon._adapt_ts = 0.0
    mon.notify_oom()
    mon.adapt()
    assert mon.max_cpu_percent < old
    assert mon._oom is False  # the OOM only backs off one step


@pytest.mark.parametrize(
    ("mode", "samples"),
    [
        ("performance", [80.0, 82.0, 84.0, 86.0]),  # rising, mid-range -> step_up * 0.5
        ("performance", [96.0, 96.0, 96.0, 96.0]),  # above target -> step_down
        ("performance", [85.0, 85.0, 85.0, 85.0]),  # flat, mid-range -> unchanged
        ("resources", [40.0, 40.0, 40.0, 40.0]),  # above target -> step_down
        ("resources", [10.0, 10.0, 10.0, 10.0]),  # below target -> step_up * 0.5
        ("resources", [25.0, 25.0, 25.0, 25.0]),  # flat, mid-range -> unchanged
        ("balance", [40.0, 40.0, 40.0, 40.0]),  # below target -> step_up
        ("balance", [76.0, 76.0, 76.0, 76.0]),  # above target -> step_down
        ("balance", [56.0, 58.0, 60.0, 62.0]),  # rising, mid-range -> step_up * 0.5
        ("balance", [70.0, 68.0, 66.0, 64.0]),  # falling, mid-range -> step_down * 0.5
        ("balance", [65.0, 65.0, 65.0, 65.0]),  # flat, mid-range -> unchanged
    ],
)
def test_adapt_covers_every_mode_branch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, samples: list[float]) -> None:
    cfg = parse(Config, {"mode": mode, "output_dir": str(tmp_path)})
    mon = ResourceMonitor()
    mon.configure(cfg)
    mon.cpus = 8
    monkeypatch.setattr(mon, "snapshot", lambda: Snapshot(10.0, 32_000 * MB, 30_000 * MB, 100 * MB))
    mon._cpu_samples.clear()
    mon._cpu_samples.extend(samples)
    mon._adapt_ts = 0.0
    mon.adapt()  # exercises the branch; the exact value is clamped to the mode range


def test_worker_plan_scales_with_free_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    per_job = 400 * MB
    plenty = make_monitor(monkeypatch, 12000).plan_workers(per_job, 6)
    assert (plenty.workers, plenty.threads) == (6, 1)
    two = make_monitor(monkeypatch, 12000).plan_workers(per_job, 2)
    assert (two.workers, two.threads) == (2, 4)  # leftover cores become encoder threads
    tight = make_monitor(monkeypatch, 1000).plan_workers(per_job, 6)
    assert (tight.workers, tight.threads) == (1, 8)  # one job, but every core still used
    gpu = make_monitor(monkeypatch, 12000).plan_workers(per_job, 6, limit=3)  # a GPU allows only a few encode sessions
    assert (gpu.workers, gpu.threads) == (3, 2)
    assert make_monitor(monkeypatch, 12000, max_cpu_percent=50).cpu_budget() == 4


def test_cache_budget_follows_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    assert make_monitor(monkeypatch, 100).cache_budget() == 4 * MB
    assert 8 * MB <= make_monitor(monkeypatch, 12000).cache_budget() <= 256 * MB
    fixed = ResourceMonitor()
    fixed.configure()
    # No hard-coded tier remains now: runtime auto caps the cache from usable memory.
    assert fixed.cache_budget() >= 8 * MB


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


class FakeMsvcrt(types.ModuleType):
    """Stand-in for the Windows-only module: records the lock calls, optionally fails one."""

    LK_LOCK = 1
    LK_UNLCK = 2

    def __init__(self, *, fail_lock: bool = False, fail_unlock: bool = False) -> None:
        super().__init__("msvcrt")
        self.fail_lock = fail_lock
        self.fail_unlock = fail_unlock
        self.calls: list[tuple[int, int]] = []

    def locking(self, fd: int, mode: int, nbytes: int) -> None:
        self.calls.append((mode, nbytes))
        if (mode == self.LK_LOCK and self.fail_lock) or (mode == self.LK_UNLCK and self.fail_unlock):
            raise OSError("the byte is locked by another process")


def shared_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: FakeMsvcrt) -> Path:
    """Configure file logging as if this were Windows, where appends need an explicit lock."""
    monkeypatch.setattr(log_module, "_appends_need_locking", lambda: True)
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    log_file = tmp_path / "shared.log"
    setup_logging(log_file, "INFO", None)
    return log_file


def test_a_shared_log_file_locks_every_record_where_appends_are_not_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeMsvcrt()
    log_file = shared_log(tmp_path, monkeypatch, fake)
    LOG.info("first")
    LOG.info("second")
    assert fake.calls == [(fake.LK_LOCK, 1), (fake.LK_UNLCK, 1)] * 2
    text = log_file.read_text(encoding="utf-8")
    assert "first" in text and "second" in text


def test_a_record_is_written_unlocked_when_the_shared_lock_cannot_be_taken(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeMsvcrt(fail_lock=True)
    log_file = shared_log(tmp_path, monkeypatch, fake)
    LOG.info("still written")
    assert fake.calls == [(fake.LK_LOCK, 1)]  # there is no lock to release
    assert "still written" in log_file.read_text(encoding="utf-8")


def test_a_failed_unlock_does_not_lose_the_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeMsvcrt(fail_unlock=True)
    log_file = shared_log(tmp_path, monkeypatch, fake)
    LOG.info("written before the unlock")
    assert fake.calls == [(fake.LK_LOCK, 1), (fake.LK_UNLCK, 1)]
    assert "written before the unlock" in log_file.read_text(encoding="utf-8")


def test_a_closed_shared_handler_drops_the_record_quietly(tmp_path: Path) -> None:
    handler = log_module._SharedFileHandler(tmp_path / "closed.log", encoding="utf-8")
    handler.close()
    handler.emit(logging.LogRecord("autoclip", logging.INFO, __file__, 1, "lost", (), None))
    assert (tmp_path / "closed.log").read_text(encoding="utf-8") == ""


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


def test_kv_evict_evicts_everything_when_the_cap_is_zero(tmp_path: Path) -> None:
    s = Store(tmp_path / "s2.db", fresh=True)
    s.kv_put("a", b"x")
    s.kv_evict(0)
    assert s.kv_get("a") is None
    s.close()


def test_kv_evict_keeps_deleting_and_exhausts_the_table_under_a_negative_cap(tmp_path: Path) -> None:
    s = Store(tmp_path / "s3.db", fresh=True)
    s.kv_put("a", b"x")
    s.kv_put("b", b"y")
    s.kv_evict(-1000)  # a negative budget can never be reached: every row goes
    assert s.kv_get("a") is None and s.kv_get("b") is None
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
    mon.configure()
    mon.memory_fraction = 0.000001  # force a tiny runtime budget
    store = Store(tmp_path / "a.db")
    cache = Cache(store, mon)
    arr = np.arange(1000, dtype=np.float32)
    calls: list[int] = []

    def produce() -> NDArray[np.float32]:
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


def test_logging_file_syncs_to_disk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    synced: list[int] = []
    monkeypatch.setattr(os, "fsync", synced.append)
    log_file = tmp_path / "logs" / "sync.log"
    setup_logging(log_file, "INFO", None)
    LOG.info("sync-test")
    assert synced, "the file handler must fsync before returning"


def test_logging_with_a_non_standard_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import sys

    stream = io.StringIO()  # not a TextIOWrapper (some IDEs / service managers replace stdout like this)
    monkeypatch.setattr(sys, "stdout", stream)
    setup_logging(None, "INFO", "INFO")
    LOG.info("hello")
    assert "hello" in stream.getvalue()


def test_log_redaction_helpers() -> None:
    from urllib.parse import urlsplit

    from autoclip.core.log import redact_url, safe_path_text, system_profile

    token = "1234567890abcdefghijklmnopqrstuvwxyz"
    base = f"https://example.com/watch?v=abcd&token={token}#frag"
    redacted = redact_url(base)
    parsed = urlsplit(redacted)
    assert "frag" not in redacted
    assert f"token={token}" not in redacted
    assert parsed.scheme == "https" and parsed.netloc == "example.com"
    assert "token=%3Credacted%3E" in redacted or "token=<redacted>" in redacted
    assert "v=abcd" in redacted

    unrelated = redact_url("https://example.com/foo/bar?sig=hidden&k=v")
    assert "hidden" not in unrelated
    assert "sig=%3Credacted%3E" in unrelated or "sig=<redacted>" in unrelated

    home = Path.home()
    text = safe_path_text(home / "foo" / "bar.mp4")
    assert text.startswith("~/")
    assert str(home) not in text

    summary = system_profile(output_dir=Path("/tmp/foo"), log_path=Path("/tmp/foo/x.log"))
    assert "ram_total=" in summary and "os=" in summary and "cpus=" in summary
    assert "password" not in summary.lower()
    assert "/home/" not in summary or summary.lower().startswith("os=")
    # A URL without a query/fragment is returned unchanged; an unparsable one is returned verbatim.
    assert redact_url("https://example.com/plain") == "https://example.com/plain"
    assert redact_url("http://[::1") == "http://[::1"


def test_system_profile_and_health_include_cgroup(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core import log as log_module
    from autoclip.core import resources

    monkeypatch.setattr(resources, "cgroup_memory_available", lambda: 123 * MB)
    assert "cgroup_free=123MB" in log_module.system_profile()
    assert "cgroup_free=123MB" in log_module.health_check("x")


def test_traced_covers_async_and_bootstrap_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from autoclip.core import log as log_module

    @log_module.traced("async-op")
    async def add(a: int, b: int) -> int:
        return a + b

    assert asyncio.run(add(1, 2)) == 3

    @log_module.traced("async-fail")
    async def boom() -> int:
        raise RuntimeError("no")

    with pytest.raises(RuntimeError, match="no"):
        asyncio.run(boom())

    monkeypatch.setattr(log_module, "_resources", lambda: None)

    @log_module.traced("bootstrap-op")
    def plain() -> str:
        return "ok"

    assert plain() == "ok"

    @log_module.traced("bootstrap-async")
    async def plain_async() -> str:
        return "ok"

    assert asyncio.run(plain_async()) == "ok"


def test_stdout_tee_mirrors_and_survives_broken_streams() -> None:
    import io

    from autoclip.core.log import _StdoutTee

    mirror = io.StringIO()
    logger = logging.getLogger("autoclip.tee-test")
    tee = _StdoutTee(mirror, logger)
    assert tee.write("hello\nworld\n") == 12
    assert "hello" in mirror.getvalue() and "world" in mirror.getvalue()
    tee.write("\n\n")  # empty lines are dropped
    tee.write("partial line")  # no newline: flush must emit the buffered text
    tee.flush()
    assert tee.isatty() is False

    # A mirror that raises must not break the tee.
    class Broken:
        def write(self, s: str) -> int:
            raise OSError("closed")

        def flush(self) -> None:
            raise ValueError("closed")

        def isatty(self) -> bool:
            raise OSError("closed")

    broken = _StdoutTee(Broken(), logger)  # type: ignore[arg-type]
    assert broken.write("x\n") == 2
    broken.flush()
    assert broken.isatty() is False
    assert _StdoutTee(None, logger).isatty() is False


def test_record_job_folds_own_current_rss_into_the_peak() -> None:
    mon = ResourceMonitor()
    mon.configure()
    assert mon.peak_rss == 0
    mon.record_job(1.0, peak_rss=0, threads=1, workers=1)
    assert mon.peak_rss > 0  # the parent always folds in its own RSS


def test_record_job_keeps_an_external_peak_when_it_is_larger() -> None:
    mon = ResourceMonitor()
    mon.configure()
    mon.record_job(1.0, peak_rss=10**12, threads=1, workers=1)
    assert mon.peak_rss == 10**12


def test_configure_restores_persisted_samples(tmp_path: Path) -> None:
    import json

    state_dir = tmp_path / ".work"
    state_dir.mkdir()
    (state_dir / "adaptive.json").write_text(
        json.dumps({"mode": "balance", "cpu_percent": 50.0, "samples": [10.0, 20.0, 30.0, 999.0, "bad"]}),
        encoding="utf-8",
    )
    cfg = parse(Config, {"mode": "balance", "output_dir": str(tmp_path)})
    mon = ResourceMonitor()
    mon.configure(cfg)
    assert list(mon._cpu_samples) == [10.0, 20.0, 30.0, 999.0]  # strings and overflow are dropped


def test_admission_goes_with_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    mon = make_monitor(monkeypatch, free_mb=12000)
    d = mon.admission("render", 100)
    assert (d.value, d.source) == ("go", "headroom")


def test_admission_defers_when_free_memory_is_below_the_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    mon = make_monitor(monkeypatch, free_mb=100)
    d = mon.admission("render", 10 * 1024 * 1024 * 1024)
    assert (d.value, d.source) == ("defer", "cost")


def test_admission_skips_a_cosmetic_stage_under_high_pressure(monkeypatch: pytest.MonkeyPatch) -> None:
    mon = make_monitor(monkeypatch, free_mb=10)  # 10 MB of 16 GB: high pressure
    assert mon.admission("thumbnail", 0).value == "skip"
    assert mon.admission("render", 0).value == "go"  # not cosmetic: never skipped


def test_admission_defers_after_an_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    mon = make_monitor(monkeypatch, free_mb=12000)
    mon._oom = True
    d = mon.admission("render", 0)
    assert (d.value, d.source) == ("defer", "ram-oom")


def test_admission_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    mon = ResourceMonitor()

    def broken() -> Any:
        raise RuntimeError("unmeasurable")

    monkeypatch.setattr(mon, "snapshot", broken)
    d = mon.admission("render", 0)
    assert (d.value, d.source) == ("go", "monitor-unavailable")


def test_admission_logs_its_decision(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    mon = make_monitor(monkeypatch, free_mb=10)
    with caplog.at_level(logging.INFO, logger="autoclip"):
        mon.admission("thumbnail", 0)
    assert "automation admission stage=thumbnail action=skip | source=ram-high" in caplog.text


def test_snapshot_uses_the_cgroup_limit_as_the_pressure_denominator(monkeypatch: pytest.MonkeyPatch) -> None:
    mon = ResourceMonitor()
    mon.configure()
    monkeypatch.setattr(mon, "_proc", type("_P", (), {"memory_info": lambda self: type("_M", (), {"rss": 10})()})())
    monkeypatch.setattr("autoclip.core.resources.psutil.virtual_memory", type("_V", (), {"total": 64 << 30, "available": 60 << 30}))  # noqa: PLW0108
    monkeypatch.setattr("autoclip.core.resources.cgroup_memory_available", lambda root=None: 2 << 30)
    monkeypatch.setattr("autoclip.core.resources._cgroup_memory_limit", lambda root: 2 << 30)
    snap = mon.snapshot()
    assert snap.mem_total == (2 << 30)  # the cgroup ceiling, not the host's 64 GB
    assert snap.mem_available == 2 << 30  # cgroup ceiling wins over the host's headroom
