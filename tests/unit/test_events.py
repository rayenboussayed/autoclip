"""Unit tests for autoclip.core.events."""

import threading
import time

import pytest

from autoclip.core.events import STAGES, Cancelled, CancelToken, Progress, Reporter


def test_percent_is_monotonic_and_bounded() -> None:
    seen: list[Progress] = []
    r = Reporter(seen.append)
    r("validate", 1.0)
    r("transcribe", 0.5)
    r("transcribe", 0.2)  # a lower fraction must not go backwards
    r("render", 1.0)
    r("package", 1.0)
    pcts = [p.percent for p in seen]
    assert pcts == sorted(pcts)
    assert pcts[-1] == 100.0
    assert all(0 <= p <= 100 for p in pcts)


def test_stage_ranges_are_contiguous() -> None:
    ends = [hi for _, hi in STAGES.values()]
    starts = [lo for lo, _ in STAGES.values()]
    assert starts[1:] == ends[:-1]
    assert starts[0] == 0.0
    assert ends[-1] == 100.0


def test_cancel_stops_at_next_report_and_cleanup_runs() -> None:
    tok = CancelToken()
    seen: list[Progress] = []
    cleaned: list[str] = []
    r = Reporter(seen.append, tok)

    def pipeline() -> None:
        try:
            r("validate", 1.0)
            r("transcribe", 1.0)
            tok.cancel()  # e.g. DELETE /api/jobs/{id} from another thread
            r("select", 0.5)  # must raise here, before any render work
            seen.append("RENDER STARTED")  # type: ignore[arg-type]
        finally:
            cleaned.append("store closed")

    with pytest.raises(Cancelled):
        pipeline()
    assert "RENDER STARTED" not in seen  # type: ignore[comparison-overlap]
    assert cleaned == ["store closed"]


def test_thread_safe_cancel() -> None:
    tok = CancelToken()
    r = Reporter(lambda p: None, tok)
    hit: list[bool] = []

    def worker() -> None:
        try:
            for _ in range(10_000):
                r("render", 0.5)
                time.sleep(0.0005)
        except Cancelled:
            hit.append(True)

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.05)
    tok.cancel()
    t.join(3)
    assert hit == [True]
    assert not t.is_alive()
