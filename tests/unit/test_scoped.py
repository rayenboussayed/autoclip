from __future__ import annotations

from autoclip.core.events import STAGES, Progress, Reporter


def test_scoped_reporters_advance_through_the_whole_batch() -> None:
    seen: list[Progress] = []
    root = Reporter(seen.append)
    for index in (1, 2, 3):
        scoped = root.scoped(index, 3)
        for stage in STAGES:
            scoped(stage, 1.0)
    pcts = [p.percent for p in seen]
    firsts = [p.percent for p in seen if p.stage == "package"]
    assert pcts == sorted(pcts) and pcts[-1] == 100.0
    assert firsts == [33.3, 66.7, 100.0]


def test_single_input_is_unchanged() -> None:
    seen: list[Progress] = []
    r = Reporter(seen.append).scoped(1, 1)
    r("validate", 1.0)
    r("package", 1.0)
    assert [p.percent for p in seen] == [5.0, 100.0]
