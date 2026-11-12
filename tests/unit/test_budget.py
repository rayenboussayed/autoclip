"""Unit tests for the pure admission policy (``autoclip.core.internal.budget``)."""

from __future__ import annotations

import pytest

from autoclip.core.internal.budget import Pressure, decide_admission


def test_decisions_are_frozen() -> None:
    from dataclasses import FrozenInstanceError

    d = decide_admission("thumbnail", 1, Pressure(4096, 8192, True, False))
    with pytest.raises(FrozenInstanceError):
        d.name = "other"  # type: ignore[misc]


def test_memory_oom_defers_every_stage_including_cosmetic_ones() -> None:
    oom = Pressure(4096, 8192, False, True)
    assert decide_admission("render", 10, oom).value == "defer"
    assert decide_admission("thumbnail", 0, oom).source == "ram-oom"


def test_high_pressure_skips_only_cosmetic_stages() -> None:
    high = Pressure(4096, 8192, True, False)
    assert decide_admission("thumbnail", 0, high).value == "skip"
    assert decide_admission("render", 0, high).value == "go"


def test_a_skip_loses_to_a_real_oom_reason() -> None:
    both = Pressure(4096, 8192, True, True)
    assert decide_admission("thumbnail", 0, both).value == "defer"  # the OOM reason wins


def test_cost_over_free_memory_defers() -> None:
    squeezed = Pressure(512, 8192, False, False)
    assert decide_admission("render", 10 << 30, squeezed).value == "defer"
    assert decide_admission("render", 10 << 30, squeezed).source == "cost"


def test_headroom_goes() -> None:
    fat = Pressure(1 << 34, 1 << 35, False, False)
    d = decide_admission("render", 0, fat)
    assert (d.value, d.source) == ("go", "headroom")
    assert decide_admission("thumbnail", 1 << 33, Pressure(1 << 34, 1 << 35, True, False)).value == "skip"  # cosmetic drop wins over cost
    assert decide_admission("unknown-stage", 0, Pressure(1, 100, False, False)).value == "go"


def test_low_pressure_does_not_skip() -> None:
    low = Pressure(1 << 30, 1 << 34, False, False)
    assert decide_admission("thumbnail", 0, low).value == "go"
