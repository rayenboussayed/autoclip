"""Pure decisions for the resource-aware adapter. No I/O, no state - ``ResourceMonitor`` owns the state.

The controller's log channel is ``automation <name>=<value> | source=<rule>`` (see ``internal/decisions.py``);
this module is the policy half of the admission check. It knows nothing about the engine, the file
layout or the config schema: callers hand it a measured snapshot and a stage name, and get back a
freezed decision object. That is what makes it coverable without any of the engine's harness.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["AutoDecision", "Pressure", "decide_admission"]

#: Stages whose work is cosmetic and regenerable; only these may be dropped outright under pressure.
SKIPPABLE_STAGES = frozenset({"thumbnail"})


@dataclass(frozen=True, slots=True)
class AutoDecision:
    """One automatic decision.

    ``source`` names the rule that produced it (``ram-oom``, ``ram-high``, ``cost``, ``headroom``,
    ``monitor-unavailable``), so the log line is greppable per cause.
    """

    name: str
    value: str
    source: str


@dataclass(frozen=True, slots=True)
class Pressure:
    """A snapshot the caller measured once; keeps this module syscall-free and fully testable."""

    free_bytes: int
    total_bytes: int
    high: bool
    oom: bool


def decide_admission(stage: str, cost_bytes: int, pressure: Pressure) -> AutoDecision:
    """Whether ``stage`` may run now.

    ``go`` runs it, ``defer`` waits for a finished clip to free memory (the pool never starves:
    when nothing is running the caller submits anyway), ``skip`` drops it entirely (only for
    cosmetic, regenerable stages - every other stage falls back to ``defer``).
    """
    if pressure.oom:
        return AutoDecision(stage, "defer", "ram-oom")
    if pressure.high and stage in SKIPPABLE_STAGES:
        return AutoDecision(stage, "skip", "ram-high")
    if cost_bytes > 0 and cost_bytes > pressure.free_bytes:
        return AutoDecision(stage, "defer", "cost")
    return AutoDecision(stage, "go", "headroom")
