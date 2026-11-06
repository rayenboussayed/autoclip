"""Mode profiles for the performance/resources/balance setting."""

from __future__ import annotations

from dataclasses import dataclass

VALID_MODES = ("performance", "resources", "balance")


@dataclass(frozen=True)
class ModeProfile:
    name: str
    cpu_percent_min: float
    cpu_percent_max: float
    cpu_percent_default: float
    mem_fraction: float
    memory_reserve_mb: int
    encoding_threads_cap: int
    parallel: bool
    target_cpu_low: float
    target_cpu_high: float
    step_up: float
    step_down: float
    min_workers_for_parallel: bool


PROFILES: dict[str, ModeProfile] = {
    "performance": ModeProfile(
        name="performance",
        cpu_percent_min=70.0,
        cpu_percent_max=100.0,
        cpu_percent_default=90.0,
        mem_fraction=0.75,
        memory_reserve_mb=256,
        encoding_threads_cap=4,
        parallel=True,
        target_cpu_low=80.0,
        target_cpu_high=95.0,
        step_up=5.0,
        step_down=3.0,
        min_workers_for_parallel=True,
    ),
    "resources": ModeProfile(
        name="resources",
        cpu_percent_min=10.0,
        cpu_percent_max=40.0,
        cpu_percent_default=25.0,
        mem_fraction=0.30,
        memory_reserve_mb=1024,
        encoding_threads_cap=2,
        parallel=False,
        target_cpu_low=15.0,
        target_cpu_high=35.0,
        step_up=2.0,
        step_down=5.0,
        min_workers_for_parallel=False,
    ),
    "balance": ModeProfile(
        name="balance",
        cpu_percent_min=30.0,
        cpu_percent_max=80.0,
        cpu_percent_default=50.0,
        mem_fraction=0.50,
        memory_reserve_mb=512,
        encoding_threads_cap=4,
        parallel=True,
        target_cpu_low=55.0,
        target_cpu_high=75.0,
        step_up=4.0,
        step_down=4.0,
        min_workers_for_parallel=True,
    ),
}


def profile_for(mode: str) -> ModeProfile:
    if mode not in PROFILES:
        raise ValueError(f"unknown mode {mode!r}; valid: {', '.join(VALID_MODES)}")
    return PROFILES[mode]


__all__ = ["PROFILES", "VALID_MODES", "ModeProfile", "profile_for"]
