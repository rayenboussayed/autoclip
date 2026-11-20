"""The engine's pure rules: machine measurements in, decisions out (no host state involved)."""

from __future__ import annotations

import pytest

from autoclip.core.internal.auto import (
    JOB_RAM_MB,
    auto_max_concurrent_jobs,
    auto_mode,
    auto_ollama_model,
    auto_quality,
    auto_whisper_model,
)


@pytest.mark.parametrize(
    ("cpus", "free_ram_mb", "expected"),
    [
        (8, 8192, "high"),
        (4, 3072, "high"),  # the boundary of the comfortable tier
        (4, 3071, "medium"),
        (3, 3072, "medium"),  # cores alone can't buy "high"
        (2, 2048, "medium"),
        (2, 2047, "low"),
        (1, 512, "low"),
        (1, 16384, "low"),  # many GB but one core: encode is the bottleneck
        (16, 1024, "low"),  # many cores but almost no RAM (a full disk cache/encoder would thrash)
        (0, 8192, "low"),  # degenerate reading still produces a runnable answer
    ],
)
def test_auto_quality_tiers(cpus: int, free_ram_mb: float, expected: str) -> None:
    assert auto_quality(cpus=cpus, free_ram_mb=free_ram_mb) == expected


@pytest.mark.parametrize(
    ("cpus", "free_ram_mb", "expected"),
    [
        (16, 16384, "performance"),
        (8, 8192, "performance"),  # the boundary of the ample tier
        (8, 8191, "balance"),
        (7, 16384, "balance"),
        (4, 4096, "balance"),
        (2, 512, "resources"),  # two cores is the ceiling for "resources"
        (1, 64000, "resources"),  # even a fat RAM does not rescue a single core
        (64, 1024, "resources"),  # RAM pressure overrides raw cores
        (4, 1535, "resources"),  # below 1.5 GB free, stay lean
        (4, 1536, "balance"),
    ],
)
def test_auto_mode_tiers(cpus: int, free_ram_mb: float, expected: str) -> None:
    assert auto_mode(cpus=cpus, free_ram_mb=free_ram_mb) == expected


def test_auto_max_concurrent_jobs_is_memory_and_core_bound() -> None:
    assert auto_max_concurrent_jobs(cpus=16, free_ram_mb=64 * 1024) == 4  # capped at 4
    assert auto_max_concurrent_jobs(cpus=4, free_ram_mb=2 * JOB_RAM_MB) == 2  # RAM-bound
    assert auto_max_concurrent_jobs(cpus=2, free_ram_mb=64 * 1024) == 1  # core-bound (two cores per job)
    assert auto_max_concurrent_jobs(cpus=8, free_ram_mb=8 * 1024) == 4  # not core-bound at 8 cores
    assert auto_max_concurrent_jobs(cpus=1, free_ram_mb=100) == 1  # never zero
    assert auto_max_concurrent_jobs(cpus=0, free_ram_mb=0) == 1  # degenerate: still one


def test_auto_ollama_model_prefers_the_largest_that_fits() -> None:
    sizes = {"small": 1024**3, "medium": 4 * 1024**3, "large": 20 * 1024**3}
    free = 10 * 1024  # MB; the 0.5 fraction leaves room for KV cache and the pipeline
    assert auto_ollama_model(sizes, free) == "medium"


def test_auto_ollama_model_is_deterministic_on_equal_sizes() -> None:
    free = 64 * 1024
    sizes = {"zeta": 1024**3, "alpha": 1024**3, "middle": 1024**3}
    assert auto_ollama_model(sizes, free) == "alpha"


def test_auto_ollama_model_falls_back_to_the_smallest_when_none_fits() -> None:
    sizes = {"biggest": 90 * 1024**3, "huge": 40 * 1024**3, "big": 20 * 1024**3}
    assert auto_ollama_model(sizes, 2048) == "big"  # the smallest known model still runs


def test_auto_ollama_model_uses_alphabetical_order_when_sizes_are_unknown() -> None:
    assert auto_ollama_model({"zeta": 0, "alpha": 0, "mid": -1}, 8192) == "alpha"
    assert auto_ollama_model({"only": 0}, 8192) == "only"


def test_auto_ollama_model_with_no_models_installed() -> None:
    assert auto_ollama_model({}, 8192) is None
    assert auto_ollama_model({"a": 0, "b": 0}, 8192) == "a"  # all unknown: alphabetical


def test_auto_ollama_model_mixed_known_and_unknown() -> None:
    # the unknown size is ignored for the fit decision but never wins over a known small one
    sizes = {"unknown": 0, "known-small": 1024**3}
    assert auto_ollama_model(sizes, 10 * 1024) == "known-small"


def test_auto_rules_survive_nan_and_infinity_readings() -> None:
    assert auto_quality(cpus=4, free_ram_mb=float("nan")) == "low"  # no finite RAM reading: the safest tier
    assert auto_quality(cpus=4, free_ram_mb=float("inf")) == "low"  # absurd readings never buy quality either
    assert auto_quality(cpus=4, free_ram_mb=-1.0) == "low"
    assert auto_mode(cpus=8, free_ram_mb=float("nan")) == "resources"
    assert auto_mode(cpus=8, free_ram_mb=float("inf")) == "resources"  # absurd readings are never trusted as "ample"
    assert auto_max_concurrent_jobs(cpus=2, free_ram_mb=float("nan")) == 1
    assert auto_ollama_model({"a": 0, "b": 0}, float("nan")) == "a"  # all-unknown path under nonsense RAM
    assert auto_ollama_model({"a": 1024**3}, float("nan")) == "a"  # nothing fits : fall back to the smallest known


@pytest.mark.parametrize(
    ("device", "cpus", "ram", "expected"),
    [
        ("cuda", 4, 1024.0, "large-v3"),
        ("cuda", 16, 32768.0, "large-v3"),
        ("cpu", 16, 16384.0, "medium"),
        ("cpu", 8, 8192.0, "medium"),
        ("cpu", 7, 8192.0, "small"),
        ("cpu", 8, 8191.0, "small"),
        ("cpu", 4, 4096.0, "small"),
        ("cpu", 16, float("nan"), "small"),
    ],
)
def test_auto_whisper_model(device: str, cpus: int, ram: float, expected: str) -> None:
    """CPU medium is capped to performance-class machines; CUDA always keeps large-v3."""
    assert auto_whisper_model(device=device, cpus=cpus, free_ram_mb=ram) == expected
