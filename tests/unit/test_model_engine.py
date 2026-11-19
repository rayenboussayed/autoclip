"""Automatic, resource-aware model lifecycle policy."""

from types import SimpleNamespace

import autoclip.core.model_engine as engine
from autoclip.core.resources import MB, RESOURCES, Snapshot


def _snapshot(available_mb: int) -> Snapshot:
    return Snapshot(cpu_percent=10.0, mem_total=8192 * MB, mem_available=available_mb * MB, rss=100 * MB)


def test_high_memory_pressure_selects_a_bounded_worker(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(RESOURCES, "snapshot", lambda: _snapshot(100))
    assert engine.choose_model("whisper", 4000).subprocess


def test_roomy_machine_keeps_the_model_in_process(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(RESOURCES, "snapshot", lambda: _snapshot(8000))
    monkeypatch.setattr(engine, "vram_free_mb", lambda: None)
    assert not engine.choose_model("whisper", 4000).subprocess


def test_tests_parse_nvidia_smi_output(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(engine.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="1500, 4000\n"))  # type: ignore[attr-defined]
    assert engine.vram_free_mb() == 2500
    monkeypatch.setattr(engine.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))  # type: ignore[attr-defined]
    assert engine.vram_free_mb() is None


def test_vram_pressure_selects_a_bounded_worker(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(RESOURCES, "snapshot", lambda: _snapshot(8000))
    monkeypatch.setattr(engine, "vram_free_mb", lambda: 1000)
    assert engine.choose_model("stable-diffusion", 3500).subprocess


def test_release_allocations_is_callable_with_or_without_torch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setitem(__import__("sys").modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    engine.release_allocations()


def test_release_works_when_torch_is_not_importable(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    engine.release_allocations()


def test_vram_probe_handles_a_malformed_nvidia_smi_reply(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(engine.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="not\n-a-number\n"))  # type: ignore[attr-defined]
    assert engine.vram_free_mb() is None
    monkeypatch.setattr(engine.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("missing")))  # type: ignore[attr-defined]
    assert engine.vram_free_mb() is None


def test_low_ram_with_no_vram_still_selects_a_bounded_worker(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(RESOURCES, "snapshot", lambda: _snapshot(1000))
    monkeypatch.setattr(engine, "vram_free_mb", lambda: None)
    assert engine.choose_model("large-model", 4000).subprocess


def test_release_all_tolerates_the_empty_case(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    engine.release_allocations()
