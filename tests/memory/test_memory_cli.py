"""CLI-subprocess memory-regression budgets (M4, M5) — Linux-only, sequential."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from . import budgets
from .harness import peak_of, run_command


def _sidecar_for(video: Path) -> None:
    if video.with_suffix(".json").exists():
        return
    Path(str(video.with_suffix(".json"))).write_text(
        json.dumps({"words": [{"start": i * 4.0, "end": i * 4.0 + 3.0, "text": f"sentence number {i} is here."} for i in range(10)]}),
        encoding="utf-8",
    )


def _write_config(target: Path, video: Path) -> None:
    target.write_text(
        json.dumps({"input": [str(video)], "clips": {"preset": "shorts", "scenes": "best", "ranges": ["00:00:02-00:00:12"]}}),
        encoding="utf-8",
    )


def _run_cli(video: Path, workdir: Path, out_dir: Path) -> tuple[int, list[int]]:
    cfg = workdir / "run.json"
    _sidecar_for(video)
    _write_config(cfg, video)
    return run_command(
        [sys.executable, "-m", "autoclip", "--output-dir", str(out_dir), "--quality", "lowest", "--skip", "caption,title,llm", str(cfg)],
        out_dir=out_dir,
    )


@pytest.mark.memory
def test_cli_fixture_stays_within_budget(media, tmp_path: Path, memory_results: Path) -> None:  # type: ignore[no-untyped-def]
    """A real subprocess of the CLI must stay within the tree budget on the fixture video."""
    out_dir = memory_results / "cli-fixture-out"
    code, samples = _run_cli(media.video, tmp_path, out_dir)
    assert code == 0
    assert list(out_dir.glob("short/*/clips.json")), "no clips were produced"
    peak = peak_of(samples)
    assert len(samples) >= budgets.MIN_SAMPLES
    assert peak >= budgets.MIN_PEAK_MB * 1024 * 1024
    assert peak / (1024 * 1024) <= budgets.CLI_TREE_PEAK_MB, (
        f"tree peak {peak / (1024 * 1024):.1f} MB exceeded {budgets.CLI_TREE_PEAK_MB} MB"
    )


@pytest.mark.memory
def test_cli_many_parts_stays_within_budget(many_parts_video: Path, tmp_path: Path, memory_results: Path) -> None:
    """The CLI subprocess on the 120 s many-parts fixture stays within the tree budget (no whisper required)."""
    out_dir = memory_results / "cli-many-parts-out"
    code, samples = _run_cli(many_parts_video, tmp_path, out_dir)
    assert code == 0
    assert list(out_dir.glob("short/*/clips.json")), "no clips were produced"
    peak = peak_of(samples)
    assert len(samples) >= budgets.MIN_SAMPLES
    assert peak >= budgets.MIN_PEAK_MB * 1024 * 1024
    assert peak / (1024 * 1024) <= budgets.CLI_MANY_PARTS_TREE_MB, (
        f"tree peak {peak / (1024 * 1024):.1f} MB exceeded {budgets.CLI_MANY_PARTS_TREE_MB} MB"
    )
