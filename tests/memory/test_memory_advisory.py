"""Advisory memory tests — informational, never fail the suite."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from tests.memory import budgets
from tests.memory.harness import CaseMetrics, peak_of, write_metrics


@pytest.mark.memory
def test_cli_fixture_wall_budget(media, tmp_path: Path, memory_results: Path) -> None:  # type: ignore[no-untyped-def]
    """The fixture lands the render far below the advisory wall budget on this machine."""
    import subprocess
    import sys

    cfg = tmp_path / "run.json"
    out = memory_results / "advisory-wall-out"
    out.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        json.dumps({"input": [str(media.video)], "clips": {"preset": "shorts", "scenes": "best", "ranges": ["00:00:02-00:00:12"]}}),
        encoding="utf-8",
    )
    begin = time.perf_counter()
    subprocess.run(
        [sys.executable, "-m", "autoclip", "--output-dir", str(out), "--quality", "lowest", "--skip", "caption,title,llm", str(cfg)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wall = time.perf_counter() - begin
    assert wall < budgets.CLI_WALL_SECONDS


@pytest.mark.memory
def test_metrics_json_schema(tmp_path: Path) -> None:
    """A metrics file written by the harness must round-trip through the schema the docs describe."""
    case = CaseMetrics("fixture", 1.0, 40.0, 80.0, 512.0, 1024.0, True)
    path = write_metrics(tmp_path, [case])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["cases"][0]["name"] == "fixture"
    assert payload["cases"][0]["budget_parent_mb"] == 512.0
    assert payload["python"] and payload["platform"]


@pytest.mark.memory
def test_enhance_audio_window_bounded(media, memory_results: Path, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """enhance_audio on the fixture stays within its (generous) advisory budget."""
    from .harness import run_python

    workdir = memory_results / "enhance-audio-out"
    snippet = """
import sys
from pathlib import Path
from autoclip.core.audio import enhance_audio
from autoclip.core.log import setup_logging

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
setup_logging(out / "run.log", "INFO", None)
audio = enhance_audio(src, 0.0, 40.0, 1)
print("size:", audio.size)
"""
    code, samples = run_python(snippet, [str(media.video), str(workdir)], out_dir=tmp_path / "case-enhance-audio")
    assert code == 0
    peak = peak_of(samples)
    assert len(samples) >= budgets.MIN_SAMPLES
    assert peak / (1024 * 1024) <= budgets.ENHANCE_AUDIO_PEAK_MB
