"""In-process memory-regression budgets — measured in a child Python process for isolation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from . import budgets
from .harness import peak_of, run_python


def _ensure_sidecar(target: Path, words: list[str], *, start: float = 2.0, step: float = 4.0) -> None:

    target.with_suffix(".json").write_text(
        json.dumps({"words": [{"start": start + i * step, "end": start + i * step + 3.0, "text": s} for i, s in enumerate(words)]}),
        encoding="utf-8",
    )


@pytest.mark.memory
def test_process_video_stays_within_budget(media, tmp_path: Path, memory_results: Path) -> None:  # type: ignore[no-untyped-def]
    """process_video on the 40 s fixture — parent+children tree RSS must fit the budget."""
    _ensure_sidecar(media.video, [f"sentence number {i} is here." for i in range(10)])
    workdir = memory_results / "process-video-out"
    snippet = """
import sys
import time
from pathlib import Path
from autoclip.core.config import FileConfig, RuntimeOverrides, parse
from autoclip.core.media import probe
from autoclip.core.models import Encoder
from autoclip.core.pipeline import process_video
from autoclip.core.log import setup_logging

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
setup_logging(out / "run.log", "INFO", None)
file_cfg = parse(FileConfig, {"input": [str(src)], "clips": {"preset": "shorts", "scenes": "best", "ranges": ["00:00:02-00:00:12"]}})
cfg = file_cfg.to_config(RuntimeOverrides(output_dir=str(out)))
cfg.quality = "lowest"
encoder = Encoder("libx264", False, "ultrafast", ())
n, bad = process_video(src, cfg, out / "run.log", encoder, name="task.mp4")
print("done:", n, "failed:", bad)
"""
    code, samples = run_python(snippet, [str(media.video), str(workdir)], out_dir=tmp_path / "case-process-video")
    assert code == 0, f"the probe exited {code}"
    assert list(workdir.glob("short/*/clips.json")), "no clips were produced"
    peak = peak_of(samples)
    assert len(samples) >= budgets.MIN_SAMPLES
    assert peak >= budgets.MIN_PEAK_MB * 1024 * 1024
    assert peak / (1024 * 1024) <= budgets.PROCESS_VIDEO_PEAK_MB, (
        f"tree peak {peak / (1024 * 1024):.1f} MB exceeded {budgets.PROCESS_VIDEO_PEAK_MB} MB"
    )


@pytest.mark.memory
def test_unsilence_many_parts_stays_flat(many_parts_video: Path, tmp_path: Path, memory_results: Path) -> None:
    """The many-parts unsilence re-encode must stay tiny (chain, not compose)."""
    workdir = memory_results / "unsilence-out"
    snippet = """
import sys
from pathlib import Path
import autoclip.core.unsilence as unsilence_module
unsilence_module.UNSILENCE_MIN_TOTAL_SECONDS = 0.1
from autoclip.core.config import parse, Config
from autoclip.core.unsilence import remove_silence
from autoclip.core.log import setup_logging
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
setup_logging(out / "run.log", "INFO", None)
cfg = parse(Config, {"quality": "lowest"})
result = remove_silence(src, cfg, out / "clean.mp4")
print("out:", result)
"""
    code, samples = run_python(snippet, [str(many_parts_video), str(workdir)], out_dir=tmp_path / "case-unsilence")
    assert code == 0
    peak = peak_of(samples)
    assert len(samples) >= budgets.MIN_SAMPLES
    assert peak >= budgets.MIN_PEAK_MB * 1024 * 1024
    assert peak / (1024 * 1024) <= budgets.UNSILENCE_MANY_PARTS_PEAK_MB, (
        f"tree peak {peak / (1024 * 1024):.1f} MB exceeded {budgets.UNSILENCE_MANY_PARTS_PEAK_MB} MB"
    )


@pytest.mark.memory
def test_render_720p_medium_stays_within_budget(p720_video: Path, tmp_path: Path, memory_results: Path) -> None:
    """Rendering a short clip from a 720p source at quality medium must fit the render budget."""
    _ensure_sidecar(p720_video, [f"sentence number {i} is here." for i in range(6)], start=2.0, step=3.0)
    workdir = memory_results / "render-720p-out"
    snippet = """
import sys
from pathlib import Path
from autoclip.core.config import FileConfig, RuntimeOverrides, parse
from autoclip.core.models import Encoder
from autoclip.core.pipeline import process_video
from autoclip.core.log import setup_logging

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
setup_logging(out / "run.log", "INFO", None)
file_cfg = parse(FileConfig, {"input": [str(src)], "clips": {"preset": "shorts", "scenes": "best", "ranges": ["00:00:02-00:00:12"]}})
cfg = file_cfg.to_config(RuntimeOverrides(output_dir=str(out)))
cfg.quality = "medium"
encoder = Encoder("libx264", False, "veryfast", ())
n, bad = process_video(src, cfg, out / "run.log", encoder, name="task.mp4")
print("done:", n, "failed:", bad)
"""
    code, samples = run_python(snippet, [str(p720_video), str(workdir)], out_dir=tmp_path / "case-render-720p")
    assert code == 0
    assert list(workdir.glob("short/*/clips.json")), "no clips were produced"
    peak = peak_of(samples)
    assert len(samples) >= budgets.MIN_SAMPLES
    assert peak >= budgets.MIN_PEAK_MB * 1024 * 1024
    assert peak / (1024 * 1024) <= budgets.RENDER_720P_PEAK_MB, (
        f"tree peak {peak / (1024 * 1024):.1f} MB exceeded {budgets.RENDER_720P_PEAK_MB} MB"
    )
