"""End-to-end check that a cue stays locked to the video when silence removal rebuilds the source.

Offline: uses the synthetic `media` fixture and a sidecar transcript, no YouTube or Whisper.
Opt in with ``AUTOCLIP_E2E_TESTS=1 pytest tests/e2e/test_timeline_e2e.py -m e2e --no-cov``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoclip.core.media import probe
from autoclip.core.transcript import parse_srt

from ..support import Media
from .harness import E2ERun, pipeline_config

pytestmark = pytest.mark.e2e


def test_tail_cue_matches_the_rendered_clip(media: Media, e2e_run: E2ERun, tmp_path: Path) -> None:
    """The last caption of an exact range must not drift past the end of the rendered clip."""
    source = tmp_path / "cue.mp4"
    source.write_bytes(media.video.read_bytes())
    srt = tmp_path / "cue.srt"
    srt.write_text("1\n00:00:36,000 --> 00:00:39,000\nLAST CUE\n", encoding="utf-8")
    out = tmp_path / "out"
    config = pipeline_config(
        out,
        source,
        transcript=srt,
        clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 2, "max_seconds": 6}, "ranges": ["00:00:36-00:00:39"]},
    )
    e2e_run.execute("tail-cue", config).require_ok()

    clips = sorted(out.glob("short/*/*.mp4"))
    assert len(clips) == 1
    sidecar = clips[0].with_suffix(".srt")
    cues = parse_srt(sidecar.read_text(encoding="utf-8"))
    assert cues and " ".join(cue["text"] for cue in cues) == "LAST CUE"
    assert cues[-1]["end"] == pytest.approx(probe(clips[0]).duration, abs=0.1)
