"""End-to-end check that a cue stays locked to the video when silence removal rebuilds the source.

Offline: uses the synthetic `media` fixture and a sidecar transcript, no YouTube or Whisper.
Opt in with ``AUTOCLIP_E2E_TESTS=1 pytest tests/e2e/test_timeline_e2e.py -m e2e --no-cov``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoclip.core.fonts import find_font
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


def test_raw_clip_is_full_length_and_captioned(media: Media, e2e_run: E2ERun, tmp_path: Path) -> None:
    """The raw preset produces one captioned file covering the whole silence-removed source."""
    source = tmp_path / "raw.mp4"
    source.write_bytes(media.video.read_bytes())
    srt = tmp_path / "raw.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:39,000\nRAW CAPTION\n", encoding="utf-8")
    out = tmp_path / "out-raw"
    config = pipeline_config(out, source, transcript=srt, clips={"preset": "raw"})
    e2e_run.execute("raw-caption", config).require_ok()

    clips = sorted(p for p in out.glob("raw/*/*.mp4") if not p.stem.endswith("_uncaptioned"))
    assert len(clips) == 1
    assert not (out / "short").exists() and not (out / "long").exists()
    cues = parse_srt(clips[0].with_suffix(".srt").read_text(encoding="utf-8"))
    assert cues and " ".join(cue["text"] for cue in cues) == "RAW CAPTION"
    assert cues[-1]["end"] == pytest.approx(probe(clips[0]).duration, abs=0.3)

    copies = sorted(p for p in out.glob("raw/*/*.mp4") if p.stem.endswith("_uncaptioned"))
    assert len(copies) == 1
    full = copies[0].with_suffix(".srt")
    assert full.is_file()
    full_cues = parse_srt(full.read_text(encoding="utf-8"))
    assert full_cues and " ".join(cue["text"] for cue in full_cues) == "RAW CAPTION"
    assert full_cues[-1]["end"] == pytest.approx(probe(copies[0]).duration, abs=2.0)


def test_default_run_also_exports_the_raw_folder(media: Media, e2e_run: E2ERun, tmp_path: Path) -> None:
    """Even a default, non-raw preset writes the silence-removed source and its full transcript to raw/."""
    source = tmp_path / "default.mp4"
    source.write_bytes(media.video.read_bytes())
    srt = tmp_path / "default.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:39,000\nDEFAULT CAPTION\n", encoding="utf-8")
    out = tmp_path / "out-default"
    config = pipeline_config(out, source, transcript=srt)  # the harness default preset is 'shorts'
    e2e_run.execute("always-raw", config).require_ok()

    raw = next(out.glob("raw/*"))
    stems = {p.stem for p in raw.glob("*.mp4")}
    stem = source.stem
    assert {f"{stem}_raw_01_0s", f"{stem}_raw_01_0s_uncaptioned"} <= stems
    assert (raw / f"{stem}_raw_01_0s.jpg").is_file()
    full = (raw / f"{stem}_raw_01_0s_uncaptioned.srt").read_text(encoding="utf-8")
    assert parse_srt(full) and "DEFAULT CAPTION" in full
    assert len(list(out.glob("short/*/*.mp4"))) == 1


def test_all_captions_share_one_font(media: Media, e2e_run: E2ERun, tmp_path: Path) -> None:
    """A real CLI render chooses one caption font for the whole clip, logged once per build."""
    if find_font("HELLO 123 CAFÉ") is None:
        pytest.skip("no system font installed")
    source = tmp_path / "font.mp4"
    source.write_bytes(media.video.read_bytes())
    srt = tmp_path / "font.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\nHELLO 123\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\nCAFÉ\n\n"
        "3\n00:00:08,000 --> 00:00:10,000\nWORLD AGAIN\n",
        encoding="utf-8",
    )
    out = tmp_path / "out-font"
    config = pipeline_config(out, source, transcript=srt, clips={"preset": "raw"})
    result = e2e_run.execute("one-font", config).require_ok()

    assert result.log_path is not None
    chosen = [
        line.split("caption font: ", 1)[1] for line in result.log_path.read_text(encoding="utf-8").splitlines() if "caption font: " in line
    ]
    assert len(chosen) == 1 and chosen[0] != "None"
    assert sorted(out.glob("raw/*/*.mp4"))
