from __future__ import annotations

import av
from pathlib import Path

import pytest

from autoclip.core.config import Config
from autoclip.core.hardware import cpu_encoder
from autoclip.core.media import probe
from autoclip.core.pipeline import process_video
from autoclip.core.unsilence import _pydub_available, remove_silence

from ..support import Media, make_cfg, write_config


pytest.importorskip("pydub")


def _skip_without_audioop() -> None:
    if not _pydub_available():
        pytest.skip("pydub/audioop is not available")


def test_remove_silence_creates_cleaner_video(media: Media, tmp_path: Path) -> None:
    _skip_without_audioop()
    cfg = make_cfg(media, tmp_path, None)
    cfg.unsilence.enabled = True
    destination = tmp_path / "long" / (media.video.stem + ".mp4")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = destination.parent / f"unsilence_{media.video.stem}.mp4"
    returned = remove_silence(media.video, cfg, destination)
    assert returned == destination
    assert destination.is_file() and destination.stat().st_size > 0
    assert probe(destination).duration <= probe(media.video).duration + 1e-6


def test_disabled_by_default_returns_original(media: Media, tmp_path: Path) -> None:
    cfg = make_cfg(media, tmp_path, None)
    cfg.unsilence.enabled = False
    destination = tmp_path / "unsilence_test.mp4"
    returned = remove_silence(media.video, cfg, destination)
    assert returned is media.video
    assert not destination.exists()


def test_process_video_runs_after_unsilence(media: Media, tmp_path: Path) -> None:
    _skip_without_audioop()
    cfg = make_cfg(media, tmp_path, None, unsilence={"enabled": True, "min_silence_ms": 300})
    out = tmp_path / "out"
    cfg.output_dir = str(out)
    (done, failed) = process_video(media.video, cfg, None, cpu_encoder(cfg.video), name=media.video.stem, consent=None)
    assert failed == 0
    assert done > 0
    expected = out / "long" / media.video.stem / f"unsilence_{media.video.stem}.mp4"
    assert expected.is_file()
