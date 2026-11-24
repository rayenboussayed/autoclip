"""Integration check: a restarted run reads its checkpoint instead of rebuilding finished work."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from autoclip.core.hardware import pick_encoder
from autoclip.core.pipeline import process_video
from tests.support import Media, install_stub_whisper, make_cfg


def test_second_run_resumes_instead_of_rendering_again(media: Media, tmp_path: Path, monkeypatch: Any, caplog: Any) -> None:
    """Real encoder and real files: the second run must leave every rendered MP4 untouched."""
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    out = tmp_path / "out"
    cfg = make_cfg(media, out, clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    cfg.quality = "low"
    encoder = pick_encoder(cfg.video, cfg.quality, allow_gpu=False)

    assert process_video(media.video, cfg, None, encoder) == (2, 0)  # one short + one raw
    rendered = sorted(out.rglob("*.mp4"))
    assert len(rendered) == 3  # one short + the raw render + its uncaptioned copy
    before = {path: path.stat().st_mtime_ns for path in rendered}

    with caplog.at_level(logging.INFO, logger="autoclip"):
        assert process_video(media.video, cfg, None, encoder) == (2, 0)
    assert {path: path.stat().st_mtime_ns for path in rendered} == before
    assert "checkpoint: reusing 2 planned clips" in caplog.text
