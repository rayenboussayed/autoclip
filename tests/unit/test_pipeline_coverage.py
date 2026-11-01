from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autoclip.core.pipeline import process_video
from tests.support import install_stub_whisper, make_cfg


def test_process_video_unsilence_returns_destination(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    import shutil

    out = tmp_path / "out"
    out.mkdir()
    cfg = make_cfg(media, out, clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    cfg.unsilence.enabled = True
    cfg.captions.enabled = False
    cfg.title.enabled = False

    destination = out / "long" / media.video.stem / f"unsilence_{media.video.stem}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(media.video, destination)
    monkeypatch.setattr("autoclip.core.unsilence.remove_silence", lambda src, cfg, dst, *a, **k: dst)

    rc = process_video(media.video, cfg, None, __import__("autoclip.core.hardware", fromlist=["cpu_encoder"]).cpu_encoder(cfg.video))
    assert rc[0] >= 0


def test_process_video_unsilence_exception(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    out = tmp_path / "out2"
    out.mkdir()
    cfg = make_cfg(media, out, clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    cfg.unsilence.enabled = True
    cfg.captions.enabled = False
    cfg.title.enabled = False

    def boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("autoclip.core.unsilence.remove_silence", boom)
    rc = process_video(media.video, cfg, None, __import__("autoclip.core.hardware", fromlist=["cpu_encoder"]).cpu_encoder(cfg.video))
    assert rc[0] >= 0


def test_process_video_unsilence_returns_original(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    out = tmp_path / "out4"
    out.mkdir()
    cfg = make_cfg(media, out, clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    cfg.unsilence.enabled = True
    cfg.captions.enabled = False
    cfg.title.enabled = False
    monkeypatch.setattr("autoclip.core.unsilence.remove_silence", lambda src, cfg, dst, *a, **k: src)
    from autoclip.core.hardware import cpu_encoder as ce

    rc = process_video(media.video, cfg, None, ce(cfg.video))
    assert rc[0] >= 0


def test_process_video_no_transcript_needed(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    out = tmp_path / "out3"
    out.mkdir()
    cfg = make_cfg(
        media,
        out,
        clips={
            "mode": "ranges",
            "kinds": ["short"],
            "short": {"count": 1, "min_seconds": 5, "max_seconds": 10, "ranges": ["00:00:00-00:00:05"]},
        },
    )
    cfg.captions.enabled = False
    cfg.title.enabled = False
    from autoclip.core.pipeline import needs_text as _needs_text

    assert _needs_text(cfg) is False
    from autoclip.core.hardware import cpu_encoder as ce

    rc = process_video(media.video, cfg, None, ce(cfg.video))
    assert rc[0] >= 0
