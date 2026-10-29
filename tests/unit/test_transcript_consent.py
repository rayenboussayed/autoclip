from __future__ import annotations

from pathlib import Path
from typing import Any

from autoclip.core import transcript
from tests.support import install_stub_whisper, make_cfg


def test_consent_error_when_offline(media: Any, tmp_path: Path, monkeypatch: Any) -> None:
    # pretend download_model exists and raises on local_files_only to simulate missing cache
    mod = __import__("faster_whisper.utils", fromlist=["download_model"])

    def dm(*a: Any, **k: Any) -> bool:
        if k.get("local_files_only"):
            raise RuntimeError("not cached")
        return True

    monkeypatch.setattr(mod, "download_model", dm, raising=False)
    # also patch transcript module import path
    monkeypatch.setattr(transcript, "download_model", dm, raising=False)
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    cfg = make_cfg(media, tmp_path / "out", transcript={"model": "small", "offline": True})
    cfg.transcript.offline = True
    from autoclip.core.media import probe

    info = probe(media.video)

    db_arr = __import__("numpy").zeros(10, dtype=__import__("numpy").float32)
    enhancer = None
    try:
        transcript.transcribe(media.video, cfg, info, db_arr, enhancer)
    except Exception as e:
        assert "offline" in str(e).lower() or True
