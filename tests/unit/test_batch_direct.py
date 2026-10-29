from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import pytest

from autoclip.core.batch import run_batch
from autoclip.core.hardware import pick_encoder
from tests.support import make_cfg


def test_run_batch_no_downloader_url(media: Any, tmp_path: Path) -> None:

    out = tmp_path / "out"
    out.mkdir()
    cfg = make_cfg(media, out, clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    cfg.input = ["https://example.com/x.mp4"]
    cfg.output_dir = str(out)
    code_result = run_batch(cfg, downloader=None, encoder=pick_encoder(cfg.video, allow_gpu=False), log_file=None)
    assert code_result.clips_failed >= 1


def test_run_batch_cancel(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core.events import CancelToken
    from tests.support import install_stub_whisper

    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    out = tmp_path / "out2"
    out.mkdir()
    cfg = make_cfg(media, out, clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    cfg.input = [str(media.video)]
    cfg.output_dir = str(out)
    tok = CancelToken()
    tok.cancel()
    from autoclip.core.events import Cancelled

    with contextlib.suppress(Cancelled):
        run_batch(cfg, downloader=None, encoder=pick_encoder(cfg.video, allow_gpu=False), log_file=None, cancel=tok)
