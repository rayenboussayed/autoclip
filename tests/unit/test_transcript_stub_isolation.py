"""A stub installed after the real faster-whisper was imported must stay a stub.

``transcribe`` used to detect the test stub by failing to import ``faster_whisper.utils``.  A module
already in ``sys.modules`` imports successfully even when the parent stub has an empty ``__path__``,
so the real model downloader ran and the consent prompt blocked a headless run.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autoclip.core import transcript
from autoclip.core.media import probe
from tests.support import SENTENCES, install_stub_whisper, make_cfg


def test_a_cached_real_utils_module_does_not_turn_the_stub_into_a_download(
    media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("faster_whisper.utils")
    real_utils = importlib.import_module("faster_whisper.utils")
    install_stub_whisper(monkeypatch, media, {})
    # the order-dependent regression: a previous test already imported the real submodule
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", real_utils)
    monkeypatch.setattr("builtins.input", lambda *a, **k: pytest.fail("the consent prompt must not run in a stub"))
    cfg = make_cfg(
        media,
        tmp_path / "out",
        transcript={"model": "small", "model_dir": str(tmp_path / "empty"), "auto_downgrade": False},
    )
    segs = transcript.transcribe(media.video, cfg, probe(media.video), np.zeros(10, dtype=np.float32), None)
    assert [s.text for s in segs] == SENTENCES
