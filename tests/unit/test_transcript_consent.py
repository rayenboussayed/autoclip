"""Whisper models download automatically: a cached model runs offline, an uncached one is fetched on first use."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from autoclip.core import transcript
from autoclip.core.media import probe
from autoclip.core.models import MediaInfo, Segment
from tests.support import SENTENCES, install_stub_whisper, make_cfg


class Downloader:
    """A faster-whisper downloader with an empty (or populated) cache that records every call."""

    def __init__(self, *, cached: bool = False) -> None:
        self.cached = cached
        self.calls: list[dict[str, Any]] = []

    def __call__(self, model_name: str, **kwargs: Any) -> bool:
        self.calls.append({"model": model_name, **kwargs})
        if kwargs.get("local_files_only") and not self.cached:
            raise RuntimeError("not cached")
        return True

    @property
    def online(self) -> list[dict[str, Any]]:
        return [call for call in self.calls if not call.get("local_files_only")]


def _run(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, downloader: Downloader, **sections: Any) -> list[Segment]:
    install_stub_whisper(monkeypatch, media, {}, download_model=downloader)
    cfg = make_cfg(media, tmp_path / "out", transcript_file=None, **sections)
    info: MediaInfo = probe(media.video)
    return transcript.transcribe(media.video, cfg, info, np.zeros(10, dtype=np.float32), None)


def test_a_cached_model_needs_no_download(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = Downloader(cached=True)
    segs = _run(media, tmp_path, monkeypatch, downloader)
    assert [s.text for s in segs] == SENTENCES
    assert len(downloader.calls) == 1 and downloader.calls[0]["local_files_only"] is True
    assert downloader.online == []


def test_an_uncached_model_downloads_on_first_use(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = Downloader()
    segs = _run(media, tmp_path, monkeypatch, downloader)
    assert [s.text for s in segs] == SENTENCES
    assert [call["local_files_only"] for call in downloader.calls] == [True, False]
    assert downloader.online[0]["model"] == "tiny"


def _utils(monkeypatch: pytest.MonkeyPatch, download_model: Any) -> None:
    module = ModuleType("faster_whisper.utils")
    module.download_model = download_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", module)


def test_the_seams_ask_faster_whisper_with_the_right_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The two module-level seams are the only place that imports the real downloader."""
    calls: list[dict[str, Any]] = []

    cached = {"v": False}

    def download_model(model_name: str, **kwargs: Any) -> None:
        calls.append({"model": model_name, **kwargs})
        if kwargs["local_files_only"] and not cached["v"]:
            raise RuntimeError("not cached")

    _utils(monkeypatch, download_model)
    assert transcript._model_is_cached("small") is False
    cached["v"] = True
    assert transcript._model_is_cached("small") is True
    transcript._download_model("small")
    assert [call["local_files_only"] for call in calls] == [True, True, False]
    assert all(call["model"] == "small" for call in calls)
