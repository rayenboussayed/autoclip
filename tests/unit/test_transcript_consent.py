"""Download consent: cached models run, uncached ones need an explicit yes (and offline never downloads)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from autoclip.core import transcript
from autoclip.core.errors import AutoclipError
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


def _run(
    media: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    downloader: Downloader,
    *,
    offline: bool = False,
    consent: str = "ask",
    answer: Any = None,
    assume_yes: bool = False,
    consent_provider: Any = None,
) -> list[Segment]:
    install_stub_whisper(monkeypatch, media, {}, download_model=downloader)
    monkeypatch.setenv("AUTOCLIP_ASSUME_YES", "1" if assume_yes else "0")
    if answer is not None:
        monkeypatch.setattr("builtins.input", answer)
    cfg = make_cfg(
        media,
        tmp_path / "out",
        transcript={
            "model": "small",
            "model_dir": str(tmp_path / "models"),
            "offline": offline,
            "consent": consent,
            "auto_downgrade": False,
        },
    )
    info: MediaInfo = probe(media.video)
    return transcript.transcribe(media.video, cfg, info, np.zeros(10, dtype=np.float32), None, consent=consent_provider)


def test_a_cached_model_needs_no_prompt_and_no_download(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = Downloader(cached=True)
    segs = _run(media, tmp_path, monkeypatch, downloader, answer=lambda *a, **k: pytest.fail("cached: no prompt"))
    assert [s.text for s in segs] == SENTENCES
    assert len(downloader.calls) == 1 and downloader.calls[0]["local_files_only"] is True
    assert downloader.calls[0]["cache_dir"] == str(tmp_path / "models")
    assert downloader.online == []


def test_offline_with_an_uncached_model_fails_without_prompting(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = Downloader()
    with pytest.raises(AutoclipError, match="offline"):
        _run(media, tmp_path, monkeypatch, downloader, offline=True, answer=lambda *a, **k: pytest.fail("offline: no prompt"))
    assert len(downloader.calls) == 1 and downloader.calls[0]["local_files_only"] is True
    assert downloader.online == []


@pytest.mark.parametrize(
    "answer",
    [
        lambda *a, **k: "n",
        lambda *a, **k: "",
        lambda *a, **k: (_ for _ in ()).throw(EOFError()),
        lambda *a, **k: (_ for _ in ()).throw(OSError("stdin is unavailable")),
    ],
    ids=["no", "empty", "eof", "no-stdin"],
)
def test_an_uncached_model_is_not_downloaded_without_consent(
    media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: Any
) -> None:
    downloader = Downloader()
    with pytest.raises(AutoclipError, match="not cached"):
        _run(media, tmp_path, monkeypatch, downloader, answer=answer)
    assert len(downloader.calls) == 1 and downloader.calls[0]["local_files_only"] is True
    assert downloader.online == []


def test_yes_downloads_the_model_once_and_then_transcribes(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Pretend the stubbed stdin is a real terminal so the prompt branch runs.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    downloader = Downloader()
    segs = _run(media, tmp_path, monkeypatch, downloader, answer=lambda *a, **k: "yes please")
    assert [s.text for s in segs] == SENTENCES
    assert [call["local_files_only"] for call in downloader.calls] == [True, False]
    assert downloader.online[0]["model"] == "small" and downloader.online[0]["cache_dir"] == str(tmp_path / "models")


def test_autoclip_assume_yes_downloads_without_prompting(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = Downloader()
    segs = _run(media, tmp_path, monkeypatch, downloader, assume_yes=True, answer=lambda *a, **k: pytest.fail("assume yes"))
    assert [s.text for s in segs] == SENTENCES
    assert [call["local_files_only"] for call in downloader.calls] == [True, False]


def _utils(monkeypatch: pytest.MonkeyPatch, download_model: Any) -> None:
    module = ModuleType("faster_whisper.utils")
    module.download_model = download_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", module)


def test_the_seams_ask_faster_whisper_with_the_right_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The two module-level seams are the only place that imports the real downloader."""
    calls: list[dict[str, Any]] = []

    def download_model(model_name: str, **kwargs: Any) -> None:
        calls.append({"model": model_name, **kwargs})
        if kwargs["local_files_only"] and not (tmp_path / "models").exists():
            raise RuntimeError("not cached")

    _utils(monkeypatch, download_model)
    assert transcript._model_is_cached("small", str(tmp_path / "models")) is False
    (tmp_path / "models").mkdir()
    assert transcript._model_is_cached("small", str(tmp_path / "models")) is True
    transcript._download_model("small", str(tmp_path / "models"))
    assert [call["local_files_only"] for call in calls] == [True, True, False]
    assert all(call["model"] == "small" for call in calls)


def test_auto_downloads_without_prompting(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = Downloader()
    segs = _run(media, tmp_path, monkeypatch, downloader, consent="auto", answer=lambda *a, **k: pytest.fail("auto: no prompt"))
    assert [s.text for s in segs] == SENTENCES
    assert [call["local_files_only"] for call in downloader.calls] == [True, False]


def test_never_fails_without_downloading(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = Downloader()
    with pytest.raises(AutoclipError, match="consent is 'never'"):
        _run(media, tmp_path, monkeypatch, downloader, consent="never", answer=lambda *a, **k: pytest.fail("never: no prompt"))
    assert downloader.online == []


def test_ask_with_a_consent_provider_false_fails(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = Downloader()
    provider_calls: list[tuple[str, str]] = []

    def provider(title: str, message: str) -> bool:
        provider_calls.append((title, message))
        return False

    with pytest.raises(AutoclipError, match="not cached"):
        _run(media, tmp_path, monkeypatch, downloader, consent_provider=provider, answer=lambda *a, **k: pytest.fail("ask: no TTY input"))
    assert provider_calls and "Download Whisper" in provider_calls[0][1]
    assert downloader.online == []


def test_ask_with_a_consent_provider_true_downloads(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = Downloader()

    def provider(title: str, message: str) -> bool:
        return True

    segs = _run(
        media, tmp_path, monkeypatch, downloader, consent_provider=provider, answer=lambda *a, **k: pytest.fail("ask: no TTY input")
    )
    assert [s.text for s in segs] == SENTENCES
    assert [call["local_files_only"] for call in downloader.calls] == [True, False]
