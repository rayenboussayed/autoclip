"""Pure helpers shared by the unit and end-to-end suites (fixtures live in ``conftest.py``)."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from autoclip.core.config import Config, parse

SENTENCES = [
    "Welcome everyone to this big talk today.",
    "Why does this matter so much, honestly?",
    "Um I never listened to my customers ever.",
    "haha that was so funny lol I laughed.",
    "My mother passed away and I cried alone.",
    "Here is how to learn step by step today.",
    "Wow that was incredible we finally won!",
    "Never give up on your dream and believe.",
    "They say it is a scam but they are wrong.",
    "Thank you so much for listening goodbye.",
]
SR = 22050
FIXTURE_SAMPLES = 40 * 16000  # the fixture video is 40 s long
MANIFEST_PATH = Path(__file__).resolve().parent / "assets" / "youtube.json"


@dataclass(frozen=True)
class Media:
    """Paths of the generated fixtures."""

    video: Path  # 40 s, 320x180, speech-like tone bursts (3 s) separated by 1 s of silence
    silent: Path  # same picture, no audio track
    transcript: Path  # word-level JSON transcript matching ``video``
    srt: Path  # sentence-level SRT matching ``video``


@dataclass(frozen=True)
class ManifestEntry:
    """One video of ``tests/assets/youtube.json`` (``duration`` bounds the download ladder)."""

    title: str
    url: str
    duration: float | None = None


def _segments() -> list[dict[str, Any]]:
    segs = []
    for k, text in enumerate(SENTENCES):
        start, words = k * 4.0, text.split()
        step = 3.0 / len(words)
        segs.append(
            {"start": start, "end": start + 3.0, "text": text,
             "words": [{"start": start + n * step, "end": start + (n + 1) * step, "word": w} for n, w in enumerate(words)]}
        )  # fmt: skip
    return segs


def _srt_time(t: float) -> str:
    """A WebVTT/SRT timestamp; rounding to the millisecond carries into the next second/hour."""
    total_ms = round(t * 1000)
    seconds, millis = divmod(total_ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """``base`` with ``over`` laid on top, recursing into nested objects (so ``clips={"short": {...}}`` keeps the rest)."""
    return {k: merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v for k, v in {**base, **over}.items()}


def make_cfg(media: Media, out: Path, source: Path | None = None, **sections: dict[str, Any]) -> Config:
    """A small, fast configuration for ``source`` (default: the fixture video) with per-section overrides."""
    data: dict[str, Any] = {
        "clips": {
            "kinds": ["short"],
            "short": {"count": 2, "min_seconds": 6, "max_seconds": 12},
            "long": {"min_seconds": 12, "max_seconds": 30},
        },
        "transcript": {"mode": "file", "file": str(media.transcript)},
        "video": {"short_side": 144, "preset": "ultrafast", "thumbnail": False, "gpu": "off"},
        "logging": {"enabled": False, "console_level": "ERROR"},
        "resources": {"max_workers": 1},
    }
    cfg = parse(Config, merge(data, sections))
    cfg.input = [str(source or media.video)]
    cfg.output_dir = str(out)
    return cfg


def load_manifest(path: Path | None = None) -> list[ManifestEntry]:
    """Load and validate the YouTube manifest used by the E2E suite."""
    data = json.loads((path or MANIFEST_PATH).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path or MANIFEST_PATH}: expected a non-empty JSON list")
    entries = []
    for index, item in enumerate(data, 1):
        if not isinstance(item, dict):
            raise ValueError(f"{path or MANIFEST_PATH}: entry {index} is not an object")
        title, url = item.get("title"), item.get("video_link")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"{path or MANIFEST_PATH}: entry {index} has no title")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ValueError(f"{path or MANIFEST_PATH}: entry {index} has no http(s) video_link")
        duration = item.get("duration_seconds")
        if duration is not None and (not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0):
            raise ValueError(f"{path or MANIFEST_PATH}: entry {index} has an invalid duration_seconds")
        entries.append(ManifestEntry(title, url, float(duration) if duration is not None else None))
    return entries


def install_stub_whisper(
    monkeypatch: pytest.MonkeyPatch,
    media: Media,
    record: dict[str, Any],
    cuda: int = 0,
    full_samples: int = FIXTURE_SAMPLES,
    download_model: Callable[..., Any] | None = None,
    cuda_fails: bool = False,
) -> None:
    """A stand-in for faster-whisper / ctranslate2 with the real call signatures (the model host is offline in CI).

    ``download_model`` installs a ``faster_whisper.utils`` module; by default the stub has none, so the
    download-consent path is skipped exactly as it is when the real package is absent.
    """
    segments = json.loads(media.transcript.read_text(encoding="utf-8"))

    class WhisperModel:
        def __init__(self, name: str, device: str, compute_type: str, cpu_threads: int, **kwargs: Any) -> None:
            record["init"] = (name, device, compute_type, cpu_threads)
            self.device = device

        def transcribe(self, audio: Any, **options: Any) -> tuple[Any, Any]:
            if cuda_fails and self.device == "cuda":
                raise RuntimeError("CUDA driver/runtime mismatch")
            record["options"] = options
            # The 1 s CUDA warm-up passes no language; a real chunk call always does.
            if isinstance(audio, np.ndarray) and len(audio) < full_samples - 16000 and "language" in options:  # a chunk
                record.setdefault("chunks", []).append((len(audio), options["language"]))
                seg = SimpleNamespace(start=0.5, end=2.5, text=" chunk", words=[SimpleNamespace(start=0.5, end=1.5, word=" chunk")])
                return iter([seg]), SimpleNamespace(language="en", language_probability=0.9, duration=len(audio) / 16000)
            record["samples"] = len(audio)
            record["audio_sum"] = float(np.asarray(audio, dtype=np.float64).sum())
            gen = (
                SimpleNamespace(
                    start=s["start"],
                    end=s["end"],
                    text=f" {s['text']}",
                    words=[SimpleNamespace(start=w["start"], end=w["end"], word=f" {w['word']}") for w in s["words"]],
                )
                for s in segments
            )
            return gen, SimpleNamespace(language="en", language_probability=0.99, duration=40.0)

    fw, ct = ModuleType("faster_whisper"), ModuleType("ctranslate2")
    fw.WhisperModel = WhisperModel  # type: ignore[attr-defined]
    ct.get_cuda_device_count = lambda: cuda  # type: ignore[attr-defined]
    # An empty package path makes ``faster_whisper.utils`` unimportable: the real submodule may already
    # be cached by another test, and ``transcribe`` must see the stub, not the real model downloader.
    fw.__dict__["__path__"] = []
    monkeypatch.delitem(sys.modules, "faster_whisper.utils", raising=False)
    if download_model is not None:
        utils = ModuleType("faster_whisper.utils")
        utils.download_model = download_model  # type: ignore[attr-defined]
        fw.__dict__["utils"] = utils
        monkeypatch.setitem(sys.modules, "faster_whisper.utils", utils)
    # ``transcribe`` reaches faster-whisper through two seams, so a test never depends on whether the
    # real ``faster_whisper.utils`` happens to be cached in ``sys.modules`` by another test.
    from autoclip.core import transcript

    if download_model is None:
        monkeypatch.setattr(transcript, "_model_is_cached", lambda *a, **k: True, raising=False)
        monkeypatch.setattr(transcript, "_download_model", lambda *a, **k: None, raising=False)
    else:

        def _model_is_cached(model_name: str, cache_dir: str | None) -> bool:
            try:
                download_model(model_name, cache_dir=cache_dir, local_files_only=True)
            except Exception:
                return False
            return True

        def _download_model(model_name: str, cache_dir: str | None) -> None:
            download_model(model_name, cache_dir=cache_dir, local_files_only=False)

        monkeypatch.setattr(transcript, "_model_is_cached", _model_is_cached, raising=False)
        monkeypatch.setattr(transcript, "_download_model", _download_model, raising=False)
    monkeypatch.setitem(sys.modules, "faster_whisper", fw)
    monkeypatch.setitem(sys.modules, "ctranslate2", ct)


BASE_CONFIG: dict[str, Any] = {
    "clips": {
        "kinds": ["short"],
        "short": {"count": 1, "min_seconds": 6, "max_seconds": 12},
        "long": {"count": 1, "min_seconds": 12, "max_seconds": 30},
    },
    "video": {"short_side": 96, "preset": "ultrafast", "gpu": "off"},
    "audio": {"denoise": "none"},
    "logging": {"console_level": "ERROR"},
    "resources": {"max_workers": 1},
}


def write_config(tmp_path: Path, media: Media, **overrides: Any) -> Path:
    """Write a CLI config for the fixture video and its two extra transcript files."""
    (tmp_path / "subs").mkdir(parents=True, exist_ok=True)
    for stem in ("talk", "silent", "broken"):
        (tmp_path / "subs" / f"{stem}.json").write_text(media.transcript.read_text(encoding="utf-8"), encoding="utf-8")
    data = merge(
        {**BASE_CONFIG, "input": [str(media.video), str(media.silent)], "output_dir": str(tmp_path / "out"),
         "transcript": {"mode": "file", "file": str(tmp_path / "subs" / "{stem}.json")}},
        overrides,
    )  # fmt: skip
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path
