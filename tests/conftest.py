"""Shared fixtures: a tiny generated video (pip-only tools, so it runs on Linux, macOS and Windows)."""

from __future__ import annotations

import json
import subprocess
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autoclip.config import Config, parse
from autoclip.log import LOG, setup_logging
from autoclip.media import ffmpeg_exe

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


@dataclass(frozen=True)
class Media:
    """Paths of the generated fixtures."""

    video: Path  # 40 s, 320x180, speech-like tone bursts (3 s) separated by 1 s of silence
    silent: Path  # same picture, no audio track
    transcript: Path  # word-level JSON transcript matching ``video``
    srt: Path  # sentence-level SRT matching ``video``


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
    return f"00:00:{int(t):02d},{round(t % 1 * 1000):03d}"


@pytest.fixture(autouse=True)
def _restore_logger() -> Iterator[None]:
    """``setup_logging`` mutates the shared logger; give every test a clean one (and let caplog see it)."""
    yield
    setup_logging(None, "INFO", None)
    LOG.propagate = True


@pytest.fixture(scope="session")
def media(tmp_path_factory: pytest.TempPathFactory) -> Media:
    """Generate the fixture video once per test session."""
    root = tmp_path_factory.mktemp("media")
    t = np.arange(int(40 * SR)) / SR
    gate = ((t % 4.0) < 3.0).astype(np.float64)
    voice = 0.35 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    pcm = (np.clip(voice * gate + np.random.default_rng(0).normal(0, 0.003, t.size), -1, 1) * 32767).astype(np.int16)
    with wave.open(str(root / "voice.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    base = [ffmpeg_exe(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=15:d=40"]
    enc = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    subprocess.run([*base, "-i", str(root / "voice.wav"), *enc, "-c:a", "aac", "-shortest", str(root / "talk.mp4")], check=True)
    subprocess.run([*base, *enc, str(root / "silent.mp4")], check=True)
    segs = _segments()
    (root / "talk.json").write_text(json.dumps(segs), encoding="utf-8")
    blocks = [f"{n}\n{_srt_time(s['start'])} --> {_srt_time(s['end'])}\n{s['text']}\n" for n, s in enumerate(segs, 1)]
    (root / "talk.srt").write_text("\n".join(blocks), encoding="utf-8")
    return Media(root / "talk.mp4", root / "silent.mp4", root / "talk.json", root / "talk.srt")


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


@pytest.fixture
def cfg_factory(media: Media, tmp_path: Path):  # type: ignore[no-untyped-def]
    """``cfg_factory(**sections)`` -> Config writing into this test's tmp folder."""

    def factory(source: Path | None = None, **sections: dict[str, Any]) -> Config:
        return make_cfg(media, tmp_path / "out", source, **sections)

    return factory
