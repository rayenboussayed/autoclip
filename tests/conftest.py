"""Shared fixtures: a tiny generated video (pip-only tools, so it runs on Linux, macOS and Windows)."""

from __future__ import annotations

import json
import os
import subprocess
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autoclip.core.config import Config
from autoclip.core.log import LOG, setup_logging
from autoclip.core.media import ffmpeg_exe

from .support import SR, Media, _segments, _srt_time, make_cfg


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


@pytest.fixture
def cfg_factory(media: Media, tmp_path: Path):  # type: ignore[no-untyped-def]
    """``cfg_factory(**sections)`` -> Config writing into this test's tmp folder."""

    def factory(source: Path | None = None, **sections: dict[str, Any]) -> Config:
        return make_cfg(media, tmp_path / "out", source, **sections)

    return factory


def _annotation(path: str, line: int, message: str) -> None:
    """Print one GitHub Actions annotation (ASCII, one command per line, escaped by the runner's rules)."""
    detail = message.encode("ascii", "backslashreplace").decode("ascii")
    detail = detail.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    # the command has to start its own line: pytest has just written the progress character
    print(f"\n::error file={path},line={max(line, 1)}::{detail}", flush=True)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``e2e`` and ``network`` tests unless their environment flag opts in (the suite is offline by default)."""
    e2e_on = os.environ.get("AUTOCLIP_E2E_TESTS") == "1"
    network_on = e2e_on or os.environ.get("AUTOCLIP_NETWORK_TESTS") == "1"
    skip_e2e = pytest.mark.skip(reason="e2e test: set AUTOCLIP_E2E_TESTS=1 to run it")
    skip_network = pytest.mark.skip(reason="network test: set AUTOCLIP_NETWORK_TESTS=1 (or AUTOCLIP_E2E_TESTS=1) to run it")
    for item in items:
        if "e2e" in item.keywords and not e2e_on:
            item.add_marker(skip_e2e)
        elif "network" in item.keywords and not network_on:
            item.add_marker(skip_network)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Under GitHub Actions, report a failure as an annotation (the job log needs an account to read)."""
    if not report.failed or os.environ.get("GITHUB_ACTIONS") != "true":
        return
    path, line, _ = report.location
    line = (line or 0) + 1
    text = str(report.longrepr)
    summary = next((row.strip() for row in reversed(text.splitlines()) if row.strip()), text)
    _annotation(path, line, f"{report.nodeid} ({report.when}): {summary[:200]}")
    _annotation(path, line, text[:4000])
