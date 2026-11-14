"""Session fixtures for the memory-regression suite."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pytest

ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(os.environ.get("AUTOCLIP_MEMORY_RESULTS", str(ROOT / "tests" / "results")))


def ffmpeg_exe() -> str:
    """The ffmpeg binary the engine uses."""
    return str(imageio_ffmpeg.get_ffmpeg_exe())


def _write_json_sidecar(target: Path, count: int = 10) -> None:
    import json

    target.write_text(
        json.dumps({"words": [{"start": i * 4.0, "end": i * 4.0 + 3.0, "text": f"sentence number {i} is here."} for i in range(count)]}),
        encoding="utf-8",
    )


@pytest.fixture(scope="session")
def many_parts_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 120 s 320x180 video with a 3 s voice-on / 1 s voice-off rhythm (~30 kept intervals)."""
    root = tmp_path_factory.mktemp("memory-media")
    import wave

    import numpy as np

    sr = 16000
    t = np.arange(int(120 * sr)) / sr
    gate = ((t % 4.0) < 3.0).astype(np.float64)
    voice = 0.35 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    pcm = (np.clip(voice * gate + np.random.default_rng(0).normal(0, 0.003, t.size), -1, 1) * 32767).astype(np.int16)
    wav = root / "many-parts.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    target = root / "many-parts.mp4"
    subprocess.run(
        [
            ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x180:r=15:d=120:duration=120",
            "-i",
            str(wav),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(target),
        ],
        check=True,
    )
    return target


@pytest.fixture(scope="session")
def p720_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 20 s 1280x720 video (same source as media, scaled) + a toy sidecar transcript."""
    root = tmp_path_factory.mktemp("memory-media-720")
    target = root / "hd.mp4"
    subprocess.run(
        [
            ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=1280x720:r=30:d=20",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=20",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-shortest",
            str(target),
        ],
        check=True,
    )
    _write_json_sidecar(target.with_suffix(".json"), 12)
    return target


@pytest.fixture(scope="session")
def memory_results() -> Path:
    """A timestamped folder under ``tests/results/`` for this run's metrics output."""
    from tests.memory.harness import new_run_directory

    root = RESULTS_ROOT / "memory"
    root.mkdir(parents=True, exist_ok=True)
    return new_run_directory(root)


__all__ = [
    "RESULTS_ROOT",
    "ROOT",
    "ffmpeg_exe",
    "memory_results",
    "many_parts_video",
    "p720_video",
]
