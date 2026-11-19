"""Visible cut detection and boundary snapping helpers."""

import subprocess
from pathlib import Path
from typing import Any

import pytest

from autoclip.core.media import ffmpeg_exe, probe
from autoclip.core.scenes import detect_scenes


def test_detect_scenes_finds_a_clear_cut(tmp_path: Path) -> None:
    video = tmp_path / "cuts.mp4"
    command = [
        ffmpeg_exe(),
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=black:s=32x32:r=10:d=1",
        "-f",
        "lavfi",
        "-i",
        "color=white:s=32x32:r=10:d=1",
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1[out]",
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video),
    ]
    subprocess.run(command, check=True)
    cuts = detect_scenes(video, probe(video))
    assert cuts and cuts[0] == pytest.approx(1.0, abs=0.2)


def test_detect_scenes_returns_no_boundaries_when_detection_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "blank.mp4"
    command = [ffmpeg_exe(), "-y", "-v", "error", "-f", "lavfi", "-i", "color=black:s=32x32:r=1:d=1", str(video)]
    subprocess.run(command, check=True)
    import scenedetect

    monkeypatch.setattr(scenedetect, "open_video", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("broken")))
    assert detect_scenes(video, probe(video)) == []


def test_detect_scenes_ignores_a_non_visual_input(media: Any) -> None:
    from autoclip.core.models import MediaInfo

    assert detect_scenes(media.video, MediaInfo(0.0, True, 320, 180, 15.0)) == []


def test_detect_scenes_handles_a_missing_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, media: Any) -> None:
    import sys

    silent = tmp_path / "silent.mp4"
    silent.write_bytes(media.silent.read_bytes())
    monkeypatch.setitem(sys.modules, "scenedetect", None)
    assert detect_scenes(silent, probe(silent)) == []


def test_detect_scenes_closes_the_video_even_when_closing_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, media: Any) -> None:
    class FakeVideo:
        def close(self) -> None:
            raise OSError("close failed")

    class FakeTimecode:
        def __init__(self, seconds: float) -> None:
            self.seconds = seconds

    class FakeManager:
        auto_downscale = False

        def add_detector(self, _detector: object) -> None:
            pass

        def detect_scenes(self, video: object) -> None:
            pass

        def get_scene_list(self) -> list[object]:
            return [(FakeTimecode(0.0), None), (FakeTimecode(1.0), None)]

    import scenedetect

    video = tmp_path / "fake.mp4"
    video.write_bytes(media.silent.read_bytes())
    monkeypatch.setattr(scenedetect, "open_video", lambda *args, **kwargs: FakeVideo())
    monkeypatch.setattr(scenedetect, "SceneManager", FakeManager)
    monkeypatch.setattr(scenedetect, "AdaptiveDetector", lambda **kwargs: object())
    assert detect_scenes(video, probe(video)) == [1.0]
