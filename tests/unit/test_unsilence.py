from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autoclip.core.errors import AutoclipError
from autoclip.core.hardware import cpu_encoder
from autoclip.core.media import probe
from autoclip.core.pipeline import process_video
from autoclip.core.unsilence import _pydub_available, remove_silence

from ..support import Media, make_cfg

pytest.importorskip("pydub")


def _skip_without_audioop() -> None:
    if not _pydub_available():
        pytest.skip("pydub/audioop is not available")


def test_remove_silence_creates_cleaner_video(media: Media, tmp_path: Path) -> None:
    _skip_without_audioop()
    cfg = make_cfg(media, tmp_path, None)
    cfg.unsilence.enabled = True
    destination = tmp_path / "long" / (media.video.stem + ".mp4")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = destination.parent / f"unsilence_{media.video.stem}.mp4"
    returned = remove_silence(media.video, cfg, destination)
    assert returned == destination
    assert destination.is_file() and destination.stat().st_size > 0
    assert probe(destination).duration <= probe(media.video).duration + 1e-6


def test_remove_silence_does_not_need_system_ffprobe(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The bundled imageio-ffmpeg ships no ffprobe; unsilence must not depend on one (regression)."""
    _skip_without_audioop()
    from autoclip.core.media import ffmpeg_exe

    cfg = make_cfg(media, tmp_path, None)
    cfg.unsilence.enabled = True
    cfg.unsilence.quality_preset = "medium"
    destination = tmp_path / "long" / media.video.stem / f"unsilence_{media.video.stem}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))
    ffmpeg_exe.cache_clear()
    try:
        returned = remove_silence(media.video, cfg, destination)
    finally:
        ffmpeg_exe.cache_clear()
    assert returned == destination
    assert destination.is_file() and destination.stat().st_size > 0


def test_disabled_by_default_returns_original(media: Media, tmp_path: Path) -> None:
    cfg = make_cfg(media, tmp_path, None)
    cfg.unsilence.enabled = False
    destination = tmp_path / "unsilence_test.mp4"
    returned = remove_silence(media.video, cfg, destination)
    assert returned is media.video
    assert not destination.exists()


def test_process_video_runs_after_unsilence(media: Media, tmp_path: Path) -> None:
    _skip_without_audioop()
    cfg = make_cfg(media, tmp_path, None, unsilence={"enabled": True, "min_silence_ms": 300})
    out = tmp_path / "out"
    cfg.output_dir = str(out)
    (done, failed) = process_video(media.video, cfg, None, cpu_encoder(cfg.video), name=media.video.stem, consent=None)
    assert failed == 0
    assert done > 0
    expected = out / "long" / media.video.stem / f"unsilence_{media.video.stem}.mp4"
    assert expected.is_file()


def test_when_pydub_is_unavailable_source_is_copied(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(media, tmp_path, None)
    cfg.unsilence.enabled = True
    destination = tmp_path / "long" / media.video.stem / f"unsilence_{media.video.stem}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("autoclip.core.unsilence._pydub_available", lambda: False)
    returned = remove_silence(media.video, cfg, destination)
    assert returned == destination
    assert destination.is_file()


def test_no_audio_track_source_is_copied(media: Media, tmp_path: Path) -> None:
    cfg = make_cfg(media, tmp_path, None)
    cfg.unsilence.enabled = True
    destination = tmp_path / "long" / "silent" / "unsilence_silent.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    returned = remove_silence(media.silent, cfg, destination)
    assert returned == destination
    assert destination.is_file()


def test_audio_decode_failure_falls_back(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _skip_without_audioop()
    cfg = make_cfg(media, tmp_path, None)
    cfg.unsilence.enabled = True
    destination = tmp_path / "long" / media.video.stem / f"unsilence_{media.video.stem}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("autoclip.core.unsilence.decode_audio", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad")))

    returned = remove_silence(media.video, cfg, destination)
    assert returned is media.video
    assert not destination.exists()


def test_no_removable_silence_copies_source(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _skip_without_audioop()
    cfg = make_cfg(media, tmp_path, None)
    cfg.unsilence.enabled = True
    destination = tmp_path / "long" / media.video.stem / f"unsilence_{media.video.stem}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("autoclip.core.unsilence._non_silent_intervals", lambda cfg, audio: [])
    returned = remove_silence(media.video, cfg, destination)
    assert returned == destination
    assert destination.is_file()


def test_impossible_source_raises(media: Media, tmp_path: Path) -> None:
    cfg = make_cfg(media, tmp_path, None)
    cfg.unsilence.enabled = True
    destination = tmp_path / "long" / media.video.stem / f"unsilence_{media.video.stem}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(AutoclipError):
        remove_silence(tmp_path / "missing.mp4", cfg, destination)


def test_pydub_import_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:

    from autoclip.core import unsilence

    monkeypatch.setattr("importlib.import_module", lambda name: (_ for _ in ()).throw(RuntimeError("boom")))
    assert unsilence._pydub_available() is False


def test_audioop_fallback_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sys
    import types

    from autoclip.core import unsilence

    importlib_original = importlib.import_module
    fake_audioop = types.ModuleType("audioop_lts")
    monkeypatch.delitem(sys.modules, "audioop", raising=False)
    monkeypatch.setitem(sys.modules, "audioop_lts", fake_audioop)

    def fake_module(name: str) -> Any:
        if name == "pydub":
            return importlib_original(name)
        if name == "audioop":
            raise ImportError("missing")
        if name == "audioop_lts":
            return fake_audioop
        return importlib_original(name)

    monkeypatch.setattr("importlib.import_module", fake_module)
    assert unsilence._pydub_available() is True


def test_non_silent_interval_helpers(monkeypatch: pytest.MonkeyPatch, media: Media) -> None:
    from pydub import AudioSegment

    from autoclip.core import unsilence

    from ..support import make_cfg

    cfg = make_cfg(media, Path("."), None)
    cfg.unsilence.enabled = True
    audio = AudioSegment.silent(duration=1000)

    monkeypatch.setattr("pydub.silence.detect_nonsilent", lambda audio, **kw: [])
    assert unsilence._non_silent_intervals(cfg, audio) == []

    raw = [(0, 1000), (900, 2000), (2500, 2600)]
    monkeypatch.setattr("pydub.silence.detect_nonsilent", lambda audio, **kw: raw)
    intervals = unsilence._non_silent_intervals(cfg, audio)
    assert len(intervals) == 1
    cfg_one = cfg
    monkeypatch.setattr("pydub.silence.detect_nonsilent", lambda audio, **kw: [(0, 1000)])
    intervals = unsilence._non_silent_intervals(cfg_one, AudioSegment.silent(duration=1000))
    assert intervals == [(0.0, 1.0)]

    cfg_long_min = cfg
    cfg_long_min.unsilence.min_segment_seconds = 10
    monkeypatch.setattr("pydub.silence.detect_nonsilent", lambda audio, **kw: [(0, 100)])
    assert unsilence._non_silent_intervals(cfg_long_min, AudioSegment.silent(duration=1000)) == []


def test_cancel_and_video_failure_branches(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _skip_without_audioop()
    cfg = make_cfg(media, tmp_path, None)
    cfg.unsilence.enabled = True
    destination = tmp_path / "long" / media.video.stem / f"unsilence_{media.video.stem}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)

    from autoclip.core.events import CancelToken

    token = CancelToken()
    returned = remove_silence(media.video, cfg, destination, cancel=token)
    assert returned == destination
    assert destination.is_file()

    import moviepy

    destination2 = tmp_path / "long" / media.video.stem / "boom.mp4"
    destination2.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(moviepy, "VideoFileClip", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    returned2 = remove_silence(media.video, cfg, destination2)
    assert returned2 == destination2
    assert destination2.is_file()


def test_moviepy_invalid_output_branch(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _skip_without_audioop()
    cg = make_cfg(media, tmp_path, None)
    cg.unsilence.enabled = True
    destination = tmp_path / "long" / media.video.stem / f"invalid_{media.video.stem}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)

    import types

    class FakeClip:
        fps = 10.0
        duration = 10.0
        audio = types.SimpleNamespace(fps=22050.0)

        def subclipped(self: Any, s: float, e: float) -> Any:
            return self

        def close(self) -> None:
            return None

    class FakeAssembled:
        def write_videofile(self: Any, *a: Any, **k: Any) -> None:
            return None

        def close(self) -> None:
            return None

    import moviepy

    monkeypatch.setattr(moviepy, "VideoFileClip", lambda *a, **k: FakeClip())
    monkeypatch.setattr(moviepy, "concatenate_videoclips", lambda *a, **k: FakeAssembled())
    returned = remove_silence(media.video, cg, destination)
    assert returned == destination
    assert destination.is_file()


def test_exception_copy_failure_branch(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(media, tmp_path, None)
    cfg.unsilence.enabled = True
    destination = tmp_path / "long" / media.video.stem / f"except_{media.video.stem}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)

    import moviepy

    monkeypatch.setattr("autoclip.core.unsilence._pydub_available", lambda: True)
    monkeypatch.setattr(moviepy, "VideoFileClip", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr("shutil.copyfile", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    monkeypatch.setattr("pathlib.Path.unlink", lambda self: (_ for _ in ()).throw(OSError("boom")))
    returned = remove_silence(media.video, cfg, destination)
    assert returned is media.video


def test_exception_same_source_branch(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core import unsilence

    _skip_without_audioop()
    cfg = make_cfg(media, tmp_path, None)
    cfg.unsilence.enabled = True
    monkeypatch.setattr(unsilence, "_pydub_available", lambda: True)
    import moviepy

    monkeypatch.setattr(moviepy, "VideoFileClip", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    returned = remove_silence(media.video, cfg, media.video)
    assert returned is media.video


def test_audioop_lts_unavailable_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import sys

    from autoclip.core import unsilence

    importlib_original = importlib.import_module

    monkeypatch.delitem(sys.modules, "audioop", raising=False)

    def fake(name: str) -> Any:
        if name == "pydub":
            return importlib_original(name)
        if name == "audioop":
            raise ImportError("missing")
        if name == "audioop_lts":
            raise RuntimeError("boom")
        return importlib_original(name)

    monkeypatch.setattr("importlib.import_module", fake)
    assert unsilence._pydub_available() is False


def test_non_silent_merge_branch(media: Media, monkeypatch: pytest.MonkeyPatch) -> None:
    from pydub import AudioSegment

    from autoclip.core import unsilence

    from ..support import make_cfg

    cfg = make_cfg(media, Path("."), None)
    cfg.unsilence.enabled = True
    audio = AudioSegment.silent(duration=2000)
    monkeypatch.setattr("pydub.silence.detect_nonsilent", lambda audio, **kw: [(0, 1000), (500, 1500)])
    intervals = unsilence._non_silent_intervals(cfg, audio)
    assert len(intervals) == 1


def test_no_audio_copy_failure(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(media, tmp_path, None)
    cfg.unsilence.enabled = True
    destination = tmp_path / "long" / "silent" / "unsilence_silent.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("shutil.copyfile", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    returned = remove_silence(media.silent, cfg, destination)
    assert returned is media.silent
    assert not destination.exists()


def test_moviepy_failure_with_copy_failure(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _skip_without_audioop()
    cfg = make_cfg(media, tmp_path, None)
    cfg.unsilence.enabled = True
    destination = tmp_path / "long" / media.video.stem / f"unsilence_{media.video.stem}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)

    import moviepy

    monkeypatch.setattr(moviepy, "VideoFileClip", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr("shutil.copyfile", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    returned = remove_silence(media.video, cfg, destination)
    assert returned is media.video
