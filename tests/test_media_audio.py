"""Media probing / analysis and the audio pipeline (real ffmpeg, tiny fixtures)."""

from __future__ import annotations

import builtins
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autoclip.core import audio, media
from autoclip.core.audio import Timeline, build_audio, compute_keep, enhance_audio, frame_db, is_filler, load_window_audio, piece_bounds
from autoclip.core.config import AudioCfg, SilenceCfg
from autoclip.core.errors import AutoclipError
from autoclip.core.media import AUDIO_SR, FRAME, decode_audio, ffmpeg_exe, frame_energy_db, probe, scene_cuts, snap_to_cut, snap_to_quiet
from autoclip.core.models import Word

from .conftest import Media

TONE = (0.3 * np.sin(2 * np.pi * 220 * np.arange(2 * AUDIO_SR) / AUDIO_SR)).astype(np.float32)


# ---- media ---------------------------------------------------------------------------------------
def test_ffmpeg_exe_prefers_path_then_bundle_then_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    media.ffmpeg_exe.cache_clear()
    monkeypatch.setattr(shutil, "which", lambda _n: "/usr/bin/ffmpeg")
    assert media.ffmpeg_exe() == "/usr/bin/ffmpeg"
    media.ffmpeg_exe.cache_clear()
    monkeypatch.setattr(shutil, "which", lambda _n: None)
    assert Path(media.ffmpeg_exe()).exists()  # imageio-ffmpeg's bundled binary
    media.ffmpeg_exe.cache_clear()
    real_import = builtins.__import__

    def no_imageio(name: str, *a: Any, **k: Any) -> Any:
        if name == "imageio_ffmpeg":
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_imageio)
    with pytest.raises(AutoclipError, match="ffmpeg not found"):
        media.ffmpeg_exe()
    monkeypatch.undo()
    media.ffmpeg_exe.cache_clear()


def test_probe(media: Media, tmp_path: Path) -> None:
    info = probe(media.video)
    assert (info.width, info.height, info.has_audio) == (320, 180, True)
    assert info.duration == pytest.approx(40, abs=0.5) and info.fps == pytest.approx(15)
    assert not probe(media.silent).has_audio
    wav = tmp_path / "a.wav"
    subprocess.run([ffmpeg_exe(), "-y", "-v", "error", "-f", "lavfi", "-i", "sine=d=1", str(wav)], check=True)
    with pytest.raises(AutoclipError, match="no video stream"):
        probe(wav)
    (tmp_path / "junk.mp4").write_text("not a video")
    with pytest.raises(AutoclipError, match="cannot read"):
        probe(tmp_path / "junk.mp4")


def test_energy_track_sees_speech_and_pauses(media: Media) -> None:
    db = frame_energy_db(media.video)
    assert db is not None and len(db) == pytest.approx(80, abs=2)
    assert db[1] > db[7] + 20  # 0.5-1 s is speech, 3.5-4 s is the pause
    assert frame_energy_db(media.silent) is None


def test_snap_helpers() -> None:
    db = np.array([-10, -10, -50, -10, -10, -10], dtype=np.float32)
    assert snap_to_quiet(db, 1.2, 1.5, 3.0) == pytest.approx(2 * FRAME + FRAME / 2)
    assert snap_to_quiet(db, 0.0, 0.0, 3.0) == 0.0  # empty search range keeps t
    assert snap_to_cut(10.0, [3.0, 11.0, 12.5], 2.5) == 11.0
    assert snap_to_cut(10.0, [1.0], 2.5) == 10.0


def test_scene_cuts_with_and_without_scenedetect(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    two = tmp_path / "two.mp4"  # a hard cut between two different pictures
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=160x90:r=15:d=3",
        "-f",
        "lavfi",
        "-i",
        "color=c=red:s=160x90:r=15:d=3",
    ]
    subprocess.run([*cmd, "-filter_complex", "[0][1]concat=n=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(two)], check=True)
    assert scene_cuts(two) == pytest.approx([3.0], abs=0.3)
    monkeypatch.setitem(sys.modules, "scenedetect", None)  # simulates "not installed"
    assert scene_cuts(two) == []


# ---- audio ---------------------------------------------------------------------------------------
def test_load_window_audio_length_and_filters(media: Media) -> None:
    x = load_window_audio(media.video, 0.0, 2.0)
    assert x.dtype == np.float32 and len(x) == 2 * AUDIO_SR and np.abs(x).max() > 0.1
    assert len(load_window_audio(media.video, 0.0, 1.0, highpass=80, fft_denoise=True)) == AUDIO_SR
    assert len(load_window_audio(media.video, 39.5, 3.0)) == 3 * AUDIO_SR  # past the end: zero padded


def test_frame_db_handles_short_input() -> None:
    assert len(frame_db(np.zeros(5, np.float32))) == 1
    assert len(frame_db(np.zeros(AUDIO_SR, np.float32))) == 50


@pytest.mark.parametrize("denoise", ["none", "ffmpeg", "noisereduce", "auto"])
def test_enhance_audio_normalises_loudness_and_limits_peaks(media: Media, denoise: str) -> None:
    cfg = AudioCfg(denoise=denoise)  # type: ignore[arg-type]
    x = enhance_audio(media.video, 0.0, 3.0, cfg, threads=1)
    assert len(x) == 3 * AUDIO_SR and np.abs(x).max() <= 0.95
    speech = x[: AUDIO_SR * 2]
    assert 20 * np.log10(np.sqrt(np.mean(speech**2))) == pytest.approx(cfg.target_dbfs, abs=4)


def test_enhance_audio_off_and_auto_without_noisereduce(media: Media, monkeypatch: pytest.MonkeyPatch) -> None:
    assert len(enhance_audio(media.video, 0.0, 1.0, AudioCfg(enhance=False), 1)) == AUDIO_SR
    monkeypatch.setattr(importlib.util, "find_spec", lambda _n: None)
    assert len(enhance_audio(media.video, 0.0, 1.0, AudioCfg(denoise="auto"), 1)) == AUDIO_SR  # falls back to ffmpeg afftdn


def test_is_filler() -> None:
    assert is_filler("Um,", {"um"}) and is_filler("UH.", {"uh"}) and not is_filler("umbrella", {"um"})


def test_compute_keep_shortens_pauses_and_cuts_fillers() -> None:
    x = np.concatenate([TONE, np.zeros(2 * AUDIO_SR, np.float32), TONE])
    keep = compute_keep(x, [Word(0.2, 0.5, "um,")], 6.0, SilenceCfg())
    assert len(keep) >= 2 and sum(b - a for a, b in keep) < 5.0
    assert not any(a < 0.3 < b for a, b in keep)
    assert compute_keep(x, [], 6.0, SilenceCfg(threshold_db=-60, remove_fillers=False))[0][0] == 0.0
    assert compute_keep(x, [], 6.0, SilenceCfg(remove=False)) == [(0.0, 6.0)]
    assert compute_keep(x, [Word(0.2, 0.5, "um", pseudo=True)], 6.0, SilenceCfg(min_silence=9)) == [(0.0, 6.0)]
    assert compute_keep(np.zeros(2 * AUDIO_SR, np.float32), [], 2.0, SilenceCfg()) == [(0.0, 2.0)]  # never gut a clip


def test_compute_keep_never_cuts_below_the_requested_length() -> None:
    """The trimming may not shorten the clip below the minimum the user asked for."""
    x = np.concatenate([TONE, np.zeros(2 * AUDIO_SR, np.float32), TONE])
    words = [Word(0.2, 0.5, "um,"), Word(0.6, 0.9, "uh,")]
    free = compute_keep(x, words, 6.0, SilenceCfg())
    assert sum(b - a for a, b in free) < 5.0
    limited = compute_keep(x, words, 6.0, SilenceCfg(), 5.5)
    assert sum(b - a for a, b in limited) >= 5.5  # the clip keeps its minimum
    # the fillers are given back before the long pause
    assert any(a < 0.3 < b for a, b in limited)


def test_compute_keep_gives_back_as_little_as_possible() -> None:
    """Only as much as needed is given back: a longer minimum keeps more of the trimming."""
    x = np.concatenate([TONE, np.zeros(2 * AUDIO_SR, np.float32), TONE])
    words = [Word(0.2, 0.5, "um,")]
    a = compute_keep(x, words, 6.0, SilenceCfg(), 4.0)
    b = compute_keep(x, words, 6.0, SilenceCfg(), 5.0)
    assert sum(x1 - x0 for x0, x1 in a) <= sum(x1 - x0 for x0, x1 in b)


def test_timeline_maps_time_and_drops_removed_parts() -> None:
    tl = Timeline([(0.0, 2.0), (5.0, 8.0)])
    assert tl.total == 5.0
    assert tl.map_range(1.0, 1.5) == (1.0, 1.5) and tl.map_range(5.5, 6.0) == (2.5, 3.0)
    assert tl.map_range(1.0, 6.0) == (1.0, 3.0)
    assert tl.map_range(3.0, 4.0) is None


def test_build_audio_pads_cuts_and_declicks() -> None:
    x = np.ones(5 * AUDIO_SR, np.float32)
    y = build_audio(x, [(0.0, 1.0), (2.0, 3.0)], 2.5)
    assert len(y) == round(2.5 * AUDIO_SR) and y[0] == 0.0 and y[AUDIO_SR // 2] == 1.0 and y[-1] == 0.0
    assert len(build_audio(x, [(0.0, 1.0)], 0.5)) == AUDIO_SR // 2
    assert build_audio(x, [(0.0, 1.0)], 1.0)[0] == 1.0  # a single range is not ramped


def test_ffmpeg_failure_is_loud(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        load_window_audio(tmp_path / "missing.mp4", 0.0, 1.0)


def test_decode_audio_at_any_sample_rate(media: Media) -> None:
    whisper = decode_audio(media.video, 1.0, 2.0, 16000)
    assert whisper.dtype == np.float32 and len(whisper) == 32000 and np.abs(whisper).max() > 0.1
    assert len(decode_audio(media.video, 0.0, 1.0, 44100, ["highpass=f=80"])) == 44100
    assert len(decode_audio(media.video, 39.0, 4.0, 16000)) == 64000  # past the end: zero padded to the requested length


def test_piece_bounds_never_leave_a_tiny_tail() -> None:
    assert piece_bounds(100, 40, 10) == [(0, 40), (40, 80), (80, 100)]
    assert piece_bounds(85, 40, 10) == [(0, 40), (40, 85)]  # the 5-sample tail joins the previous piece
    assert piece_bounds(30, 40, 10) == [(0, 30)]


def test_noise_reduction_runs_in_pieces_and_skips_tiny_clips(media: Media, monkeypatch: pytest.MonkeyPatch) -> None:
    import noisereduce

    calls: list[int] = []
    real = noisereduce.reduce_noise

    def counting(**kwargs: Any) -> Any:
        calls.append(len(kwargs["y"]))
        return real(**kwargs)

    monkeypatch.setattr(noisereduce, "reduce_noise", counting)
    monkeypatch.setattr(audio, "DENOISE_CHUNK_SECONDS", 2)
    x = enhance_audio(media.video, 0.0, 5.0, AudioCfg(denoise="noisereduce"), threads=1)
    assert len(x) == 5 * AUDIO_SR and calls == [2 * AUDIO_SR, 2 * AUDIO_SR, AUDIO_SR]  # a tail of >= 1 s is its own piece
    calls.clear()
    enhance_audio(media.video, 0.0, 4.5, AudioCfg(denoise="noisereduce"), threads=1)
    assert calls == [2 * AUDIO_SR, round(2.5 * AUDIO_SR)]  # a 0.5 s tail would be too short: it joins the previous piece
    calls.clear()
    enhance_audio(media.video, 0.0, 0.5, AudioCfg(denoise="noisereduce"), threads=1)
    assert calls == []  # under a second: no noise estimate possible, denoise skipped
