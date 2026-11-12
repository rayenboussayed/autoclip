"""Speech enhancement before transcription: backends, model download and graceful degradation."""

from __future__ import annotations

import contextlib
import logging
import sys
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from autoclip.core import denoise
from autoclip.core.denoise import SpeechEnhancer, fetch_model


def tone(seconds: float = 0.5, sr: int = 16000) -> NDArray[np.float32]:
    """A short noisy tone, standing in for speech."""
    t = np.arange(int(seconds * sr)) / sr
    return cast(
        NDArray[np.float32],
        (0.2 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.default_rng(0).standard_normal(len(t))).astype(np.float32),
    )


def speech_like(seconds: float = 4.0, sr: int = 16000) -> NDArray[np.float32]:
    """Tone bursts separated by digital silence - a clean recording with pauses."""
    audio = np.zeros(int(seconds * sr), dtype=np.float32)
    for start in (0.0, 2.0):
        a = int(start * sr)
        audio[a : a + int(1.5 * sr)] = tone(1.5, sr)
    return audio


def noisy(seconds: float = 4.0, sr: int = 16000) -> NDArray[np.float32]:
    """The same speech with a constant noise floor of -30 dB."""
    rng = np.random.default_rng(1)
    hiss = np.asarray(10 ** (-30 / 20) * rng.standard_normal(int(seconds * sr)), dtype=np.float32)
    return np.asarray(speech_like(seconds, sr) + hiss, dtype=np.float32)


def test_the_noise_measurement_separates_clean_and_noisy_audio() -> None:
    assert denoise.snr_db(speech_like()) > 60  # digital silence between the words
    assert denoise.snr_db(noisy()) < 30  # a constant hiss
    assert denoise.snr_db(noisy()) < denoise.snr_db(speech_like())


def test_default_models_dir_follows_each_platform(tmp_path: Path) -> None:
    from autoclip.core.denoise import default_models_dir

    home = tmp_path / "home"
    linux = default_models_dir("linux", {"XDG_CACHE_HOME": str(tmp_path / "cache")}, home)
    assert linux == tmp_path / "cache" / "autoclip" / "models"
    mac = default_models_dir("darwin", {}, home)
    assert mac == home / "Library" / "Caches" / "autoclip" / "models"
    windows = default_models_dir("win32", {"LOCALAPPDATA": str(tmp_path / "local")}, home)
    assert windows == tmp_path / "local" / "autoclip" / "models"
    windows_default = default_models_dir("win32", {}, home)
    assert windows_default == home / "AppData" / "Local" / "autoclip" / "models"


def test_the_noise_gate_can_be_disabled_entirely(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    enhancer = SpeechEnhancer(True, tmp_path)
    monkeypatch.setattr(enhancer, "_denoise", lambda audio: audio * 0.5)
    monkeypatch.setattr(denoise, "ENHANCE_SNR_THRESHOLD_DB", None)
    assert np.array_equal(enhancer(noisy()), noisy() * 0.5)


def test_clean_audio_is_left_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    enhancer = SpeechEnhancer(True, tmp_path)
    called: list[int] = []

    def denoise(audio: NDArray[np.float32]) -> NDArray[np.float32]:
        called.append(1)
        return audio

    monkeypatch.setattr(enhancer, "_denoise", denoise)
    audio = speech_like()
    assert np.array_equal(enhancer(audio), audio)
    assert not called  # the denoiser never ran


def test_noisy_audio_is_enhanced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    enhancer = SpeechEnhancer(True, tmp_path)
    monkeypatch.setattr(enhancer, "_denoise", lambda audio: audio * 0.5)
    audio = noisy()
    assert np.array_equal(enhancer(audio), audio * 0.5)


def test_the_noise_gate_can_be_switched_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(denoise, "snr_db", lambda audio: 0.0)
    enhancer = SpeechEnhancer(True, tmp_path)
    monkeypatch.setattr(enhancer, "_denoise", lambda audio: audio * 0.5)
    audio = speech_like()
    assert np.array_equal(enhancer(audio), audio * 0.5)


def test_a_broken_noise_measurement_still_enhances(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    enhancer = SpeechEnhancer(True, tmp_path)
    monkeypatch.setattr(enhancer, "_denoise", lambda audio: audio * 0.5)
    monkeypatch.setattr(denoise, "snr_db", lambda audio: (_ for _ in ()).throw(ValueError("broken")))
    audio = speech_like()
    assert np.array_equal(enhancer(audio), audio * 0.5)  # an unknown noise level is treated as noisy


def test_disabled_enhancement_passes_the_audio_through(tmp_path: Path) -> None:
    audio = tone()
    enhancer = SpeechEnhancer(False, tmp_path)
    assert not enhancer.active
    assert np.array_equal(enhancer(audio), audio)


def test_missing_backend_falls_back_to_the_original_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)  # importing it fails
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        enhancer = SpeechEnhancer(True, tmp_path)
    assert not enhancer.active
    assert "speech enhancement unavailable" in caplog.text
    audio = tone()
    assert np.array_equal(enhancer(audio), audio)


def test_a_failing_backend_is_disabled_after_the_first_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(_audio: NDArray[np.float32]) -> NDArray[np.float32]:
        raise RuntimeError("model exploded")

    enhancer = SpeechEnhancer(True, tmp_path)
    monkeypatch.setattr(enhancer, "_denoise", boom)
    audio = tone()
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        assert np.array_equal(enhancer(audio), audio)
    assert "speech enhancement failed" in caplog.text
    assert not enhancer.active  # the broken backend is not tried again


def test_fetch_model_downloads_once_and_reuses_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class Response:
        def read(self) -> bytes:
            return b"onnx-bytes"

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

    def fake_urlopen(url: str, timeout: float = 0) -> Response:
        calls.append(url)
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    path = fetch_model("gtcrn", tmp_path)
    assert path.name == "gtcrn_simple.onnx" and path.read_bytes() == b"onnx-bytes"
    assert len(calls) == 1 and "gtcrn_simple.onnx" in calls[0]
    assert fetch_model("gtcrn", tmp_path) == path  # the second call reuses the file
    assert len(calls) == 1


def test_fetch_model_names_map_to_the_release_files() -> None:
    assert denoise.SHERPA_MODELS["gtcrn"] == "gtcrn_simple"
    assert set(denoise.SHERPA_MODELS) == {"gtcrn", "dpdfnet_baseline", "dpdfnet2", "dpdfnet4", "dpdfnet8"}
    assert denoise.MODEL_URL.startswith("https://")


def test_an_invalid_model_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Config:
        def __init__(self, model: Any) -> None:
            pass

        def validate(self) -> bool:
            return False

    class Gtcrn:
        model = ""

    class Dpdf:
        model = ""
        attenuation_limit_db = 0.0

    class ModelConfig:
        def __init__(self, *args: Any) -> None:
            pass

    module = ModuleType("sherpa_onnx")
    module.OfflineSpeechDenoiserGtcrnModelConfig = Gtcrn  # type: ignore[attr-defined]
    module.OfflineSpeechDenoiserDpdfNetModelConfig = Dpdf  # type: ignore[attr-defined]
    module.OfflineSpeechDenoiserModelConfig = ModelConfig  # type: ignore[attr-defined]
    module.OfflineSpeechDenoiserConfig = Config  # type: ignore[attr-defined]
    module.OfflineSpeechDenoiser = lambda config: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
    monkeypatch.setattr(denoise, "fetch_model", lambda model, directory, timeout=120.0: directory / "m.onnx")

    enhancer = SpeechEnhancer(True, tmp_path)
    assert not enhancer.active  # the invalid model disables the enhancement instead of failing the run


def test_fetch_model_retries_then_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    with contextlib.suppress(Exception):
        fetch_model("gtcrn", tmp_path)
