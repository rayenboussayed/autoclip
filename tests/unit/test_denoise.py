"""Speech enhancement before transcription: backends, model download and graceful degradation."""

from __future__ import annotations

import logging
import sys
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from autoclip.core import denoise
from autoclip.core.config import EnhanceCfg
from autoclip.core.denoise import SpeechEnhancer, fetch_model


def tone(seconds: float = 0.5, sr: int = 16000) -> NDArray[np.float32]:
    """A short noisy tone, standing in for speech."""
    t = np.arange(int(seconds * sr)) / sr
    return (0.2 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.default_rng(0).standard_normal(len(t))).astype(np.float32)


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


def test_clean_audio_is_left_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    enhancer = SpeechEnhancer(EnhanceCfg(backend="noisereduce"), tmp_path)
    called: list[int] = []

    def denoise(audio: NDArray[np.float32]) -> NDArray[np.float32]:
        called.append(1)
        return audio

    monkeypatch.setattr(enhancer, "_denoise", denoise)
    audio = speech_like()
    assert np.array_equal(enhancer(audio), audio)
    assert not called  # the denoiser never ran


def test_noisy_audio_is_enhanced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    enhancer = SpeechEnhancer(EnhanceCfg(backend="noisereduce"), tmp_path)
    monkeypatch.setattr(enhancer, "_denoise", lambda audio: audio * 0.5)
    audio = noisy()
    assert np.array_equal(enhancer(audio), audio * 0.5)


def test_the_noise_gate_can_be_switched_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    enhancer = SpeechEnhancer(EnhanceCfg(backend="noisereduce", snr_threshold_db=None), tmp_path)
    monkeypatch.setattr(enhancer, "_denoise", lambda audio: audio * 0.5)
    audio = speech_like()
    assert np.array_equal(enhancer(audio), audio * 0.5)


def test_a_broken_noise_measurement_still_enhances(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    enhancer = SpeechEnhancer(EnhanceCfg(backend="noisereduce"), tmp_path)
    monkeypatch.setattr(enhancer, "_denoise", lambda audio: audio * 0.5)
    monkeypatch.setattr(denoise, "snr_db", lambda audio: (_ for _ in ()).throw(ValueError("broken")))
    audio = speech_like()
    assert np.array_equal(enhancer(audio), audio * 0.5)  # an unknown noise level is treated as noisy


def test_disabled_enhancement_passes_the_audio_through(tmp_path: Path) -> None:
    audio = tone()
    enhancer = SpeechEnhancer(EnhanceCfg(enabled=False), tmp_path)
    assert not enhancer.active
    assert np.array_equal(enhancer(audio), audio)


def test_missing_backend_falls_back_to_the_original_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)  # importing it fails
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        enhancer = SpeechEnhancer(EnhanceCfg(backend="sherpa"), tmp_path)
    assert not enhancer.active
    assert "speech enhancement unavailable" in caplog.text
    audio = tone()
    assert np.array_equal(enhancer(audio), audio)


def test_noisereduce_backend_enhances_the_audio(tmp_path: Path) -> None:
    enhancer = SpeechEnhancer(EnhanceCfg(backend="noisereduce"), tmp_path)
    assert enhancer.active
    audio = tone()
    out = enhancer(audio)
    assert out.shape == audio.shape and out.dtype == np.float32
    assert not np.array_equal(out, audio)


def test_a_failing_backend_is_disabled_after_the_first_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(_audio: NDArray[np.float32]) -> NDArray[np.float32]:
        raise RuntimeError("model exploded")

    enhancer = SpeechEnhancer(EnhanceCfg(backend="noisereduce"), tmp_path)
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


def test_sherpa_backend_builds_a_denoiser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The sherpa-onnx model config is assembled as the library expects (both model families)."""
    built: dict[str, Any] = {}

    class Gtcrn:
        model = ""

    class Dpdf:
        model = ""
        attenuation_limit_db = 0.0

    class ModelConfig:
        def __init__(self, gtcrn: Any, dpdfnet: Any, num_threads: int, debug: bool, provider: str) -> None:
            built["gtcrn"], built["dpdfnet"], built["threads"], built["provider"] = gtcrn, dpdfnet, num_threads, provider

        def validate(self) -> bool:
            return True

    class Config:
        def __init__(self, model: Any) -> None:
            built["model"] = model

        def validate(self) -> bool:
            return True

    class Denoiser:
        def __init__(self, config: Any) -> None:
            built["denoiser"] = config

        def run(self, audio: NDArray[np.float32], sr: int) -> Any:
            built["run"] = (len(audio), sr)
            return type("Out", (), {"samples": audio * 0.5})()

    module = ModuleType("sherpa_onnx")
    module.OfflineSpeechDenoiserGtcrnModelConfig = Gtcrn  # type: ignore[attr-defined]
    module.OfflineSpeechDenoiserDpdfNetModelConfig = Dpdf  # type: ignore[attr-defined]
    module.OfflineSpeechDenoiserModelConfig = ModelConfig  # type: ignore[attr-defined]
    module.OfflineSpeechDenoiserConfig = Config  # type: ignore[attr-defined]
    module.OfflineSpeechDenoiser = Denoiser  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
    monkeypatch.setattr(denoise, "fetch_model", lambda model, directory, timeout=120.0: directory / "m.onnx")

    enhancer = SpeechEnhancer(EnhanceCfg(model="dpdfnet2", attenuation_limit_db=9.0, num_threads=2), tmp_path)
    assert enhancer.active
    assert built["dpdfnet"].model.endswith("m.onnx") and built["dpdfnet"].attenuation_limit_db == 9.0
    assert built["threads"] == 2 and built["provider"] == "cpu"
    out = enhancer(tone())
    assert np.allclose(out, tone() * 0.5)  # the model halved the samples
    assert built["run"][1] == 16000


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

    enhancer = SpeechEnhancer(EnhanceCfg(model="gtcrn"), tmp_path)
    assert not enhancer.active  # the invalid model disables the enhancement instead of failing the run
