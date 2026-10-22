"""Speech enhancement before transcription: the ASR model should hear clear words.

Two backends are available.  ``sherpa`` runs a small ONNX denoiser from the sherpa-onnx
release (no torch, no separate environment); ``noisereduce`` is spectral gating, which
the project already depends on.  Both are optional: when a backend cannot be used the
untouched audio is transcribed instead and a warning is logged - a missing model must
never stop a run.

Denoising only helps when there is noise to remove.  A clean recording (a screen
recording with a good microphone, for instance) is damaged by it, and the ASR model then
drifts into the wrong language.  ``enhance.snr_threshold_db`` therefore skips the
enhancement for every chunk that is already clean enough.
"""

from __future__ import annotations

import logging
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .audio import frame_db
from .config import EnhanceCfg
from .errors import AutoclipError
from .media import F32

log = logging.getLogger("autoclip")

#: The sample rate the denoisers and faster-whisper work with.
SR = 16000

#: The ONNX denoisers of the sherpa-onnx release: config name -> release file name.
SHERPA_MODELS: dict[str, str] = {
    "gtcrn": "gtcrn_simple",
    "dpdfnet_baseline": "dpdfnet_baseline",
    "dpdfnet2": "dpdfnet2",
    "dpdfnet4": "dpdfnet4",
    "dpdfnet8": "dpdfnet8",
}
MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/{name}.onnx"


def snr_db(audio: F32) -> float:
    """The speech-to-noise ratio of ``audio`` in dB (speech level minus the noise floor).

    The noise floor is the quietest tenth of the frames (the pauses between the words), the
    speech level the loudest tenth.  Clean recordings score far above 20 dB.
    """
    db = frame_db(audio)
    return float(np.percentile(db, 90) - np.percentile(db, 10))


def fetch_model(model: str, model_dir: Path, timeout: float = 120.0) -> Path:
    """The ONNX file of ``model``, downloaded into ``model_dir`` once."""
    name = SHERPA_MODELS[model]
    dest = model_dir / f"{name}.onnx"
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    model_dir.mkdir(parents=True, exist_ok=True)
    url = MODEL_URL.format(name=name)
    log.info("downloading speech enhancement model %s", name)
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed https url
        data = response.read()
    partial = dest.with_suffix(".part")
    partial.write_bytes(data)
    partial.replace(dest)
    return dest


def _sherpa_denoise(cfg: EnhanceCfg, model_dir: Path) -> Callable[[F32], F32]:
    """A denoiser that runs the sherpa-onnx ONNX model of ``cfg.model``."""
    import sherpa_onnx

    path = fetch_model(cfg.model, model_dir)
    gtcrn = sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig()
    dpdfnet = sherpa_onnx.OfflineSpeechDenoiserDpdfNetModelConfig()
    if cfg.model.startswith("dpdfnet"):
        dpdfnet.model = str(path)
        dpdfnet.attenuation_limit_db = cfg.attenuation_limit_db
    else:
        gtcrn.model = str(path)
    config = sherpa_onnx.OfflineSpeechDenoiserConfig(
        model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(gtcrn, dpdfnet, cfg.num_threads, False, "cpu")
    )
    if not config.validate():
        raise AutoclipError(f"invalid speech enhancement model '{cfg.model}'")
    denoiser = sherpa_onnx.OfflineSpeechDenoiser(config)

    def denoise(audio: F32) -> F32:
        return np.asarray(denoiser.run(audio, SR).samples, dtype=np.float32)

    return denoise


def _noisereduce_denoise(cfg: EnhanceCfg) -> Callable[[F32], F32]:
    """A denoiser that removes stationary noise with spectral gating."""
    import noisereduce

    def denoise(audio: F32) -> F32:
        return np.asarray(
            noisereduce.reduce_noise(y=audio, sr=SR, stationary=False, n_jobs=cfg.num_threads, prop_decrease=cfg.denoise_strength),
            dtype=np.float32,
        )

    return denoise


class SpeechEnhancer:
    """Denoises the audio that is handed to the ASR model.

    The backend is built once (model download included) and reused for every chunk.  Any
    problem - the package is missing, the download failed, the model cannot run - falls
    back to the original audio with a warning, so a run never fails because of it.
    """

    def __init__(self, cfg: EnhanceCfg, model_dir: Path) -> None:
        self.cfg, self.model_dir = cfg, model_dir
        self._denoise: Callable[[F32], F32] | None = None
        if not cfg.enabled:
            return
        try:
            self._denoise = _sherpa_denoise(cfg, model_dir) if cfg.backend == "sherpa" else _noisereduce_denoise(cfg)
        except Exception as exc:  # optional dependency, download or model problem
            log.warning("speech enhancement unavailable (%s: %s) - transcribing the original audio", type(exc).__name__, exc)

    @property
    def active(self) -> bool:
        """Whether a backend is available (the audio may still be left alone when it is clean)."""
        return self._denoise is not None

    def __call__(self, audio: F32) -> F32:
        """The enhanced audio (the original one when the backend is unavailable)."""
        if self._denoise is None:
            return audio
        limit = self.cfg.snr_threshold_db
        if limit is not None:
            try:
                ratio = snr_db(audio)
            except Exception:  # a broken measurement must not stop the run
                ratio = 0.0
            if ratio >= limit:
                log.info("audio is clean (%.0f dB signal-to-noise): not enhancing", ratio)
                return audio  # the audio is already clean: denoising would only remove signal
            log.info("enhancing noisy audio (%.0f dB signal-to-noise)", ratio)
        try:
            return self._denoise(audio)
        except Exception as exc:
            log.warning("speech enhancement failed (%s: %s) - using the original audio", type(exc).__name__, exc)
            self._denoise = None
            return audio
