"""Speech enhancement before transcription: the ASR model should hear clear words.

Two backends are available.  ``sherpa`` runs a small ONNX denoiser from the sherpa-onnx
release (no torch, no separate environment); ``noisereduce`` is spectral gating, which
the project already depends on.  Both are optional: when a backend cannot be used the
untouched audio is transcribed instead and a warning is logged - a missing model must
never stop a run.

Denoising only helps when there is noise to remove.  A clean recording (a screen
recording with a good microphone, for instance) is damaged by it, and the ASR model then
drifts into the wrong language.  ``skip.denoise`` turns it off and every chunk cleaner
than ``ENHANCE_SNR_THRESHOLD_DB`` is transcribed untouched.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from .audio import frame_db
from .errors import AutoclipError
from .internal.automation import DENOISE_STRENGTH, ENHANCE_ATTENUATION_DB, ENHANCE_SNR_THRESHOLD_DB, denoiser_threads
from .media import F32
from .resources import RESOURCES

log = logging.getLogger("autoclip")

#: The sample rate the denoisers and faster-whisper work with.
SR = 16000


def default_models_dir(system: str | None = None, env: Mapping[str, str] | None = None, home: Path | None = None) -> Path:
    """The shared per-user cache for the speech enhancement model (never inside the output folder).

    Keeps the ONNX denoiser once per machine instead of once per output folder, so changing the
    output location does not re-download it.  ``system``/``home`` make the per-OS layout testable.
    """
    import os

    system = system or sys.platform
    env = dict(os.environ if env is None else env)
    home = home or Path.home()
    if system == "darwin":
        base = home / "Library" / "Caches"
    elif system == "win32":
        base = Path(env.get("LOCALAPPDATA") or (home / "AppData" / "Local"))
    else:
        base = Path(env.get("XDG_CACHE_HOME") or (home / ".cache"))
    return base / "autoclip" / "models"


#: The ONNX denoisers of the sherpa-onnx release: config name -> release file name.
SHERPA_MODELS: dict[str, str] = {
    "gtcrn": "gtcrn_simple",
    "dpdfnet_baseline": "dpdfnet_baseline",
    "dpdfnet2": "dpdfnet2",
    "dpdfnet4": "dpdfnet4",
    "dpdfnet8": "dpdfnet8",
}
#: SHA256 of each model file, used to verify the download before it is trusted.
#: Update together with SHERPA_MODELS when upstream re-releases.
#: SHA256 digests of the five sherpa-onnx speech-enhancement models, verified at download.
#: Pinned from upstream releases on 2026-11-13; bump intentionally with a re-download check.
SHERPA_MODEL_SHA256: dict[str, str] = {
    "gtcrn_simple": "e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534",
    "dpdfnet_baseline": "debaf0e8893479fca91b8b4c1eae8db195aa8980ccc9012f5809fde2b738151a",
    "dpdfnet2": "ce35d6025fc71df0ef10d1540e1b7916837bbfe5f6896deb744508d2cad487a9",
    "dpdfnet4": "71b588bc26163941aa82a592cce924b08c1fbdc0879fe44f5a2d4eac44bd8420",
    "dpdfnet8": "2751c1f5a4e849d23a07c675b4c838158b249b42152f10cc318522dd339134f0",
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
    want_sha = SHERPA_MODEL_SHA256.get(name)
    log.info("downloading speech enhancement model %s", name)
    last_exc: Exception | None = None
    for _attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed https url
                data = response.read()
            if want_sha is not None and (got := hashlib.sha256(data).hexdigest()) != want_sha:
                raise OSError(f"SHA-256 mismatch for {name}: got {got}, expected {want_sha}")
            break
        except Exception as exc:  # transient network problems are allowed to retry
            last_exc = exc
    else:
        raise last_exc if last_exc is not None else OSError("download failed")
    partial = dest.with_suffix(".part")
    partial.write_bytes(data)
    partial.replace(dest)
    return dest


def _sherpa_denoise(model_dir: Path) -> Callable[[F32], F32]:
    """A denoiser that runs the sherpa-onnx gtcrn ONNX model (the default, fastest at this quality class)."""
    import sherpa_onnx

    path = fetch_model("gtcrn", model_dir)
    gtcrn = sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig()
    gtcrn.model = str(path)
    dpdfnet = sherpa_onnx.OfflineSpeechDenoiserDpdfNetModelConfig()
    dpdfnet.attenuation_limit_db = ENHANCE_ATTENUATION_DB
    threads = denoiser_threads(getattr(RESOURCES, "mode", "balance"))
    config = sherpa_onnx.OfflineSpeechDenoiserConfig(
        model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(gtcrn, dpdfnet, threads, False, "cpu")
    )
    if not config.validate():
        raise AutoclipError("invalid speech enhancement model 'gtcrn'")
    denoiser = sherpa_onnx.OfflineSpeechDenoiser(config)

    def denoise(audio: F32) -> F32:  # pragma: no cover
        return np.asarray(denoiser.run(audio, SR).samples, dtype=np.float32)  # pragma: no cover

    return denoise  # pragma: no cover


def _noisereduce_denoise() -> Callable[[F32], F32]:
    """A denoiser that removes stationary noise with spectral gating."""
    import noisereduce  # pragma: no cover

    def denoise(audio: F32) -> F32:  # pragma: no cover
        n_jobs = denoiser_threads(getattr(RESOURCES, "mode", "balance"))  # pragma: no cover
        result = noisereduce.reduce_noise(
            y=audio, sr=SR, stationary=False, n_jobs=n_jobs, prop_decrease=DENOISE_STRENGTH
        )  # pragma: no cover
        return np.asarray(result, dtype=np.float32)  # pragma: no cover

    return denoise  # pragma: no cover


class SpeechEnhancer:
    """Denoises the audio that is handed to the ASR model.

    The backend is built once (model download included) and reused for every chunk.  Any
    problem - the package is missing, the download failed, the model cannot run - falls
    back to the original audio with a warning, so a run never fails because of it.
    """

    def __init__(self, enabled: bool, model_dir: Path) -> None:
        self.enabled, self.model_dir = enabled, model_dir
        self._denoise: Callable[[F32], F32] | None = None
        if not enabled:
            return
        try:
            self._denoise = _sherpa_denoise(model_dir)
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
        limit = ENHANCE_SNR_THRESHOLD_DB
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
