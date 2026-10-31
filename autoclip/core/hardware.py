"""Choosing the video encoder: a GPU one that *really works* on this machine, else the CPU's libx264.

Listing an encoder is not enough (ffmpeg builds list ``h264_nvenc`` on machines without NVIDIA hardware), so
each candidate is probed by encoding a short test clip with exactly the arguments - and the same ffmpeg binary -
that MoviePy will use later.
"""

from __future__ import annotations

import functools
import logging
import platform
import subprocess
from collections.abc import Callable

from .config import VideoCfg
from .media import external_env
from .models import Encoder

log = logging.getLogger("autoclip")
#: GPU encoders to try per operating system, best first (``video.gpu = "auto"``).
PRIORITY = {"Darwin": ("videotoolbox",), "Windows": ("nvenc", "qsv", "amf"), "Linux": ("nvenc", "qsv")}
#: Consumer GPUs limit the number of simultaneous encode sessions; never run more workers than this on a GPU.
HW_MAX_WORKERS = 3
#: What MoviePy is given as ``-preset`` for GPU encoders; encoders with real presets override it in their arguments,
#: the others (VideoToolbox, AMF) just ignore it with a warning.
PLACEHOLDER_PRESET = "medium"


def _cap(cfg: VideoCfg) -> tuple[str, ...]:
    """Bitrate cap arguments (upload friendly); empty when uncapped."""
    m = cfg.maxrate_mbps
    return ("-maxrate", f"{m:g}M", "-bufsize", f"{2 * m:g}M") if m > 0 else ()


def _hevc(cfg: VideoCfg) -> int:
    """Index into the per-family codec names: 0 = H.264, 1 = H.265."""
    return 1 if cfg.codec == "h265" else 0


def _profile(cfg: VideoCfg) -> str:
    """``high`` for H.264, ``main`` for H.265 (HEVC has no High profile in yuv420p 8-bit)."""
    return "main" if cfg.codec == "h265" else "high"


def _videotoolbox(cfg: VideoCfg) -> Encoder:
    """VideoToolbox: ``-q:v`` (1-100, higher is better) when uncapped, a bitrate when capped.

    A capped bitrate used to be its only mode, so ``maxrate_mbps = 0`` silently meant "10 Mbps". The quality
    mode maps CRF (0 best .. 51 worst) onto VideoToolbox's 1-100 scale.
    """
    quality = max(1, min(100, round((51 - cfg.crf) * 100 / 51)))
    mode = ("-b:v", f"{cfg.maxrate_mbps:g}M", *_cap(cfg)) if cfg.maxrate_mbps > 0 else ("-q:v", str(quality))
    return Encoder(VIDEOTOOLBOX[_hevc(cfg)], True, PLACEHOLDER_PRESET, (*mode, "-profile:v", _profile(cfg)))


def cpu_encoder(cfg: VideoCfg) -> Encoder:
    """libx264 / libx265 (constant quality)."""
    name = "libx265" if cfg.codec == "h265" else "libx264"
    return Encoder(name, False, cfg.preset, ("-crf", str(cfg.crf), "-profile:v", _profile(cfg), *_cap(cfg)))


#: Per-family codec names: index 0 = H.264, index 1 = H.265 (HEVC).
NVENC = ("h264_nvenc", "hevc_nvenc")
VIDEOTOOLBOX = ("h264_videotoolbox", "hevc_videotoolbox")
QSV = ("h264_qsv", "hevc_qsv")
AMF = ("h264_amf", "hevc_amf")

#: Builders of the supported GPU encoders; ``cfg.crf`` is reused as their quality level.
HARDWARE: dict[str, Callable[[VideoCfg], Encoder]] = {
    "nvenc": lambda c: Encoder(
        NVENC[_hevc(c)],
        True,
        PLACEHOLDER_PRESET,
        ("-preset", "p4", "-rc", "vbr", "-cq", str(c.crf), "-b:v", "0", "-profile:v", _profile(c), *_cap(c)),
    ),
    "videotoolbox": _videotoolbox,
    "qsv": lambda c: Encoder(
        QSV[_hevc(c)], True, PLACEHOLDER_PRESET, ("-global_quality", str(c.crf), "-preset", "medium", "-profile:v", _profile(c))
    ),
    "amf": lambda c: Encoder(
        AMF[_hevc(c)],
        True,
        PLACEHOLDER_PRESET,
        ("-quality", "balanced", "-rc", "cqp", "-qp_i", str(c.crf), "-qp_p", str(c.crf), "-profile:v", _profile(c)),
    ),
}


def encoding_ffmpeg() -> str:
    """The ffmpeg MoviePy writes with: its bundled binary unless ``FFMPEG_BINARY`` names another one.

    ``media.ffmpeg_exe`` prefers the ffmpeg on ``PATH`` (used to decode), but MoviePy has always imported the
    bundled one instead. Probing an encoder with a different binary than the one that will encode is how a GPU
    encoder gets "proven" and then fails at render time with an option the other build does not know.
    """
    from moviepy.video.io import ffmpeg_writer  # heavy: only imported when an encoder is probed

    return str(ffmpeg_writer.FFMPEG_BINARY)


@functools.cache
def works(encoder: Encoder) -> bool:
    """Can the ffmpeg MoviePy writes with encode a short test clip with this encoder and these arguments?"""
    cmd = [encoding_ffmpeg(), "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=5:d=0.4"]
    cmd += ["-pix_fmt", "yuv420p", "-c:v", encoder.name]
    try:  # same argument order as MoviePy's writer: preset first, then our parameters
        return (
            subprocess.run(
                [*cmd, "-preset", encoder.preset, *encoder.params, "-f", "null", "-"],
                capture_output=True,
                timeout=60,
                check=False,
                env=external_env(),
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def pick_encoder(cfg: VideoCfg, system: str | None = None, allow_gpu: bool = True) -> Encoder:
    """The encoder to use: the first working GPU encoder (``video.gpu``), otherwise libx264 on the CPU.

    ``allow_gpu`` is false when ``compute.device`` is ``"cpu"``: then the GPU is not touched at all.
    """
    cpu = cpu_encoder(cfg)
    if not allow_gpu:
        log.info("compute.device is 'cpu': using libx264 on the CPU")
        return cpu
    if cfg.gpu == "off":
        return cpu
    names = PRIORITY.get(system or platform.system(), ()) if cfg.gpu == "auto" else (cfg.gpu,)
    for name in names:
        candidate = HARDWARE[name](cfg)
        if works(candidate):
            log.info("GPU video encoder: %s", candidate.name)
            return candidate
        log.debug("GPU encoder %s is not usable on this machine", candidate.name)
    if cfg.gpu != "auto":
        log.warning("video.gpu = '%s' is not usable on this machine: falling back to the CPU encoder", cfg.gpu)
    log.info("no usable GPU video encoder: using libx264 on the CPU")
    return cpu
