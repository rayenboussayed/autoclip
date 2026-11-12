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
from .internal.quality import quality_profile
from .media import external_env
from .models import Encoder

log = logging.getLogger("autoclip")
#: GPU encoders to try per operating system, best first.
PRIORITY = {"Darwin": ("videotoolbox",), "Windows": ("nvenc", "qsv", "amf"), "Linux": ("nvenc", "qsv")}
#: Consumer GPUs limit the number of simultaneous encode sessions; never run more workers than this on a GPU.
HW_MAX_WORKERS = 3
#: What MoviePy is given as ``-preset`` for GPU encoders; encoders with real presets override it in their arguments,
#: the others (VideoToolbox, AMF) just ignore it with a warning.
PLACEHOLDER_PRESET = "medium"


def _profile(cfg: VideoCfg) -> str:
    """``high`` for H.264, ``main`` for H.265 (HEVC has no High profile in yuv420p 8-bit)."""
    return "main" if cfg.codec == "h265" else "high"


def _hevc(cfg: VideoCfg) -> int:
    """Index into the per-family codec names: 0 = H.264, 1 = H.265."""
    return 1 if cfg.codec == "h265" else 0


def _videotoolbox(cfg: VideoCfg, quality: str) -> Encoder:
    """VideoToolbox: quality 100 best, quality 1 most efficient."""
    profile = quality_profile(quality)
    return Encoder(VIDEOTOOLBOX[_hevc(cfg)], True, PLACEHOLDER_PRESET, ("-q:v", str(profile.videotoolbox_qv), "-profile:v", _profile(cfg)))


def cpu_encoder(cfg: VideoCfg, quality: str) -> Encoder:
    """libx264 / libx265 (constant quality)."""
    profile = quality_profile(quality)
    name = "libx265" if cfg.codec == "h265" else "libx264"
    return Encoder(name, False, profile.preset, ("-crf", str(profile.crf), "-profile:v", _profile(cfg)))


#: Per-family codec names: index 0 = H.264, index 1 = H.265 (HEVC).
NVENC = ("h264_nvenc", "hevc_nvenc")
VIDEOTOOLBOX = ("h264_videotoolbox", "hevc_videotoolbox")
QSV = ("h264_qsv", "hevc_qsv")
AMF = ("h264_amf", "hevc_amf")


#: Builders of the supported GPU encoders; the top-level ``quality`` selects their native quality level.
def _nvenc_preset(quality: str) -> str:
    return {"highest": "p5", "high": "p4", "medium": "p3", "low": "p2", "lowest": "p1"}[quality]


def _amf_quality(quality: str) -> str:
    return {"highest": "quality", "high": "balanced", "medium": "balanced", "low": "speed", "lowest": "speed"}[quality]


HARDWARE: dict[str, Callable[[VideoCfg, str], Encoder]] = {
    "nvenc": lambda c, q: Encoder(
        NVENC[_hevc(c)],
        True,
        PLACEHOLDER_PRESET,
        (
            "-preset",
            _nvenc_preset(q),
            "-rc",
            "vbr",
            "-cq",
            str(quality_profile(q).nvenc_cq),
            "-b:v",
            "0",
            "-profile:v",
            _profile(c),
        ),
    ),
    "videotoolbox": _videotoolbox,
    "qsv": lambda c, q: Encoder(
        QSV[_hevc(c)],
        True,
        PLACEHOLDER_PRESET,
        ("-global_quality", str(quality_profile(q).nvenc_cq), "-preset", "medium", "-profile:v", _profile(c)),
    ),
    "amf": lambda c, q: Encoder(
        AMF[_hevc(c)],
        True,
        PLACEHOLDER_PRESET,
        (
            "-quality",
            _amf_quality(q),
            "-rc",
            "cqp",
            "-qp_i",
            str(quality_profile(q).nvenc_cq),
            "-qp_p",
            str(quality_profile(q).nvenc_cq),
            "-profile:v",
            _profile(c),
        ),
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


def pick_encoder(cfg: VideoCfg, quality: str, system: str | None = None, allow_gpu: bool = True) -> Encoder:
    """The encoder to use: the first working GPU encoder of this OS, otherwise the CPU's libx264/libx265.

    ``allow_gpu`` is false when ``compute.device`` is ``"cpu"``: then the GPU is not touched at all.
    """
    cpu = cpu_encoder(cfg, quality)
    if not allow_gpu:
        log.info("compute.device is 'cpu': using %s on the CPU", cpu.name)
        return cpu
    for name in PRIORITY.get(system or platform.system(), ()):
        candidate = HARDWARE[name](cfg, quality)
        if works(candidate):
            log.info("GPU video encoder: %s", candidate.name)
            return candidate
        log.debug("GPU encoder %s is not usable on this machine", candidate.name)
    log.info("no usable GPU video encoder: using %s on the CPU", cpu.name)
    return cpu
