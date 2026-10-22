"""Choosing the video encoder: a GPU one that *really works* on this machine, else the CPU's libx264.

Listing an encoder is not enough (ffmpeg builds list ``h264_nvenc`` on machines without NVIDIA hardware), so
each candidate is probed by encoding a short test clip with exactly the arguments that will be used later.
"""

from __future__ import annotations

import functools
import logging
import platform
import subprocess
from collections.abc import Callable

from .config import VideoCfg
from .media import ffmpeg_exe
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


def cpu_encoder(cfg: VideoCfg) -> Encoder:
    """libx264 (High profile, constant quality)."""
    return Encoder("libx264", False, cfg.preset, ("-crf", str(cfg.crf), "-profile:v", "high", *_cap(cfg)))


#: Builders of the supported GPU encoders; ``cfg.crf`` is reused as their quality level.
HARDWARE: dict[str, Callable[[VideoCfg], Encoder]] = {
    "nvenc": lambda c: Encoder(
        "h264_nvenc",
        True,
        PLACEHOLDER_PRESET,
        ("-preset", "p4", "-rc", "vbr", "-cq", str(c.crf), "-b:v", "0", "-profile:v", "high", *_cap(c)),
    ),
    "videotoolbox": lambda c: Encoder(
        "h264_videotoolbox", True, PLACEHOLDER_PRESET, ("-b:v", f"{c.maxrate_mbps or 10:g}M", "-profile:v", "high")
    ),
    "qsv": lambda c: Encoder(
        "h264_qsv", True, PLACEHOLDER_PRESET, ("-global_quality", str(c.crf), "-preset", "medium", "-profile:v", "high")
    ),
    "amf": lambda c: Encoder(
        "h264_amf",
        True,
        PLACEHOLDER_PRESET,
        ("-quality", "balanced", "-rc", "cqp", "-qp_i", str(c.crf), "-qp_p", str(c.crf), "-profile:v", "high"),
    ),
}


@functools.cache
def works(encoder: Encoder) -> bool:
    """Can ffmpeg encode a short test clip with this encoder and these arguments on this machine?"""
    cmd = [ffmpeg_exe(), "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=5:d=0.4", "-pix_fmt", "yuv420p", "-c:v", encoder.name]
    try:  # same argument order as MoviePy's writer: preset first, then our parameters
        return (
            subprocess.run(
                [*cmd, "-preset", encoder.preset, *encoder.params, "-f", "null", "-"], capture_output=True, timeout=60, check=False
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
