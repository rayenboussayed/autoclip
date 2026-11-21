"""Rules that turn measured resources into engine decisions, and the logs that explain them.

Every rule is pure: measurements are passed in or read from :mod:`probes` at call time, so tests can
inject exact values by patching ``derive.policy.detect_cpu_count`` / ``free_ram_mb``. Nothing here is
user-facing configuration; each choice is logged as ``automation <name>=<value> | source=<rule>``.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from ..errors import ConfigError
from .constants import (
    CACHE_DISK_MB,
    CHUNK_MINUTES,
    DEFAULT_FILLERS,  # re-exported for the old decision-log callers
    JOB_RAM_MB,
    LEXICON_FILE,
    OLLAMA_RAM_FRACTION,
    denoiser_threads,
    download_quality_for,
    profile_for,
    transcription_beam_size,
)
from .probes import detect_cpu_count, detect_device, free_ram_mb

__all__ = [
    "JOB_RAM_MB",
    "LEXICON_FILE",
    "AutoDecision",
    "Pressure",
    "RuntimeDecision",
    "auto_max_concurrent_jobs",
    "auto_mode",
    "auto_ollama_model",
    "auto_quality",
    "auto_whisper_model",
    "background_kind",
    "decide_admission",
    "detect_cpu_count",
    "detect_device",
    "fillers_from",
    "free_ram_mb",
    "is_color",
    "load_lexicon",
    "log_decisions",
    "resolve_background",
    "runtime_decisions",
    "whisper_model_for",
]

log = logging.getLogger("autoclip")

_MIB = 1 << 20


def _free_mb(free_ram_mb: float) -> float:
    """A finite free-RAM reading in MB, or ``0.0`` for NaN/infinity/nonsense (the safe direction)."""
    return free_ram_mb if math.isfinite(free_ram_mb) and free_ram_mb > 0.0 else 0.0


def auto_quality(*, cpus: int, free_ram_mb: float) -> Literal["high", "medium", "low"]:
    """The encode tier the machine sustains: ``high`` on any normal desktop, downgraded when it cannot cope.

    ``high`` (the pre-4.x default) needs 4 cores and 3 GB of free RAM; ``medium`` (2 cores, 2 GB) still renders
    comfortably; ``low`` works everywhere else. The tier is never above ``high``: a GPU speeds the encode up,
    it is not asked to make the picture better than what a machine without one would produce.
    """
    if cpus >= 4 and _free_mb(free_ram_mb) >= 3072:
        return "high"
    if cpus >= 2 and _free_mb(free_ram_mb) >= 2048:
        return "medium"
    return "low"


def auto_mode(*, cpus: int, free_ram_mb: float) -> Literal["performance", "resources", "balance"]:
    """The resource profile the machine is given: ``resources`` when it is tight, ``performance`` when ample.

    The adaptive tuner keeps refining the CPU budget inside the chosen profile during the run, exactly as
    when ``balance`` was the user-visible default.
    """
    free = _free_mb(free_ram_mb)
    if cpus <= 2 or free < 1536:
        return "resources"
    if cpus >= 8 and free >= 8192:
        return "performance"
    return "balance"


def auto_whisper_model(*, device: str, cpus: int, free_ram_mb: float) -> str:
    """CUDA: ``large-v3``. CPU: ``medium`` only on a performance-class machine, else ``small``.

    CPU ``medium`` is about five times slower than ``small`` and ``large-v3`` takes hours on a long
    video, so the larger pick is gated by the same ample-machine rule as the resource profile. The
    RAM fit in ``transcript.fit_whisper_model`` still shrinks the result when memory is tight.
    """
    if device == "cuda":
        return "large-v3"
    return "medium" if auto_mode(cpus=cpus, free_ram_mb=free_ram_mb) == "performance" else "small"


def auto_max_concurrent_jobs(*, cpus: int, free_ram_mb: float) -> int:
    """How many pipeline jobs the HTTP server may run at once (1-4).

    Memory-bound first (each job peaks around ``JOB_RAM_MB``), then core-bound (a job is not free on
    one core). Measured once per server process, at the first job; restart to re-evaluate.
    """
    free = _free_mb(free_ram_mb)
    jobs = min(int(cpus) // 2, int(free // JOB_RAM_MB), 4)
    return max(1, jobs)


def auto_ollama_model(installed: Mapping[str, int], free_ram_mb: float) -> str | None:
    """Choose which installed model the pipeline should use for ranking, titles and captions.

    Args:
        installed: model name -> size on disk in bytes (Ollama's ``/api/tags`` ``size`` field); sizes
            of ``0`` or less mean "unknown".
        free_ram_mb: free RAM in MB at probe time.

    Returns:
        The largest model whose size fits inside :data:`OLLAMA_RAM_FRACTION` of free RAM; when none
        fits, the smallest known model; when every size is unknown, the first name alphabetically
        (``None`` for an empty selection, i.e. the AI helpers stay off). With zero usable free RAM the
        smallest known model is returned; whether it fits in practice is the runner's business.
    """
    if not installed:
        return None
    free = _free_mb(free_ram_mb)
    known = {name: size for name, size in installed.items() if size > 0}
    if not known:
        return sorted(installed)[0]
    budget = int(free * _MIB * OLLAMA_RAM_FRACTION)
    fits = {name: size for name, size in known.items() if size <= budget}
    if fits:
        # largest that fits; ties broken alphabetically so the choice is deterministic
        return max(sorted(fits), key=fits.__getitem__)
    return min(sorted(known), key=known.__getitem__)


# --- admission policy -----------------------------------------------------------------------------

#: Stages whose work is cosmetic and regenerable; only these may be dropped outright under pressure.
SKIPPABLE_STAGES = frozenset({"thumbnail"})


@dataclass(frozen=True, slots=True)
class AutoDecision:
    """One automatic decision.

    ``source`` names the rule that produced it (``ram-oom``, ``ram-high``, ``cost``, ``headroom``,
    ``monitor-unavailable``), so the log line is greppable per cause.
    """

    name: str
    value: str
    source: str


@dataclass(frozen=True, slots=True)
class Pressure:
    """A snapshot the caller measured once; keeps this module syscall-free and fully testable."""

    free_bytes: int
    total_bytes: int
    high: bool
    oom: bool


def decide_admission(stage: str, cost_bytes: int, pressure: Pressure) -> AutoDecision:
    """Whether ``stage`` may run now.

    ``go`` runs it, ``defer`` waits for a finished clip to free memory (the pool never starves:
    when nothing is running the caller submits anyway), ``skip`` drops it entirely (only for
    cosmetic, regenerable stages - every other stage falls back to ``defer``).
    """
    if pressure.oom:
        return AutoDecision(stage, "defer", "ram-oom")
    if pressure.high and stage in SKIPPABLE_STAGES:
        return AutoDecision(stage, "skip", "ram-high")
    if cost_bytes > 0 and cost_bytes > pressure.free_bytes:
        return AutoDecision(stage, "defer", "cost")
    return AutoDecision(stage, "go", "headroom")


# --- runtime decision log -------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeDecision:
    """One computed value plus why it was chosen and what it affects."""

    name: str
    value: object
    source: str


def whisper_model_for(device: str = "cpu", *, cpus: int | None = None, ram_mb: float | None = None) -> str:
    """The Whisper model for a machine: ``large-v3`` on CUDA; on CPU ``medium`` only on ample hardware.

    ``cpus``/``ram_mb`` let tests pass exact measurements; production uses this process's readings.
    The RAM fit in ``transcript.fit_whisper_model`` is the final arbiter.
    """
    return auto_whisper_model(
        device=device,
        cpus=detect_cpu_count() if cpus is None else cpus,
        free_ram_mb=free_ram_mb() if ram_mb is None else ram_mb,
    )


def runtime_decisions(quality: str = "high", mode: str = "balance", device: str = "cpu") -> list[RuntimeDecision]:
    """The derived values logged for this run (diagnostics, not a contract)."""
    profile = profile_for(mode)
    return [
        RuntimeDecision("quality", quality, "derived from cpus/free-ram"),
        RuntimeDecision("mode", mode, "derived from cpus/free-ram"),
        RuntimeDecision("chunk_minutes", CHUNK_MINUTES, "fixed memory-safe chunk"),
        RuntimeDecision("cache_disk_mb", CACHE_DISK_MB, "fixed SQLite cache ceiling"),
        RuntimeDecision("encoding_threads", profile.encoding_threads_cap, f"derived from mode={profile.name}"),
        RuntimeDecision("max_cpu_percent", profile.cpu_percent_default, f"derived from mode={profile.name}"),
        RuntimeDecision("memory_reserve_mb", profile.memory_reserve_mb, f"derived from mode={profile.name}"),
        RuntimeDecision("memory_fraction", profile.mem_fraction, f"derived from mode={profile.name}"),
        RuntimeDecision("download_quality", download_quality_for(quality), f"derived from quality={quality}"),
        RuntimeDecision("transcript_model", whisper_model_for(device), f"derived from device={device} (pre free-RAM fit)"),
        RuntimeDecision("transcription_beam_size", transcription_beam_size(device), f"derived from device={device}"),
        RuntimeDecision("denoiser_threads", denoiser_threads(mode), f"derived from mode={mode}"),
        RuntimeDecision("silence_threshold", "auto", "per-input percentile for pause removal (the unsilence pre-pass uses a fixed -40 dB)"),
    ]


class _HasQualityAndMode(Protocol):
    """The minimal surface of a runtime config the decision log needs (``Config`` satisfies it)."""

    @property
    def quality(self) -> str: ...
    @property
    def mode(self) -> str: ...


def log_decisions(cfg: _HasQualityAndMode | None = None) -> None:
    """Record the derived values in the run log (a best-effort snapshot, not a contract).

    The Whisper model row names the requested model (before the free-memory fit can shrink it),
    and ``max_cpu_percent`` reports the profile default (the adaptive value, if any, is logged by
    the monitor). Without ``cfg`` the snapshot is the high/balance defaults on the detected device.
    """
    quality, mode = ("high", "balance")
    if cfg is not None:
        quality, mode = cfg.quality, cfg.mode
    for decision in runtime_decisions(quality, mode, detect_device()):
        log.info(
            "automation %s=%s | source=%s",
            decision.name,
            decision.value,
            decision.source,
        )


# --- background and lexicon values ----------------------------------------------------------------

#: ``#RGB`` or ``#RRGGBB`` (the two forms Pillow's ``ImageColor`` reads unambiguously).
_COLOR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def is_color(value: str) -> bool:
    """Whether ``value`` is a ``#RGB``/``#RRGGBB`` colour."""
    return bool(_COLOR.fullmatch(value))


def background_kind(value: str) -> str:
    """``"blur"``, ``"color"`` or ``"image"`` for a ``video.background`` value.

    Raises:
        ConfigError: for an empty value or a ``#`` value that is not a colour.
    """
    if value == "blur":
        return "blur"
    if value.startswith("#"):
        if is_color(value):
            return "color"
        raise ConfigError(f"video.background: '{value}' is not a colour (use #RGB, #RRGGBB, 'blur' or an image path)")
    if not value.strip():
        raise ConfigError("video.background must be 'blur', a #RGB/#RRGGBB colour or an image path")
    return "image"


def resolve_background(value: str, base: Path) -> str:
    """``value`` with an image path made absolute against the config file's folder."""
    if background_kind(value) != "image":
        return value
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else base / path)


def load_lexicon(base: Path | None) -> dict[str, list[str]]:
    """The word lists in ``<base>/lexicon.json`` (empty when the folder has no such file).

    Raises:
        ConfigError: if the file is unreadable, not a JSON object of string lists, or holds an empty word.
    """
    if base is None:
        return {}
    path = base / LEXICON_FILE
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"cannot read '{path}': {e.strerror or e}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a JSON object")
    words: dict[str, list[str]] = {}
    for name, entries in raw.items():
        if not isinstance(entries, list) or not all(isinstance(w, str) and w.strip() for w in entries):
            raise ConfigError(f"{path}: '{name}' must be a list of non-empty strings")
        words[str(name)] = list(entries)
    return words


def fillers_from(lexicon: dict[str, list[str]]) -> frozenset[str]:
    """The filler sounds: the lexicon's ``fillers`` list or the built-in defaults (an empty list disables them)."""
    return frozenset(lexicon["fillers"]) if "fillers" in lexicon else frozenset(DEFAULT_FILLERS)
