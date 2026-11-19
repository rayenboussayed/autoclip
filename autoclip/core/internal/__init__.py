"""Derived runtime configuration and helpers (import submodules directly; only the stable re-exports live here)."""

from . import adaptive, modes
from .quality import output_fps, quality_profile, quality_short_side
from .runtime import CACHE_DISK_MB, CHUNK_MINUTES, detect_cpu_count

__all__ = [
    "CACHE_DISK_MB",
    "CHUNK_MINUTES",
    "adaptive",
    "detect_cpu_count",
    "modes",
    "output_fps",
    "quality_profile",
    "quality_short_side",
]
