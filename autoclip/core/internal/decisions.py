"""Logging for the machine-derived decisions of a run (see ``internal.auto`` for the rules)."""

from __future__ import annotations

import logging
from typing import Protocol

from .runtime import detect_device, runtime_decisions

log = logging.getLogger("autoclip")


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


__all__ = ["log_decisions"]
