"""Logging helpers for runtime decisions derived internally in autoclip."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Protocol

from .runtime import RuntimeDecision, runtime_decisions

log = logging.getLogger("autoclip")


def decision_text(decisions: list[RuntimeDecision]) -> str:
    """A stable, JSON-safe diagnostic representation."""
    return json.dumps(
        [{"name": d.name, "value": d.value, "source": d.source} for d in decisions],
        sort_keys=True,
        separators=(",", ":"),
    )


if TYPE_CHECKING:
    pass  # kept for import symmetry; the protocol below is what the callers must satisfy


class _HasQualityAndMode(Protocol):
    @property
    def quality(self) -> str: ...
    @property
    def mode(self) -> str: ...


def log_decisions(cfg: _HasQualityAndMode | None = None) -> None:
    """Record every auto-derived value in the run log.

    With ``cfg`` the recorded values match what this run will actually use; without it they are the high/balance/CPU defaults.
    """
    quality, mode, device = ("high", "balance", "cpu")
    if cfg is not None:
        quality, mode = cfg.quality, cfg.mode
    for decision in runtime_decisions(quality, mode, device):
        log.info(
            "automation %s=%s | source=%s",
            decision.name,
            decision.value,
            decision.source,
        )


__all__ = ["decision_text", "log_decisions"]
