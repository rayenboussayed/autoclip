"""Logging helpers for runtime decisions derived internally in autoclip."""

from __future__ import annotations

import json
import logging

from .runtime import RuntimeDecision, runtime_decisions

log = logging.getLogger("autoclip")


def decision_text(decisions: list[RuntimeDecision]) -> str:
    """A stable, JSON-safe diagnostic representation."""
    return json.dumps(
        [{"name": d.name, "value": d.value, "source": d.source} for d in decisions],
        sort_keys=True,
        separators=(",", ":"),
    )


def log_decisions() -> None:
    """Record every auto-derived value in the run log."""
    for decision in runtime_decisions():
        log.info(
            "automation %s=%s | source=%s",
            decision.name,
            decision.value,
            decision.source,
        )


__all__ = ["decision_text", "log_decisions"]
