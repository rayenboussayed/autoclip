"""Adaptive tuning helpers: samples, persistence and trend detection.

Pure helpers only; the mutable state lives in ``ResourceMonitor``.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from pathlib import Path
from typing import Any

SAMPLE_CAP = 8


def load_state(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def save_state(path: Path, state: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except (OSError, UnboundLocalError):  # pragma: no cover
            logging.getLogger("autoclip").debug("could not remove temporary adaptive state", exc_info=True)  # pragma: no cover


def trend(samples: deque[float]) -> int:
    """-1 falling, 0 flat, +1 rising over the recent samples."""
    if len(samples) < 3:
        return 0
    first = sum(list(samples)[: len(samples) // 2]) / (len(samples) // 2)
    second = sum(list(samples)[len(samples) // 2 :]) / (len(samples) - len(samples) // 2)
    if second - first > 0.05:
        return 1
    if first - second > 0.05:
        return -1
    return 0


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


__all__ = ["SAMPLE_CAP", "clamp", "load_state", "save_state", "trend"]
