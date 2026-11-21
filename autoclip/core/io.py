"""Small, dependency-free I/O and naming helpers shared across the engine.

These used to be copy-pasted in the downloader, the assembly, the runner and the front ends.
Keeping one definition means the SRT timestamps, the originals manifest and the batch name
deduplication cannot drift apart. They stay private to the engine: nothing here is user-facing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    """Read a small JSON object, raising on any failure (the strict downloader contract)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def srt_time(seconds: float) -> str:
    """Seconds -> SubRip ``HH:MM:SS,mmm``."""
    millis = round(max(seconds, 0.0) * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def unique_name(name: str, used: set[str]) -> str:
    """``name`` unless ``used`` already holds it; then ``name (2)``, ``name (3)`` ... (case-insensitive).

    ``used`` is updated with the chosen name so a batch never reuses one.
    """
    candidate, n = name, 2
    while candidate.casefold() in used:
        candidate = f"{name} ({n})"
        n += 1
    used.add(candidate.casefold())
    return candidate
