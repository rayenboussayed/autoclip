"""Clean a yt-dlp info dict before it is used or cached."""

from __future__ import annotations

import copy
import logging
from typing import Any

log = logging.getLogger("autoclip")

STALE_KEYS = (
    "requested_downloads",
    "requested_formats",
    "requested_subtitles",
    "filepath",
    "_filename",
    "__files_to_move",
    "__finaldir",
    "section_start",
    "section_end",
)


def _clean_info(full: dict[str, Any]) -> dict[str, Any]:
    """A deep copy of an extracted info dict without the bookkeeping of a previous pass."""
    info = copy.deepcopy(full)
    for key in STALE_KEYS:
        info.pop(key, None)
    return info
