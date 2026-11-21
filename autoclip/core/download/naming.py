"""Turn a URL, title and range into the stable originals-cache prefix."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..config import DownloadCfg, parse_hms
from ..formats import Choice

log = logging.getLogger("autoclip")

_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
#: Device names Windows refuses to use as a file name, with or without an extension.
_RESERVED_NAMES = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))})


def _ext_from_url(url: str) -> str:
    """The extension in the path of a subtitle URL (``""`` when there is none)."""
    if not url:
        return ""
    return Path(urlsplit(url).path).suffix.lower().lstrip(".")


#: Path prefixes of a single video (a watch URL often carries a playlist id as well).
_VIDEO_PATHS = ("/watch", "/shorts/", "/live/", "/embed/", "/v/", "/video/", "/clip/")
#: Path prefixes that list several videos.
_PLAYLIST_PATHS = (
    "/playlist",
    "/channel/",
    "/c/",
    "/user/",
    "/@",
    "/album/",
    "/sets/",
    "/show/",
    "/results",
)


def wants_all_entries(url: str) -> bool:
    """Whether a URL names a list of videos (playlist/channel/album/user) rather than one video.

    The shape decides, so a YouTube watch URL that also carries ``list=`` stays a single video,
    while a channel or a playlist page downloads every entry (up to ``PLAYLIST_MAX_VIDEOS``).
    """
    parts = urlsplit(url)
    path = parts.path.lower()
    if any(path.startswith(prefix) for prefix in _PLAYLIST_PATHS):
        return True
    return not any(path.startswith(prefix) for prefix in _VIDEO_PATHS) and path.count("/") > 1


def _entry_url(entry: dict[str, Any]) -> str | None:
    """The URL of a playlist entry (a flat entry may only carry an id and an extractor key).

    yt-dlp normally reports a full URL, but some extractors emit the bare video id.  A bare
    YouTube id is turned back into a watch URL; anything else keeps the raw value, because
    ``process_ie_result`` uses ``ie_key`` to pick the extractor that understands it.
    """
    for key in ("webpage_url", "url", "original_url"):
        value = entry.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    video_id = entry.get("id")
    ie_key = entry.get("ie_key") or entry.get("extractor_key")
    if not isinstance(video_id, str) or not video_id:
        return None
    if isinstance(ie_key, str) and ie_key == "Youtube":
        return f"https://www.youtube.com/watch?v={video_id}"
    raw = entry.get("url")
    return raw if isinstance(raw, str) and raw else None


def _safe_title(text: str, limit: int = 80) -> str:
    """A title that is safe on every file system and inside a yt-dlp output template.

    yt-dlp only sanitises the *values* it interpolates, not the literal template text, so the
    prefix must be clean by itself: no path separator, no Windows-invalid character, no leading
    dot or dash, and at most ``limit`` UTF-8 bytes.
    """
    name = _INVALID_CHARS.sub("_", text)
    name = re.sub(r"\s+", " ", name).strip(" .-")
    if not name:
        name = "video"
    if name.split(".")[0].upper() in _RESERVED_NAMES:
        name = f"_{name}"
    encoded = name.encode("utf-8")
    if len(encoded) > limit:
        name = encoded[:limit].decode("utf-8", "ignore").rstrip(" .-") or "video"
    return name


def _safe_id(text: str) -> str:
    """A video/format id rendered as a file-name fragment (``""`` when there is no id)."""
    if not text.strip():
        return ""
    return _safe_title(text, limit=40)


def _range(settings: DownloadCfg) -> tuple[float, float] | None:
    """The configured time range in seconds, or ``None``."""
    if not settings.time_range:
        return None
    low, high = (parse_hms(part) for part in settings.time_range.split("-"))
    return low, high


def _range_suffix(rng: tuple[float, float] | None) -> str:
    """``" [60-120s]"`` (empty without a range), added to a cache prefix and an output name."""
    if rng is None:
        return ""
    return f" [{rng[0]:g}-{rng[1]:g}s]"


def _prefix(base: str, choice: Choice, suffix: str) -> str:
    """The file-name prefix of one download: title, id, chosen format and range."""
    return f"{base} [{_safe_id(choice.format_id)}]{suffix}"
