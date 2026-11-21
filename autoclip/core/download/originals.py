"""Find and validate previously downloaded originals in the zero-network cache."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ..errors import AutoclipError
from ..media import probe

log = logging.getLogger("autoclip")

MEDIA_EXTS = frozenset(
    {
        "3gp",
        "avi",
        "flv",
        "m4a",
        "m4v",
        "mkv",
        "mov",
        "mp3",
        "mp4",
        "mpeg",
        "mpg",
        "ogg",
        "ogv",
        "opus",
        "ts",
        "wav",
        "webm",
        "wmv",
    }
)


#: Subtitle formats, most convenient first.
def _media_in(directory: Path, prefix: str) -> Path | None:
    """The media file of ``prefix`` in ``directory`` (final, merged or a single stream)."""
    for ext in sorted(MEDIA_EXTS):
        exact = directory / f"{prefix}.{ext}"
        if exact.is_file():
            return exact
    for ext in sorted(MEDIA_EXTS):
        merged = directory / f"{prefix}.media.{ext}"
        if merged.is_file():
            return merged
    streams = [
        item
        for item in directory.iterdir()
        if item.is_file() and item.name.startswith(f"{prefix}.media.f") and item.suffix.lower().lstrip(".") in MEDIA_EXTS
    ]
    return streams[0] if len(streams) == 1 else None


def _canonical_media(directory: Path, prefix: str, media: Path) -> Path:
    """Rename a merged or per-stream media file to ``<prefix>.<ext>`` (the cache's final name)."""
    target = directory / f"{prefix}{media.suffix.lower()}"
    if media == target:
        return media
    try:
        media.replace(target)
    except OSError:
        return media
    return target


def _range_ok(media: Path, rng: tuple[float, float]) -> bool:
    """Whether ``media`` is close enough to the requested range (within 3 s or 10%)."""
    expected = rng[1] - rng[0]
    try:
        actual = probe(media).duration
    except AutoclipError:
        return False
    except Exception:
        return False  # an unreadable cache entry must be treated as a miss, never trusted
    return abs(actual - expected) <= max(3.0, 0.1 * expected)


def _exact_media(directory: Path, prefix: str, rng: tuple[float, float] | None) -> Path | None:
    """The cache file of ``prefix`` (exact ``<prefix>.<ext>`` only), or ``None`` when it is absent or unusable.

    Streams and merges are deliberately not matched here: the caller re-validates a reusable file (a
    partial write would otherwise be re-served forever), and a wrong length under a time-range prefix is
    deleted instead of reused.
    """
    for ext in sorted(MEDIA_EXTS):
        exact = directory / f"{prefix}.{ext}"
        if not exact.is_file():
            continue
        if rng is not None and not _range_ok(exact, rng):
            exact.unlink(missing_ok=True)
            return None
        return exact
    return None


def _empty(directory: Path) -> None:
    """Delete the contents of a temporary directory (best effort)."""
    for item in directory.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
        except OSError:
            log.debug("could not remove %s", item, exc_info=True)
