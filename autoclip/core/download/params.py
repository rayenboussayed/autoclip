"""yt-dlp logger and error classification shared by the downloader."""

from __future__ import annotations

import logging

log = logging.getLogger("autoclip")


class _YdlLogger:
    """Route yt-dlp's messages to the autoclip logger (never to stdout/stderr)."""

    def debug(self, msg: str) -> None:
        """yt-dlp's ``[debug]`` lines and its ``info`` messages."""
        log.debug("yt-dlp: %s", msg[8:] if msg.startswith("[debug] ") else msg)

    def info(self, msg: str) -> None:
        """Informational messages (yt-dlp sends them to ``debug`` when a logger is set)."""
        log.debug("yt-dlp: %s", msg)

    def warning(self, msg: str) -> None:
        """A recoverable problem (a missing optional format, a slow site ...)."""
        log.warning("yt-dlp: %s", msg)

    def error(self, msg: str) -> None:
        """The reason a download failed; it is re-raised with the exception that ends the pass."""
        log.debug("yt-dlp: %s", msg)


def _clean_error(message: str) -> str:
    """The message yt-dlp raised without its ``ERROR:`` prefix."""
    text = message.strip()
    while text.startswith("ERROR:"):
        text = text[len("ERROR:") :].strip()
    return text


#: Lower-case fragments yt-dlp and its HTTP stack use when they are told to slow down.
_RATE_LIMIT_MARKERS = (
    "http 429",
    "http error 429",
    "too many requests",
    "rate limit",
    "ratelimit",
    "temporarily blocked",
    "try again later",
)


def _rate_limited(exc: BaseException) -> bool:
    """Whether a yt-dlp failure means "you are being rate limited" (walks the exception chain)."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = getattr(current, "code", None) or getattr(current, "status", None)
        if code == 429:
            return True
        text = str(current).lower()
        if any(marker in text for marker in _RATE_LIMIT_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False
