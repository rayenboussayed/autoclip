"""Downloading the http(s) URLs of ``input`` with yt-dlp.

The whole flow lives in :class:`Downloader`; this module is the stable import surface for the CLI,
the app and the tests. ``yt_dlp`` is imported inside :meth:`Downloader.__init__`, so a local-only
run never imports it or makes a network call.
"""

from __future__ import annotations

from .core import Downloader, _ChosenSub
from .extraction import STALE_KEYS, _clean_info
from .naming import _entry_url, _ext_from_url, _prefix, _range, _range_suffix, _safe_id, _safe_title, wants_all_entries
from .originals import MEDIA_EXTS, _canonical_media, _empty, _exact_media, _media_in, _range_ok
from .params import _clean_error, _rate_limited, _YdlLogger
from .reporting import FormatReport, FormatRow, _cell, _row, format_report_text
from .subtitles import (
    NON_CONVERTIBLE,
    SUB_EXTS,
    _pick_sub,
    _pick_variant,
    _shift_srt,
    _srt_text,
    _subtitle_file,
    _subtitle_languages,
    _valid_srt,
    _write_srt,
)

__all__ = [
    "MEDIA_EXTS",
    "NON_CONVERTIBLE",
    "STALE_KEYS",
    "SUB_EXTS",
    "Downloader",
    "FormatReport",
    "FormatRow",
    "_ChosenSub",
    "_YdlLogger",
    "_canonical_media",
    "_cell",
    "_clean_error",
    "_clean_info",
    "_empty",
    "_entry_url",
    "_exact_media",
    "_ext_from_url",
    "_media_in",
    "_pick_sub",
    "_pick_variant",
    "_prefix",
    "_range",
    "_range_ok",
    "_range_suffix",
    "_rate_limited",
    "_row",
    "_safe_id",
    "_safe_title",
    "_shift_srt",
    "_srt_text",
    "_subtitle_file",
    "_subtitle_languages",
    "_valid_srt",
    "_write_srt",
    "format_report_text",
    "wants_all_entries",
]
