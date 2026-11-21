"""Human-readable format reports from yt-dlp info dicts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("autoclip")


@dataclass(frozen=True, slots=True)
class FormatRow:
    """One line of a format report."""

    format_id: str
    ext: str
    quality: str
    codec: str
    fps: str
    bitrate: str
    notes: str


@dataclass(frozen=True, slots=True)
class FormatReport:
    """What ``--formats`` shows for one URL."""

    url: str
    title: str
    platform: str
    chosen: str
    rows: tuple[FormatRow, ...]
    warnings: tuple[str, ...]


def _cell(value: str, width: int) -> str:
    """One table cell cut to ``width`` so a long value never shifts the columns after it."""
    text = value if len(value) <= width else value[: width - 1] + "…"
    return f"{text:<{width}}"


def format_report_text(report: FormatReport) -> str:
    """A readable table of the formats of one URL (``--formats``)."""
    lines = [
        f"title:    {report.title or '-'}",
        f"url:      {report.url}",
        f"platform: {report.platform}",
        f"chosen:   {report.chosen}",
        "",
        f"{_cell('ID', 16)}{_cell('EXT', 7)}{_cell('QUALITY', 12)}{_cell('CODEC', 12)}{_cell('FPS', 6)}{_cell('BITRATE', 10)}NOTES",
    ]
    lines.extend(
        f"{_cell(row.format_id, 16)}{_cell(row.ext, 7)}{_cell(row.quality, 12)}{_cell(row.codec, 12)}"
        f"{_cell(row.fps, 6)}{_cell(row.bitrate, 10)}{row.notes}".rstrip()
        for row in report.rows
    )
    if report.warnings:
        lines.append("")
        lines.extend(f"! {warning}" for warning in report.warnings)
    return "\n".join(lines)


def _row(fmt: dict[str, Any]) -> FormatRow:
    """One format as a report line."""
    vcodec = str(fmt.get("vcodec") or "")
    acodec = str(fmt.get("acodec") or "")
    width, height = fmt.get("width"), fmt.get("height")
    if vcodec == "none":
        quality = "audio"
    elif width and height:
        quality = f"{width}x{height}"
    else:
        quality = "?"
    fps = fmt.get("fps")
    bitrate = fmt.get("tbr") or fmt.get("abr")
    notes: list[str] = []
    if fmt.get("has_drm"):
        notes.append("DRM")
    if vcodec == "none":
        notes.append("audio only")
    elif acodec == "none":
        notes.append("video only")
    if str(fmt.get("ext") or "") == "mhtml":
        notes.append("storyboard")
    return FormatRow(
        format_id=str(fmt.get("format_id") or "?"),
        ext=str(fmt.get("ext") or "?"),
        quality=quality,
        codec=vcodec if vcodec and vcodec != "none" else acodec or "?",
        fps=f"{fps:g}" if isinstance(fps, (int, float)) else "-",
        bitrate=f"{bitrate:g}k" if isinstance(bitrate, (int, float)) else "-",
        notes=", ".join(notes),
    )
