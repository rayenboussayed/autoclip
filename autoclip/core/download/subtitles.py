"""Subtitle discovery and SRT conversion (yt-dlp's dfxp converter is injected)."""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..config import DownloadCfg
from ..errors import DownloadError
from ..io import srt_time as _srt_time
from ..media import external_env, ffmpeg_exe
from .naming import _ext_from_url

log = logging.getLogger("autoclip")

SUB_EXTS = ("srt", "vtt", "ttml", "dfxp", "tt", "ass", "ssa", "lrc", "json3", "srv1", "srv2", "srv3", "mhtml")
#: Subtitle formats autoclip cannot turn into an SRT.
NON_CONVERTIBLE = frozenset({"json", "json3", "srv1", "srv2", "srv3", "mhtml"})


#: Keys a previous download pass left on an info dict; a fresh pass must not see them.
def _srt_text(cues: list[dict[str, Any]]) -> str:
    """Cues back to SRT text (normalized: no rolling duplicates or overlaps)."""
    from ..transcript import normalize_cues

    blocks = [
        f"{index}\n{_srt_time(float(cue['start']))} --> {_srt_time(float(cue['end']))}\n{str(cue['text']).strip()}\n"
        for index, cue in enumerate(normalize_cues(cues), 1)
    ]
    return "\n".join(blocks)


def _shift_srt(text: str, rng: tuple[float, float], duration: float) -> str:
    """Move the cues of a range download back to zero and drop the ones outside the range."""
    from ..transcript import parse_srt

    low, _ = rng
    cues: list[dict[str, Any]] = []
    for cue in parse_srt(text):
        start = float(cue["start"]) - low
        end = float(cue["end"]) - low
        if end <= 0 or start >= duration:
            continue
        cues.append({"start": max(start, 0.0), "end": min(end, duration), "text": cue["text"]})
    if not cues:
        raise DownloadError("the downloaded subtitles do not cover the requested time range")
    return _srt_text(cues)


def _write_srt(source: Path, target: Path, rng: tuple[float, float] | None, dfxp: Callable[[bytes], str]) -> None:
    """Convert one subtitle file of any format to SRT (and shift it when a range was cut)."""
    from ..transcript import parse_srt

    ext = source.suffix.lower().lstrip(".")
    temp = target.with_name(f"{target.name}.tmp.srt")
    try:
        if ext == "srt":
            text = source.read_text(encoding="utf-8", errors="replace")
        elif ext in ("ttml", "dfxp", "tt"):
            text = dfxp(source.read_bytes())
        else:
            result = subprocess.run(
                [ffmpeg_exe(), "-y", "-v", "error", "-i", str(source), "-c:s", "srt", str(temp)],
                capture_output=True,
                text=True,
                check=False,
                env=external_env(),
            )
            if result.returncode != 0 or not temp.is_file():
                raise DownloadError(f"ffmpeg cannot convert the {ext} subtitles: {result.stderr.strip()[:200]}")
            text = temp.read_text(encoding="utf-8", errors="replace")
        if rng is not None:
            text = _shift_srt(text, rng, rng[1] - rng[0])
        cues = parse_srt(text)
        if not cues:
            raise DownloadError("the subtitle file is empty")
        target.write_text(_srt_text(cues), encoding="utf-8")
    finally:
        temp.unlink(missing_ok=True)


def _subtitle_file(directory: Path, prefix: str, key: str) -> Path | None:
    """The raw subtitle file yt-dlp saved for one requested key."""
    for ext in SUB_EXTS:
        path = directory / f"{prefix}.subtitle.{key}.{ext}"
        if path.is_file():
            return path
    return None


def _valid_srt(path: Path) -> bool:
    """Whether a file is a readable, non-empty SRT."""
    from ..transcript import parse_srt

    try:
        return bool(parse_srt(path.read_text(encoding="utf-8", errors="replace")))
    except OSError:
        return False


def _subtitle_languages(info: dict[str, Any], settings: DownloadCfg) -> list[str]:
    """Languages to try for a downloaded subtitle set; derived when none are configured."""
    candidates: list[str] = []
    for key in ("language", "original_language"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip().lower())
    candidates.append("en")
    return list(dict.fromkeys(candidates))


def _pick_variant(variants: Any) -> dict[str, Any] | None:
    """The best subtitle variant autoclip can convert (later entries are better in yt-dlp)."""
    if not isinstance(variants, list):
        return None
    for raw in reversed(variants):
        variant = raw if isinstance(raw, dict) else {"url": str(raw)}
        if variant.get("has_drm"):
            continue
        kind = str(variant.get("kind") or "").lower()
        if kind and kind not in ("subtitles", "captions"):
            continue
        ext = str(variant.get("ext") or "").lower().lstrip(".")
        if not ext:
            ext = _ext_from_url(str(variant.get("url") or ""))
        if not ext or ext in NON_CONVERTIBLE:
            continue
        return {**variant, "ext": ext}
    return None


def _pick_sub(store: Any, language: str) -> tuple[str, dict[str, Any]] | None:
    """The best convertible track for ``language`` in one subtitle store (exact key first)."""
    if not isinstance(store, dict):
        return None
    wanted = language.strip().lower()
    if not wanted:
        return None
    keys = [key for key, variants in store.items() if isinstance(variants, list)]
    exact = [key for key in keys if key.lower() == wanted]
    prefixed = [key for key in keys if key.lower().startswith((f"{wanted}-", f"{wanted}_"))]
    for key in (*exact, *prefixed):
        variant = _pick_variant(store[key])
        if variant is not None:
            return key, variant
    return None
