"""Choosing a concrete yt-dlp format id from the formats of an extracted URL.

The policy is deliberately separate from yt-dlp's own sorting: autoclip wants a *short side*
quality (a 1080x1920 vertical video is "1080"), a codec preference and a container, and it has
to return an explicit ``<video>+<audio>`` id because yt-dlp does not merge a callable selector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import DownloadCfg
from .errors import DownloadError

#: Video codec -> preference rank (the ``auto`` order).
CODEC_RANK = {"h264": 0, "vp9": 1, "av1": 2, "h265": 3}
#: Prefixes yt-dlp uses for each codec family.
CODEC_PATTERNS = {
    "h264": ("avc", "h264"),
    "h265": ("hev", "hvc", "h265", "hevc"),
    "vp9": ("vp9", "vp09"),
    "av1": ("av01", "av1"),
}
#: Extensions that belong to a requested container.
CONTAINER_EXT = {"mp4": ("mp4", "m4v"), "webm": ("webm",), "mkv": (), "auto": ()}


@dataclass(frozen=True, slots=True)
class Choice:
    """The format id to download and warnings about preferences that could not be met."""

    format_id: str
    warnings: tuple[str, ...] = ()
    has_video: bool = True  # False when the URL only offers audio (the pipeline needs a video stream)


def _num(value: Any) -> float:
    """A numeric field of a format dict (missing or malformed -> 0)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _short_side(fmt: dict[str, Any]) -> float:
    """The smaller dimension (the "quality" of a vertical and a horizontal video alike)."""
    width, height = _num(fmt.get("width")), _num(fmt.get("height"))
    if width and height:
        return min(width, height)
    return height or width


def _codec(fmt: dict[str, Any]) -> str:
    """The codec family of a format (``other`` when yt-dlp reports something unusual)."""
    vcodec = str(fmt.get("vcodec") or "").lower()
    for name, prefixes in CODEC_PATTERNS.items():
        if vcodec.startswith(prefixes):
            return name
    return "other"


def _video_formats(formats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Formats that contain a usable video stream.

    ``vcodec`` is ``None`` (not the string ``"none"``) for a direct media URL the generic
    extractor did not inspect; such a format counts as video because it is often the only thing
    on offer.  It loses to any format with known dimensions, so it is never chosen over a real one.
    """
    return [
        f for f in formats if str(f.get("vcodec") or "").lower() != "none" and not f.get("has_drm") and str(f.get("ext") or "") != "mhtml"
    ]


def _audio_formats(formats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Audio-only formats (a progressive format is handled as a video format)."""
    return [
        f
        for f in formats
        if str(f.get("acodec") or "none").lower() != "none"
        and str(f.get("vcodec") or "none").lower() == "none"
        and not f.get("has_drm")
        and str(f.get("ext") or "") != "mhtml"
    ]


def _quality_key(fmt: dict[str, Any], cfg: DownloadCfg) -> tuple[float, ...]:
    """Sort key for the quality preference (smaller is better)."""
    quality = _short_side(fmt)
    target = cfg.quality
    if quality <= 0:  # unknown dimensions always lose to a known format
        return (2, 0)
    if target == "best":
        return (0, -quality)
    if target == "worst":
        return (0, quality)
    wanted = float(target)
    if cfg.prefer == "at_most":
        return (0, -quality) if quality <= wanted else (1, quality)
    if cfg.prefer == "at_least":
        return (0, quality) if quality >= wanted else (1, -quality)
    return (0, abs(quality - wanted), quality)  # closest; equal distances prefer the lower quality


def _fps_key(fmt: dict[str, Any], cfg: DownloadCfg) -> tuple[float, ...]:
    """Sort key for the fps preference (smaller is better; unknown fps loses)."""
    if cfg.fps is None:
        if cfg.quality == "best":  # best quality also means the smoothest of two otherwise equal formats
            fps = _num(fmt.get("fps"))
            return (0, -fps) if fps else (1, 0)
        return (0,)
    fps = _num(fmt.get("fps"))
    if not fps:
        return (2, 0)
    wanted = cfg.fps
    if cfg.prefer == "at_most":
        return (0, -fps) if fps <= wanted else (1, fps)
    if cfg.prefer == "at_least":
        return (0, fps) if fps >= wanted else (1, -fps)
    return (0, abs(fps - wanted), fps)


def _container_rank(fmt: dict[str, Any], container: str) -> int:
    """0 when the format already has the requested container, 1 otherwise."""
    if container in ("auto", "mkv"):
        return 0
    return 0 if str(fmt.get("ext") or "").lower() in CONTAINER_EXT[container] else 1


def _audio_rank(fmt: dict[str, Any], cfg: DownloadCfg) -> tuple[float, int, str]:
    """Sort key for an audio format (smaller is better)."""
    bitrate = _num(fmt.get("abr") or fmt.get("tbr"))
    container = cfg.container
    ext = str(fmt.get("ext") or "").lower()
    if container == "mp4":
        container_rank = 0 if ext in ("m4a", "mp4") else 1
    elif container == "webm":
        container_rank = 0 if ext in ("webm", "opus") else 1
    else:
        container_rank = 0
    return (-bitrate if cfg.audio_quality == "best" else bitrate, container_rank, str(fmt.get("format_id") or ""))


def _language_matches(fmt: dict[str, Any], language: str) -> bool:
    """Whether an audio format is in ``language`` (``en`` matches ``en-US``)."""
    value = str(fmt.get("language") or "").lower()
    return bool(value) and (value == language or value.startswith((language + "-", language + "_")))


def _pick_audio(formats: list[dict[str, Any]], cfg: DownloadCfg) -> tuple[dict[str, Any] | None, list[str]]:
    """The best (or worst) audio-only format, preferring the requested language."""
    audios = _audio_formats(formats)
    if not audios:
        return None, []
    warnings: list[str] = []
    language = (cfg.audio_language or "").strip().lower()
    if language:
        matching = [f for f in audios if _language_matches(f, language)]
        if matching:
            audios = matching
        else:
            warnings.append(f"no audio track in language {cfg.audio_language!r}; using the default")
    return min(audios, key=lambda f: _audio_rank(f, cfg)), warnings


def _quality_warnings(fmt: dict[str, Any], cfg: DownloadCfg) -> list[str]:
    """Warn when an at_most/at_least quality preference had to be broken."""
    if cfg.quality in ("best", "worst"):
        return []
    quality = _short_side(fmt)
    wanted = float(cfg.quality)
    shown = f"{quality:.0f}p" if quality else "an unknown resolution"
    if cfg.prefer == "at_most" and quality > wanted:
        return [f"no format at or below {wanted:.0f}p; using {shown}"]
    if cfg.prefer == "at_least" and quality < wanted:
        return [f"no format at or above {wanted:.0f}p; using {shown}"]
    return []


def _fps_warnings(fmt: dict[str, Any], cfg: DownloadCfg) -> list[str]:
    """Warn when an at_most/at_least fps preference had to be broken."""
    if cfg.fps is None:
        return []
    fps = _num(fmt.get("fps"))
    if not fps:
        return ["the chosen format has no fps information"]
    if cfg.prefer == "at_most" and fps > cfg.fps:
        return [f"no format at or below {cfg.fps:g} fps; using {fps:g} fps"]
    if cfg.prefer == "at_least" and fps < cfg.fps:
        return [f"no format at or above {cfg.fps:g} fps; using {fps:g} fps"]
    return []


def choose(info: dict[str, Any], cfg: DownloadCfg) -> Choice:
    """Pick the format id for one extracted URL.

    Args:
        info: the ``extract_info(..., download=False)`` result (its ``formats`` list is used).
        cfg: the effective download settings for the platform.

    Raises:
        DownloadError: if there is nothing downloadable or the requested codec is absent.
    """
    formats = [f for f in info.get("formats") or [] if isinstance(f, dict)]
    videos = _video_formats(formats)
    warnings: list[str] = []
    if not videos:
        audio, audio_warnings = _pick_audio(formats, cfg)
        if audio is None:
            raise DownloadError("the URL offers no downloadable video or audio format")
        return Choice(str(audio.get("format_id") or "best"), tuple(audio_warnings), has_video=False)
    if cfg.codec != "auto":
        videos = [f for f in videos if _codec(f) == cfg.codec]
        if not videos:
            available = sorted({_codec(f) for f in _video_formats(formats)})
            raise DownloadError(f"no {cfg.codec} video format; available: {', '.join(available) or 'none'}")
    best = min(
        videos,
        key=lambda f: (
            *_quality_key(f, cfg),
            *_fps_key(f, cfg),
            CODEC_RANK.get(_codec(f), 4) if cfg.codec == "auto" else 0,
            _container_rank(f, cfg.container),
            -_num(f.get("tbr")),
            str(f.get("format_id") or ""),
        ),
    )
    warnings.extend(_quality_warnings(best, cfg))
    warnings.extend(_fps_warnings(best, cfg))
    format_id = str(best.get("format_id") or "best")
    if str(best.get("acodec") or "").lower() == "none":
        audio, audio_warnings = _pick_audio(formats, cfg)
        warnings.extend(audio_warnings)
        if audio is None:
            warnings.append("the chosen video format has no audio track")
        else:
            format_id = f"{format_id}+{audio.get('format_id')}"
    return Choice(format_id, tuple(warnings))
