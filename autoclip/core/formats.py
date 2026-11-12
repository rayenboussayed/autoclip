"""Choosing a concrete yt-dlp format id from the formats of an extracted URL.

The policy is deliberately separate from yt-dlp's own sorting: autoclip wants a *short side*
quality (a 1080x1920 vertical video is "1080"), and it has to return an explicit
``<video>+<audio>`` id because yt-dlp does not merge a callable selector.

The quality target is derived from the run's quality tier (``internal.quality.download_quality_for``);
the only user preference left is the audio language.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

from .errors import DownloadError


class YtFormat(TypedDict, total=False):
    """One entry of yt-dlp's ``formats`` list (every field is optional)."""

    format_id: str | int | None
    vcodec: str | None
    acodec: str | None
    width: float | int | None
    height: float | int | None
    fps: float | int | None
    tbr: float | int | None
    abr: float | int | None
    protocol: str | None
    ext: str | None
    filesize: int | None
    language: str | None
    has_drm: bool


class YtInfo(TypedDict, total=False):
    """The ``extract_info(..., download=False)`` result, as far as autoclip reads it."""

    formats: list[YtFormat]
    language: str | None


#: Video codec -> preference rank (h264 merges and decodes everywhere, h265 least of all).
CODEC_RANK = {"h264": 0, "vp9": 1, "av1": 2, "h265": 3}
#: Prefixes yt-dlp uses for each codec family.
CODEC_PATTERNS = {
    "h264": ("avc", "h264"),
    "h265": ("hev", "hvc", "h265", "hevc"),
    "vp9": ("vp9", "vp09"),
    "av1": ("av01", "av1"),
}


@dataclass(frozen=True, slots=True)
class DownloadPrefs:
    """What the format chooser needs to know about the run."""

    quality: int  # target short side in pixels
    audio_language: str | None = None  # None = the video's original language


@dataclass(frozen=True, slots=True)
class Choice:
    """The format id to download and warnings about preferences that could not be met."""

    format_id: str
    warnings: tuple[str, ...] = ()
    has_video: bool = True  # False when the URL only offers audio (the pipeline needs a video stream)


def _num(value: object) -> float:
    """A numeric field of a format dict (missing or malformed -> 0)."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _short_side(fmt: YtFormat) -> float:
    """The smaller dimension (the "quality" of a vertical and a horizontal video alike)."""
    width, height = _num(fmt.get("width")), _num(fmt.get("height"))
    if width and height:
        return min(width, height)
    return height or width


def _codec(fmt: YtFormat) -> str:
    """The codec family of a format (``other`` when yt-dlp reports something unusual)."""
    vcodec = str(fmt.get("vcodec") or "").lower()
    for name, prefixes in CODEC_PATTERNS.items():
        if vcodec.startswith(prefixes):
            return name
    return "other"


def _video_formats(formats: list[YtFormat]) -> list[YtFormat]:
    """Formats that contain a usable video stream.

    ``vcodec`` is ``None`` (not the string ``"none"``) for a direct media URL the generic
    extractor did not inspect; such a format counts as video because it is often the only thing
    on offer.  It loses to any format with known dimensions, so it is never chosen over a real one.
    """
    return [
        f for f in formats if str(f.get("vcodec") or "").lower() != "none" and not f.get("has_drm") and str(f.get("ext") or "") != "mhtml"
    ]


def _audio_formats(formats: list[YtFormat]) -> list[YtFormat]:
    """Audio-only formats (a progressive format is handled as a video format)."""
    return [
        f
        for f in formats
        if str(f.get("acodec") or "none").lower() != "none"
        and str(f.get("vcodec") or "none").lower() == "none"
        and not f.get("has_drm")
        and str(f.get("ext") or "") != "mhtml"
    ]


def _quality_key(fmt: YtFormat, target: int) -> tuple[float, ...]:
    """Sort key for the quality preference (smaller is better): closest, then the lower size."""
    quality = _short_side(fmt)
    if quality <= 0:  # unknown dimensions always lose to a known format
        return (1, 0.0, 0.0)
    return (0, abs(quality - target), quality)


def _protocol_rank(fmt: YtFormat) -> int:
    """0 for a direct download, 1 for a segmented one (HLS/DASH).

    YouTube hands out an m3u8 variant next to the direct one for the same resolution. The segmented
    variant reports an inflated bitrate (the fragments' peak) and throttles badly on a range download,
    so a direct URL wins whenever the quality is equal.
    """
    protocol = str(fmt.get("protocol") or "").lower()
    return 1 if any(part in protocol for part in ("m3u8", "dash", "frag")) else 0


def _audio_rank(fmt: YtFormat) -> tuple[float, str]:
    """Sort key for an audio format (smaller is better): the highest bitrate wins."""
    bitrate = _num(fmt.get("abr") or fmt.get("tbr"))
    return (-bitrate, str(fmt.get("format_id") or ""))


def _language_matches(fmt: YtFormat, language: str) -> bool:
    """Whether an audio format is in ``language`` (``en`` matches ``en-US``)."""
    value = str(fmt.get("language") or "").lower()
    return bool(value) and (value == language or value.startswith((language + "-", language + "_")))


def _audio_ranked(formats: list[YtFormat], prefs: DownloadPrefs, language: str) -> tuple[list[YtFormat], list[str]]:
    """The audio-only formats best first (after the language filter), plus any warnings.

    ``language`` is the original audio language of the video (``info["language"]``). It is only a
    fallback for when ``audio_language`` is not set: a dubbed track must never win merely because
    it has a slightly higher bitrate.
    """
    audios = _audio_formats(formats)
    if not audios:
        return [], []
    warnings: list[str] = []
    wanted = (prefs.audio_language or "").strip().lower()
    preferred = wanted or language
    if preferred:
        matching = [f for f in audios if _language_matches(f, preferred)]
        if matching:
            audios = matching
        elif wanted:
            warnings.append(f"no audio track in language {prefs.audio_language!r}; using the default")
    return sorted(audios, key=_audio_rank), warnings


def _pick_audio(formats: list[YtFormat], prefs: DownloadPrefs, language: str) -> tuple[YtFormat | None, list[str]]:
    """The best audio-only format, preferring the requested or original language."""
    ranked, warnings = _audio_ranked(formats, prefs, language)
    return (ranked[0] if ranked else None), warnings


def choose(info: YtInfo, prefs: DownloadPrefs) -> Choice:
    """Pick the format id for one extracted URL.

    Args:
        info: the ``extract_info(..., download=False)`` result (its ``formats`` list is used).
        prefs: the derived quality target and the requested audio language.

    Raises:
        DownloadError: if there is nothing downloadable.
    """
    return choose_candidates(info, prefs)[0]


def choose_candidates(info: YtInfo, prefs: DownloadPrefs) -> list[Choice]:
    """Every usable candidate, best first (the first is what :func:`choose` returns).

    A network stall on the media stream is the failure this list exists for: a downloader
    that cannot fetch one format can simply try the next instead of giving up on the video.
    Candidates are ordered by the same preference as :func:`choose` (quality, direct
    downloads, codec, least bitrate), so the alternates are also quality-sensible. A URL
    that offers only audio produces one audio-only candidate.

    Raises:
        DownloadError: if there is nothing downloadable.
    """
    formats = [f for f in info.get("formats") or [] if isinstance(f, dict)]
    videos = _video_formats(formats)
    language = str(info.get("language") or "").strip().lower()
    if not videos:
        audio, audio_warnings = _pick_audio(formats, prefs, language)
        if audio is None:
            raise DownloadError("the URL offers no downloadable video or audio format")
        return [Choice(str(audio.get("format_id") or "best"), tuple(audio_warnings), has_video=False)]
    ordered = sorted(
        videos,
        key=lambda f: (
            *_quality_key(f, prefs.quality),
            _protocol_rank(f),
            CODEC_RANK.get(_codec(f), 4),
            -_num(f.get("fps")) if _num(f.get("fps")) > 0 else math.inf,  # a higher frame rate wins; unknown fps loses
            _num(f.get("tbr")) if _num(f.get("tbr")) > 0 else math.inf,  # least-burdensome bitrate; unknown bitrate loses
            str(f.get("format_id") or ""),
        ),
    )
    audios, audio_warnings = _audio_ranked(formats, prefs, language)
    candidates: list[Choice] = []
    audio_slot = 0
    for video in ordered:
        format_id = str(video.get("format_id") or "best")
        if str(video.get("acodec") or "").lower() != "none":
            candidates.append(Choice(format_id, tuple(audio_warnings)))
            continue
        if not audios:
            candidates.append(Choice(format_id, tuple([*audio_warnings, "the chosen video format has no audio track"])))
            continue
        # A stall can hit the audio stream too (see the SABR ``140-N`` suffixes): give every video-only
        # candidate the next-best audio id instead of pairing all of them with the one that failed.
        audio = audios[min(audio_slot, len(audios) - 1)]
        audio_slot += 1
        candidates.append(Choice(f"{format_id}+{audio.get('format_id')}", tuple(audio_warnings)))
    return candidates


def describe(info: YtInfo, choice: Choice) -> str:
    """A short human-readable summary of a choice (for the log line when a download starts)."""
    by_id = {str(f.get("format_id") or ""): f for f in info.get("formats") or [] if isinstance(f, dict)}
    video, *audio = choice.format_id.split("+")
    parts: list[str] = []
    chosen = by_id.get(video) if choice.has_video else None
    if chosen is not None:
        codec = _codec(chosen)
        width, height = int(_num(chosen.get("width"))), int(_num(chosen.get("height")))
        protocol = chosen.get("protocol") or "unknown"
        parts.append(
            f"{width}x{height} {codec if codec != 'other' else '?'} "
            f"{_num(chosen.get('fps')):.0f}fps {_num(chosen.get('tbr')):.0f} kbps ({protocol})"
        )
    else:
        parts.append(f"audio {video}")
    if audio:
        parts.append(f"audio {'+'.join(audio)}")
    return " + ".join(parts)
