"""Downloading the http(s) URLs of ``input`` with yt-dlp.

The whole flow lives in :class:`Downloader`: the platform policy, the per-platform settings, the
cache of originals and the yt-dlp parameters, so the command line and the desktop app behave
identically.  ``yt_dlp`` is imported inside :meth:`Downloader.__init__` - a run with only local
files never imports it and never makes a network call.
"""

from __future__ import annotations

import copy
import logging
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .config import Config, DownloadCfg, JsRuntimeCfg, parse, parse_hms
from .errors import AutoclipError, DownloadError
from .formats import Choice, choose
from .log import traced
from .media import ffmpeg_exe, probe, trim_video
from .models import Downloaded
from .platforms import (
    GENERIC,
    allowed,
    canonical,
    check_families,
    effective_download,
    extractor_patterns,
    family_names,
    identify,
)

log = logging.getLogger("autoclip")

#: Extensions of a media file (used to tell media apart from subtitles and thumbnails).
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
#: Extensions of a thumbnail.
IMAGE_EXTS = frozenset({"jpg", "jpeg", "png", "webp"})
#: Subtitle formats, most convenient first.
SUB_EXTS = ("srt", "vtt", "ttml", "dfxp", "tt", "ass", "ssa", "lrc", "json3", "srv1", "srv2", "srv3", "mhtml")
#: Subtitle formats autoclip cannot turn into an SRT.
NON_CONVERTIBLE = frozenset({"json", "json3", "srv1", "srv2", "srv3", "mhtml"})
#: Keys a previous download pass left on an info dict; a fresh pass must not see them.
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

#: Characters Windows does not allow in a file name (and the path separators everywhere).
_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
#: Device names Windows refuses to use as a file name, with or without an extension.
_RESERVED_NAMES = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))})


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


@dataclass(frozen=True, slots=True)
class _ChosenSub:
    """One subtitle track the downloader will ask yt-dlp for."""

    language: str  # the language as written in the config (it names the final file)
    key: str  # the key in yt-dlp's subtitles/automatic_captions dict
    manual: bool
    variant: dict[str, Any]


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


def _clean_error(message: str) -> str:
    """The message yt-dlp raised without its ``ERROR:`` prefix."""
    text = message.strip()
    while text.startswith("ERROR:"):
        text = text[len("ERROR:") :].strip()
    return text


def _ext_from_url(url: str) -> str:
    """The extension in the path of a subtitle URL (``""`` when there is none)."""
    if not url:
        return ""
    return Path(urlsplit(url).path).suffix.lower().lstrip(".")


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


def _prefix(base: str, choice: Choice, settings: DownloadCfg, suffix: str) -> str:
    """The file-name prefix of one download: title, id, chosen format, container and range."""
    text = f"{base} [{_safe_id(choice.format_id)}]"
    if settings.container != "auto":
        text += f" [{settings.container}]"
    return text + suffix


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


def _srt_time(seconds: float) -> str:
    """Seconds -> ``HH:MM:SS,mmm``."""
    millis = round(max(seconds, 0.0) * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _srt_text(cues: list[dict[str, Any]]) -> str:
    """Cues back to SRT text."""
    blocks = [
        f"{index}\n{_srt_time(float(cue['start']))} --> {_srt_time(float(cue['end']))}\n{str(cue['text']).strip()}\n"
        for index, cue in enumerate(cues, 1)
    ]
    return "\n".join(blocks)


def _shift_srt(text: str, rng: tuple[float, float], duration: float) -> str:
    """Move the cues of a range download back to zero and drop the ones outside the range."""
    from .transcript import parse_srt

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
    from .transcript import parse_srt

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
            )
            if result.returncode != 0 or not temp.is_file():
                raise DownloadError(f"ffmpeg cannot convert the {ext} subtitles: {result.stderr.strip()[:200]}")
            text = temp.read_text(encoding="utf-8", errors="replace")
        if rng is not None:
            text = _shift_srt(text, rng, rng[1] - rng[0])
        if not parse_srt(text):
            raise DownloadError("the subtitle file is empty")
        target.write_text(text, encoding="utf-8")
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
    from .transcript import parse_srt

    try:
        return bool(parse_srt(path.read_text(encoding="utf-8", errors="replace")))
    except OSError:
        return False


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


def _clean_info(full: dict[str, Any]) -> dict[str, Any]:
    """A deep copy of an extracted info dict without the bookkeeping of a previous pass."""
    info = copy.deepcopy(full)
    for key in STALE_KEYS:
        info.pop(key, None)
    return info


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


class Downloader:
    """Downloads the http(s) URLs of ``input`` with the platform policy of one config."""

    def __init__(self, cfg: Config) -> None:
        """Import yt-dlp, check the platform names and resolve the policy.

        Raises:
            DownloadError: if yt-dlp is missing.
            ConfigError: if ``download.platforms`` names an unknown platform.
        """
        try:
            import yt_dlp
            from yt_dlp.extractor import get_info_extractor
            from yt_dlp.globals import supported_js_runtimes
            from yt_dlp.utils import YoutubeDLError, dfxp2srt, download_range_func
        except ImportError as e:
            raise DownloadError('downloading URLs needs yt-dlp: pip install "yt-dlp[default]"') from e
        self.cfg = cfg
        self.root = Path(cfg.download.original_dir) if cfg.download.original_dir else Path(cfg.output_dir) / ".work" / "downloads"
        self._YoutubeDL = yt_dlp.YoutubeDL
        self._get_info_extractor = get_info_extractor
        self._runtime_classes = supported_js_runtimes.value
        self._dfxp2srt = dfxp2srt
        self._download_range_func = download_range_func
        self._ytdl_error = YoutubeDLError
        self._logger = _YdlLogger()
        self._settings_cache: dict[str, DownloadCfg] = {}
        self._used: set[tuple[str, str]] = set()
        platforms = cfg.download.platforms
        check_families(platforms.allow, platforms.deny, platforms.groups, platforms.settings)
        known = set(family_names())
        self.permitted_families = {family for family in known if allowed(family, platforms.allow, platforms.deny, platforms.groups)}
        if self.permitted_families >= known:
            self.allowed_extractors = ["default", "end"]
        elif self.permitted_families == known - {GENERIC}:
            self.allowed_extractors = ["default", "-generic", "-html5", "end"]
        else:
            self.allowed_extractors = extractor_patterns(self.permitted_families)
        self.js_runtimes = self._runtime_config(cfg.download.js_runtime)
        self.remote_components = ["ejs:github"] if cfg.download.js_runtime.remote_components == "ejs" else []
        self._range_supported = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))

    # ------------------------------------------------------------------ policy

    def platform(self, url: str) -> str:
        """The canonical extractor family of a URL (offline: no request is made)."""
        return identify(url)

    def permitted(self, family: str) -> bool:
        """Whether the allow/deny policy lets this family be used."""
        return canonical(family) in self.permitted_families

    def describe(self, url: str) -> str:
        """One line saying whether ``url`` may be downloaded and why."""
        family = self.platform(url)
        if self.permitted(family):
            return f"{family}: allowed"
        names = ", ".join(self.cfg.download.platforms.allow) or "nothing"
        return f"{family}: blocked (download.platforms.allow: {names})"

    def policy_text(self) -> str:
        """The allow/deny policy as a few readable lines (the desktop app shows it)."""
        platforms = self.cfg.download.platforms
        lines = [
            f"allow: {', '.join(platforms.allow) or '-'}",
            f"deny: {', '.join(platforms.deny) or '-'}",
        ]
        if platforms.groups:
            lines.append("groups: " + "; ".join(f"{name} = {', '.join(members)}" for name, members in platforms.groups.items()))
        if platforms.settings:
            lines.append("settings: " + ", ".join(platforms.settings))
        lines.append("permitted: " + (", ".join(sorted(self.permitted_families)) or "-"))
        return "\n".join(lines)

    # ----------------------------------------------------------------- formats

    @traced("formats")
    def formats(self, url: str) -> FormatReport:
        """The formats of one URL and the one ``download`` would choose (``--formats``)."""
        family = self._require(url)
        settings = self._settings(family)
        entries = self._entries(url, settings, limit=2)
        if not entries:
            raise DownloadError(f"{url} has no downloadable entries")
        entry = entries[0]
        entry_url = _entry_url(entry)
        entry_family = self._entry_family(entry, family, entry_url)
        if not self.permitted(entry_family):
            raise DownloadError(f"platform '{entry_family}' is blocked by download.platforms")
        effective = settings if entry_family == family else self._settings(entry_family)
        full = self._full_info(entry, entry_url, effective)
        choice = choose(full, effective)
        warnings = list(choice.warnings)
        if len(entries) > 1:
            warnings.append("the URL is a playlist; showing the first entry")
        if not choice.has_video:
            warnings.append("the URL only offers audio; autoclip needs a video track")
        formats = [fmt for fmt in full.get("formats") or [] if isinstance(fmt, dict)]
        formats.sort(
            key=lambda fmt: (
                str(fmt.get("vcodec") or "").lower() == "none",
                -float(fmt.get("height") or 0),
                -float(fmt.get("tbr") or 0),
            )
        )
        return FormatReport(
            url=url,
            title=str(full.get("title") or ""),
            platform=entry_family,
            chosen=choice.format_id,
            rows=tuple(_row(fmt) for fmt in formats),
            warnings=tuple(warnings),
        )

    # ---------------------------------------------------------------- download

    @traced("download")
    def download(self, url: str) -> list[Downloaded]:
        """Download one URL (or every entry of its playlist) into the original cache.

        Returns:
            One :class:`Downloaded` per saved original, with the subtitle to use for it (if any).

        Raises:
            DownloadError: if the platform is blocked, yt-dlp fails, or nothing could be saved.
        """
        family = self._require(url)
        settings = self._settings(family)
        entries = self._entries(url, settings, limit=None)
        if not entries:
            raise DownloadError(f"{url} has no downloadable entries")
        results: list[Downloaded] = []
        for index, entry in enumerate(entries):
            if index and settings.network.sleep_between_seconds:
                time.sleep(settings.network.sleep_between_seconds)
            try:
                results.extend(self._one_entry(entry, family, settings))
            except DownloadError as e:
                if len(entries) == 1:
                    raise
                log.warning("skipping a playlist entry of %s: %s", url, e)
        if not results:
            raise DownloadError(f"nothing could be downloaded from {url}")
        return results

    def _require(self, url: str) -> str:
        """The family of ``url``; raises before any request when the policy blocks it."""
        family = self.platform(url)
        if not self.permitted(family):
            raise DownloadError(self.describe(url))
        return family

    def _one_entry(self, entry: dict[str, Any], parent: str, parent_settings: DownloadCfg) -> list[Downloaded]:
        """Extract, choose, download (or reuse) one playlist entry."""
        url = _entry_url(entry)
        family = self._entry_family(entry, parent, url)
        if not self.permitted(family):
            raise DownloadError(f"platform '{family}' is blocked by download.platforms")
        settings = parent_settings if family == parent else self._settings(family)
        full = self._full_info(entry, url, settings)
        if full.get("_type") in ("playlist", "multi_video") and full.get("entries"):
            return [downloaded for sub in full["entries"] if isinstance(sub, dict) for downloaded in self._one_entry(sub, family, settings)]
        choice = choose(full, settings)
        if not choice.has_video:
            raise DownloadError(f"{full.get('title') or url or 'the URL'} only offers audio; autoclip needs a video track")
        rng = _range(settings)
        name, directory, prefix, cached = self._prepare(full, family, settings, choice, rng)
        if cached is not None:
            media = _canonical_media(directory, prefix, cached)
            if not self._complete(directory, prefix, settings):
                self._sidecars(full, settings, directory, prefix, choice, rng)
        else:
            temporary = Path(tempfile.mkdtemp(prefix=".download-", dir=directory))
            try:
                media = self._media(full, settings, directory, prefix, choice, rng, temporary)
                self._sidecars(full, settings, directory, prefix, choice, rng)
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
        subtitle, manual = self._subtitle_for(directory, prefix, settings)
        log.info("downloaded %s (%s)", name, family)
        return [Downloaded(path=media, name=name, platform=family, subtitle=subtitle, subtitle_manual=manual)]

    def _prepare(
        self, full: dict[str, Any], family: str, settings: DownloadCfg, choice: Choice, rng: tuple[float, float] | None
    ) -> tuple[str, Path, str, Path | None]:
        """The output name, cache directory, file prefix and the cached media (if present)."""
        title = _safe_title(str(full.get("title") or full.get("id") or "video"))
        video_id = _safe_id(str(full.get("id") or ""))
        base = f"{title} [{video_id}]" if video_id else title
        suffix = _range_suffix(rng)
        directory = self._directory(family)
        prefix = _prefix(base, choice, settings, suffix)
        media = _media_in(directory, prefix)
        if media is not None:
            return base + suffix, directory, prefix, media
        key = (str(directory), base + suffix)
        if key in self._used:
            number = 2
            while (str(directory), f"{base} ({number}){suffix}") in self._used:
                number += 1
            base = f"{base} ({number})"
        self._used.add((str(directory), base + suffix))
        return base + suffix, directory, _prefix(base, choice, settings, suffix), None

    def _directory(self, family: str) -> Path:
        """The folder of the originals of one family (``group_by_platform`` adds a sub-folder)."""
        root = self.root / family if self.cfg.download.platforms.group_by_platform else self.root
        root.mkdir(parents=True, exist_ok=True)
        return root

    # ------------------------------------------------------------------ media

    def _media(
        self,
        full: dict[str, Any],
        settings: DownloadCfg,
        directory: Path,
        prefix: str,
        choice: Choice,
        rng: tuple[float, float] | None,
        temporary: Path,
    ) -> Path:
        """Download the media (and cut the configured range out of it)."""
        params = self._media_params(settings, directory, prefix, temporary, choice)
        if rng is not None and self._range_supported:
            ranged = dict(params)
            ranged["download_ranges"] = self._download_range_func(None, [rng])
            ranged["force_keyframes_at_cuts"] = True
            try:
                with self._YoutubeDL(ranged) as ydl:
                    self._call(ydl.process_ie_result, _clean_info(full), download=True)
                media = self._collect(directory, temporary, prefix)
                if media is not None and self._duration_ok(media, rng):
                    return media
                log.warning("the %s download does not have the expected length; downloading the whole video", settings.time_range)
                if media is not None:
                    media.unlink(missing_ok=True)
                _empty(temporary)
            except DownloadError as e:
                log.warning("the %s range download failed (%s); downloading the whole video", settings.time_range, e)
                _empty(temporary)
        with self._YoutubeDL(params) as ydl:
            self._call(ydl.process_ie_result, _clean_info(full), download=True)
        media = self._collect(directory, temporary, prefix)
        if media is None:
            raise DownloadError(f"yt-dlp did not save a video file for {prefix}")
        if rng is None:
            return media
        target = directory / f"{prefix}.mkv"
        try:
            trim_video(media, target, rng[0], rng[1])
        except AutoclipError as e:
            media.unlink(missing_ok=True)
            raise DownloadError(f"cannot cut {settings.time_range} out of the downloaded video: {e}") from e
        media.unlink(missing_ok=True)
        return target

    def _duration_ok(self, media: Path, rng: tuple[float, float]) -> bool:
        """Whether a range download is close enough to the requested length."""
        expected = rng[1] - rng[0]
        try:
            actual = probe(media).duration
        except AutoclipError:
            return False
        return abs(actual - expected) <= max(3.0, 0.1 * expected)

    def _collect(self, directory: Path, temporary: Path, prefix: str) -> Path | None:
        """The downloaded media, moved from the temporary folder into the cache when needed."""
        media = _media_in(directory, prefix)
        if media is not None:
            return _canonical_media(directory, prefix, media)
        media = _media_in(temporary, prefix)
        if media is None:
            return None
        target = directory / f"{prefix}{media.suffix.lower()}"
        try:
            shutil.move(str(media), str(target))
        except OSError as e:
            raise DownloadError(f"cannot move the downloaded video into place: {e}") from e
        return target

    # ---------------------------------------------------------------- sidecars

    def _sidecars(
        self,
        full: dict[str, Any],
        settings: DownloadCfg,
        directory: Path,
        prefix: str,
        choice: Choice,
        rng: tuple[float, float] | None,
    ) -> None:
        """Save the requested subtitles, thumbnail and metadata next to the original."""
        metadata = settings.metadata
        if not (settings.subtitles.enabled or metadata.info_json or metadata.thumbnail or metadata.description):
            return
        plan = self._plan(full, settings) if settings.subtitles.enabled else []
        temporary = Path(tempfile.mkdtemp(prefix=".sidecar-", dir=directory))
        try:
            info = _clean_info(full)
            self._apply_plan(info, plan)
            params = self._sidecar_params(settings, directory, prefix, temporary, choice, plan)
            with self._YoutubeDL(params) as ydl:
                self._call(ydl.process_ie_result, info, download=True)
            self._sweep(temporary, directory, prefix)
        except Exception as e:  # a missing sidecar must never cost the downloaded video
            log.warning("could not save every extra file for %s: %s", prefix, e)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        if plan:
            self._convert_subtitles(directory, prefix, plan, rng)

    def _plan(self, full: dict[str, Any], settings: DownloadCfg) -> list[_ChosenSub]:
        """The subtitle track to request for every configured language."""
        plan: list[_ChosenSub] = []
        for language in settings.subtitles.languages:
            manual = _pick_sub(full.get("subtitles") or {}, language)
            automatic = _pick_sub(full.get("automatic_captions") or {}, language) if manual is None else None
            picked = manual or automatic
            if picked is None:
                log.warning("no usable %s subtitles for %s", language, full.get("title") or full.get("id") or "the URL")
                continue
            plan.append(_ChosenSub(language=language, key=picked[0], manual=manual is not None, variant=picked[1]))
        return plan

    def _apply_plan(self, info: dict[str, Any], plan: list[_ChosenSub]) -> None:
        """Leave only the chosen tracks in the info dict, each with only its convertible variant."""
        manual: dict[str, list[dict[str, Any]]] = {}
        automatic: dict[str, list[dict[str, Any]]] = {}
        for item in plan:
            (manual if item.manual else automatic)[item.key] = [item.variant]
        info["subtitles"] = manual
        info["automatic_captions"] = automatic

    def _convert_subtitles(self, directory: Path, prefix: str, plan: list[_ChosenSub], rng: tuple[float, float] | None) -> None:
        """Turn the downloaded subtitle files into the SRTs the pipeline can read."""
        for item in plan:
            source = _subtitle_file(directory, prefix, item.key)
            if source is None:
                log.warning("no subtitle file was saved for language %s", item.language)
                continue
            target = directory / f"{prefix}.{'manual' if item.manual else 'auto'}.{item.language}.srt"
            try:
                _write_srt(source, target, rng, self._dfxp2srt)
            except Exception as e:  # one broken subtitle file must not cost the downloaded video
                log.warning("the %s subtitles cannot be used (%s: %s)", item.language, type(e).__name__, e)
                continue
            source.unlink(missing_ok=True)

    def _sweep(self, temporary: Path, directory: Path, prefix: str) -> None:
        """Move sidecars yt-dlp left in the temporary folder into the cache (best effort)."""
        for item in temporary.iterdir():
            if not item.is_file() or not item.name.startswith(f"{prefix}.media."):
                continue
            rest = item.name[len(prefix) + len(".media.") :]
            ext = item.suffix.lower().lstrip(".")
            if ext in IMAGE_EXTS:
                target = directory / f"{prefix}.thumbnail.{ext}"
            elif ext in SUB_EXTS and "." in rest:
                target = directory / f"{prefix}.subtitle.{rest}"
            else:
                continue
            if target.exists():
                continue
            try:
                shutil.move(str(item), str(target))
            except OSError:
                log.debug("could not move %s", item, exc_info=True)

    def _complete(self, directory: Path, prefix: str, settings: DownloadCfg) -> bool:
        """Whether a cached original already has every requested sidecar."""
        metadata = settings.metadata
        if metadata.info_json and not (directory / f"{prefix}.info.json").is_file():
            return False
        if metadata.description and not (directory / f"{prefix}.description").is_file():
            return False
        if metadata.thumbnail and not self._any(directory, (f"{prefix}.thumbnail.",)):
            return False
        if settings.subtitles.enabled:
            return self._any(directory, (f"{prefix}.manual.", f"{prefix}.auto.", f"{prefix}.subtitle."))
        return True

    @staticmethod
    def _any(directory: Path, prefixes: tuple[str, ...]) -> bool:
        """Whether ``directory`` holds a file whose name starts with one of ``prefixes``."""
        return any(item.is_file() and item.name.startswith(prefixes) for item in directory.iterdir())

    def _subtitle_for(self, directory: Path, prefix: str, settings: DownloadCfg) -> tuple[Path | None, bool]:
        """The SRT the pipeline may use as this video's transcript, and whether it is human-made."""
        if not settings.subtitles.enabled:
            return None, False
        policy = settings.subtitles.use_as_transcript
        if policy == "never":
            return None, False
        if self.cfg.transcript.mode == "whisper" and policy != "always":
            return None, False
        for language in settings.subtitles.languages:
            for kind, manual in (("manual", True), ("auto", False)):
                if kind == "auto" and policy != "always":
                    continue
                path = directory / f"{prefix}.{kind}.{language}.srt"
                if path.is_file() and _valid_srt(path):
                    return path, manual
        return None, False

    # --------------------------------------------------------------- yt-dlp IO

    def _entries(self, url: str, settings: DownloadCfg, limit: int | None) -> list[dict[str, Any]]:
        """The top-level video or the entries of its playlist (flat extraction)."""
        params = self._base_params(settings)
        params["extract_flat"] = "in_playlist"
        if limit is not None:
            params["playlistend"] = limit
        with self._YoutubeDL(params) as ydl:
            info = self._call(ydl.extract_info, url, download=False)
        if not isinstance(info, dict):
            raise DownloadError(f"yt-dlp returned no information about {url}")
        items = info.get("entries") if info.get("_type") in ("playlist", "multi_video") else None
        if items is None:
            return [info]
        entries = [entry for entry in items if isinstance(entry, dict)]
        if not entries:
            raise DownloadError(f"{url} has no downloadable entries")
        return entries[:1] if settings.playlist == "single" and limit is None else entries

    def _full_info(self, entry: dict[str, Any], url: str | None, settings: DownloadCfg) -> dict[str, Any]:
        """The full extraction of one entry (a resolved entry is reused as it is)."""
        if entry.get("formats") and entry.get("_type", "video") == "video":
            return entry
        if url is None:
            raise DownloadError("a playlist entry has no URL")
        item = copy.deepcopy(entry)
        item["_type"] = "url" if item.get("_type") in (None, "video") else item["_type"]
        item["url"] = url
        with self._YoutubeDL(self._base_params(settings)) as ydl:
            info = self._call(ydl.process_ie_result, item, download=False)
        if not isinstance(info, dict):
            raise DownloadError(f"yt-dlp returned no information about {url}")
        return info

    def _entry_family(self, entry: dict[str, Any], parent: str, url: str | None) -> str:
        """The family of a playlist entry; the generic scraper defers to its playlist."""
        name = str(entry.get("extractor") or "").split(":")[0].strip().lower()
        if name:
            family = canonical(name)
        else:
            key = str(entry.get("extractor_key") or entry.get("ie_key") or "")
            if key:
                try:
                    family = canonical(str(self._get_info_extractor(key).IE_NAME).split(":")[0])
                except KeyError:
                    log.debug("yt-dlp reported an unknown extractor key %r", key, exc_info=True)
                    family = identify(url) if url else GENERIC
            else:
                family = identify(url) if url else GENERIC
        if family == GENERIC and parent != GENERIC:
            return parent
        return family

    def _base_params(self, settings: DownloadCfg) -> dict[str, Any]:
        """The yt-dlp parameters every pass shares (network, cookies, policy, runtime)."""
        params: dict[str, Any] = {
            "quiet": True,
            "noprogress": True,
            "color": "never",
            "logger": self._logger,
            "windowsfilenames": True,
            "restrictfilenames": False,
            "overwrites": True,
            "socket_timeout": settings.network.socket_timeout,
            "retries": settings.network.retries,
            "fragment_retries": settings.network.retries,
            "noplaylist": settings.playlist == "single",
            "playlistend": 1 if settings.playlist == "single" else settings.max_videos,
            "allowed_extractors": list(self.allowed_extractors),
            "js_runtimes": copy.deepcopy(self.js_runtimes),
            "remote_components": list(self.remote_components),
            "ffmpeg_location": ffmpeg_exe(),
        }
        if settings.network.proxy:
            params["proxy"] = settings.network.proxy
        if settings.network.rate_limit_mbps:
            params["ratelimit"] = settings.network.rate_limit_mbps * 1024 * 1024 / 8
        if settings.cookies.file:
            params["cookiefile"] = settings.cookies.file
        elif settings.cookies.from_browser:
            params["cookiesfrombrowser"] = (settings.cookies.from_browser, None, None, None)
        return params

    def _media_params(self, settings: DownloadCfg, directory: Path, prefix: str, temporary: Path, choice: Choice) -> dict[str, Any]:
        """The parameters of a media pass (chosen format, file names, directories)."""
        params = self._base_params(settings)
        params["format"] = choice.format_id
        params["outtmpl"] = self._outtmpl(prefix)
        params["paths"] = {"home": str(directory), "temp": str(temporary)}
        return params

    def _sidecar_params(
        self,
        settings: DownloadCfg,
        directory: Path,
        prefix: str,
        temporary: Path,
        choice: Choice,
        plan: list[_ChosenSub],
    ) -> dict[str, Any]:
        """The parameters of a sidecar pass (no media, only the requested files)."""
        params = self._media_params(settings, directory, prefix, temporary, choice)
        params.update(
            {
                "skip_download": True,
                "writesubtitles": any(item.manual for item in plan),
                "writeautomaticsub": any(not item.manual for item in plan),
                "subtitleslangs": [re.escape(item.key) for item in plan],
                "subtitlesformat": "best",
                "writethumbnail": settings.metadata.thumbnail,
                "writeinfojson": settings.metadata.info_json,
                "writedescription": settings.metadata.description,
            }
        )
        return params

    @staticmethod
    def _outtmpl(prefix: str) -> dict[str, str]:
        """The output templates (a literal ``%`` must be escaped for yt-dlp)."""
        safe = prefix.replace("%", "%%")
        return {
            "default": f"{safe}.%(ext)s",
            "temp": f"{safe}.media.%(ext)s",
            "subtitle": f"{safe}.subtitle.%(ext)s",
            "thumbnail": f"{safe}.thumbnail.%(ext)s",
            "description": safe,
            "infojson": safe,
        }

    def _call(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call yt-dlp and translate its errors into :class:`DownloadError`."""
        try:
            return function(*args, **kwargs)
        except self._ytdl_error as e:
            raise DownloadError(_clean_error(str(e))) from e

    def _settings(self, family: str) -> DownloadCfg:
        """The effective download settings of one family (top-level, its groups, its own entry)."""
        fam = canonical(family)
        cached = self._settings_cache.get(fam)
        if cached is None:
            base = asdict(self.cfg.download)
            base.pop("platforms", None)
            base.pop("js_runtime", None)
            platforms = self.cfg.download.platforms
            cached = parse(DownloadCfg, effective_download(base, fam, platforms.groups, platforms.settings))
            self._settings_cache[fam] = cached
        return cached

    def _runtime_config(self, cfg: JsRuntimeCfg) -> dict[str, dict[str, str]]:
        """The ``js_runtimes`` parameter: the chosen runtime, or ``{}`` when none is available."""
        if cfg.name == "auto":
            for name in ("deno", "node", "bun", "quickjs"):
                runtime = self._runtime_classes.get(name)
                if runtime is not None and runtime().info is not None:
                    log.info("using the %s JavaScript runtime for yt-dlp", name)
                    return {name: {}}
            log.warning("no JavaScript runtime found: YouTube may not offer every format (install deno or node)")
            return {}
        runtime = self._runtime_classes.get(cfg.name)
        if runtime is None:
            log.warning("yt-dlp does not support the %s JavaScript runtime", cfg.name)
            return {}
        if cfg.path:
            path = Path(cfg.path)
            if not path.is_file():
                raise DownloadError(f"download.js_runtime.path: '{cfg.path}' is not a file")
            return {cfg.name: {"path": str(path)}}
        if runtime().info is None:
            log.warning("the %s JavaScript runtime was not found on PATH", cfg.name)
            return {}
        return {cfg.name: {}}
