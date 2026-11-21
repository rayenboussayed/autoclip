"""The downloader itself: platform policy, originals, subtitles and yt-dlp orchestration."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import random
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from ..config import Config, DownloadCfg, parse
from ..derive.constants import (
    COOKIE_BROWSERS,
    DOWNLOAD_FORMAT_CANDIDATES,
    DOWNLOAD_PAUSE_MAX_SECONDS,
    DOWNLOAD_PAUSE_MIN_SECONDS,
    DOWNLOAD_PLAYLIST_PAUSE_MAX_SECONDS,
    DOWNLOAD_PLAYLIST_PAUSE_MIN_SECONDS,
    DOWNLOAD_RETRIES,
    DOWNLOAD_SOCKET_TIMEOUT,
    JS_RUNTIME_ORDER,
    PLAYLIST_MAX_VIDEOS,
    download_quality_for,
)
from ..errors import AutoclipError, DownloadError, RateLimitError
from ..formats import Choice, DownloadPrefs, YtInfo, choose, choose_candidates, describe
from ..io import read_json as _read_json
from ..log import health_check, safe_path_text, traced
from ..media import bundled_ffmpeg_exe, ffmpeg_exe, trim_video
from ..models import Downloaded
from ..platforms import (
    GENERIC,
    allowed,
    canonical,
    check_families,
    effective_download,
    extractor_patterns,
    family_names,
    identify,
)
from .extraction import _clean_info
from .naming import _entry_url, _prefix, _range, _range_suffix, _safe_id, _safe_title, wants_all_entries
from .originals import _canonical_media, _empty, _exact_media, _media_in, _range_ok
from .params import _clean_error, _rate_limited, _YdlLogger
from .reporting import FormatReport, _row
from .subtitles import (
    SUB_EXTS,
    _pick_sub,
    _subtitle_file,
    _subtitle_languages,
    _valid_srt,
    _write_srt,
)

log = logging.getLogger("autoclip")


@dataclass(frozen=True, slots=True)
class _ChosenSub:
    """One subtitle track the downloader will ask yt-dlp for."""

    language: str  # the language as written in the config (it names the final file)
    key: str  # the key in yt-dlp's subtitles/automatic_captions dict
    manual: bool
    variant: dict[str, Any]


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
        self.root = Path(cfg.output_dir) / ".work" / "downloads"
        self._quality = download_quality_for(cfg.quality)
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
        self.js_runtimes = self._runtime_config()
        # A frozen build has no ffprobe next to its bundled ffmpeg (and a system ffmpeg would
        # inherit PyInstaller's LD_LIBRARY_PATH), so yt-dlp cannot cut a range safely there:
        # download the whole file and trim it locally with media.trim_video instead.
        frozen = bool(getattr(sys, "frozen", False))
        self._range_supported = not frozen and bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))

    # -------------------------------- zero-network cache (shared across runs) -----------------
    def _prefs(self, settings: DownloadCfg) -> DownloadPrefs:
        """The derived format preferences of one effective platform settings block."""
        return DownloadPrefs(quality=self._quality, audio_language=None)

    def _cache_key(self, family: str, url: str, settings: DownloadCfg) -> str:
        """Fingerprint of ``url`` + the settings that decide what is downloaded (paths/policy excluded)."""
        effective = asdict(settings)
        effective.pop("platforms", None)
        prefs = asdict(self._prefs(settings))
        blob = json.dumps({"url": url, "settings": effective, "prefs": prefs}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_file(self) -> Path:
        return self.root / "autoclip-cache.json"

    def _cache_load(self, family: str, url: str, settings: DownloadCfg) -> list[Downloaded] | None:
        """The cached originals for this URL/settings, or ``None`` when nothing can be reused."""
        try:
            raw = self._cache_file().read_text(encoding="utf-8")
            data = json.loads(raw)
            entries = data.get(self._cache_key(family, url, settings)) if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None  # a missing or unreadable cache never stops a download
        if not isinstance(entries, list) or not entries:
            return None
        resolved = self.root.resolve()
        out: list[Downloaded] = []
        for entry in entries:
            try:
                path = (self.root / str(entry["path"])).resolve()
                subtitle = (self.root / str(entry["subtitle"])).resolve() if entry.get("subtitle") else None
            except (TypeError, KeyError):
                return None
            if not path.is_relative_to(resolved) or not path.is_file():
                return None
            if subtitle is not None and (not subtitle.is_relative_to(resolved) or not subtitle.is_file()):
                return None
            prefix = path.stem
            if not (self.root / f"{prefix}.info.json").is_file():
                return None
            out.append(
                Downloaded(
                    path=path,
                    name=str(entry.get("name") or path.stem),
                    platform=family,
                    subtitle=subtitle,
                    subtitle_manual=bool(entry.get("subtitle_manual")),
                )
            )
        return out

    def _cache_store(self, family: str, url: str, settings: DownloadCfg, results: list[Downloaded]) -> None:
        """Remember the media files this URL/settings produced, so a later run needs no network."""
        self.root.mkdir(parents=True, exist_ok=True)
        key = self._cache_key(family, url, settings)
        file = self._cache_file()
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (OSError, json.JSONDecodeError):
            data = {}
        data[key] = [
            {
                "path": str(item.path.relative_to(self.root)),
                "name": item.name,
                "platform": item.platform,
                "subtitle": str(item.subtitle.relative_to(self.root)) if item.subtitle else None,
                "subtitle_manual": item.subtitle_manual,
            }
            for item in results
        ]
        tmp = file.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(file)
        except OSError as exc:  # a cache that cannot be written is not fatal
            log.warning("cannot write the download cache %s: %s", file, exc)

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
        return f"{family}: blocked (download.allow: {names})"

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
        choice = choose(cast(YtInfo, full), self._prefs(effective))
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
        log.info("system health before download: %s", health_check("download"))
        settings = self._settings(family)
        cached = self._cache_load(family, url, settings)
        if cached is not None:
            log.info("reusing %d cached original(s) for %s (no network needed)", len(cached), url)
            return cached
        try:
            self._pause(DOWNLOAD_PAUSE_MIN_SECONDS, DOWNLOAD_PAUSE_MAX_SECONDS)
            entries = self._entries(url, settings, limit=None)
        except RateLimitError as e:
            log.warning("%s: %s (skipping this URL)", url, e)
            raise
        if not entries:
            raise DownloadError(f"{url} has no downloadable entries")
        results: list[Downloaded] = []
        for index, entry in enumerate(entries):
            if index:
                self._pause(DOWNLOAD_PLAYLIST_PAUSE_MIN_SECONDS, DOWNLOAD_PLAYLIST_PAUSE_MAX_SECONDS)
            try:
                results.extend(self._one_entry(entry, family, settings))
            except RateLimitError:
                raise
            except DownloadError as e:
                if len(entries) == 1:
                    raise
                log.warning("skipping a playlist entry of %s: %s", url, e)
        if not results:
            raise DownloadError(f"nothing could be downloaded from {url}")
        self._cache_store(family, url, settings, results)
        return results

    @staticmethod
    def _pause(min_seconds: float, max_seconds: float) -> None:
        """Sleep a random moment so a batch of URLs does not look like a burst (patchable in tests)."""
        time.sleep(random.uniform(min_seconds, max_seconds))  # noqa: S311 - jitter, not a secret

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
        candidates = choose_candidates(cast(YtInfo, full), self._prefs(settings))[:DOWNLOAD_FORMAT_CANDIDATES]
        choice = candidates[0]
        if not choice.has_video:
            raise DownloadError(f"{full.get('title') or url or 'the URL'} only offers audio; autoclip needs a video track")
        rng = _range(settings)
        name, directory, prefix, cached = self._prepare(full, family, settings, choice, rng)
        if cached is not None:
            used = choice
            media = _canonical_media(directory, prefix, cached)
            if not self._complete(directory, prefix, settings):
                self._sidecars(full, settings, directory, prefix, choice, rng)
        else:
            log.debug("download: candidates in preference order: %s", ", ".join(c.format_id for c in candidates))
            temporary = Path(tempfile.mkdtemp(prefix=".download-", dir=directory))
            try:
                media, used, prefix = self._media(full, settings, directory, name, rng, temporary, candidates)
                self._sidecars(full, settings, directory, prefix, used, rng)
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
        for warning in used.warnings:
            log.warning("%s (format %s)", warning, used.format_id)
        log.info("downloaded with format %s (%s)", used.format_id, describe(cast(YtInfo, full), used))
        subtitle, manual = self._subtitle_for(directory, prefix, settings)
        log.info(
            "downloaded %s | path=%s | sidecar=%s | platform=%s | subtitle=%s",
            name,
            safe_path_text(media),
            f"{prefix}.info.json",
            family,
            safe_path_text(subtitle) if subtitle else "-",
        )
        return [Downloaded(path=media, name=name, platform=family, subtitle=subtitle, subtitle_manual=manual)]

    def _prepare(
        self, full: dict[str, Any], family: str, settings: DownloadCfg, choice: Choice, rng: tuple[float, float] | None
    ) -> tuple[str, Path, str, Path | None]:
        """The output name, cache directory, file prefix and the cached media (if present)."""
        title = _safe_title(str(full.get("title") or full.get("id") or "video"))
        video_id = _safe_id(str(full.get("id") or ""))
        base = f"{title} [{video_id}]" if video_id else title
        suffix = _range_suffix(rng)
        directory = self._directory()
        prefix = _prefix(base, choice, suffix)
        media = _exact_media(directory, prefix, rng)
        if media is not None:
            return base + suffix, directory, prefix, media
        key = (str(directory), base + suffix)
        if key in self._used:
            number = 2
            while (str(directory), f"{base} ({number}){suffix}") in self._used:
                number += 1
            base = f"{base} ({number})"
        self._used.add((str(directory), base + suffix))
        return base + suffix, directory, _prefix(base, choice, suffix), None

    def _directory(self) -> Path:
        """The folder of the downloaded originals."""
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    # ------------------------------------------------------------------ media

    def _media(
        self,
        full: dict[str, Any],
        settings: DownloadCfg,
        directory: Path,
        name: str,
        rng: tuple[float, float] | None,
        temporary: Path,
        candidates: list[Choice],
    ) -> tuple[Path, Choice, str]:
        """Download the media. Returns ``(media, candidate used, its prefix)``.

        ``name`` is the deduplicated output base (``base``/``base (2)``, with a range suffix);
        every candidate gets its own ``<name> [<format id>][<range>]`` prefix. With a time range
        and a real ffmpeg the preferred candidate is tried once (its own fallback to the whole
        video stays); every other failure moves on to the next candidate (at most two
        alternates), so a stalling or unreachable format costs the video nothing.
        """
        suffix = _range_suffix(rng)
        base = name[: -len(suffix)] if suffix else name
        ranged_error: DownloadError | None = None
        if rng is not None and self._range_supported:
            first = candidates[0]
            prefix = _prefix(base, first, suffix)
            ranged = dict(self._media_params(settings, directory, prefix, temporary, first))
            ranged["download_ranges"] = self._download_range_func(None, [rng])
            ranged["force_keyframes_at_cuts"] = True
            try:
                with self._YoutubeDL(ranged) as ydl:
                    self._call(ydl.process_ie_result, _clean_info(full), download=True)
                media = self._collect(directory, temporary, prefix)
                if media is not None and self._duration_ok(media, rng):
                    return media, first, prefix
                log.warning("the %s download does not have the expected length; downloading the whole video", settings.time_range)
                if media is not None:
                    media.unlink(missing_ok=True)
                _empty(temporary)
            except RateLimitError:
                _empty(temporary)
                raise  # the platform asked to stop; the whole-video fallback must not hide it
            except DownloadError as e:
                ranged_error = e
                log.warning("the %s range download failed (%s); trying the other candidates", settings.time_range, e)
                _empty(temporary)
        # The range attempt already failed this exact format id; when there is another candidate to try,
        # skip straight to it (a single-candidate URL keeps the old whole-video recovery instead).
        skipped = candidates[0].format_id if ranged_error is not None and len(candidates) > 1 else None
        last_error: DownloadError | None = ranged_error
        attempted: list[str] = [f"{skipped} (range download)"] if skipped is not None else []
        for choice in candidates:
            if choice.format_id == skipped:  # the range attempt already proved this exact format id unusable
                continue
            prefix = _prefix(base, choice, suffix)
            cached = _exact_media(directory, prefix, rng)
            if cached is not None:  # a different candidate was downloaded by an earlier run
                return cached, choice, prefix
            try:
                with self._YoutubeDL(self._media_params(settings, directory, prefix, temporary, choice)) as ydl:
                    self._call(ydl.process_ie_result, _clean_info(full), download=True)
                media = self._collect(directory, temporary, prefix)
                if media is None:
                    raise DownloadError(f"yt-dlp did not save a video file for {prefix}")
                if rng is None:
                    return media, choice, prefix
                target = directory / f"{prefix}.mkv"
                try:
                    trim_video(media, target, rng[0], rng[1])
                except AutoclipError as e:
                    media.unlink(missing_ok=True)
                    target.unlink(missing_ok=True)
                    raise DownloadError(f"cannot cut {settings.time_range} out of the downloaded video: {e}") from e
                media.unlink(missing_ok=True)
                return target, choice, prefix
            except RateLimitError:
                raise  # the platform asked to stop; trying another format would ignore it
            except Exception as e:  # every failure of a candidate costs nothing but the candidate
                wrapped = e if isinstance(e, DownloadError) else DownloadError(f"{type(e).__name__}: {e}")
                attempted.append(f"{choice.format_id} ({wrapped})")
                last_error = wrapped
                remaining = [c for c in candidates if c.format_id not in {choice.format_id, skipped}]
                if remaining:
                    log.warning(
                        "the media format %s failed (%s); trying the next best format (%s)",
                        choice.format_id,
                        wrapped,
                        remaining[0].format_id,
                    )
                    _empty(temporary)
                    continue
        if len(attempted) == 1:  # keep the original failure for the single candidate that was tried
            raise last_error if last_error is not None else DownloadError("download failed")
        joined = "; ".join(attempted)
        raise DownloadError(f"none of the {len(attempted)} attempt(s) could be downloaded ({joined})") from last_error

    def _duration_ok(self, media: Path, rng: tuple[float, float]) -> bool:
        """Whether a range download is close enough to the requested length."""
        return _range_ok(media, rng)

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
        """Save the requested subtitles and the metadata next to the original."""
        plan = self._plan(full, settings)
        temporary = Path(tempfile.mkdtemp(prefix=".sidecar-", dir=directory))
        try:
            info = _clean_info(full)
            self._apply_plan(info, plan)
            params = self._sidecar_params(settings, directory, prefix, temporary, choice, plan)
            with self._YoutubeDL(params) as ydl:
                self._call(ydl.process_ie_result, info, download=True)
            self._sweep(temporary, directory, prefix)
        except RateLimitError:
            raise
        except Exception as e:  # a missing sidecar must never cost the downloaded video
            log.warning("could not save every extra file for %s: %s", prefix, e)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        if plan:
            self._convert_subtitles(directory, prefix, plan, rng)

    def _plan(self, full: dict[str, Any], settings: DownloadCfg) -> list[_ChosenSub]:
        """The subtitle track to request for every configured language."""
        plan: list[_ChosenSub] = []
        for language in _subtitle_languages(full, settings):
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
            if ext not in SUB_EXTS or "." not in rest:
                continue
            target = directory / f"{prefix}.subtitle.{rest}"
            if target.exists():
                continue
            try:
                shutil.move(str(item), str(target))
            except OSError:
                log.debug("could not move %s", item, exc_info=True)

    def _complete(self, directory: Path, prefix: str, settings: DownloadCfg) -> bool:
        """Whether a cached original already has every requested sidecar."""
        if not (directory / f"{prefix}.info.json").is_file():
            return False
        return self._any(directory, (f"{prefix}.manual.", f"{prefix}.auto.", f"{prefix}.subtitle."))

    @staticmethod
    def _any(directory: Path, prefixes: tuple[str, ...]) -> bool:
        """Whether ``directory`` holds a file whose name starts with one of ``prefixes``."""
        return any(item.is_file() and item.name.startswith(prefixes) for item in directory.iterdir())

    def _subtitle_for(self, directory: Path, prefix: str, settings: DownloadCfg) -> tuple[Path | None, bool]:
        """The SRT the pipeline may use as this video's transcript, and whether it is human-made.

        A manual track is preferred; automatic captions are only used when no language has one.
        The requested languages follow the video's original language when the user leaves it alone.
        """
        try:
            info = _read_json(directory / f"{prefix}.info.json")
        except Exception:
            info = {}
        for kind, manual in (("manual", True), ("auto", False)):
            for language_code in _subtitle_languages(info, settings):
                path = directory / f"{prefix}.{kind}.{language_code}.srt"
                if path.is_file() and _valid_srt(path):
                    return path, manual
        for pattern in (f"{prefix}.manual.*.srt", f"{prefix}.auto.*.srt"):
            for item in sorted(directory.glob(pattern)):
                if item.is_file() and _valid_srt(item):
                    return item, pattern.startswith(f"{prefix}.manual.")
        return None, False

    # --------------------------------------------------------------- yt-dlp IO

    def _entries(self, url: str, settings: DownloadCfg, limit: int | None) -> list[dict[str, Any]]:
        """The top-level video or the entries of its playlist (flat extraction).

        The URL shape decides: a watch/shorts/video URL is one video even when it carries a
        playlist id, while a playlist/channel/album/user URL expands to its entries (capped at
        :data:`PLAYLIST_MAX_VIDEOS`). ``limit`` overrides the cap for an internal caller.
        """
        all_entries = limit is None and wants_all_entries(url)
        params = self._base_params(settings)
        params["extract_flat"] = "in_playlist"
        params["noplaylist"] = not all_entries
        params["playlistend"] = limit if limit is not None else (PLAYLIST_MAX_VIDEOS if all_entries else 1)
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
        return entries[:limit] if limit is not None else (entries if all_entries else entries[:1])

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
            "socket_timeout": DOWNLOAD_SOCKET_TIMEOUT,
            "retries": DOWNLOAD_RETRIES,
            "fragment_retries": DOWNLOAD_RETRIES,
            "sleep_interval": DOWNLOAD_PAUSE_MIN_SECONDS,
            "max_sleep_interval": DOWNLOAD_PAUSE_MAX_SECONDS,
            "allowed_extractors": list(self.allowed_extractors),
            "js_runtimes": copy.deepcopy(self.js_runtimes),
            "ffmpeg_location": bundled_ffmpeg_exe() if getattr(sys, "frozen", False) else ffmpeg_exe(),
        }
        if settings.cookies:
            if settings.cookies in COOKIE_BROWSERS:
                params["cookiesfrombrowser"] = (settings.cookies, None, None, None)
            else:
                params["cookiefile"] = settings.cookies
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
                "writeinfojson": True,
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
            "infojson": safe,
        }

    def _call(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call yt-dlp and translate its errors into :class:`RateLimitError`/:class:`DownloadError`."""
        try:
            return function(*args, **kwargs)
        except self._ytdl_error as e:
            message = _clean_error(str(e))
            if _rate_limited(e):
                raise RateLimitError(f"the platform is rate limiting requests: {message}") from e
            raise DownloadError(message) from e

    def _settings(self, family: str) -> DownloadCfg:
        """The effective download settings of one family (top-level, its groups, its own entry)."""
        fam = canonical(family)
        cached = self._settings_cache.get(fam)
        if cached is None:
            base = asdict(self.cfg.download)
            base.pop("platforms", None)
            platforms = self.cfg.download.platforms
            cached = parse(DownloadCfg, effective_download(base, fam, platforms.groups, platforms.settings))
            self._settings_cache[fam] = cached
        return cached

    def _runtime_config(self) -> dict[str, dict[str, str]]:
        """The ``js_runtimes`` parameter: the first runtime found on PATH, or ``{}`` when none is available."""
        for name in JS_RUNTIME_ORDER:
            runtime = self._runtime_classes.get(name)
            if runtime is None:
                continue
            try:
                available = runtime().info is not None
            except Exception:  # a broken runtime must not break the downloader
                log.debug("could not check the %s JavaScript runtime", name, exc_info=True)
                continue
            if available:
                log.info("automation js_runtime=%s | source=first runtime found on PATH", name)
                return {name: {}}
        log.warning("no JavaScript runtime found: YouTube may not offer every format (install deno or node)")
        return {}
