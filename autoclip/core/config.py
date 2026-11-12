"""Typed configuration: JSON -> validated dataclasses (Pydantic owns the schema).

Every option lives in exactly one place - its dataclass field - so defaults, ``--init-full`` output and
validation can never drift apart. ``load_config`` also resolves all paths relative to the config file
and expands ``input`` globs into the list of videos to process.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import platform
import re
import typing
from collections.abc import Mapping
from dataclasses import asdict, field
from glob import glob
from pathlib import Path
from typing import Annotated, Literal, TypeVar
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, Strict, StringConstraints, TypeAdapter, ValidationError, model_validator
from pydantic.dataclasses import dataclass

from .errors import ConfigError
from .internal.automation import COOKIE_BROWSERS
from .internal.background import background_kind, resolve_background
from .internal.content import load_lexicon
from .platforms import BUILTIN_GROUPS, canonical, deep_merge
from .styles import STYLES

log = logging.getLogger("autoclip")

#: A value that can appear in a JSON document (used instead of ``Any`` for config-shaped data).
JsonValue = dict[str, "JsonValue"] | list["JsonValue"] | str | int | float | bool | None

Level = Literal["DEBUG", "INFO", "WARNING", "ERROR", "OFF"]
Position = Literal["top", "center", "lower", "bottom"]

#: The schema: an unknown key is an error, a value keeps the type the JSON gave it.
SCHEMA = ConfigDict(extra="forbid")
#: JSON has no int/float distinction: a float field takes both, an int field only integers, a bool only true/false.
Int = Annotated[int, Strict()]
Float = Annotated[float, Strict()]
Bool = Annotated[bool, Strict()]
#: "9:16" (any numbers) or "original".
Aspect = Annotated[str, StringConstraints(pattern=r"^(\d+(\.\d+)?:\d+(\.\d+)?|original)$")]
#: "00:01:00-00:01:45".
Range = Annotated[str, StringConstraints(pattern=r"^[\d:.]+-[\d:.]+$")]
#: The URL of the LLM server: http or https.
Url = Annotated[str, StringConstraints(pattern=r"(?i)^https?://\S+$")]
#: A proxy URL for the downloader (http(s) or SOCKS).
Proxy = Annotated[str, StringConstraints(pattern=r"(?i)^(https?|socks[45]h?)://\S+$")]


def parse_hms(text: str) -> float:
    """``[[HH:]MM:]SS[.ms]`` -> seconds."""
    seconds = 0.0
    for part in text.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


@dataclass(config=SCHEMA)
class Section:
    """Base of every config section: unknown keys are an error, ``"_comment"`` keys are ignored."""

    @model_validator(mode="before")
    @classmethod
    def _drop_comments(cls, data: JsonValue) -> JsonValue:
        """Comments in the config file are keys starting with an underscore - they are not options."""
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if not k.startswith("_")}
        return data


@dataclass(config=SCHEMA)
class KindCfg(Section):
    """One kind of output clip ("short" or "long"): the length range."""

    count: Annotated[Int, Field(ge=1)] = 5  # clips per scene type
    min_seconds: Annotated[Float, Field(gt=0)] = 20  # a clip is never shorter than this
    max_seconds: Annotated[Float, Field(gt=0)] = 60  # a clip is never longer than this
    aspect: Aspect = "9:16"  # 9:16 | 1:1 | 4:5 | 16:9 | original
    ranges: list[Range] = field(default_factory=list)  # clips.mode = "ranges", e.g. ["00:01:00-00:01:45"]

    @model_validator(mode="after")
    def _lengths_fit(self) -> KindCfg:
        """The length range has to make sense."""
        if self.min_seconds > self.max_seconds:
            raise ConfigError("min_seconds must be <= max_seconds")
        return self


@dataclass(config=SCHEMA)
class ClipsCfg(Section):
    """Which moments to cut."""

    mode: Literal["highlights", "ranges", "split"] = "highlights"
    kinds: list[Literal["short", "long"]] = field(default_factory=lambda: ["short", "long"])
    scenes: str | list[str] = "best"  # "all", one scene type or a list; each type gets its own clips
    short: KindCfg = field(default_factory=KindCfg)
    long: KindCfg = field(default_factory=lambda: KindCfg(count=2, min_seconds=180, max_seconds=600, aspect="16:9"))
    keywords: list[str] = field(default_factory=list)  # extra words/phrases (any language) that boost a moment

    def scene_list(self) -> list[str]:
        """The scene types to generate, in order (``"all"`` expands to every type)."""
        if self.scenes == "all":
            return list(STYLES)
        return [self.scenes] if isinstance(self.scenes, str) else self.scenes

    def kind(self, name: str) -> KindCfg:
        """Settings of the ``"short"`` or ``"long"`` kind."""
        return self.short if name == "short" else self.long

    @model_validator(mode="after")
    def _scenes_and_kinds(self) -> ClipsCfg:
        """The scene types must exist and every kind must be listed once."""
        bad = [s for s in self.scene_list() if s not in STYLES]
        if bad:
            raise ConfigError(f"unknown clips.scenes {bad}; valid: 'all' or {', '.join(STYLES)}")
        if not self.kinds or len(set(self.kinds)) != len(self.kinds):
            raise ConfigError("clips.kinds must list 'short' and/or 'long' once each")
        return self


@dataclass(config=SCHEMA)
class LlmCfg(Section):
    """Optional local re-ranking (Ollama or any OpenAI-compatible server). Ollama is local and needs no API key."""

    base_url: Url = "http://localhost:11434"
    model: str = "llama3.2"


@dataclass(config=SCHEMA)
class VideoCfg(Section):
    """Output encoding (the canvas shape comes from each clip kind's ``aspect``)."""

    layout: Literal["fit", "fill"] = "fit"  # fit = whole frame on a background, fill = crop to fill
    background: str = "blur"  # "blur", a #RGB/#RRGGBB colour or an image file (relative to the config file)
    codec: Literal["h264", "h265"] = "h264"

    @model_validator(mode="after")
    def _background_is_usable(self) -> VideoCfg:
        """``blur``, a colour or a path; a malformed ``#`` value is rejected here rather than at render time."""
        background_kind(self.background)
        return self


@dataclass(config=SCHEMA)
class CaptionsCfg(Section):
    """Burned-in captions."""

    enabled: Bool = True
    theme: Literal["social", "plain", "outline", "no_box"] = "social"  # colour and box as one preset
    position: Position = "lower"
    uppercase: Bool = False

    def __post_init__(self) -> None:
        style = {
            "social": ("#FFFFFF", "#000000", 0.6, 0, "#000000", "#FFD400"),
            "plain": ("#FFFFFF", "#000000", 0.0, 0, "#000000", "#FFFFFF"),
            "outline": ("#FFFFFF", "#000000", 0.0, 3, "#000000", "#FFFFFF"),
            "no_box": ("#FFFFFF", "#000000", 0.0, 0, "#000000", "#FFFFFF"),
            "blur": ("#FFFFFF", "#000000", 0.6, 0, "#000000", "#FFD400"),
        }[self.theme]
        self.color, self.bg_color, self.bg_opacity, self.stroke_width, self.stroke_color, self.highlight_color = style


@dataclass(config=SCHEMA)
class TitleCfg(Section):
    """Optional hook title shown at the start of each clip."""

    enabled: Bool = False


@dataclass(config=SCHEMA)
class LoggingCfg(Section):
    """Log file and console verbosity."""

    level: Level = "INFO"  # written to the log file and the console; OFF disables both


@dataclass(config=SCHEMA)
class SkipCfg(Section):
    """Leave individual pipeline steps out (nothing is skipped by default)."""

    stt: Bool = False  # no Whisper; an explicit, downloaded or cached transcript is still used
    denoise: Bool = False  # transcribe raw audio, without the speech enhancer
    audio_enhance: Bool = False  # keep the clip audio untouched (no denoise/loudness/limiter)
    audio: Bool = False  # mute the clips (and skip audio decoding, so pauses cannot be removed)
    pauses: Bool = False  # keep long pauses and filler sounds in the clips
    caption: Bool = False  # no burned-in captions
    title: Bool = False  # no overlay title, even when the model generated one
    fade: Bool = False  # no video or audio fades
    thumbnail: Bool = False  # keep the frame JPEG instead of generating an AI thumbnail
    llm: Bool = False  # no Ollama calls (ranking, titles, social captions)
    snap: Bool = False  # do not snap cut points to shot changes


SkipName = Literal["stt", "denoise", "audio_enhance", "audio", "pauses", "caption", "title", "fade", "thumbnail", "llm", "snap"]
SKIP_NAMES: tuple[SkipName, ...] = (
    "stt",
    "denoise",
    "audio_enhance",
    "audio",
    "pauses",
    "caption",
    "title",
    "fade",
    "thumbnail",
    "llm",
    "snap",
)


@dataclass(config=SCHEMA)
class PlatformsCfg(Section):
    """Which platforms may be downloaded, and their per-platform overrides."""

    allow: list[str] = field(default_factory=lambda: ["@popular"])  # families or @groups that may be used
    deny: list[str] = field(default_factory=list)  # families or @groups that never may (deny wins)
    groups: dict[str, list[str]] = field(default_factory=dict)  # user groups: {"@news": ["youtube", ...]}
    settings: dict[str, dict[str, object]] = field(default_factory=dict)  # partial download settings per family/@group

    @model_validator(mode="after")
    def _names_and_groups(self) -> PlatformsCfg:
        """Group names, references and cycles are checked here; families need yt-dlp and are checked at run time."""
        if not self.allow:
            raise ConfigError("download.platforms.allow must name at least one platform or group")
        for name in (*self.allow, *self.deny):
            if not name.strip():
                raise ConfigError("download.platforms.allow/deny must not contain empty names")
        known = set(BUILTIN_GROUPS) | set(self.groups)
        for name in (*self.allow, *self.deny):
            if name.startswith("@") and name not in known:
                raise ConfigError(f"download.platforms: unknown group {name}")
        for name, members in self.groups.items():
            if not re.fullmatch(r"@[a-z0-9][a-z0-9_-]*", name):
                raise ConfigError(f"download.platforms.groups.{name}: a group name must look like '@name'")
            if name in BUILTIN_GROUPS:
                raise ConfigError(f"download.platforms.groups.{name}: a built-in group cannot be redefined")
            if not members:
                raise ConfigError(f"download.platforms.groups.{name}: a group needs at least one member")
            for member in members:
                if not member.strip():
                    raise ConfigError(f"download.platforms.groups.{name}: a member must not be empty")
                if member.startswith("@") and member not in known:
                    raise ConfigError(f"download.platforms.groups.{name}: unknown group {member}")
        done: set[str] = set()

        def visit(name: str, path: tuple[str, ...]) -> None:
            if name in done:
                return
            if name in path:
                raise ConfigError("download.platforms.groups has a cycle: " + " -> ".join((*path, name)))
            for member in self.groups[name]:
                if member.startswith("@") and member in self.groups:
                    visit(member, (*path, name))
            done.add(name)

        for name in self.groups:
            visit(name, ())
        seen: dict[str, str] = {}
        for name in self.settings:
            if not name.strip():
                raise ConfigError("download.platforms.settings must not contain an empty platform name")
            if name.startswith("@"):
                if name not in known:
                    raise ConfigError(f"download.platforms.settings: unknown group {name}")
                continue
            family = canonical(name)
            if family in seen:
                raise ConfigError(f"download.platforms.settings: '{seen[family]}' and '{name}' name the same platform")
            seen[family] = name
        return self


@dataclass(config=SCHEMA)
class DownloadCfg(Section):
    """Downloading the http(s) URLs found in ``input`` with yt-dlp."""

    time_range: Range | None = None  # "MM:SS-MM:SS" / "HH:MM:SS-HH:MM:SS"
    cookies: str | None = None  # a browser name (chrome, firefox, ...) or a Netscape-format cookies.txt
    proxy: Proxy | None = None  # runtime override (never in the file)
    platforms: PlatformsCfg = field(default_factory=PlatformsCfg)

    @model_validator(mode="after")
    def _range_and_platform_settings(self) -> DownloadCfg:
        """The time range must make sense and every per-platform override must be a valid download section."""
        if self.time_range:
            try:
                lo, hi = (parse_hms(part) for part in self.time_range.split("-"))
            except ValueError as e:
                raise ConfigError(f"download.time_range is not a valid time range: {e}") from e
            if hi <= lo:
                raise ConfigError("download.time_range must end after it starts")
        base = asdict(self)
        base.pop("platforms")
        for name, entry in self.platforms.settings.items():
            if "platforms" in entry:
                raise ConfigError(f"download.platforms.settings.{name}: platforms cannot be overridden per platform")
            try:
                parse(DownloadCfg, deep_merge(base, entry))
            except ConfigError as e:
                raise ConfigError(f"download.platforms.settings.{name}: {e}") from e
        return self


def _default_output_dir(system: str | None = None, env: Mapping[str, str] | None = None, home: Path | None = None) -> str:
    """The output folder when none is given: the platform's Videos/Movies folder plus ``clips``."""
    system = system or platform.system()
    env = os.environ if env is None else env
    home = Path.home() if home is None else home
    if system == "Darwin":
        return str(home / "Movies" / "clips")
    if system == "Windows":
        return str(home / "Videos" / "clips")
    videos = env.get("XDG_VIDEOS_DIR")
    return str((Path(videos).expanduser() if videos else home / "Videos") / "clips")


@dataclass(config=SCHEMA)
class Config(Section):
    """The whole configuration file."""

    @model_validator(mode="before")
    @classmethod
    def _normalise_mode(cls, data: JsonValue) -> JsonValue:
        """An unknown or empty ``mode`` quietly falls back to ``balance`` instead of failing validation."""
        if isinstance(data, dict) and "mode" in data:
            mode = data["mode"]
            if mode not in ("performance", "resources", "balance"):
                data = {**data, "mode": "balance"}
        return data

    @model_validator(mode="after")
    def _mode_is_valid(self) -> Config:
        """Keep the runtime choice consistent: ``profile_for`` never raises on a misconfigured value."""
        if self.mode not in ("performance", "resources", "balance"):
            self.mode = "balance"  # pragma: no cover
        return self

    input: list[str] = field(default_factory=lambda: ["input.mp4"])  # files, glob patterns or http(s) URLs
    output_dir: str = field(default_factory=_default_output_dir)  # one sub-folder per video is created inside
    mode: Literal["performance", "resources", "balance"] = "balance"  # performance/resource/balance policy
    quality: Literal["highest", "high", "medium", "low", "lowest"] = "high"  # encode resolution/fps/audio quality
    download: DownloadCfg = field(default_factory=DownloadCfg)
    clips: ClipsCfg = field(default_factory=ClipsCfg)
    transcript_file: str | None = None  # runtime override: explicit .srt/.json transcript
    llm: LlmCfg = field(default_factory=LlmCfg)
    video: VideoCfg = field(default_factory=VideoCfg)
    captions: CaptionsCfg = field(default_factory=CaptionsCfg)
    title: TitleCfg = field(default_factory=TitleCfg)
    skip: SkipCfg = field(default_factory=SkipCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)


@dataclass(config=SCHEMA)
class FileClipsCfg(Section):
    """The four clip options that are worth asking about."""

    preset: Literal["both", "shorts", "long"] = "both"  # which families of clips to render
    scenes: str | list[str] = "best"  # "all", one scene type or a list
    keywords: list[str] = field(default_factory=list)  # extra words/phrases that boost a moment
    ranges: list[Range] = field(default_factory=list)  # exact cuts: ["00:01:00-00:01:45"]

    def scene_list(self) -> list[str]:
        """The scene types to generate, in order (``"all"`` expands to every type)."""
        if self.scenes == "all":
            return list(STYLES)
        return [self.scenes] if isinstance(self.scenes, str) else self.scenes

    @model_validator(mode="after")
    def _scenes_exist(self) -> FileClipsCfg:
        """The scene types must exist (``"all"`` expands to every type)."""
        bad = [s for s in self.scene_list() if s not in STYLES]
        if bad:
            raise ConfigError(f"unknown clips.scenes {bad}; valid: 'all' or {', '.join(STYLES)}")
        return self


@dataclass(config=SCHEMA)
class FilePlatformsCfg(Section):
    """The only download policy a user ever needs to write down."""

    allow: list[str] = field(default_factory=lambda: ["@popular"])
    deny: list[str] = field(default_factory=list)

    @model_validator(mode="after")
    def _names(self) -> FilePlatformsCfg:
        """No empty names and no unknown built-in groups (families are checked against yt-dlp at run time)."""
        if not self.allow:
            raise ConfigError("download.platforms.allow must name at least one platform or group")
        for name in (*self.allow, *self.deny):
            if not name.strip():
                raise ConfigError("download.platforms.allow/deny must not contain empty names")
            if name.startswith("@") and name not in BUILTIN_GROUPS:
                raise ConfigError(f"download.platforms: unknown group {name}")
        return self


@dataclass(config=SCHEMA)
class FileDownloadCfg(Section):
    """Downloading URLs found in ``input`` (everything else follows the platform's own conventions)."""

    platforms: FilePlatformsCfg = field(default_factory=FilePlatformsCfg)


#: Keys older versions accepted, with the replacement to use instead. A removed *section* covers its children.
REMOVED_KEYS: dict[str, str] = {
    "output_dir": "set AUTOCLIP_OUTPUT_DIR, pass --output-dir, or choose the folder in the app",
    "mode": "pass --mode performance|resources|balance",
    "quality": "pass --quality highest|high|medium|low|lowest",
    "logging": "pass --log-level DEBUG|INFO|WARNING|ERROR|OFF",
    "clips.mode": "clips.preset chooses shorts/long/both; use clips.ranges for exact cuts or pass --split",
    "clips.kinds": "use clips.preset",
    "clips.short": "use clips.preset (the short profile is built in)",
    "clips.long": "use clips.preset (the long profile is built in)",
    "clips.lexicon_file": "drop a lexicon.json next to the config file instead",
    "clips.skip_intro_seconds": "no longer configurable",
    "clips.snap_to_cuts": "shot-change snapping is automatic",
    "clips.pad_start": "padding is chosen automatically",
    "clips.pad_end": "padding is chosen automatically",
    "clips.split_seconds": "pass --split",
    "transcript": "pass --transcript PATH (consent is automatic; language is auto-detected)",
    "transcript_file": "pass --transcript PATH",
    "llm": "pass --llm-model (or AUTOCLIP_LLM_MODEL/AUTOCLIP_LLM_URL); a local Ollama is detected automatically",
    "llm.api_key": "Ollama is local and needs no API key",
    "compute": "the GPU is probed and used automatically (there is no --device flag)",
    "device": "the GPU is probed and used automatically (there is no --device flag)",
    "offline": "removed: AI is controlled by --skip llm; Whisper downloads models automatically",
    "video": "the output stack (layout, background, codec, fps) is chosen automatically",
    "captions": "captions use the automatic social style; pass --skip caption to disable",
    "title": "titles follow the LLM/kind policy; pass --skip title to disable",
    "audio": "audio processing is automatic; pass --skip audio_enhance to disable",
    "unsilence": "silence removal is automatic",
    "silence": "silence removal is automatic",
    "skip": "pass --skip NAME (see `autoclip run --help`)",
    "metadata": "metadata sidecars are always written",
    "cookies": "pass --cookies BROWSER|FILE or AUTOCLIP_COOKIES",
    "js_runtime": "the JavaScript runtime is probed automatically",
    "network": "pass --proxy URL or AUTOCLIP_PROXY",
    "download.quality": "the best matching format is chosen automatically",
    "download.fps": "the best matching format is chosen automatically",
    "download.prefer": "the best matching format is chosen automatically",
    "download.codec": "the best matching format is chosen automatically",
    "download.container": "the best matching format is chosen automatically",
    "download.audio_quality": "the best audio track is chosen automatically",
    "download.audio_language": "the video's original audio track is kept",
    "download.original_dir": "originals live in <output>/.work/downloads",
    "download.metadata": "metadata sidecars are always written",
    "download.subtitles": "a downloaded subtitle is used when it exists (pass --skip stt to forbid Whisper)",
    "download.playlist": "the URL shape decides: one video, or the whole playlist/channel (max 50)",
    "download.max_videos": "playlists are capped at 50 entries",
    "download.time_range": "cut the source beforehand or use clips.ranges",
    "download.cookies": "pass --cookies BROWSER|FILE or AUTOCLIP_COOKIES",
    "download.network": "pass --proxy URL or AUTOCLIP_PROXY",
    "download.proxy": "pass --proxy URL or AUTOCLIP_PROXY",
    "download.js_runtime": "the JavaScript runtime is probed automatically",
    "download.cache": "originals are always cached",
    "download.platforms.groups": "name families directly in download.platforms.allow/deny",
    "download.platforms.settings": "per-platform settings are chosen automatically",
}

#: Sections whose children are walked when looking for removed keys.
_REMOVED_CONTAINERS: frozenset[str] = frozenset(
    prefix for path in REMOVED_KEYS for prefix in (".".join(path.split(".")[:i]) for i in range(1, path.count(".") + 1))
) | {"clips", "download", "download.platforms"}


def _warn_removed(path: str, hint: str) -> None:
    """Warn once per removed key, in the ``parse`` call that saw it."""
    seen = _removed_seen.get()
    if seen is not None:
        if path in seen:
            return
        seen.add(path)
    log.warning("config: '%s' is no longer an option; %s", path, hint)


def _strip_removed(data: JsonValue, prefix: str = "") -> JsonValue:
    """Drop keys older versions accepted (with a warning) so an old config file still loads."""
    if not isinstance(data, dict):
        return data
    out: dict[str, JsonValue] = {}
    for key, raw in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if path in REMOVED_KEYS:
            _warn_removed(path, REMOVED_KEYS[path])
            continue
        value = _strip_removed(raw, path) if isinstance(raw, dict) and (not prefix or path in _REMOVED_CONTAINERS) else raw
        out[key] = value
    return out


@dataclass(config=SCHEMA)
class RuntimeOverrides(Section):
    """Options a front end sets outside the file: the command line, the environment, the app, the server."""

    input: list[str] | None = None
    output_dir: str | None = None
    transcript: str | None = None  # explicit .srt/.json transcript (overrides the file key, which does not exist)
    cookies: str | None = None  # a browser name or a Netscape cookies file
    proxy: Proxy | None = None
    log_level: Level | None = None
    skip: list[SkipName] | None = None
    llm_base_url: Url | None = None
    llm_model: str | None = None
    split: Bool | None = None
    mode: Literal["performance", "resources", "balance"] | None = None
    quality: Literal["highest", "high", "medium", "low", "lowest"] | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> RuntimeOverrides:
        """The overrides carried by ``AUTOCLIP_*`` environment variables (invalid values are ignored)."""
        env = os.environ if env is None else env
        data: dict[str, JsonValue] = {}
        names = {
            "AUTOCLIP_OUTPUT_DIR": "output_dir",
            "AUTOCLIP_TRANSCRIPT": "transcript",
            "AUTOCLIP_COOKIES": "cookies",
            "AUTOCLIP_PROXY": "proxy",
            "AUTOCLIP_LOG_LEVEL": "log_level",
            "AUTOCLIP_LLM_URL": "llm_base_url",
            "AUTOCLIP_LLM_MODEL": "llm_model",
            "AUTOCLIP_MODE": "mode",
            "AUTOCLIP_QUALITY": "quality",
        }
        for variable, field_name in names.items():
            value = env.get(variable, "").strip()
            if value:
                data[field_name] = value
        skips = [name.strip() for name in env.get("AUTOCLIP_SKIP", "").split(",") if name.strip()]
        if skips:
            data["skip"] = skips  # type: ignore[assignment]
        try:
            return parse(cls, data)
        except ConfigError as e:
            log.warning("config: ignoring invalid environment overrides: %s", e)
            return cls()

    @classmethod
    def combined(cls, explicit: RuntimeOverrides | None, env: Mapping[str, str] | None = None) -> RuntimeOverrides:
        """Environment values first, then explicit overrides field by field (an explicit value always wins)."""
        base = cls.from_env(env)
        if explicit is None:
            return base
        data = asdict(base)
        for name, value in asdict(explicit).items():
            if value is not None:
                data[name] = value
        return parse(cls, data)


#: Which kinds a preset renders.
PRESET_KINDS: dict[str, tuple[str, ...]] = {
    "both": ("short", "long"),
    "shorts": ("short",),
    "long": ("long",),
}


@dataclass(config=SCHEMA)
class FileConfig(Section):
    """The whole configuration file: seven keys, everything else is derived at run time."""

    @model_validator(mode="before")
    @classmethod
    def _drop_removed(cls, data: JsonValue) -> JsonValue:
        """An old config file still loads: removed keys warn and are ignored."""
        return _strip_removed(data)

    input: list[str] = field(default_factory=lambda: ["input.mp4"])  # files, glob patterns or http(s) URLs
    clips: FileClipsCfg = field(default_factory=FileClipsCfg)
    download: FileDownloadCfg = field(default_factory=FileDownloadCfg)

    def to_config(self, overrides: RuntimeOverrides | None = None) -> Config:
        """Derive the complete runtime configuration from these few choices.

        Raises:
            ConfigError: if ``--split`` is combined with explicit ``clips.ranges``.
        """
        ov = overrides or RuntimeOverrides()
        skip = SkipCfg()
        for name in ov.skip or ():
            setattr(skip, name, True)
        ranges = list(self.clips.ranges)
        if ov.split and ranges:
            raise ConfigError("--split cannot be combined with clips.ranges")
        mode: Literal["highlights", "ranges", "split"] = "split" if ov.split else ("ranges" if ranges else "highlights")
        short = KindCfg(count=5, min_seconds=20, max_seconds=60, aspect="9:16", ranges=ranges)
        long = KindCfg(count=2, min_seconds=180, max_seconds=600, aspect="16:9", ranges=ranges)
        clips = ClipsCfg(
            mode=mode,
            kinds=list(PRESET_KINDS[self.clips.preset]),  # type: ignore[arg-type]
            scenes=self.clips.scenes,
            short=short,
            long=long,
            keywords=list(self.clips.keywords),
        )
        download = DownloadCfg(
            cookies=ov.cookies,
            proxy=ov.proxy,
            platforms=PlatformsCfg(allow=list(self.download.platforms.allow), deny=list(self.download.platforms.deny)),
        )
        return Config(
            input=list(self.input if ov.input is None else ov.input),
            output_dir=ov.output_dir or _default_output_dir(),
            mode=ov.mode or "balance",
            quality=ov.quality or "high",
            download=download,
            clips=clips,
            transcript_file=ov.transcript,
            llm=LlmCfg(base_url=ov.llm_base_url or "http://localhost:11434", model=ov.llm_model or "llama3.2"),
            video=VideoCfg(),
            captions=CaptionsCfg(enabled=not skip.caption),
            title=TitleCfg(enabled=not skip.title),
            skip=skip,
            logging=LoggingCfg(level=ov.log_level or "INFO"),
        )


def _field_default(cls: type, name: str) -> JsonValue:
    """The default value of a dataclass field, evaluating factories lazily."""
    field = getattr(cls, "__dataclass_fields__", {}).get(name)
    if field is None:
        return None  # pragma: no cover
    from dataclasses import MISSING

    if field.default is not MISSING:
        import copy

        try:
            return copy.deepcopy(field.default)  # type: ignore[no-any-return]
        except Exception:  # pragma: no cover
            return field.default  # type: ignore[no-any-return]  # pragma: no cover
    if field.default_factory is not MISSING:
        factory = field.default_factory
        try:
            return factory()  # type: ignore[no-any-return]
        except TypeError:  # pragma: no cover
            return factory  # type: ignore[no-any-return]  # pragma: no cover
    return None  # pragma: no cover


def _coerce_defaults(cls: type, data: JsonValue, prefix: str = "") -> JsonValue:
    """Replace values that fail validation with the field defaults, recursively."""
    if not isinstance(data, dict):
        return data
    try:
        hints = typing.get_type_hints(cls, globalns=globals())
    except Exception:  # pragma: no cover
        return data  # pragma: no cover
    result = dict(data)
    for name in list(result):
        fields = getattr(cls, "__dataclass_fields__", {})
        if name not in fields:
            continue  # keep unknown keys so they still raise "extra_forbidden"
        hint = hints.get(name)
        if hint is None:
            continue  # pragma: no cover
        path = f"{prefix}.{name}" if prefix else name
        value = result[name]
        # try nested coercion first
        try:
            origin = typing.get_origin(hint)
            target = origin if origin is not None else hint
            if isinstance(target, type) and issubclass(target, Section) and isinstance(value, dict):
                value = _coerce_defaults(target, value, path)
                result[name] = value
        except Exception as _exc:  # pragma: no cover
            _ = str(_exc)
        try:
            TypeAdapter(hint).validate_python(value)
        except Exception:
            log.warning("config: invalid value for %s; falling back to default", path)
            result[name] = _field_default(cls, name)
    return result


T = TypeVar("T")
#: Removed keys already warned about in the current ``parse`` call (each path warns once).
_removed_seen: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar("autoclip_removed_keys", default=None)


def parse(cls: type[T], data: dict[str, JsonValue]) -> T:
    """Validate ``data`` against the schema of the config dataclass ``cls``.

    Unknown keys, wrong types and out-of-range values would normally raise ``ConfigError``; this loader
    now falls back to each field's default for values that do not fit the schema, while still rejecting
    genuinely unknown keys. Nested sections are coerced field by field.

    Raises:
        ConfigError: if the data contains unknown keys or still cannot be validated after coercion.
    """
    if isinstance(data, dict):
        data = {k: v for k, v in data.items() if not (isinstance(k, str) and k.startswith("_"))}
    adapter: TypeAdapter[T] = TypeAdapter(cls)
    token = _removed_seen.set(set()) if _removed_seen.get() is None else None
    try:
        try:
            return adapter.validate_python(data)
        except ValidationError as e:
            if any(
                str(err.get("type", "")).endswith(("extra_forbidden", "unexpected_keyword_argument"))
                or "Unexpected keyword" in str(err.get("msg", ""))
                for err in e.errors()
            ):
                raise _config_error(e) from e
            coerced = _coerce_defaults(cls, data) if isinstance(data, dict) else data
            try:
                return adapter.validate_python(coerced)
            except ValidationError as e2:
                raise _config_error(e2) from e2
    finally:
        if token is not None:
            _removed_seen.reset(token)


def _config_error(exc: ValidationError) -> ConfigError:
    """The validation failures as one readable message (``clips.short.count: Input should be ...``)."""
    return ConfigError("; ".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()))


def _abs(base: Path, value: str) -> str:
    """``value`` as an absolute path string (relative paths are taken from the config's folder)."""
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else base / p)


def _cookie_source(base: Path, value: str) -> str:
    """A browser name stays a browser name; anything else is a cookies.txt path from the config's folder."""
    return value if value in COOKIE_BROWSERS else _abs(base, value)


def is_url(value: str) -> bool:
    """True for an http(s) URL: an input item to download, never a path to resolve or glob."""
    try:
        parts = urlsplit(value)
    except ValueError:
        return False
    return parts.scheme.lower() in ("http", "https") and bool(parts.netloc)


def expand_inputs(patterns: list[str], base: Path) -> list[str]:
    """Resolve local files / glob patterns to absolute paths and keep URLs verbatim.

    Raises:
        ConfigError: if a local pattern matches nothing or two local videos share a file name (they would share outputs).
    """
    items: dict[str, None] = {}
    for pattern in patterns:
        if is_url(pattern):
            items[pattern] = None
            continue
        full = _abs(base, pattern)
        hits = [full] if Path(full).is_file() else sorted(p for p in glob(full) if Path(p).is_file())
        if not hits:
            raise ConfigError(f"input '{pattern}' matches no file or URL")
        items.update(dict.fromkeys(hits))
    stems = [Path(item).stem for item in items if not is_url(item)]
    if len(set(stems)) != len(stems):
        raise ConfigError("two input videos share the same file name; rename one (outputs are keyed by name)")
    return list(items)


def load_config(path: str | Path, overrides: RuntimeOverrides | None = None) -> Config:
    """Read, validate and resolve a JSON config file, then derive the complete runtime configuration.

    All relative paths are resolved against the config file's folder, and ``input`` becomes the final
    list of absolute video paths (a plain string is accepted for a single video). ``overrides`` (and the
    ``AUTOCLIP_*`` environment variables) replace the removed per-run keys.
    """
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"cannot read config '{p}': {e.strerror or e}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"{p} is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{p} must contain a JSON object")
    if isinstance(raw.get("input"), str):
        raw["input"] = [raw["input"]]
    base = p.resolve().parent
    file_cfg = parse(FileConfig, raw)
    cfg = file_cfg.to_config(RuntimeOverrides.combined(overrides, os.environ))
    cfg.input = expand_inputs(cfg.input, base)
    cfg.output_dir = _abs(Path.cwd(), cfg.output_dir)  # independent of where the config file lives
    if cfg.transcript_file:
        cfg.transcript_file = _abs(base, cfg.transcript_file)
        if len(cfg.input) > 1 and "{stem}" not in cfg.transcript_file:
            raise ConfigError("with several videos transcript_file must contain '{stem}' (e.g. 'subs/{stem}.srt')")
    if cfg.download.cookies:
        cfg.download.cookies = _cookie_source(base, cfg.download.cookies)
    cfg.video.background = resolve_background(cfg.video.background, base)
    load_lexicon(base)  # validate <config-dir>/lexicon.json early; moments reloads it at run time
    return cfg
