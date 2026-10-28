"""Typed configuration: JSON -> validated dataclasses (Pydantic owns the schema).

Every option lives in exactly one place - its dataclass field - so defaults, ``--init-full`` output and
validation can never drift apart. ``load_config`` also resolves all paths relative to the config file
and expands ``input`` globs into the list of videos to process.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, field
from glob import glob
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, Strict, StringConstraints, TypeAdapter, ValidationError, model_validator
from pydantic.dataclasses import dataclass

from .errors import ConfigError
from .platforms import BUILTIN_GROUPS, canonical, deep_merge
from .styles import STYLES

Level = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
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
    def _drop_comments(cls, data: Any) -> Any:
        """Comments in the config file are keys starting with an underscore - they are not options."""
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if not k.startswith("_")}
        return data


@dataclass(config=SCHEMA)
class KindCfg(Section):
    """One kind of output clip ("short" or "long"): the length range and the recommended length."""

    count: Annotated[Int, Field(ge=1)] = 5  # clips per scene type
    min_seconds: Annotated[Float, Field(gt=0)] = 20  # a clip is never shorter than this
    target_seconds: Float | None = None  # the recommended length (a preference, not a limit)
    max_seconds: Annotated[Float, Field(gt=0)] = 60  # a clip is never longer than this
    aspect: Aspect = "9:16"  # 9:16 | 1:1 | 4:5 | 16:9 | original
    ranges: list[Range] = field(default_factory=list)  # clips.mode = "ranges", e.g. ["00:01:00-00:01:45"]

    @model_validator(mode="after")
    def _lengths_fit(self) -> KindCfg:
        """The three lengths have to make sense together."""
        if self.min_seconds > self.max_seconds:
            raise ConfigError("min_seconds must be <= max_seconds")
        if self.target_seconds is not None and not self.min_seconds <= self.target_seconds <= self.max_seconds:
            raise ConfigError("target_seconds must be between min_seconds and max_seconds")
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
    lexicon_file: str | None = None  # JSON {"funny": [...], ...} extending the built-in English lexicons
    skip_intro_seconds: Float = 0
    snap_to_cuts: Bool = False  # snap cut points to shot changes (needs scenedetect)
    pad_start: Float = 0.15
    pad_end: Float = 0.30
    split_seconds: Annotated[Float, Field(gt=0)] = 600  # mode=split

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
class EnhanceCfg(Section):
    """Speech enhancement applied to the audio before it is transcribed."""

    enabled: Bool = True  # denoise the audio so the ASR model hears clear words
    backend: Literal["sherpa", "noisereduce"] = "sherpa"  # ONNX denoiser or spectral gating
    model: Literal["gtcrn", "dpdfnet_baseline", "dpdfnet2", "dpdfnet4", "dpdfnet8"] = "gtcrn"
    attenuation_limit_db: Annotated[Float, Field(ge=0, le=100)] = 12.0  # how much noise may be removed (0 = off, 100 = no suppression)
    denoise_strength: Annotated[Float, Field(ge=0, le=1)] = 0.75  # only used by the noisereduce backend
    snr_threshold_db: Annotated[Float, Field(gt=0)] | None = 20.0  # only enhance below this speech-to-noise ratio (null = always)
    num_threads: Annotated[Int, Field(ge=1)] = 4
    model_dir: str | None = None  # where the ONNX model is kept; default <output_dir>/.work/models

    def fingerprint(self) -> dict[str, Any]:
        """The settings that change the transcript (part of the on-disk cache key)."""
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "model": self.model,
            "attenuation_limit_db": self.attenuation_limit_db,
            "denoise_strength": self.denoise_strength,
            "snr_threshold_db": self.snr_threshold_db,
        }


@dataclass(config=SCHEMA)
class TranscriptCfg(Section):
    """Where the transcript comes from."""

    mode: Literal["auto", "whisper", "file", "off"] = "auto"
    file: str | None = None  # .srt/.json; '{stem}' is replaced by the video's file name without extension
    cache: Bool = True  # reuse <video_dir>/<name>.transcript.json while input, model and language are unchanged
    model: str = "large-v3"  # tiny base small medium large-v3 (multilingual, NOT *.en)
    auto_downgrade: Bool = True  # pick a smaller model when free RAM cannot hold the chosen one
    language: str | None = None  # None = auto-detect
    multilingual: Bool = True  # detect the language per segment (mixed-language audio)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    beam_size: Int = 5  # 1 = fastest, 5 = most accurate
    word_timestamps: Bool = True
    enhance: EnhanceCfg = field(default_factory=EnhanceCfg)


@dataclass(config=SCHEMA)
class LlmCfg(Section):
    """Optional local re-ranking (Ollama or any OpenAI-compatible server)."""

    enabled: Bool = False
    base_url: Url = "http://localhost:11434"
    model: str = "llama3.2"
    api_key: str | None = field(default=None, repr=False)
    top_k: Int = 12
    timeout: Float = 180


@dataclass(config=SCHEMA)
class BackgroundCfg(Section):
    """Background behind a letter-boxed video."""

    type: Literal["blur", "color", "image"] = "blur"
    blur: Float = 28
    dim: Float = 0.35  # 0 = untouched, 1 = black
    color: str = "#101010"
    image: str | None = None


@dataclass(config=SCHEMA)
class VideoCfg(Section):
    """Output encoding (the canvas shape comes from each clip kind's ``aspect``)."""

    short_side: Int = 1080  # 1080 -> 1080x1920 for 9:16 and 1920x1080 for 16:9
    upscale: Bool = False  # False = never enlarge the source (a 720p source stays 720p)
    fps: Float | None = None  # None = source fps
    layout: Literal["fit", "fill"] = "fit"  # fit = whole frame on a background, fill = crop to fill
    background: BackgroundCfg = field(default_factory=BackgroundCfg)
    fade_seconds: Float = 0.2
    codec: Literal["h264", "h265"] = "h264"
    gpu: Literal["auto", "off", "nvenc", "videotoolbox", "qsv", "amf"] = "auto"  # GPU encoder if one really works
    crf: Int = 18  # quality (CPU: libx264/libx265 CRF; GPU: the encoder's quality level)
    preset: str = "veryslow"  # libx264/libx265 only
    maxrate_mbps: Float = 0  # bitrate cap (upload friendly); 0 = uncapped
    audio_bitrate_kbps: Int = 192  # AAC bitrate of the clip audio
    container: Literal["mp4", "mkv"] = "mp4"  # mkv = no faststart; needed for some H.265/audio combinations
    thumbnail: Bool = True  # also write <clip>.jpg


@dataclass(config=SCHEMA)
class CaptionsCfg(Section):
    """Burned-in captions."""

    enabled: Bool = True
    font: str = "auto"  # "auto" = a system font covering the script, or a font file / name
    size: Int | None = None  # None = derived from the resolution
    color: str = "#FFFFFF"
    bg_color: str = "#000000"
    bg_opacity: Float = 0.6  # 0 = no box
    stroke_width: Int = 0
    stroke_color: str = "#000000"
    highlight: Bool = True  # colour the spoken word (spaced left-to-right scripts only)
    highlight_color: str = "#FFD400"
    position: Position = "lower"
    safe_top: Float = 0.10  # fraction of a vertical frame kept free of captions (app UI)
    safe_bottom: Float = 0.20
    max_chars: Int = 24  # characters per caption screen
    max_seconds: Float = 2.6
    uppercase: Bool = False


@dataclass(config=SCHEMA)
class TitleCfg(Section):
    """Optional hook title shown at the start of each clip."""

    enabled: Bool = False
    seconds: Float = 3.0
    position: Position = "top"


@dataclass(config=SCHEMA)
class AudioCfg(Section):
    """Speech clean-up."""

    enhance: Bool = True
    denoise: Literal["auto", "noisereduce", "ffmpeg", "none"] = "auto"
    denoise_strength: Float = 0.75
    highpass_hz: Float = 80
    target_dbfs: Float = -16  # speech loudness target (about -14 LUFS)
    fade_seconds: Float = 0.12


@dataclass(config=SCHEMA)
class SilenceCfg(Section):
    """Pause and filler-word removal."""

    remove: Bool = True
    min_silence: Float = 0.5  # pauses longer than this are shortened
    keep_padding: Float = 0.12  # seconds of pause kept on each side
    threshold_db: Float | Literal["auto"] = "auto"
    remove_fillers: Bool = True  # needs word timestamps
    fillers: list[str] = field(default_factory=lambda: ["um", "uh", "umm", "uhh", "uhm", "erm", "hmm", "mmm"])


@dataclass(config=SCHEMA)
class ProcessingCfg(Section):
    """Keeping memory flat on very long videos."""

    chunk_threshold_minutes: Annotated[Float, Field(ge=0)] = 0  # 0 = always transcribe in chunks; N = only videos longer than N minutes
    chunk_minutes: Annotated[Float, Field(gt=0)] = 5  # chunk length; boundaries snap to the quietest moment nearby


@dataclass(config=SCHEMA)
class ResourcesCfg(Section):
    """Adaptive CPU / RAM balancing (re-evaluated before and after every operation)."""

    cpus: Int = 0  # 0 = auto-detect (affinity + container quota)
    max_cpu_percent: Float = 100  # share of the cores the script may use
    max_workers: Int = 0  # 0 = as many parallel renders as cores and RAM allow; 1 = sequential
    memory_fraction: Float = 0.75  # max share of the *currently free* RAM the script may claim
    memory_reserve_mb: Int = 512  # RAM always left to the OS and other programs


@dataclass(config=SCHEMA)
class StorageCfg(Section):
    """Cache sizes (the SQLite database itself is recreated empty for every video)."""

    cache_memory_mb: Int = 0  # 0 = auto (sized from free RAM)
    cache_disk_mb: Int = 2048  # cap for cached blobs inside the SQLite file


@dataclass(config=SCHEMA)
class LoggingCfg(Section):
    """Log file and console verbosity."""

    enabled: Bool = True
    level: Level = "INFO"  # written to the log file; DEBUG logs every small step
    console_level: Level = "INFO"
    file: str | None = None  # None = <output_dir>/logs/autoclip_<timestamp>.log


@dataclass(config=SCHEMA)
class SkipCfg(Section):
    """Leave individual pipeline steps out (nothing is skipped by default)."""

    stt: Bool = False  # no Whisper; an explicit, downloaded or cached transcript is still used
    denoise: Bool = False  # no speech enhancement before Whisper and no noise reduction of the clip audio
    audio_enhance: Bool = False  # keep the raw audio (no high-pass, loudness normalisation or limiter)
    audio: Bool = False  # mute the clips (and skip audio decoding, so pauses cannot be removed)
    pauses: Bool = False  # keep pauses and filler words in the clip audio
    caption: Bool = False  # no burned-in captions
    title: Bool = False  # no hook title
    fade: Bool = False  # no video or audio fades
    thumbnail: Bool = False  # no .jpg cover next to each clip
    llm: Bool = False  # no LLM re-ranking (even when llm.enabled is true)
    snap: Bool = False  # no snapping of cut points to shot changes


@dataclass(config=SCHEMA)
class ComputeCfg(Section):
    """Where the heavy work runs."""

    device: Literal["cpu", "gpu", "auto"] = "auto"  # "cpu" never touches the GPU, "gpu" prefers it


@dataclass(config=SCHEMA)
class MetadataCfg(Section):
    """Extra files saved next to a downloaded original."""

    info_json: Bool = True  # <title> [<id>].info.json (the metadata yt-dlp found)
    thumbnail: Bool = False  # <title> [<id>].webp/.jpg
    description: Bool = False  # <title> [<id>].description


@dataclass(config=SCHEMA)
class SubtitlesCfg(Section):
    """Subtitles downloaded from the platform."""

    enabled: Bool = False  # download the subtitles the platform offers
    languages: list[str] = field(default_factory=lambda: ["en"])  # language codes, e.g. ["en", "ar"]
    use_as_transcript: Literal["never", "manual", "always"] = "never"
    # never = keep the files only; manual = human-made subtitles may replace Whisper; always = auto-generated too

    @model_validator(mode="after")
    def _languages_when_enabled(self) -> SubtitlesCfg:
        """A download with no language has nothing to request, and was never asked for."""
        if self.enabled and not self.languages:
            raise ConfigError("download.subtitles.languages must not be empty when subtitles are enabled")
        if self.use_as_transcript != "never" and not self.enabled:
            raise ConfigError("download.subtitles.use_as_transcript needs download.subtitles.enabled: true")
        return self


@dataclass(config=SCHEMA)
class CookiesCfg(Section):
    """Where yt-dlp takes cookies from (some platforms need a signed-in session)."""

    from_browser: str | None = None  # chrome, firefox, edge, ... (the browser's cookie store is read locally)
    file: str | None = None  # a Netscape-format cookies.txt

    @model_validator(mode="after")
    def _one_source(self) -> CookiesCfg:
        """One cookie source is enough and two would be ambiguous."""
        if self.from_browser and self.file:
            raise ConfigError("download.cookies: set either from_browser or file, not both")
        return self


@dataclass(config=SCHEMA)
class NetworkCfg(Section):
    """Network behaviour of the downloader."""

    proxy: Proxy | None = None
    rate_limit_mbps: Annotated[Float, Field(gt=0)] | None = None  # None = no cap
    retries: Annotated[Int, Field(ge=0, le=20)] = 3
    socket_timeout: Annotated[Float, Field(gt=0, le=300)] = 20
    sleep_between_seconds: Annotated[Float, Field(ge=0, le=60)] = 0  # pause between playlist entries


@dataclass(config=SCHEMA)
class JsRuntimeCfg(Section):
    """JavaScript runtime for YouTube's signature challenges (yt-dlp's EJS support)."""

    name: Literal["auto", "deno", "node", "bun", "quickjs"] = "auto"  # auto = find one on PATH
    path: str | None = None  # an explicit binary; only with a named runtime
    remote_components: Literal["none", "ejs"] = "none"  # "ejs" lets yt-dlp fetch the EJS challenge solver

    @model_validator(mode="after")
    def _path_needs_a_name(self) -> JsRuntimeCfg:
        """``auto`` finds a runtime itself, so a path would be ignored."""
        if self.path and self.name == "auto":
            raise ConfigError("download.js_runtime.path needs download.js_runtime.name (auto cannot use a path)")
        return self


@dataclass(config=SCHEMA)
class PlatformsCfg(Section):
    """Which platforms may be downloaded, and their per-platform overrides."""

    allow: list[str] = field(default_factory=lambda: ["@popular"])  # families or @groups that may be used
    deny: list[str] = field(default_factory=list)  # families or @groups that never may (deny wins)
    groups: dict[str, list[str]] = field(default_factory=dict)  # user groups: {"@news": ["youtube", ...]}
    settings: dict[str, dict[str, Any]] = field(default_factory=dict)  # partial download settings per family/@group
    group_by_platform: Bool = False  # download to .work/downloads/<platform>/...

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

    quality: Annotated[Int, Field(ge=144, le=4320)] | Literal["best", "worst"] = "best"  # short side, or best/worst
    fps: Annotated[Float, Field(gt=0, le=240)] | None = None  # None = any
    prefer: Literal["closest", "at_most", "at_least"] = "closest"  # how quality/fps are matched
    codec: Literal["auto", "h264", "vp9", "av1", "h265"] = "auto"  # auto = h264, then vp9, av1, h265, other
    container: Literal["auto", "mp4", "mkv", "webm"] = "auto"  # auto = the format's own container
    audio_quality: Literal["best", "worst"] = "best"
    audio_language: str | None = None  # e.g. "en"; None = the platform's default track
    original_dir: str | None = None  # None = <output_dir>/.work/downloads
    metadata: MetadataCfg = field(default_factory=MetadataCfg)
    subtitles: SubtitlesCfg = field(default_factory=SubtitlesCfg)
    playlist: Literal["single", "all"] = "single"  # single = the one video; all = every entry, up to max_videos
    max_videos: Annotated[Int, Field(ge=1)] = 50
    time_range: Range | None = None  # "MM:SS-MM:SS" / "HH:MM:SS-HH:MM:SS"
    cookies: CookiesCfg = field(default_factory=CookiesCfg)
    network: NetworkCfg = field(default_factory=NetworkCfg)
    js_runtime: JsRuntimeCfg = field(default_factory=JsRuntimeCfg)
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
        base.pop("js_runtime")
        for name, entry in self.platforms.settings.items():
            if "platforms" in entry or "js_runtime" in entry:
                raise ConfigError(f"download.platforms.settings.{name}: platforms and js_runtime cannot be overridden per platform")
            try:
                parse(DownloadCfg, deep_merge(base, entry))
            except ConfigError as e:
                raise ConfigError(f"download.platforms.settings.{name}: {e}") from e
        return self


@dataclass(config=SCHEMA)
class Config(Section):
    """The whole configuration file."""

    input: list[str] = field(default_factory=lambda: ["input.mp4"])  # files, glob patterns or http(s) URLs
    output_dir: str = "clips"  # one sub-folder per video is created inside
    compute: ComputeCfg = field(default_factory=ComputeCfg)
    download: DownloadCfg = field(default_factory=DownloadCfg)
    clips: ClipsCfg = field(default_factory=ClipsCfg)
    transcript: TranscriptCfg = field(default_factory=TranscriptCfg)
    llm: LlmCfg = field(default_factory=LlmCfg)
    video: VideoCfg = field(default_factory=VideoCfg)
    captions: CaptionsCfg = field(default_factory=CaptionsCfg)
    title: TitleCfg = field(default_factory=TitleCfg)
    audio: AudioCfg = field(default_factory=AudioCfg)
    silence: SilenceCfg = field(default_factory=SilenceCfg)
    skip: SkipCfg = field(default_factory=SkipCfg)
    processing: ProcessingCfg = field(default_factory=ProcessingCfg)
    resources: ResourcesCfg = field(default_factory=ResourcesCfg)
    storage: StorageCfg = field(default_factory=StorageCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)

    @model_validator(mode="after")
    def _skip_is_possible(self) -> Config:
        """Skipping Whisper while forcing it would leave nothing to transcribe with."""
        if self.skip.stt and self.transcript.mode == "whisper":
            raise ConfigError("skip.stt cannot be combined with transcript.mode: 'whisper' (nothing would transcribe)")
        return self


T = TypeVar("T")


def parse(cls: type[T], data: dict[str, Any]) -> T:
    """Validate ``data`` against the schema of the config dataclass ``cls``.

    Pydantic does the work: unknown keys, wrong types and out-of-range values raise ``ConfigError``.

    Raises:
        ConfigError: if the data does not fit the schema.
    """
    adapter: TypeAdapter[T] = TypeAdapter(cls)
    try:
        return adapter.validate_python(data)
    except ValidationError as e:
        raise _config_error(e) from e


def _config_error(exc: ValidationError) -> ConfigError:
    """The validation failures as one readable message (``clips.short.count: Input should be ...``)."""
    return ConfigError("; ".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()))


def _abs(base: Path, value: str) -> str:
    """``value`` as an absolute path string (relative paths are taken from the config's folder)."""
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else base / p)


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


def load_config(path: str | Path) -> Config:
    """Read, validate and resolve a JSON config file.

    All relative paths are resolved against the config file's folder, and ``input`` becomes the final
    list of absolute video paths (a plain string is accepted for a single video).
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
    cfg = parse(Config, raw)
    base = p.resolve().parent
    cfg.input = expand_inputs(cfg.input, base)
    cfg.output_dir = _abs(base, cfg.output_dir)
    if cfg.logging.file:
        cfg.logging.file = _abs(base, cfg.logging.file)
    if cfg.transcript.file:
        cfg.transcript.file = _abs(base, cfg.transcript.file)
        if len(cfg.input) > 1 and "{stem}" not in cfg.transcript.file:
            raise ConfigError("with several videos transcript.file must contain '{stem}' (e.g. 'subs/{stem}.srt')")
    if cfg.transcript.enhance.model_dir:
        cfg.transcript.enhance.model_dir = _abs(base, cfg.transcript.enhance.model_dir)
    if cfg.clips.lexicon_file:
        cfg.clips.lexicon_file = _abs(base, cfg.clips.lexicon_file)
    if cfg.video.background.image:
        cfg.video.background.image = _abs(base, cfg.video.background.image)
    if cfg.download.original_dir:
        cfg.download.original_dir = _abs(base, cfg.download.original_dir)
    if cfg.download.cookies.file:
        cfg.download.cookies.file = _abs(base, cfg.download.cookies.file)
    if cfg.download.js_runtime.path:
        cfg.download.js_runtime.path = _abs(base, cfg.download.js_runtime.path)
    for entry in cfg.download.platforms.settings.values():
        if entry.get("original_dir"):
            entry["original_dir"] = _abs(base, entry["original_dir"])
        cookies = entry.get("cookies")
        if isinstance(cookies, dict) and cookies.get("file"):
            cookies["file"] = _abs(base, cookies["file"])
    return cfg
