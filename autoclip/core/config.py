"""Typed configuration: JSON -> validated dataclasses (Pydantic owns the schema).

Every option lives in exactly one place - its dataclass field - so defaults, ``--init-full`` output and
validation can never drift apart. ``load_config`` also resolves all paths relative to the config file
and expands ``input`` globs into the list of videos to process.
"""

from __future__ import annotations

import json
from dataclasses import field
from glob import glob
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from pydantic import ConfigDict, Field, Strict, StringConstraints, TypeAdapter, ValidationError, model_validator
from pydantic.dataclasses import dataclass

from .errors import ConfigError
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
    multilingual: Bool = False  # detect the language per segment (mixed-language audio)
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
    fps: Float | None = 30  # None = source fps
    layout: Literal["fit", "fill"] = "fit"  # fit = whole frame on a background, fill = crop to fill
    background: BackgroundCfg = field(default_factory=BackgroundCfg)
    fade_seconds: Float = 0.2
    gpu: Literal["auto", "off", "nvenc", "videotoolbox", "qsv", "amf"] = "auto"  # GPU encoder if one really works
    crf: Int = 20  # quality (CPU: libx264 CRF; GPU: the encoder's quality level)
    preset: str = "veryfast"  # libx264 only
    maxrate_mbps: Float = 12  # bitrate cap (upload friendly); 0 = uncapped
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
class ComputeCfg(Section):
    """Where the heavy work runs."""

    device: Literal["cpu", "gpu", "auto"] = "auto"  # "cpu" never touches the GPU, "gpu" prefers it


@dataclass(config=SCHEMA)
class Config(Section):
    """The whole configuration file."""

    input: list[str] = field(default_factory=lambda: ["video.mp4"])  # files or glob patterns
    output_dir: str = "clips"  # one sub-folder per video is created inside
    compute: ComputeCfg = field(default_factory=ComputeCfg)
    clips: ClipsCfg = field(default_factory=ClipsCfg)
    transcript: TranscriptCfg = field(default_factory=TranscriptCfg)
    llm: LlmCfg = field(default_factory=LlmCfg)
    video: VideoCfg = field(default_factory=VideoCfg)
    captions: CaptionsCfg = field(default_factory=CaptionsCfg)
    title: TitleCfg = field(default_factory=TitleCfg)
    audio: AudioCfg = field(default_factory=AudioCfg)
    silence: SilenceCfg = field(default_factory=SilenceCfg)
    processing: ProcessingCfg = field(default_factory=ProcessingCfg)
    resources: ResourcesCfg = field(default_factory=ResourcesCfg)
    storage: StorageCfg = field(default_factory=StorageCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)


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


def expand_inputs(patterns: list[str], base: Path) -> list[str]:
    """Resolve files / glob patterns to a sorted, de-duplicated list of absolute video paths.

    Raises:
        ConfigError: if a pattern matches nothing or two videos share a file name (they would share outputs).
    """
    videos: dict[str, None] = {}
    for pattern in patterns:
        full = _abs(base, pattern)
        hits = [full] if Path(full).is_file() else sorted(p for p in glob(full) if Path(p).is_file())
        if not hits:
            raise ConfigError(f"input '{pattern}' matches no file")
        videos.update(dict.fromkeys(hits))
    stems = [Path(v).stem for v in videos]
    if len(set(stems)) != len(stems):
        raise ConfigError("two input videos share the same file name; rename one (outputs are keyed by name)")
    return list(videos)


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
    return cfg
