"""Typed configuration: JSON -> nested dataclasses (strict keys, type-checked values, validated).

Every option lives in exactly one place - its dataclass field - so defaults, ``--init-full`` output and
validation can never drift apart. ``load_config`` also resolves all paths relative to the config file
and expands ``input`` globs into the list of videos to process.
"""

from __future__ import annotations

import json
import re
import types
from dataclasses import dataclass, field, is_dataclass
from glob import glob
from pathlib import Path
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints
from urllib.parse import urlparse

from .errors import ConfigError
from .styles import STYLES

Level = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
Position = Literal["top", "center", "lower", "bottom"]


@dataclass
class KindCfg:
    """One kind of output clip ("short" or "long"): the length range and the recommended length."""

    count: int = 5  # clips per scene type
    min_seconds: float = 20  # a clip is never shorter than this
    target_seconds: float | None = None  # the recommended length (a preference, not a limit)
    max_seconds: float = 60  # a clip is never longer than this
    aspect: str = "9:16"  # 9:16 | 1:1 | 4:5 | 16:9 | original
    ranges: list[str] = field(default_factory=list)  # clips.mode = "ranges", e.g. ["00:01:00-00:01:45"]

    def __post_init__(self) -> None:
        if not 0 < self.min_seconds <= self.max_seconds:
            raise ConfigError("min_seconds must be > 0 and <= max_seconds")
        if self.target_seconds is not None and not self.min_seconds <= self.target_seconds <= self.max_seconds:
            raise ConfigError("target_seconds must be between min_seconds and max_seconds")
        if self.count < 1:
            raise ConfigError("count must be at least 1")
        if self.aspect != "original" and not re.fullmatch(r"\d+(\.\d+)?:\d+(\.\d+)?", self.aspect):
            raise ConfigError("aspect must look like 9:16, 1:1, 4:5, 16:9 or 'original'")
        if not all(re.fullmatch(r"[\d:.]+-[\d:.]+", r) for r in self.ranges):
            raise ConfigError("ranges entries must look like 'HH:MM:SS-HH:MM:SS'")


@dataclass
class ClipsCfg:
    """Which moments to cut."""

    mode: Literal["highlights", "ranges", "split"] = "highlights"
    kinds: list[Literal["short", "long"]] = field(default_factory=lambda: ["short", "long"])
    scenes: str | list[str] = "best"  # "all", one scene type or a list; each type gets its own clips
    short: KindCfg = field(default_factory=KindCfg)
    long: KindCfg = field(default_factory=lambda: KindCfg(count=2, min_seconds=180, max_seconds=600, aspect="16:9"))
    keywords: list[str] = field(default_factory=list)  # extra words/phrases (any language) that boost a moment
    lexicon_file: str | None = None  # JSON {"funny": [...], ...} extending the built-in English lexicons
    skip_intro_seconds: float = 0
    snap_to_cuts: bool = False  # snap cut points to shot changes (needs scenedetect)
    pad_start: float = 0.15
    pad_end: float = 0.30
    split_seconds: float = 600  # mode=split

    def scene_list(self) -> list[str]:
        """The scene types to generate, in order (``"all"`` expands to every type)."""
        if self.scenes == "all":
            return list(STYLES)
        return [self.scenes] if isinstance(self.scenes, str) else self.scenes

    def kind(self, name: str) -> KindCfg:
        """Settings of the ``"short"`` or ``"long"`` kind."""
        return self.short if name == "short" else self.long

    def __post_init__(self) -> None:
        bad = [s for s in self.scene_list() if s not in STYLES]
        if bad:
            raise ConfigError(f"unknown clips.scenes {bad}; valid: 'all' or {', '.join(STYLES)}")
        if not self.kinds or len(set(self.kinds)) != len(self.kinds):
            raise ConfigError("clips.kinds must list 'short' and/or 'long' once each")
        if self.split_seconds <= 0:
            raise ConfigError("clips.split_seconds must be positive")


@dataclass
class EnhanceCfg:
    """Speech enhancement applied to the audio before it is transcribed."""

    enabled: bool = True  # denoise the audio so the ASR model hears clear words
    backend: Literal["sherpa", "noisereduce"] = "sherpa"  # ONNX denoiser or spectral gating
    model: Literal["gtcrn", "dpdfnet_baseline", "dpdfnet2", "dpdfnet4", "dpdfnet8"] = "gtcrn"
    attenuation_limit_db: float = 12.0  # how much noise may be removed (0 = off, 100 = no suppression)
    denoise_strength: float = 0.75  # only used by the noisereduce backend
    snr_threshold_db: float | None = 20.0  # only enhance below this speech-to-noise ratio (null = always)
    num_threads: int = 4
    model_dir: str | None = None  # where the ONNX model is kept; default <output_dir>/.work/models

    def __post_init__(self) -> None:
        if not 0 <= self.attenuation_limit_db <= 100:
            raise ConfigError("attenuation_limit_db must be between 0 and 100")
        if not 0 <= self.denoise_strength <= 1:
            raise ConfigError("denoise_strength must be between 0 and 1")
        if self.snr_threshold_db is not None and self.snr_threshold_db <= 0:
            raise ConfigError("snr_threshold_db must be positive or null")
        if self.num_threads < 1:
            raise ConfigError("num_threads must be at least 1")

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


@dataclass
class TranscriptCfg:
    """Where the transcript comes from."""

    mode: Literal["auto", "whisper", "file", "off"] = "auto"
    file: str | None = None  # .srt/.json; '{stem}' is replaced by the video's file name without extension
    cache: bool = True  # reuse <video_dir>/<name>.transcript.json while input, model and language are unchanged
    model: str = "large-v3"  # tiny base small medium large-v3 (multilingual, NOT *.en)
    auto_downgrade: bool = True  # pick a smaller model when free RAM cannot hold the chosen one
    language: str | None = None  # None = auto-detect
    multilingual: bool = False  # detect the language per segment (mixed-language audio)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    beam_size: int = 5  # 1 = fastest, 5 = most accurate
    word_timestamps: bool = True
    enhance: EnhanceCfg = field(default_factory=EnhanceCfg)


@dataclass
class LlmCfg:
    """Optional local re-ranking (Ollama or any OpenAI-compatible server)."""

    enabled: bool = False
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    api_key: str | None = field(default=None, repr=False)
    top_k: int = 12
    timeout: float = 180

    def __post_init__(self) -> None:
        if urlparse(self.base_url).scheme not in ("http", "https"):
            raise ConfigError("llm.base_url must be an http(s) URL")


@dataclass
class BackgroundCfg:
    """Background behind a letter-boxed video."""

    type: Literal["blur", "color", "image"] = "blur"
    blur: float = 28
    dim: float = 0.35  # 0 = untouched, 1 = black
    color: str = "#101010"
    image: str | None = None


@dataclass
class VideoCfg:
    """Output encoding (the canvas shape comes from each clip kind's ``aspect``)."""

    short_side: int = 1080  # 1080 -> 1080x1920 for 9:16 and 1920x1080 for 16:9
    fps: float | None = 30  # None = source fps
    layout: Literal["fit", "fill"] = "fit"  # fit = whole frame on a background, fill = crop to fill
    background: BackgroundCfg = field(default_factory=BackgroundCfg)
    fade_seconds: float = 0.2
    gpu: Literal["auto", "off", "nvenc", "videotoolbox", "qsv", "amf"] = "auto"  # GPU encoder if one really works
    crf: int = 20  # quality (CPU: libx264 CRF; GPU: the encoder's quality level)
    preset: str = "veryfast"  # libx264 only
    maxrate_mbps: float = 12  # bitrate cap (upload friendly); 0 = uncapped
    thumbnail: bool = True  # also write <clip>.jpg


@dataclass
class CaptionsCfg:
    """Burned-in captions."""

    enabled: bool = True
    font: str = "auto"  # "auto" = a system font covering the script, or a font file / name
    size: int | None = None  # None = derived from the resolution
    color: str = "#FFFFFF"
    bg_color: str = "#000000"
    bg_opacity: float = 0.6  # 0 = no box
    stroke_width: int = 0
    stroke_color: str = "#000000"
    highlight: bool = True  # colour the spoken word (spaced left-to-right scripts only)
    highlight_color: str = "#FFD400"
    position: Position = "lower"
    safe_top: float = 0.10  # fraction of a vertical frame kept free of captions (app UI)
    safe_bottom: float = 0.20
    max_chars: int = 24  # characters per caption screen
    max_seconds: float = 2.6
    uppercase: bool = False


@dataclass
class TitleCfg:
    """Optional hook title shown at the start of each clip."""

    enabled: bool = False
    seconds: float = 3.0
    position: Position = "top"


@dataclass
class AudioCfg:
    """Speech clean-up."""

    enhance: bool = True
    denoise: Literal["auto", "noisereduce", "ffmpeg", "none"] = "auto"
    denoise_strength: float = 0.75
    highpass_hz: float = 80
    target_dbfs: float = -16  # speech loudness target (about -14 LUFS)
    fade_seconds: float = 0.12


@dataclass
class SilenceCfg:
    """Pause and filler-word removal."""

    remove: bool = True
    min_silence: float = 0.5  # pauses longer than this are shortened
    keep_padding: float = 0.12  # seconds of pause kept on each side
    threshold_db: float | Literal["auto"] = "auto"
    remove_fillers: bool = True  # needs word timestamps
    fillers: list[str] = field(default_factory=lambda: ["um", "uh", "umm", "uhh", "uhm", "erm", "hmm", "mmm"])


@dataclass
class ProcessingCfg:
    """Keeping memory flat on very long videos."""

    chunk_threshold_minutes: float = 0  # 0 = always transcribe in chunks; N = only videos longer than N minutes
    chunk_minutes: float = 5  # chunk length; boundaries snap to the quietest moment nearby

    def __post_init__(self) -> None:
        if self.chunk_minutes <= 0 or self.chunk_threshold_minutes < 0:
            raise ConfigError("processing.chunk_minutes must be positive")


@dataclass
class ResourcesCfg:
    """Adaptive CPU / RAM balancing (re-evaluated before and after every operation)."""

    cpus: int = 0  # 0 = auto-detect (affinity + container quota)
    max_cpu_percent: float = 100  # share of the cores the script may use
    max_workers: int = 0  # 0 = as many parallel renders as cores and RAM allow; 1 = sequential
    memory_fraction: float = 0.75  # max share of the *currently free* RAM the script may claim
    memory_reserve_mb: int = 512  # RAM always left to the OS and other programs


@dataclass
class StorageCfg:
    """Cache sizes (the SQLite database itself is recreated empty for every video)."""

    cache_memory_mb: int = 0  # 0 = auto (sized from free RAM)
    cache_disk_mb: int = 2048  # cap for cached blobs inside the SQLite file


@dataclass
class LoggingCfg:
    """Log file and console verbosity."""

    enabled: bool = True
    level: Level = "INFO"  # written to the log file; DEBUG logs every small step
    console_level: Level = "INFO"
    file: str | None = None  # None = <output_dir>/logs/autoclip_<timestamp>.log


@dataclass
class ComputeCfg:
    """Where the heavy work runs."""

    device: Literal["cpu", "gpu", "auto"] = "auto"  # "cpu" never touches the GPU, "gpu" prefers it


@dataclass
class Config:
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


def _conforms(tp: Any, value: Any) -> bool:
    """Does the JSON ``value`` fit the annotation ``tp`` (``int``, ``X | None``, ``Literal``, ``list[X]``)?"""
    origin = get_origin(tp)
    if origin is Literal:
        return value in get_args(tp)
    if origin in (Union, types.UnionType):
        return any(_conforms(arg, value) for arg in get_args(tp))
    if origin is list:
        return isinstance(value, list) and all(_conforms(get_args(tp)[0], v) for v in value)
    if tp is type(None):
        return value is None
    if tp is float:  # JSON has no float/int distinction
        return isinstance(value, int | float) and not isinstance(value, bool)
    if tp is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, tp)


def build(cls: type[T], data: dict[str, Any], path: str = "") -> T:
    """Instantiate dataclass ``cls`` from ``data``; unknown keys and wrongly typed values raise ``ConfigError``.

    Keys starting with ``_`` are ignored so config files can carry ``"_comment"`` entries.
    """
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if key not in hints:
            raise ConfigError(f"unknown config key '{path}{key}'")
        tp = hints[key]
        if isinstance(tp, type) and is_dataclass(tp):
            if not isinstance(value, dict):
                raise ConfigError(f"'{path}{key}' must be a JSON object")
            kwargs[key] = build(tp, value, f"{path}{key}.")
        elif _conforms(tp, value):
            kwargs[key] = value
        else:
            raise ConfigError(f"'{path}{key}' has an invalid value: {value!r}")
    return cls(**kwargs)


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
    cfg = build(Config, raw)
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
