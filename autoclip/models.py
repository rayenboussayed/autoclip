"""Plain data shared between modules (small slotted dataclasses, picklable for worker processes)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Config


@dataclass(slots=True)
class Span:
    """A piece of text with a time range in seconds (caption line, word, segment, moment ...)."""

    start: float
    end: float
    text: str


@dataclass(slots=True)
class Word(Span):
    """One transcribed word; ``pseudo`` marks timings interpolated from a segment (no real word times)."""

    pseudo: bool = False


@dataclass(slots=True)
class Segment(Span):
    """A transcript segment (usually a sentence) with its words."""

    words: list[Word] = field(default_factory=list)
    pseudo: bool = False


@dataclass(slots=True)
class Moment(Span):
    """A clip to render: source window, scoring metadata and (after rendering) the trimmed length."""

    score: float | None = None
    llm_score: float | None = None
    title: str = ""
    scene: str = ""  # the scene type this moment was picked for ("custom" for explicit ranges)
    i: int = -1  # first/last transcript segment of the window (-1 when selected from audio only)
    j: int = -1
    trimmed: float = 0.0


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """Facts about an input video."""

    duration: float
    has_audio: bool
    width: int
    height: int
    fps: float


@dataclass(frozen=True, slots=True)
class Encoder:
    """A video encoder choice: ffmpeg codec name, the ``-preset`` MoviePy always passes, and extra arguments.

    MoviePy writes ``-preset <preset>`` unconditionally and appends ``params`` after it; ffmpeg lets the last
    repeated option win, so GPU encoders keep a placeholder ``preset`` and set their own in ``params``.
    """

    name: str
    hardware: bool
    preset: str
    params: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RenderJob:
    """Everything a (possibly separate) process needs to render one clip. Results go to SQLite."""

    clip_id: int
    moment: Moment
    src: Path
    dst: Path
    aspect: str
    encoder: Encoder
    has_audio: bool
    cfg: Config
    db_path: Path
    log_file: Path | None
    parent_pid: int
    threads: int = 1
