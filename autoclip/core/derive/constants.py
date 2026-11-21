"""Values autoclip chooses for the user and never asks about.

One table for the engine's fixed numbers, mode profiles and quality tiers. These were spread over
``internal/automation``, ``internal/modes`` and ``internal/quality``; a single module makes it
possible to see every baked-in choice at once and keeps callers from inventing their own numbers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = [
    "AUDIO_FADE_SECONDS",
    "AUDIO_HIGHPASS_HZ",
    "AUDIO_TARGET_DBFS",
    "BACKGROUND_BLUR",
    "BACKGROUND_DIM",
    "BASE_SHORT_SIDE",
    "CACHE_DISK_MB",
    "CAPTION_MAX_SECONDS",
    "CAPTION_SAFE_BOTTOM",
    "CAPTION_SAFE_TOP",
    "CHUNK_MINUTES",
    "CLIP_PAD_END",
    "CLIP_PAD_START",
    "COOKIE_BROWSERS",
    "DEFAULT_FILLERS",
    "DENOISE_STRENGTH",
    "DOWNLOAD_FORMAT_CANDIDATES",
    "DOWNLOAD_PAUSE_MAX_SECONDS",
    "DOWNLOAD_PAUSE_MIN_SECONDS",
    "DOWNLOAD_PLAYLIST_PAUSE_MAX_SECONDS",
    "DOWNLOAD_PLAYLIST_PAUSE_MIN_SECONDS",
    "DOWNLOAD_QUALITY_BY_OUTPUT",
    "DOWNLOAD_RETRIES",
    "DOWNLOAD_SOCKET_TIMEOUT",
    "ENHANCE_ATTENUATION_DB",
    "ENHANCE_SNR_THRESHOLD_DB",
    "HALLUCINATION_SILENCE_THRESHOLD",
    "JOB_RAM_MB",
    "JS_RUNTIME_ORDER",
    "LEXICON_FILE",
    "LLM_NUM_CTX",
    "LLM_TIMEOUT",
    "LLM_TOP_K",
    "OLLAMA_RAM_FRACTION",
    "PLAYLIST_MAX_VIDEOS",
    "PROFILES",
    "QUALITY_PROFILES",
    "SCENE_ADAPTIVE_THRESHOLD",
    "SCENE_MIN_CONTENT_VAL",
    "SCENE_MIN_SCENE_LEN_SECONDS",
    "SCENE_SNAP_RADIUS_SECONDS",
    "SILENCE_KEEP_PADDING",
    "SILENCE_MIN_SECONDS",
    "SPLIT_SECONDS",
    "UNSILENCE_KEEP_SILENCE_MS",
    "UNSILENCE_MIN_SEGMENT_SECONDS",
    "UNSILENCE_MIN_SILENCE_MS",
    "UNSILENCE_MIN_TOTAL_SECONDS",
    "UNSILENCE_SILENCE_THRESHOLD_DB",
    "VALID_QUALITIES",
    "VIDEO_FADE_SECONDS",
    "ModeProfile",
    "QualityProfile",
    "audio_fade_seconds",
    "caption_chars",
    "denoiser_threads",
    "download_quality_for",
    "output_fps",
    "profile_for",
    "quality_profile",
    "quality_short_side",
    "title_display_seconds",
    "transcription_beam_size",
    "video_fade_seconds",
]

CAPTION_SAFE_TOP = 0.10
"""Fraction of a vertical frame kept free at the top for captions."""

CAPTION_SAFE_BOTTOM = 0.20
"""Fraction of a vertical frame kept free at the bottom for captions."""


def transcription_beam_size(device: str = "cpu") -> int:
    """Accuracy/detail level for Whisper: 5 on CUDA (the model there is large), 3 on the CPU (it is not)."""
    return 5 if device == "cuda" else 3


def denoiser_threads(mode: str = "balance") -> int:
    """Conservative thread count: the ONNX denoiser is small and slows down with too many workers."""
    if mode == "resources":
        return 1
    cpus = os.cpu_count() or 4
    if mode == "performance":
        return min(4, max(1, cpus // 2 or 1))
    return min(2, max(1, cpus // 2 or 1))


def caption_chars(width: int = 1080) -> int:
    """Recommended number of caption characters for one screen at the rendered width."""
    return max(16, min(36, round(width / 45)))


AUDIO_HIGHPASS_HZ = 80.0
"""Frequency below which clip audio is removed."""

AUDIO_TARGET_DBFS = -16.0
"""Speech loudness target used by the clip audio limiter."""

AUDIO_FADE_SECONDS = 0.12
"""Length of the audio fade in and out of a clip."""

VIDEO_FADE_SECONDS = 0.2
"""Length of the video fade in and out of a clip."""

BACKGROUND_BLUR = 28.0
"""Blur strength for the blurred background behind letter-boxed video."""

BACKGROUND_DIM = 0.35
"""How much the blurred background is darkened (0 = untouched, 1 = black)."""

CAPTION_MAX_SECONDS = 2.6
"""Recommended how long one caption screen stays visible."""

CLIP_PAD_START = 0.15
"""Seconds kept before the first word of a text-selected clip."""

CLIP_PAD_END = 0.30
"""Seconds kept after the last word of a text-selected clip."""

UNSILENCE_SILENCE_THRESHOLD_DB = -40.0
"""Silence threshold used by the analysis-decode pre-pass."""

UNSILENCE_MIN_SILENCE_MS = 500
"""Minimum silence span removed by the optimizer pre-pass."""

UNSILENCE_KEEP_SILENCE_MS = 120
"""Silence kept on each side of a non-silent span by the optimizer pre-pass."""

UNSILENCE_MIN_SEGMENT_SECONDS = 0.25
"""Smallest non-silent span kept by the optimizer pre-pass."""

DOWNLOAD_FORMAT_CANDIDATES = 3
"""How many formats the downloader tries (the best and the next two) before giving up on a video."""

DOWNLOAD_RETRIES = 3
"""Default retry count for the downloader (yt-dlp and fragments)."""

DOWNLOAD_SOCKET_TIMEOUT = 20.0
"""How long the downloader waits for a silent connection."""

DOWNLOAD_PAUSE_MIN_SECONDS = 0.4
"""Smallest randomized pause before a URL is fetched."""

DOWNLOAD_PAUSE_MAX_SECONDS = 1.5
"""Largest randomized pause before a URL is fetched."""

DOWNLOAD_PLAYLIST_PAUSE_MIN_SECONDS = 0.5
"""Smallest randomized pause between playlist entries."""

DOWNLOAD_PLAYLIST_PAUSE_MAX_SECONDS = 2.0
"""Largest randomized pause between playlist entries."""

PLAYLIST_MAX_VIDEOS = 50
"""Hard ceiling on the entries a ``download.playlist: "all"`` request may fetch."""

JS_RUNTIME_ORDER = ("deno", "node", "bun", "quickjs")
"""The JavaScript runtimes probed for YouTube's signature challenges, in preference order."""

COOKIE_BROWSERS = frozenset({"chrome", "chromium", "firefox", "edge", "brave", "opera", "safari", "vivaldi", "whale"})
"""Values of ``download.cookies`` that name a browser store instead of a cookie file."""

DEFAULT_FILLERS = ("um", "uh", "umm", "uhh", "uhm", "erm", "hmm", "mmm")
"""Filler sounds removed from sentences and clip audio unless ``lexicon.json`` overrides them."""

ENHANCE_SNR_THRESHOLD_DB = 20.0
"""Chunks cleaner than this are transcribed untouched (denoising clean audio corrupts the text)."""

ENHANCE_ATTENUATION_DB = 12.0
"""Denoiser attenuation limit for the models that support it."""

HALLUCINATION_SILENCE_THRESHOLD = 2.0
"""Drop a low-confidence segment surrounded by silence (Whisper hallucination guard, seconds)."""

LLM_TOP_K = 12
"""How many candidates the local LLM is asked to choose from."""

LLM_NUM_CTX = 8192
"""Context window requested from Ollama (the server default can be as low as 4096, and an overflow silently truncates the prompt)."""

LLM_TIMEOUT = 180.0
"""How long to wait for the local LLM before falling back to the heuristic order."""

DENOISE_STRENGTH = 0.75
"""How hard spectral-gating denoisers work (0 = off, 1 = aggressive)."""

SILENCE_KEEP_PADDING = 0.12
"""Seconds of pause kept on each side of a shortened silence."""

SILENCE_MIN_SECONDS = 0.5
"""Pauses longer than this are shortened when a clip is rendered."""

SPLIT_SECONDS = 600.0
"""Length of each part in ``mode: "split"``."""

UNSILENCE_MIN_TOTAL_SECONDS = 30.0
"""Below this much removable silence the optimizer pre-pass would not pay for its re-encode."""

SCENE_ADAPTIVE_THRESHOLD = 3.0
"""AdaptiveDetector rolling-average sensitivity for fast cuts (lower is more sensitive)."""

SCENE_MIN_CONTENT_VAL = 15.0
"""Minimum HSV content change for a cut to qualify as a PySceneDetect shot boundary."""

SCENE_MIN_SCENE_LEN_SECONDS = 0.6
"""Shortest shot accepted by the cut detector."""

SCENE_SNAP_RADIUS_SECONDS = 2.0
"""How far a selected clip edge may move to land on a visible shot boundary."""


def audio_fade_seconds(duration: float) -> float:
    """Audio edge fade: the built-in fade, but never more than a sixth of the clip so a short clip keeps its middle."""
    return round(min(AUDIO_FADE_SECONDS, max(0.0, duration / 6.0)), 3)


def video_fade_seconds(duration: float, *, kind: str = "short") -> float:
    """Video edge fade: downscaled for short clips, capped so it never dominates a short clip."""
    kind_scale = 0.85 if kind == "short" else 1.0
    return round(min(VIDEO_FADE_SECONDS * kind_scale, max(0.0, duration / 6.0)), 3)


def title_display_seconds(text: str, duration: float) -> float:
    """Show an overlay long enough to read, but never longer than the clip itself."""
    # roughly 10-15 characters per second for a single-line overlay title
    estimated = min(3.0, max(1.0, len(text) / 12.0))
    return round(min(duration, estimated), 3)


# --- mode profiles --------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeProfile:
    """The resource policy for one of the three internal profiles."""

    name: str
    cpu_percent_min: float
    cpu_percent_max: float
    cpu_percent_default: float
    mem_fraction: float
    memory_reserve_mb: int
    encoding_threads_cap: int
    parallel: bool
    target_cpu_low: float
    target_cpu_high: float
    step_up: float
    step_down: float


PROFILES: dict[str, ModeProfile] = {
    "performance": ModeProfile(
        name="performance",
        cpu_percent_min=70.0,
        cpu_percent_max=100.0,
        cpu_percent_default=90.0,
        mem_fraction=0.75,
        memory_reserve_mb=256,
        encoding_threads_cap=4,
        parallel=True,
        target_cpu_low=80.0,
        target_cpu_high=95.0,
        step_up=5.0,
        step_down=3.0,
    ),
    "resources": ModeProfile(
        name="resources",
        cpu_percent_min=10.0,
        cpu_percent_max=40.0,
        cpu_percent_default=25.0,
        mem_fraction=0.30,
        memory_reserve_mb=1024,
        encoding_threads_cap=2,
        parallel=False,
        target_cpu_low=15.0,
        target_cpu_high=35.0,
        step_up=2.0,
        step_down=5.0,
    ),
    "balance": ModeProfile(
        name="balance",
        cpu_percent_min=30.0,
        cpu_percent_max=80.0,
        cpu_percent_default=50.0,
        mem_fraction=0.50,
        memory_reserve_mb=512,
        encoding_threads_cap=4,
        parallel=True,
        target_cpu_low=55.0,
        target_cpu_high=75.0,
        step_up=4.0,
        step_down=4.0,
    ),
}


def profile_for(mode: str) -> ModeProfile:
    """Return the profile for *mode*; unknown or empty values fall back to ``balance`` rather than raising."""
    if not mode or mode not in PROFILES:
        return PROFILES["balance"]
    return PROFILES[mode]


# --- quality tiers --------------------------------------------------------------------------------

VALID_QUALITIES = ("high", "medium", "low")
#: Short side of the canvas at the neutral quality tier; every tier scales it.
BASE_SHORT_SIDE = 1080
DOWNLOAD_QUALITY_BY_OUTPUT = {"high": 720, "medium": 480, "low": 360}


@dataclass(frozen=True)
class QualityProfile:
    """Encoder settings for one output-quality tier."""

    short_side_scale: float
    fps_cap: float | None
    aac_bitrate_kbps: int
    crf: int
    preset: str
    nvenc_cq: int
    videotoolbox_qv: int


QUALITY_PROFILES: dict[str, QualityProfile] = {
    "high": QualityProfile(0.80, None, 192, 20, "slow", 22, 85),
    "medium": QualityProfile(0.60, 30.0, 128, 24, "medium", 28, 65),
    "low": QualityProfile(0.40, 24.0, 96, 28, "fast", 33, 45),
}


def download_quality_for(output_quality: str) -> int:
    """The short side to download for an output quality tier."""
    try:
        return DOWNLOAD_QUALITY_BY_OUTPUT[output_quality]
    except KeyError as e:
        raise ValueError(f"unknown quality {output_quality!r}; valid: {', '.join(VALID_QUALITIES)}") from e


def quality_profile(quality: str) -> QualityProfile:
    """The encoder profile for a quality tier."""
    try:
        return QUALITY_PROFILES[quality]
    except KeyError as e:
        raise ValueError(f"unknown quality {quality!r}; valid: {', '.join(VALID_QUALITIES)}") from e


def quality_short_side(quality: str) -> int:
    """The rendered short side of the quality tier (never smaller than 2 px)."""
    return max(2, round(BASE_SHORT_SIDE * quality_profile(quality).short_side_scale))


def output_fps(source_fps: float | None, quality: str) -> float:
    """The encoded frame rate: the source's, capped by the quality tier."""
    source = max(float(source_fps) if source_fps and source_fps > 0 else 30.0, 1.0)
    cap = quality_profile(quality).fps_cap
    return max(1.0, min(source, cap) if cap is not None else source)


# --- other fixed values ---------------------------------------------------------------------------

CACHE_DISK_MB = 2048
"""Fixed SQLite cache ceiling."""

CHUNK_MINUTES = 5.0
"""Fixed, memory-safe transcription chunk length."""

#: RAM one server job needs at peak (Whisper + render + Python overhead), in MB.
JOB_RAM_MB = 1536.0
#: An installed model may use at most this fraction of free RAM (the rest is KV cache, the pipeline, the OS).
OLLAMA_RAM_FRACTION = 0.5
#: Name of the optional lexicon file, looked up next to the config file.
LEXICON_FILE = "lexicon.json"
