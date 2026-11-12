"""Values that autoclip chooses for the user and never asks them about."""

from __future__ import annotations

import os

__all__ = [
    "AUDIO_FADE_SECONDS",
    "AUDIO_HIGHPASS_HZ",
    "AUDIO_TARGET_DBFS",
    "BACKGROUND_BLUR",
    "BACKGROUND_DIM",
    "CAPTION_MAX_SECONDS",
    "CAPTION_SAFE_BOTTOM",
    "CAPTION_SAFE_TOP",
    "CLIP_PAD_END",
    "CLIP_PAD_START",
    "COOKIE_BROWSERS",
    "DEFAULT_FILLERS",
    "DEFAULT_TRANSCRIPT_MODELS",
    "DENOISE_STRENGTH",
    "DOWNLOAD_FORMAT_CANDIDATES",
    "DOWNLOAD_PAUSE_MAX_SECONDS",
    "DOWNLOAD_PAUSE_MIN_SECONDS",
    "DOWNLOAD_PLAYLIST_PAUSE_MAX_SECONDS",
    "DOWNLOAD_PLAYLIST_PAUSE_MIN_SECONDS",
    "DOWNLOAD_RETRIES",
    "DOWNLOAD_SOCKET_TIMEOUT",
    "ENHANCE_ATTENUATION_DB",
    "ENHANCE_SNR_THRESHOLD_DB",
    "JS_RUNTIME_ORDER",
    "LLM_NUM_CTX",
    "LLM_TIMEOUT",
    "LLM_TOP_K",
    "PLAYLIST_MAX_VIDEOS",
    "SILENCE_KEEP_PADDING",
    "SILENCE_MIN_SECONDS",
    "SPLIT_SECONDS",
    "UNSILENCE_KEEP_SILENCE_MS",
    "UNSILENCE_MIN_SEGMENT_SECONDS",
    "UNSILENCE_MIN_SILENCE_MS",
    "UNSILENCE_MIN_TOTAL_SECONDS",
    "UNSILENCE_SILENCE_THRESHOLD_DB",
    "VIDEO_FADE_SECONDS",
    "audio_fade_seconds",
    "caption_chars",
    "denoiser_threads",
    "title_display_seconds",
    "transcription_beam_size",
    "video_fade_seconds",
]


CAPTION_SAFE_TOP = 0.10
"""Fraction of a vertical frame kept free at the top for captions."""

CAPTION_SAFE_BOTTOM = 0.20
"""Fraction of a vertical frame kept free at the bottom for captions."""


def transcription_beam_size(quality: str) -> int:
    """Accuracy/detail level for Whisper: 5 when quality matters, 1 when it should be fast."""
    return {"highest": 5, "high": 5, "medium": 4, "low": 3, "lowest": 1}.get(quality, 5)


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
"""pydub silence threshold used by the unsilence pre-pass."""

UNSILENCE_MIN_SILENCE_MS = 500
"""Minimum silence span removed by the unsilence pre-pass."""

UNSILENCE_KEEP_SILENCE_MS = 120
"""Silence kept on each side of a non-silent span by the unsilence pre-pass."""

UNSILENCE_MIN_SEGMENT_SECONDS = 0.25
"""Smallest non-silent span kept by the unsilence pre-pass."""

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

DEFAULT_TRANSCRIPT_MODELS = {
    "highest": "large-v3",
    "high": "large-v3",
    "medium": "small",
    "low": "base",
    "lowest": "tiny",
}
"""Whisper model per quality tier (``high`` drops to ``small`` without a GPU)."""

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
"""Below this much removable silence the unsilence pre-pass would not pay for its re-encode."""


def audio_fade_seconds(duration: float, *, quality: str = "high") -> float:
    """Audio edge fade: follow the configured fade, but never dominate a short clip."""
    base = AUDIO_FADE_SECONDS
    scale = {"highest": 1.0, "high": 1.0, "medium": 0.95, "low": 0.85, "lowest": 0.75}.get(quality, 1.0)
    fade = min(base * scale, max(0.0, duration / 6.0))
    return round(fade, 3)


def video_fade_seconds(duration: float, *, quality: str = "high", kind: str = "short") -> float:
    """Video edge fade: downscale for very short clips and low-quality targets."""
    base = VIDEO_FADE_SECONDS
    scale = {"highest": 1.0, "high": 1.0, "medium": 0.95, "low": 0.85, "lowest": 0.75}.get(quality, 1.0)
    kind_scale = 0.85 if kind == "short" else 1.0
    fade = min(base * scale * kind_scale, max(0.0, duration / 6.0))
    return round(fade, 3)


def title_display_seconds(text: str, duration: float, *, quality: str = "high") -> float:
    """Show an overlay long enough to read, but never longer than the clip itself."""
    # roughly 10-15 characters per second for a single-line overlay title
    estimated = min(3.0, max(1.0, len(text) / 12.0))
    scale = {"highest": 1.0, "high": 1.0, "medium": 0.95, "low": 0.9, "lowest": 0.85}.get(quality, 1.0)
    return round(min(duration, estimated * scale), 3)
