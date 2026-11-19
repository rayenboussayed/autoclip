"""Quality tiers of the internal encoder profiles, chosen automatically (see ``internal.auto.auto_quality``)."""

from __future__ import annotations

from dataclasses import dataclass

VALID_QUALITIES = ("high", "medium", "low")
#: Short side of the canvas at the neutral quality tier; every tier scales it.
BASE_SHORT_SIDE = 1080
DOWNLOAD_QUALITY_BY_OUTPUT = {"high": 720, "medium": 480, "low": 360}


@dataclass(frozen=True)
class QualityProfile:
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
    try:
        return QUALITY_PROFILES[quality]
    except KeyError as e:
        raise ValueError(f"unknown quality {quality!r}; valid: {', '.join(VALID_QUALITIES)}") from e


def quality_short_side(quality: str) -> int:
    """The rendered short side of the quality tier (never smaller than 2 px)."""
    return max(2, round(BASE_SHORT_SIDE * quality_profile(quality).short_side_scale))


def output_fps(source_fps: float | None, quality: str) -> float:
    source = max(float(source_fps) if source_fps and source_fps > 0 else 30.0, 1.0)
    cap = quality_profile(quality).fps_cap
    return max(1.0, min(source, cap) if cap is not None else source)


__all__ = [
    "BASE_SHORT_SIDE",
    "DOWNLOAD_QUALITY_BY_OUTPUT",
    "VALID_QUALITIES",
    "QualityProfile",
    "download_quality_for",
    "output_fps",
    "quality_profile",
    "quality_short_side",
]
