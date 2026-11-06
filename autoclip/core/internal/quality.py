"""User-facing video quality tiers mapped to concrete encoder settings."""

from __future__ import annotations

from dataclasses import dataclass

VALID_QUALITIES = ("highest", "high", "medium", "low", "lowest")


@dataclass(frozen=True)
class QualityProfile:
    name: str
    short_side_scale: float
    fps_cap: float | None
    aac_bitrate_kbps: int
    crf: int
    preset: str
    nvenc_cq: int
    videotoolbox_qv: int


QUALITY_PROFILES: dict[str, QualityProfile] = {
    "highest": QualityProfile("highest", 1.00, None, 256, 16, "veryslow", 18, 100),
    "high": QualityProfile("high", 0.80, None, 192, 20, "slow", 22, 85),
    "medium": QualityProfile("medium", 0.60, 30.0, 128, 24, "medium", 28, 65),
    "low": QualityProfile("low", 0.40, 24.0, 96, 28, "fast", 33, 45),
    "lowest": QualityProfile("lowest", 0.20, 15.0, 64, 32, "ultrafast", 38, 25),
}


def quality_profile(quality: str) -> QualityProfile:
    try:
        return QUALITY_PROFILES[quality]
    except KeyError as e:
        raise ValueError(f"unknown quality {quality!r}; valid: {', '.join(VALID_QUALITIES)}") from e


def quality_short_side(short_side: int, quality: str) -> int:
    return max(2, round(short_side * quality_profile(quality).short_side_scale))


def output_fps(source_fps: float | None, requested_fps: float | None, quality: str) -> float:
    source = max(float(source_fps) if source_fps and source_fps > 0 else 30.0, 1.0)
    if requested_fps is not None:
        return max(1.0, min(float(requested_fps), source))
    cap = quality_profile(quality).fps_cap
    return max(1.0, min(source, cap) if cap is not None else source)


__all__ = ["VALID_QUALITIES", "QualityProfile", "output_fps", "quality_profile", "quality_short_side"]
