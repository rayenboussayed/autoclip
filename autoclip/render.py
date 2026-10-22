"""Rendering one clip: layout on the output canvas, captions, title, fades, encode and cover image."""

from __future__ import annotations

import functools
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .audio import Timeline, build_audio, compute_keep, enhance_audio
from .captions import Y_CENTER, build_captions, render_text_image, safe_top
from .config import VideoCfg
from .fonts import find_font
from .log import speed_text, traced
from .media import AUDIO_SR
from .models import Encoder, RenderJob, Span, Word
from .store import Cache, Store

log = logging.getLogger("autoclip")


def even(v: float) -> int:
    """Nearest even integer >= 2 (H.264 with yuv420p needs even dimensions)."""
    return max(2, round(v / 2) * 2)


def canvas_size(aspect: str, short_side: int, src_w: int, src_h: int) -> tuple[int, int]:
    """Output ``(width, height)`` for an ``aspect`` such as "9:16" ("original" keeps the source size)."""
    if aspect == "original":
        return even(src_w), even(src_h)
    a, _, b = aspect.partition(":")
    ratio = float(a) / float(b)
    return (even(short_side), even(short_side / ratio)) if ratio < 1 else (even(short_side * ratio), even(short_side))


def estimate_render_bytes(size: tuple[int, int], seconds: float) -> int:
    """Rough peak RAM of one render job: interpreter + libraries, ~12 frame buffers, ~8 audio copies, caption band."""
    w, h = size
    return int(220 * (1 << 20) + 12 * w * h * 3 + 8 * seconds * AUDIO_SR * 4 + 3 * w * (h // 4) * 4)


def blur_function(sigma: float, dim: float) -> Any:
    """Frame function: Gaussian blur followed by dimming (OpenCV, which is much faster than Pillow's)."""
    import cv2

    return lambda frame: (cv2.GaussianBlur(frame, (0, 0), sigma) * (1 - dim)).astype(np.uint8)


@functools.cache
def _resize_effect() -> Any:
    """MoviePy ``Resize`` using OpenCV instead of Pillow-LANCZOS (about 5x faster per frame)."""
    import cv2
    from moviepy import vfx

    class CvResize(vfx.Resize):  # type: ignore[misc]
        """``vfx.Resize`` with ``cv2.resize`` (INTER_AREA when shrinking, INTER_LINEAR when enlarging)."""

        def resizer(self, pic: Any, new_size: Any) -> Any:
            w, h = (round(v) for v in new_size)
            shrinking = w * h < pic.shape[0] * pic.shape[1]
            return cv2.resize(pic, (w, h), interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)

    return CvResize


def resize_to(clip: Any, w: int, h: int) -> Any:
    """Resize a MoviePy clip to exactly ``w x h``."""
    return clip.with_effects([_resize_effect()(new_size=(int(w), int(h)))])


def fill_to(clip: Any, w: int, h: int) -> Any:
    """Scale ``clip`` to cover ``w x h`` and centre-crop."""
    from moviepy import vfx

    scale = max(w / clip.w, h / clip.h)
    big = resize_to(clip, max(w, math.ceil(clip.w * scale)), max(h, math.ceil(clip.h * scale)))
    return big.with_effects([vfx.Crop(x_center=big.w / 2, y_center=big.h / 2, width=w, height=h)])


def layout(fg: Any, size: tuple[int, int], cfg: VideoCfg, duration: float) -> list[Any]:
    """Video layers on the canvas: filled, or the whole frame on a blurred / coloured / image background."""
    from moviepy import ColorClip, ImageClip
    from PIL import ImageColor

    w, h = size
    if cfg.layout == "fill":
        return [fill_to(fg, w, h)]
    scale = min(w / fg.w, h / fg.h)
    front = fg if abs(scale - 1) < 1e-3 else resize_to(fg, even(fg.w * scale), even(fg.h * scale))
    if (front.w, front.h) == (w, h):
        return [front]
    bg = cfg.background
    if bg.type == "color":
        back = ColorClip((w, h), color=ImageColor.getrgb(bg.color)[:3]).with_duration(duration)
    elif bg.type == "image" and bg.image:
        back = fill_to(ImageClip(bg.image).with_duration(duration), w, h)
    else:  # blur at 1/6 resolution, then scale up: 6x less work, same look
        small = fill_to(fg, max(2, even(w / 6)), max(2, even(h / 6)))
        back = resize_to(small.image_transform(blur_function(max(0.5, bg.blur / 6), bg.dim)), w, h)
    return [back, front.with_position("center")]


@traced("encode_video")
def encode(comp: Any, dst: Path, fps: float, encoder: Encoder, threads: int) -> None:
    """Write H.264 + AAC (yuv420p, faststart) with the CPU or GPU ``encoder``."""
    t0 = time.perf_counter()
    comp.write_videofile(
        str(dst),
        fps=fps,
        codec=encoder.name,
        audio_codec="aac",
        audio_bitrate="160k",
        preset=encoder.preset,
        pixel_format="yuv420p",
        threads=threads,
        ffmpeg_params=[*encoder.params, "-movflags", "+faststart"],
        logger=None,
    )
    log.info(
        "  encoded %.1fs of video in %.1fs (%s, %s, %d threads)",
        comp.duration,
        time.perf_counter() - t0,
        speed_text(comp.duration, time.perf_counter() - t0),
        encoder.name,
        threads,
    )


@traced("render_clip")
def render_clip(vclip: Any, job: RenderJob, store: Store, cache: Cache) -> tuple[tuple[int, int], list[Span]]:
    """Render one clip: enhance audio, cut pauses/fillers, lay out, caption, fade, encode, cover image.

    Args:
        vclip: Open MoviePy ``VideoFileClip`` of the source (audio disabled).
        job: What to render and with which settings; ``job.moment.trimmed`` is set to the final length.
        store / cache: Transcript words come from ``store``; caption sprites are cached in ``cache``.

    Returns:
        The output ``(width, height)`` and the caption lines on the trimmed timeline.
    """
    from moviepy import AudioArrayClip, CompositeVideoClip, ImageClip, afx, concatenate_videoclips, vfx

    moment, cfg, src, dst, threads, has_audio = job.moment, job.cfg, job.src, job.dst, job.threads, job.has_audio
    start, dur = moment.start, moment.end - moment.start
    words = [Word(w.start - start, w.end - start, w.text, w.pseudo) for w in store.window_words(moment.start, moment.end)]
    audio = enhance_audio(src, start, dur, cfg.audio, threads) if has_audio else None
    keep = compute_keep(audio, words, dur, cfg.silence) if audio is not None else [(0.0, dur)]

    parts, kept = [], []  # clamp to the real video stream: containers often report a slightly longer duration
    for a, b in keep:
        end = min(start + b, float(vclip.duration))
        if end - (start + a) > 0.05:
            parts.append(vclip.subclipped(start + a, end))
            kept.append((a, end - start))
    if not parts:
        raise ValueError(f"window {start:.1f}-{moment.end:.1f}s lies outside the video")
    fg = concatenate_videoclips(parts) if len(parts) > 1 else parts[0]
    moment.trimmed = round(float(fg.duration), 2)
    size = canvas_size(job.aspect, cfg.video.short_side, vclip.w, vclip.h)
    overlays = []
    captions, lines = build_captions(words, Timeline(kept), cfg, cache, size, fg.duration) if cfg.captions.enabled and words else (None, [])
    if captions is not None:
        overlays.append(captions)
    if cfg.title.enabled and moment.title:
        cc = cfg.captions
        art = render_text_image(moment.title, find_font(moment.title, cc.font), int(min(size) / 13), cc, int(size[0] * 0.92))
        top = safe_top(Y_CENTER[cfg.title.position], size[1], size[0], cc, art.shape[0])
        overlays.append(ImageClip(art).with_duration(min(cfg.title.seconds, fg.duration)).with_position(("center", top)))

    comp = CompositeVideoClip(layout(fg, size, cfg.video, fg.duration) + overlays, size=size).with_duration(fg.duration)
    if cfg.video.fade_seconds > 0:
        comp = comp.with_effects([vfx.FadeIn(cfg.video.fade_seconds), vfx.FadeOut(cfg.video.fade_seconds)])
    if audio is not None:
        mono = build_audio(audio, kept, fg.duration)
        sound = AudioArrayClip(np.stack([mono, mono], axis=1), fps=AUDIO_SR)
        if cfg.audio.fade_seconds > 0:
            sound = sound.with_effects([afx.AudioFadeIn(cfg.audio.fade_seconds), afx.AudioFadeOut(cfg.audio.fade_seconds)])
        comp = comp.with_audio(sound)
    fps = min(cfg.video.fps or vclip.fps, vclip.fps)
    encode(comp, dst, float(fps), job.encoder, threads)
    if cfg.video.thumbnail:
        cover_t = min(max(0.5, cfg.title.seconds / 2 if cfg.title.enabled else 1.0), max(0.0, fg.duration - 0.1))
        comp.save_frame(str(dst.with_suffix(".jpg")), t=cover_t, with_mask=False)
    return size, lines
