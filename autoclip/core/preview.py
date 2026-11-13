"""A one-frame preview of what the final clip will look like.

The preview takes one representative frame from the source video and paints the configured
layout and captions (real :func:`autoclip.core.captions.render_text_image` sprites, real font
discovery, real :func:`autoclip.core.render.canvas_size` geometry) over it, with a mock sentence
and a mock title. It never decodes audio, never transcribes and never encodes a video; the
output is a single PNG. Everything about the caption is read from the runtime config, so what
the user sees is what the render will produce.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .captions import Y_CENTER, render_text_image, safe_top
from .config import Config
from .errors import AutoclipError
from .fonts import find_font
from .internal.automation import BACKGROUND_BLUR, BACKGROUND_DIM
from .internal.background import background_kind
from .internal.quality import quality_short_side
from .log import traced
from .media import probe
from .render import blur_function, canvas_size, even

log = logging.getLogger("autoclip")

U8 = NDArray[np.uint8]

#: The mock caption the preview shows (deliberately generic English; style/layout is what the preview is for).
MOCK_CAPTION = "this is how your captions will look"
#: The mock overlay title (titles only render when one was generated; skip.title turns this off).
MOCK_TITLE = "Example title"
#: The active word of the mock caption (index 3 = "your"); the highlight colour is part of what is previewed.
MOCK_ACTIVE = 3


def _sample_frame(path: Path, t: float) -> U8:
    """One RGB frame of ``path`` nearest to ``t`` seconds (the first frame when the seek fails)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if abs(t) > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if not ok and abs(t) > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, 0.0)
            ok, frame = cap.read()
        if not ok:
            raise AutoclipError(f"cannot read a frame from '{path.name}' (empty or unsupported video?)")
        return np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


def _cover(frame: U8, w: int, h: int) -> U8:
    """``frame`` scaled to cover ``w x h`` and centre-cropped (the ``fill`` layout)."""
    import cv2

    fh, fw = frame.shape[:2]
    scale = max(w / fw, h / fh)
    big = cv2.resize(frame, (max(w, math.ceil(fw * scale)), max(h, math.ceil(fh * scale))), interpolation=cv2.INTER_LINEAR)
    y0 = (big.shape[0] - h) // 2
    x0 = (big.shape[1] - w) // 2
    return np.ascontiguousarray(big[y0 : y0 + h, x0 : x0 + w])


def _fit(frame: U8, w: int, h: int) -> U8:
    """``frame`` scaled to fit inside ``w x h`` (the ``fit`` layout's foreground)."""
    import cv2

    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    return np.ascontiguousarray(cv2.resize(frame, (even(fw * scale), even(fh * scale)), interpolation=cv2.INTER_AREA))


def _blurred(frame: U8, w: int, h: int) -> U8:
    """The 1/6-resolution blur background of ``layout()`` (blur + dim, then upscale)."""
    import cv2

    small = _cover(frame, max(2, even(w / 6)), max(2, even(h / 6)))
    small = blur_function(max(0.5, BACKGROUND_BLUR / 6), BACKGROUND_DIM)(small)
    return np.ascontiguousarray(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR))


def _canvas(frame: U8, w: int, h: int, cfg: Config) -> U8:
    """The layout composite: the frame alone (``fill`` or no spare room) or on its background."""
    if cfg.video.layout == "fill":
        return _cover(frame, w, h)
    fg = _fit(frame, w, h)
    if fg.shape == (h, w, 3):
        return np.ascontiguousarray(fg)
    bg = cfg.video.background
    kind = background_kind(bg)
    if kind == "color":
        from PIL import ImageColor

        canvas = np.full((h, w, 3), ImageColor.getrgb(bg)[:3], np.uint8)
    elif kind == "image":
        try:
            from PIL import Image

            img = np.asarray(Image.open(bg).convert("RGB"))
            canvas = _cover(img, w, h)
        except Exception as e:
            log.warning("background image '%s' unavailable (%s): using blur", bg, e)
            canvas = _blurred(frame, w, h)
    else:
        canvas = _blurred(frame, w, h)
    y0 = (h - fg.shape[0]) // 2
    x0 = (w - fg.shape[1]) // 2
    canvas[y0 : y0 + fg.shape[0], x0 : x0 + fg.shape[1]] = fg
    return canvas


def _blit(canvas: U8, art: U8, top: int) -> None:
    """Alpha-composite a sprite horizontally centred at row ``top``."""
    h, w = art.shape[:2]
    x0 = max(0, (canvas.shape[1] - w) // 2)
    if x0 + w > canvas.shape[1] or top + h > canvas.shape[0]:
        return  # the sprite would not fit the canvas; drop it rather than draw garbage
    roi = canvas[top : top + h, x0 : x0 + w, :]
    alpha = art[:h, :w, 3:4].astype(np.float32) / 255.0
    roi[:] = (art[:h, :w, :3].astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


@traced("preview")
def preview_canvas(source: Path, cfg: Config, kind: str = "short", at: float | None = None) -> U8:
    """The output frame (RGB) of a clip of ``kind`` from ``source`` with the configured layout and captions.

    Args:
        source: The video file (a frame is read, not the whole file).
        cfg: The runtime configuration the job would use (quality, aspect, captions, layout...).
        kind: ``"short"`` (9:16 by default) or ``"long"``.
        at: Where to take the frame in seconds; the default is 40% of the duration.
    """
    try:
        info = probe(source)
    except Exception as e:
        raise AutoclipError(f"cannot probe '{source.name}': {e}") from e
    t = at if at is not None else max(0.0, float(info.duration) * 0.4)
    frame = _sample_frame(source, t)
    kcfg = cfg.clips.kind(kind)
    fill = cfg.video.layout == "fill"
    size = canvas_size(kcfg.aspect, quality_short_side(cfg.quality), info.width, info.height, fill=fill)
    canvas = _canvas(frame, size[0], size[1], cfg)
    if cfg.captions.enabled:
        text = MOCK_CAPTION.upper() if cfg.captions.uppercase else MOCK_CAPTION
        sprite = render_text_image(text, find_font(text), int(min(size) / 15), cfg.captions, int(size[0] * 0.9), active=MOCK_ACTIVE)
        band = sprite.shape[0] + (sprite.shape[0] % 2)
        top = safe_top(Y_CENTER[cfg.captions.position], size[1], size[0], cfg.captions, band)
        _blit(canvas, sprite, top)
    if not cfg.skip.title:
        sprite = render_text_image(MOCK_TITLE, find_font(MOCK_TITLE), int(min(size) / 13), cfg.captions, int(size[0] * 0.92))
        top = safe_top(Y_CENTER["top"], size[1], size[0], cfg.captions, sprite.shape[0])
        _blit(canvas, sprite, top)
    return canvas


def preview_image(source: Path, cfg: Config, kind: str = "short", at: float | None = None) -> bytes:
    """``preview_canvas`` encoded as PNG bytes."""
    import io

    from PIL import Image

    img = Image.fromarray(preview_canvas(source, cfg, kind, at))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
