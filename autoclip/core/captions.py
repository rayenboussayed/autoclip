"""Burned-in captions: word-highlighted sprites rendered once, cached compressed, blitted per frame."""

from __future__ import annotations

import bisect
import functools
import hashlib
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageColor, ImageDraw, ImageFont, features

from .audio import Timeline, is_filler
from .config import CaptionsCfg, Config
from .fonts import dominant_script, find_font, has_rtl
from .log import traced
from .media import F32
from .models import Span, Word
from .store import Cache

log = logging.getLogger("autoclip")
U8 = NDArray[np.uint8]
ENDERS = (".", "?", "!", "\u2026", "\u3002", "\uff01", "\uff1f", "\u061f", "\u0964", "\u0965")
CJK_CHAR = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\u0e00-\u0e7f\u3000-\u303f\uff00-\uffef]")
SIMPLE_SCRIPTS = {"LATIN", "CYRILLIC", "GREEK", "ARMENIAN", "GEORGIAN", "HANGUL"}  # shape correctly one word at a time
Y_CENTER = {"top": 0.14, "center": 0.5, "lower": 0.68, "bottom": 0.78}


@dataclass(slots=True)
class Chunk(Span):
    """One caption screen: its words and the joined text."""

    words: list[Word] = field(default_factory=list)


def join_tokens(tokens: Sequence[str]) -> str:
    """Join with spaces, except around CJK/Thai characters which are written without spaces."""
    out = ""
    for tok in tokens:
        out = tok if not out else out + ("" if CJK_CHAR.match(out[-1]) or CJK_CHAR.match(tok[0]) else " ") + tok
    return out


#: Shortest time a caption stays on screen; without it a one-word cue can flash for a few frames.
MIN_CHUNK_SECONDS = 0.8


def make_chunks(words: Sequence[Word], cfg: CaptionsCfg) -> list[Chunk]:
    """Group words into caption screens by length, duration, long pauses and punctuation."""
    chunks: list[Chunk] = []
    current: list[Word] = []

    def flush() -> None:
        if not current:
            return
        text, end = join_tokens([w.text for w in current]), max(current[-1].end, current[0].start + MIN_CHUNK_SECONDS)
        if chunks and not re.search(r"\w", text):  # punctuation only: glue onto the previous screen
            chunks[-1].text += text
            chunks[-1].end = max(chunks[-1].end, end)
        else:
            chunks.append(Chunk(current[0].start, end, text, list(current)))
        current.clear()

    for w in words:
        if current and (
            len(join_tokens([x.text for x in current] + [w.text])) > cfg.max_chars
            or w.end - current[0].start > cfg.max_seconds
            or w.start - current[-1].end > 0.8
        ):
            flush()
        current.append(w)
        if w.text.endswith(ENDERS) or (
            re.search(r"[,;\uff0c\u3001\u060c]$", w.text) and len(join_tokens([x.text for x in current])) > 0.6 * cfg.max_chars
        ):
            flush()
    flush()
    for a, b in pairwise(chunks):
        a.end = max(a.end, b.start)  # keep the screen until the next one appears: no flicker between cues
    return chunks


def can_highlight(text: str) -> bool:
    """Word-by-word colouring needs spaced, left-to-right text in a script that shapes correctly per word."""
    return " " in text.strip() and not has_rtl(text) and not CJK_CHAR.search(text) and dominant_script(text) in SIMPLE_SCRIPTS


def balanced_lines(widths: Sequence[float], space: float, avail: float) -> list[list[int]]:
    """Break words into the fewest lines that fit ``avail`` pixels, then even the lines out.

    Greedy wrapping leaves orphans ("... tells / you"); this keeps the greedy line count but picks the breaks
    that minimise the sum of squared slack, so subtitle lines get similar widths. An over-long word gets its own line.

    Returns:
        Word indices per line.
    """
    n = len(widths)

    def span(i: int, j: int) -> float:
        return sum(widths[i:j]) + space * (j - i - 1)

    lines, used = 1, 0.0
    for w in widths:
        if used and used + space + w > avail:
            lines, used = lines + 1, w
        else:
            used += w + (space if used else 0)
    # best[k][j]: minimal cost of laying words 0..j-1 on k lines
    best = [[float("inf")] * (n + 1) for _ in range(lines + 1)]
    back = [[0] * (n + 1) for _ in range(lines + 1)]
    best[0][0] = 0.0
    for k in range(1, lines + 1):
        for j in range(1, n + 1):
            for i in range(k - 1, j):
                if span(i, j) <= avail or j - i == 1:
                    cost = best[k - 1][i] + (avail - span(i, j)) ** 2
                    if cost < best[k][j]:
                        best[k][j], back[k][j] = cost, i
    out: list[list[int]] = []
    j = n
    for k in range(lines, 0, -1):
        out.append(list(range(back[k][j], j)))
        j = back[k][j]
    return out[::-1]


@functools.lru_cache(maxsize=64)
def _load_font(path: Path | None, size: int, raqm: bool) -> Any:
    """A PIL font (the built-in Latin font when ``path`` is ``None``)."""
    if path is None:
        return ImageFont.load_default(size)
    return ImageFont.truetype(str(path), size, layout_engine=ImageFont.Layout.RAQM if raqm else ImageFont.Layout.BASIC)


@functools.cache
def _warn_no_raqm(script: str) -> None:
    log.warning("Pillow lacks libraqm: %s captions may not shape correctly (see README: Windows/macOS)", script.title())


class Face:
    """A font plus the text-direction handling: Raqm when available, else pure-Python Arabic/Hebrew shaping."""

    def __init__(self, path: Path | None, size: int, text: str) -> None:
        raqm = bool(features.check("raqm"))
        rtl = has_rtl(text)
        script = dominant_script(text)
        if not raqm and script not in SIMPLE_SCRIPTS | {"ARABIC", "HEBREW", "CJK", "HIRAGANA", "KATAKANA"}:
            _warn_no_raqm(script)
        # Pillow's built-in default font (path is None) is always BASIC layout and rejects a text
        # direction, so Raqm may only be used for a real font file.
        usable_raqm = raqm and path is not None
        self.font = _load_font(path, size, usable_raqm)
        self.direction = "rtl" if rtl and usable_raqm else None
        self.reshape = rtl and not usable_raqm

    def prepare(self, s: str) -> str:
        """``s`` in the form the renderer needs (reshaped and reordered when Raqm is missing)."""
        if not self.reshape:
            return s
        import arabic_reshaper
        from bidi.algorithm import get_display

        return str(get_display(arabic_reshaper.reshape(s)))

    def width(self, s: str) -> float:
        """Pixel width of ``s``."""
        return float(self.font.getlength(self.prepare(s), direction=self.direction))


def render_text_image(text: str, font: Path | None, size: int, style: CaptionsCfg, max_w: int, active: int | None = None) -> U8:
    """Render ``text`` (any script, wrapped to ``max_w``) as an RGBA array with clean anti-aliased edges.

    Text, outline and highlight are drawn on separate transparent layers pre-filled with their own colour and
    then alpha-composited - drawing straight onto RGBA leaves dark fringes.

    Args:
        active: Index of the word to colour with ``style.highlight_color`` (``None`` = no highlight).
    """
    face = Face(font, size, text)
    spaced = " " in text.strip()
    tokens, sep = (text.split(), " ") if spaced else (list(text), "")
    pad, sw = int(size * 0.28), style.stroke_width
    sep_w = face.width(sep) if sep else 0.0
    lines = balanced_lines([face.width(t) for t in tokens], sep_w, max(50, max_w - 2 * pad - 2 * sw))
    highlight = active is not None and can_highlight(text)
    line_w = [
        sum(face.width(tokens[k]) for k in ln) + sep_w * (len(ln) - 1) if highlight else face.width(sep.join(tokens[k] for k in ln))
        for ln in lines
    ]
    ascent, descent = face.font.getmetrics()
    line_h = ascent + descent + int(size * 0.12)
    w_px, h_px = int(max(line_w)) + 2 * (pad + sw), line_h * len(lines) + 2 * pad

    canvas = Image.new("RGBA", (w_px, h_px), (0, 0, 0, 0))
    if style.bg_opacity > 0:
        box = (*ImageColor.getrgb(style.bg_color)[:3], int(255 * style.bg_opacity))
        ImageDraw.Draw(canvas).rounded_rectangle([0, 0, w_px - 1, h_px - 1], radius=int(size * 0.3), fill=box)
    colors = {
        name: ImageColor.getrgb(color)[:3]
        for name, color in (("stroke", style.stroke_color), ("text", style.color), ("active", style.highlight_color))
    }
    layers = {name: Image.new("RGBA", (w_px, h_px), (*rgb, 0)) for name, rgb in colors.items()}

    def draw(layer: str, xy: tuple[float, float], s: str, stroke: int = 0) -> None:
        ink = (*colors[layer], 255)
        ImageDraw.Draw(layers[layer]).text(
            xy, face.prepare(s), font=face.font, fill=ink, stroke_width=stroke, stroke_fill=ink, direction=face.direction
        )

    for row, ln in enumerate(lines):
        x, y = (w_px - line_w[row]) / 2, pad + row * line_h
        if highlight:
            for k in ln:
                if sw:
                    draw("stroke", (x, y), tokens[k], sw)
                draw("active" if k == active else "text", (x, y), tokens[k])
                x += face.width(tokens[k]) + sep_w
        else:
            if sw:
                draw("stroke", (x, y), sep.join(tokens[k] for k in ln), sw)
            draw("text", (x, y), sep.join(tokens[k] for k in ln))
    for name, used in (("stroke", bool(sw)), ("text", True), ("active", highlight)):
        if used:
            canvas = Image.alpha_composite(canvas, layers[name])
    return np.array(canvas, dtype=np.uint8)


@dataclass
class Sprite:
    """One caption image shown during ``[start, end)`` of the trimmed clip; ``render`` rebuilds it on a cache miss."""

    start: float
    end: float
    key: str
    render: Callable[[], U8]
    height: int = 0


class CaptionTrack:
    """A band-sized video track that blits the sprite for the current time.

    Instead of one full-size clip per caption (hundreds of arrays in RAM) the sprites live compressed in the
    :class:`Cache` and only the one on screen is decoded.
    """

    def __init__(self, cache: Cache, sprites: list[Sprite], width: int, band_h: int) -> None:
        self.cache, self.sprites = cache, sorted(sprites, key=lambda s: s.start)
        self.starts = [s.start for s in self.sprites]
        self.band = (band_h, width)
        self._blank = (np.zeros((*self.band, 3), np.uint8), np.zeros(self.band, np.float32))
        self._shown = -2  # index of the sprite currently composed (-1 = none)
        self._frame = self._blank

    def _compose(self, t: float) -> tuple[U8, F32]:
        k = bisect.bisect_right(self.starts, t) - 1
        k = k if k >= 0 and t < self.sprites[k].end else -1
        if k != self._shown:
            self._shown = k
            if k < 0:
                self._frame = self._blank
            else:
                art = self.cache.memoize_array(self.sprites[k].key, self.sprites[k].render)
                h, w = min(art.shape[0], self.band[0]), min(art.shape[1], self.band[1])
                y0, x0 = (self.band[0] - h) // 2, (self.band[1] - w) // 2
                rgb, alpha = np.zeros((*self.band, 3), np.uint8), np.zeros(self.band, np.float32)
                rgb[y0 : y0 + h, x0 : x0 + w], alpha[y0 : y0 + h, x0 : x0 + w] = art[:h, :w, :3], art[:h, :w, 3] / 255.0
                self._frame = (rgb, alpha)
        return self._frame

    def clip(self, duration: float) -> Any:
        """The MoviePy clip (RGB + alpha mask) for this track."""
        from moviepy import VideoClip

        mask = VideoClip(lambda t: self._compose(t)[1], is_mask=True, duration=duration)
        return VideoClip(lambda t: self._compose(t)[0], duration=duration).with_mask(mask)


def safe_top(centre: float, H: int, W: int, cfg: CaptionsCfg, band_h: int) -> int:
    """Top pixel of a ``band_h`` band centred at ``centre`` (fraction of H), kept inside the app-UI safe zone."""
    vertical = H > 1.2 * W
    low = (cfg.safe_top if vertical else 0.03) * H
    high = (1 - (cfg.safe_bottom if vertical else 0.03)) * H - band_h
    return int(min(max(centre * H - band_h / 2, low), max(low, high)))


def sprite_key(*parts: object) -> str:
    """Stable cache key for a sprite's rendering parameters."""
    return "cap:" + hashlib.sha256(repr(parts).encode()).hexdigest()


def without_fillers(words: Sequence[Word], fillers: set[str]) -> list[Word]:
    """Drop filler words (they are cut from the audio) and close the gap each one leaves."""
    out: list[Word] = []
    skipped = False
    for w in words:
        if not w.pseudo and is_filler(w.text, fillers):
            skipped = True
            continue
        if skipped and out and w.start - out[-1].end < 1.0:
            out[-1] = Word(out[-1].start, w.start, out[-1].text, out[-1].pseudo)
        skipped = False
        out.append(w)
    return out


@traced("build_captions")
def build_captions(
    words: Sequence[Word], tl: Timeline, cfg: Config, cache: Cache, size: tuple[int, int], duration: float
) -> tuple[Any | None, list[Span]]:
    """Render the caption sprites into the cache.

    Returns:
        The positioned caption clip (``None`` if there is nothing to show) and the caption lines on the trimmed
        timeline (for the ``.srt`` sidecar).
    """
    W, H = size
    cc = cfg.captions
    if cfg.silence.remove and cfg.silence.remove_fillers:
        words = without_fillers(words, {f.lower() for f in cfg.silence.fillers})
    font_size, max_w = cc.size or int(min(W, H) / 15), int(W * 0.9)
    sprites: list[Sprite] = []
    lines: list[Span] = []

    def add(start: float, end: float, text: str, active: int | None) -> None:
        params = (text, font, font_size, cc, max_w, active)
        sprites.append(Sprite(start, end, sprite_key(*params), functools.partial(render_text_image, *params)))

    for chunk in make_chunks(words, cc):
        span = tl.map_range(chunk.start, chunk.end)
        if span is None or span[1] - span[0] < 0.08:
            continue
        text = chunk.text.upper() if cc.uppercase else chunk.text
        font = find_font(text, cc.font)
        lines.append(Span(round(span[0], 2), round(span[1], 2), text))
        if cc.highlight and can_highlight(text) and len(text.split()) == len(chunk.words) and not any(w.pseudo for w in chunk.words):
            for k, w in enumerate(chunk.words):
                nxt = chunk.words[k + 1].start if k + 1 < len(chunk.words) else chunk.end
                if (part := tl.map_range(w.start, max(nxt, w.start + 0.05))) and part[1] - part[0] >= 0.03:
                    add(part[0], part[1], text, k)
        else:
            add(span[0], span[1], text, None)
    if not sprites:
        return None, lines
    band_h = max(int(cache.memoize_array(s.key, s.render).shape[0]) for s in sprites)  # also warms the cache
    band_h += band_h % 2
    track = CaptionTrack(cache, sprites, W, band_h)
    top = safe_top(Y_CENTER[cc.position], H, W, cc, band_h)
    return track.clip(duration).with_position((0, top)), lines
