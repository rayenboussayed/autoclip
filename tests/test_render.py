"""Layout helpers and real end-to-end clip rendering."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from moviepy import ColorClip, VideoFileClip
from PIL import Image

from autoclip.config import VideoCfg, build
from autoclip.hardware import cpu_encoder
from autoclip.media import probe
from autoclip.models import Encoder, Moment, RenderJob
from autoclip.render import blur_function, canvas_size, estimate_render_bytes, even, fill_to, layout, render_clip, resize_to
from autoclip.resources import RESOURCES
from autoclip.store import Cache, Store
from autoclip.transcript import load_transcript_file

from .conftest import Media, make_cfg


def test_even_and_canvas_size() -> None:
    assert even(7.2) == 8 and even(0) == 2 and even(1079.9) == 1080
    assert canvas_size("9:16", 1080, 1280, 720) == (1080, 1920)
    assert canvas_size("16:9", 1080, 1, 1) == (1920, 1080)  # long-form: horizontal
    assert canvas_size("4:5", 800, 1, 1) == (800, 1000) and canvas_size("1:1", 500, 1, 1) == (500, 500)
    assert canvas_size("original", 1080, 1279, 719) == (1280, 720)


def test_estimate_grows_with_size_and_duration() -> None:
    small, big = estimate_render_bytes((540, 960), 10), estimate_render_bytes((1080, 1920), 60)
    assert 200 * (1 << 20) < small < big < 2 * (1 << 30)


def test_blur_function_blurs_and_dims() -> None:
    frame = np.zeros((40, 40, 3), np.uint8)
    frame[:, 20:] = 255
    out = blur_function(3.0, 0.5)(frame)
    assert out.dtype == np.uint8 and out.shape == frame.shape
    assert 0 < out[20, 20, 0] < 128 and out.max() <= 128  # edge softened, everything dimmed by half


def test_resize_and_fill() -> None:
    clip = ColorClip((64, 36), color=(255, 0, 0)).with_duration(1)
    assert resize_to(clip, 32, 18).get_frame(0).shape == (18, 32, 3)
    assert resize_to(clip, 128, 72).get_frame(0).shape == (72, 128, 3)  # enlarging uses the other interpolation
    assert fill_to(clip, 20, 40).get_frame(0).shape == (40, 20, 3)
    assert fill_to(clip, 128, 36).get_frame(0).shape == (36, 128, 3)


def test_layout_variants(tmp_path: Path) -> None:
    fg = ColorClip((160, 90), color=(0, 0, 255)).with_duration(1)
    Image.new("RGB", (30, 30), (0, 255, 0)).save(tmp_path / "bg.png")

    def layers(**video: Any) -> list[Any]:
        return layout(fg, (72, 128), build(VideoCfg, video), 1.0)

    assert len(layers(layout="fill")) == 1 and layers(layout="fill")[0].size == (72, 128)
    blur = layers()
    assert len(blur) == 2 and blur[0].size == (72, 128) and blur[1].size == (72, 40)
    colour = layers(background={"type": "color", "color": "#ff0000"})
    assert tuple(colour[0].get_frame(0)[0, 0]) == (255, 0, 0)
    image = layers(background={"type": "image", "image": str(tmp_path / "bg.png")})
    assert tuple(image[0].get_frame(0)[0, 0]) == (0, 255, 0)
    assert len(layers(background={"type": "image", "image": None})) == 2  # no image given: falls back to blur
    exact = layout(ColorClip((72, 128), color=(1, 2, 3)).with_duration(1), (72, 128), VideoCfg(), 1.0)
    assert len(exact) == 1  # already the right size: nothing to letter-box


# ---- render_clip end to end ----------------------------------------------------------------------
def render(
    media: Media,
    tmp_path: Path,
    moment: Moment,
    src: Path | None = None,
    aspect: str = "9:16",
    encoder: Encoder | None = None,
    **sections: dict[str, Any],
) -> tuple[Path, tuple[int, int], list[Any]]:
    src = src or media.video
    cfg = make_cfg(media, tmp_path, source=src, **{"audio": {"denoise": "none"}, **sections})
    store = Store(tmp_path / "r.db", fresh=True)
    store.put_segments(load_transcript_file(media.transcript))
    cache = Cache(store, RESOURCES)
    dst = tmp_path / "clip.mp4"
    job = RenderJob(
        1, moment, src, dst, aspect, encoder or cpu_encoder(cfg.video), probe(src).has_audio, cfg, store.path, None, os.getpid(), threads=1
    )
    vclip = VideoFileClip(str(src), audio=False)
    try:
        size, lines = render_clip(vclip, job, store, cache)
    finally:
        vclip.close()
        cache.close()
        store.close()
    return dst, size, lines


def moment(start: float = 8.0, end: float = 19.0, title: str = "A great title") -> Moment:
    return Moment(start, end, "", title=title)


def test_render_full_featured_clip(media: Media, tmp_path: Path) -> None:
    m = moment()
    dst, size, lines = render(media, tmp_path, m, title={"enabled": True}, video={"thumbnail": True})
    info = probe(dst)
    assert (info.width, info.height) == size == (144, 256) and info.has_audio
    assert info.duration == pytest.approx(m.trimmed, abs=0.4) and m.trimmed < (m.end - m.start) - 1  # pauses were cut
    assert dst.with_suffix(".jpg").stat().st_size > 500
    assert lines and all(ln.end <= m.trimmed + 0.1 for ln in lines)
    cover = np.array(Image.open(dst.with_suffix(".jpg")))
    assert cover.shape[:2] == (256, 144)


def test_render_without_audio_track_and_without_captions(media: Media, tmp_path: Path) -> None:
    dst, size, lines = render(media, tmp_path, moment(2, 9), src=media.silent, captions={"enabled": False})
    assert not probe(dst).has_audio and lines == [] and size == (144, 256)


def test_render_fill_layout_plain_settings(media: Media, tmp_path: Path) -> None:
    m = moment(0, 6, title="")
    dst, size, _ = render(
        media,
        tmp_path,
        m,
        video={"layout": "fill", "maxrate_mbps": 0, "fade_seconds": 0, "fps": None},
        audio={"enhance": False, "fade_seconds": 0, "denoise": "none"},
        silence={"remove": False},
        title={"enabled": True},
    )
    assert probe(dst).duration == pytest.approx(6, abs=0.4) and m.trimmed == pytest.approx(6, abs=0.1)
    assert size == (144, 256)


def test_render_original_aspect_needs_no_letterbox_and_colour_background(media: Media, tmp_path: Path) -> None:
    _, size, _ = render(media, tmp_path, moment(0, 5), aspect="original")
    assert size == (320, 180)
    dst, size2, _ = render(media, tmp_path / "c", moment(0, 5), video={"background": {"type": "color", "color": "#202020"}})
    assert size2 == (144, 256) and dst.exists()


def test_render_with_noise_reduction(media: Media, tmp_path: Path) -> None:
    dst, _, _ = render(media, tmp_path, moment(0, 4), audio={"denoise": "noisereduce"})
    assert probe(dst).has_audio


def test_render_clamps_to_video_end_and_rejects_outside_windows(media: Media, tmp_path: Path) -> None:
    dst, _, _ = render(media, tmp_path, moment(36, 44))  # container is a hair longer than the video stream, window overruns
    assert probe(dst).duration < 5
    with pytest.raises(ValueError, match="outside the video"):
        render(media, tmp_path / "x", moment(60, 70))


def test_render_long_form_horizontal_clip_needs_no_letterbox(media: Media, tmp_path: Path) -> None:
    dst, size, lines = render(media, tmp_path, moment(0, 9), aspect="16:9")
    info = probe(dst)
    assert size == (256, 144) and (info.width, info.height) == (256, 144) and lines and info.has_audio  # captions kept, 16:9 canvas


def test_render_with_a_gpu_style_encoder_definition(media: Media, tmp_path: Path) -> None:
    """GPU encoders override MoviePy's placeholder preset in their arguments; software libx264 stands in for the hardware."""
    stand_in = Encoder("libx264", True, "no-such-preset", ("-preset", "ultrafast", "-crf", "40", "-maxrate", "1M", "-bufsize", "2M"))
    low, _, _ = render(media, tmp_path / "a", moment(0, 5), encoder=stand_in)
    cpu, _, _ = render(media, tmp_path / "b", moment(0, 5), encoder=cpu_encoder(VideoCfg(preset="ultrafast", crf=18)))
    assert probe(low).duration == pytest.approx(probe(cpu).duration, abs=0.3)
    assert low.stat().st_size < cpu.stat().st_size  # the placeholder preset was overridden and the custom arguments reached ffmpeg
