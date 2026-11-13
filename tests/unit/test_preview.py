"""The one-frame preview: real frame in, PNG out, configured layout/captions applied."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from autoclip.core.config import Config
from autoclip.core.errors import AutoclipError
from autoclip.core.internal.quality import quality_short_side
from autoclip.core.preview import MOCK_ACTIVE, MOCK_CAPTION, MOCK_TITLE, preview_canvas, preview_image
from autoclip.core.render import canvas_size
from tests.support import Media, make_cfg


def _png_bytes(cfg: Config, media: Media, kind: str = "short", at: float | None = None) -> bytes:
    return preview_image(media.video, cfg, kind, at)


def test_preview_returns_a_valid_png_of_the_short_canvas(media: Media, tmp_path: Path) -> None:
    cfg = make_cfg(media, tmp_path)
    png = _png_bytes(cfg, media, "short")
    with Image.open(io.BytesIO(png)) as img:
        img.load()
        assert img.format == "PNG"
        info = (320, 180)  # the fixture video
        assert img.size == canvas_size("9:16", quality_short_side(cfg.quality), *info)


def test_preview_long_kind_uses_the_long_aspect(media: Media, tmp_path: Path) -> None:
    cfg = make_cfg(media, tmp_path, clips={"long": {"aspect": "16:9"}})
    png = _png_bytes(cfg, media, "long")
    with Image.open(io.BytesIO(png)) as img:
        img.load()
        assert img.size == canvas_size("16:9", quality_short_side(cfg.quality), 320, 180)


def test_preview_canvas_matches_preview_image(media: Media, tmp_path: Path) -> None:
    cfg = make_cfg(media, tmp_path, quality="high")
    canvas = preview_canvas(media.video, cfg, "short", at=1.0)
    assert canvas.shape[2] == 3
    png = preview_image(media.video, cfg, "short", at=1.0)
    with Image.open(io.BytesIO(png)) as img:
        img.load()
        assert img.size == (canvas.shape[1], canvas.shape[0])


def test_preview_captions_change_the_output(media: Media, tmp_path: Path) -> None:
    base = make_cfg(media, tmp_path)
    plain = make_cfg(media, tmp_path, captions={"enabled": False})
    assert _png_bytes(base, media) != _png_bytes(plain, media)


def test_preview_caption_theme_changes_the_output(media: Media, tmp_path: Path) -> None:
    social = make_cfg(media, tmp_path, captions={"theme": "social"})
    plain = make_cfg(media, tmp_path, captions={"theme": "plain"})
    assert _png_bytes(social, media) != _png_bytes(plain, media)


def test_preview_uppercase_changes_the_output(media: Media, tmp_path: Path) -> None:
    normal = make_cfg(media, tmp_path, captions={"uppercase": False})
    upper = make_cfg(media, tmp_path, captions={"uppercase": True})
    assert _png_bytes(normal, media) != _png_bytes(upper, media)


def test_preview_title_flag_changes_the_output(media: Media, tmp_path: Path) -> None:
    with_title = make_cfg(media, tmp_path)
    without_title = make_cfg(media, tmp_path, skip={"title": True})
    assert _png_bytes(with_title, media) != _png_bytes(without_title, media)


def test_preview_color_background(media: Media, tmp_path: Path) -> None:
    cfg = make_cfg(media, tmp_path, video={"background": "#112233"})
    with Image.open(io.BytesIO(_png_bytes(cfg, media))) as img:
        img.load()
    assert img.size[0] > 0


def test_preview_missing_background_image_falls_back_to_blur(media: Media, tmp_path: Path) -> None:
    cfg = make_cfg(media, tmp_path, video={"background": "definitely-not-here-12345.png"})
    with Image.open(io.BytesIO(_png_bytes(cfg, media))) as img:
        img.load()


def test_preview_fill_layout(media: Media, tmp_path: Path) -> None:
    cfg = make_cfg(media, tmp_path, video={"layout": "fill"})
    with Image.open(io.BytesIO(_png_bytes(cfg, media))) as img:
        img.load()
        assert img.size == canvas_size("9:16", quality_short_side(cfg.quality), 320, 180, fill=True)


def test_preview_unreadable_video_raises(media: Media, tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video at all")
    cfg = make_cfg(media, tmp_path)
    with pytest.raises(AutoclipError, match="cannot probe"):
        preview_image(junk, cfg)


def test_mock_caption_covers_several_screens() -> None:
    words = MOCK_CAPTION.split()
    assert len(words) >= 6  # several screens' worth of words on one line
    assert words[MOCK_ACTIVE] == "your"  # the highlighted word of the mock
    assert MOCK_TITLE.strip()


def test_preview_with_zero_timestamp_skips_the_seek(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(media, tmp_path)
    seeked: list[float] = []

    class _Cap:
        def __init__(self, path: str) -> None:
            self.frame = np.zeros((8, 8, 3), np.uint8) + 3

        def set(self, _prop: int, value: float) -> bool:
            seeked.append(value)
            return True

        def read(self):  # type: ignore[no-untyped-def]
            return True, self.frame

        def release(self) -> None:
            pass

    import cv2

    monkeypatch.setattr(cv2, "VideoCapture", _Cap)
    png = _png_bytes(cfg, media, at=0.0)
    assert seeked == []
    assert png.startswith(b"\x89PNG")


def test_preview_seek_failure_falls_back_to_the_first_frame(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(media, tmp_path)

    class _Cap:
        def __init__(self, path: str) -> None:
            self.frame = np.zeros((8, 8, 3), np.uint8) + 3
            self.pos = 0.0

        def set(self, _prop: int, value: float) -> bool:
            self.pos = float(value)
            return True

        def read(self):  # type: ignore[no-untyped-def]
            if self.pos > 0.0:  # the seeked position yields nothing; position 0 works
                return False, None
            return True, self.frame

        def release(self) -> None:
            pass

    import cv2

    monkeypatch.setattr(cv2, "VideoCapture", _Cap)
    png = _png_bytes(cfg, media, at=5.0)
    assert png.startswith(b"\x89PNG")


def test_preview_unreadable_stream_raises_autoclip_error(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(media, tmp_path)

    class _Cap:
        def __init__(self, path: str) -> None:
            pass

        def set(self, _prop: int, value: float) -> bool:
            return False

        def read(self):  # type: ignore[no-untyped-def]
            return False, None

        def release(self) -> None:
            pass

    import cv2

    monkeypatch.setattr(cv2, "VideoCapture", _Cap)
    with pytest.raises(AutoclipError, match="cannot read a frame"):
        _png_bytes(cfg, media, at=5.0)


def test_preview_with_an_image_background(media: Media, tmp_path: Path) -> None:
    bg = tmp_path / "bg.png"
    Image.new("RGB", (64, 32), (10, 20, 30)).save(bg)
    cfg = make_cfg(media, tmp_path, video={"background": str(bg)})
    with Image.open(io.BytesIO(_png_bytes(cfg, media))) as img:
        img.load()


def test_preview_drops_a_sprite_that_does_not_fit(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(media, tmp_path)
    big = np.zeros((50, 10_000, 4), np.uint8)  # far wider than the canvas

    import autoclip.core.preview as preview

    monkeypatch.setattr(preview, "render_text_image", lambda *a, **k: big)
    import autoclip.core.render  # noqa: F401  (keeps cv2/blur imports resolved deterministically)

    # the canvas still comes out (title is skipped by its own natural failure to fit: huge sprites are dropped)
    png = _png_bytes(cfg, media)
    assert png.startswith(b"\x89PNG")
