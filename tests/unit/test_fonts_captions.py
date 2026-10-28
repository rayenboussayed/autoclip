"""Font discovery and caption rendering. Machine-independent: real fonts are used only when present."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import features

from autoclip.core import captions, fonts
from autoclip.core.audio import Timeline
from autoclip.core.captions import (
    CaptionTrack,
    Chunk,
    Face,
    Sprite,
    balanced_lines,
    build_captions,
    can_highlight,
    join_tokens,
    make_chunks,
    render_text_image,
    safe_top,
    sprite_key,
    without_fillers,
)
from autoclip.core.config import CaptionsCfg, Config, parse
from autoclip.core.fonts import covers, dominant_script, find_font, font_dirs, font_index, has_rtl
from autoclip.core.models import Span, Word
from autoclip.core.resources import RESOURCES
from autoclip.core.store import Cache, Store


@pytest.fixture(scope="session")
def real_font() -> Path:
    path = find_font("Hello world 123")
    if path is None:
        pytest.skip("no system font installed")
    return path


# ---- fonts ---------------------------------------------------------------------------------------
def test_font_dirs_per_platform(tmp_path: Path) -> None:
    win = font_dirs("Windows", {"WINDIR": r"D:\Win", "LOCALAPPDATA": r"D:\Local"}, tmp_path)
    assert win[0] == Path(r"D:\Win") / "Fonts" and win[1].parts[-3:] == ("Microsoft", "Windows", "Fonts")
    assert font_dirs("Windows", {}, tmp_path)[1] == tmp_path / "AppData" / "Local" / "Microsoft" / "Windows" / "Fonts"
    mac = font_dirs("Darwin", {}, tmp_path)
    assert Path("/System/Library/Fonts") in mac and tmp_path / "Library" / "Fonts" in mac
    linux = font_dirs("Linux", {}, tmp_path)
    assert Path("/usr/share/fonts") in linux and tmp_path / ".fonts" in linux


def test_font_index_walks_folders_case_insensitively(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "a" / "deep").mkdir(parents=True)
    for name in ("a/Arial.TTF", "a/deep/Noto.otf", "a/c.ttc", "a/readme.txt"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr(fonts, "font_dirs", lambda *_a: [tmp_path, tmp_path / "missing"])
    font_index.cache_clear()
    try:
        assert sorted(font_index()) == ["arial.ttf", "c.ttc", "noto.otf"]
    finally:
        font_index.cache_clear()


def test_script_and_direction_detection() -> None:
    assert (
        dominant_script("hello") == "LATIN"
        and dominant_script("123") == "LATIN"
        and dominant_script("\u0645\u0631\u062d\u0628\u0627") == "ARABIC"
    )
    assert (
        dominant_script("\u4eca\u5929") == "CJK"
        and dominant_script("\u3053\u3093\u306b\u3061\u306f \u4eca") == "HIRAGANA"
        and dominant_script("\u043f\u0440\u0438\u0432\u0435\u0442") == "CYRILLIC"
    )
    assert has_rtl("\u0645\u0631\u062d\u0628\u0627") and has_rtl("\u05e9\u05dc\u05d5\u05dd") and not has_rtl("hello")


def test_covers(real_font: Path, tmp_path: Path) -> None:
    assert covers(real_font, "Hello") and covers(real_font, "!!!")
    assert not covers(real_font, "\U00013000\U00013001")  # Egyptian hieroglyphs: no text font has them
    junk = tmp_path / "junk.ttf"
    junk.write_bytes(b"not a font")
    assert not covers(junk, "Hello")


def test_find_font_preferences_and_fallbacks(real_font: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    index = {real_font.name.lower(): real_font}
    monkeypatch.setattr(fonts, "font_index", lambda: index)
    monkeypatch.setattr(fonts, "UNIVERSAL", [real_font.name])
    assert find_font("Hello") == real_font
    assert find_font("Hello", str(real_font)) == real_font  # explicit file
    assert find_font("Hello", real_font.stem) is not None  # by name without extension
    assert find_font("Hello", real_font.name.upper()) == real_font  # by file name, any case
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        assert find_font("Hello", "no-such-font") == real_font
    assert "not found or lacks" in caplog.text
    caplog.clear()
    monkeypatch.setattr(fonts, "font_index", lambda: {})
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        assert find_font("\u4eca\u5929") is None
    assert "no installed font covers Cjk" in caplog.text


def test_find_font_tries_script_specific_names_first(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    arabic, universal = tmp_path / "tahomabd.ttf", tmp_path / "arialbd.ttf"
    monkeypatch.setattr(fonts, "font_index", lambda: {"tahomabd.ttf": arabic, "arialbd.ttf": universal})
    monkeypatch.setattr(fonts, "covers", lambda _p, _t: True)
    assert find_font("\u0645\u0631\u062d\u0628\u0627") == arabic
    assert find_font("hello") == universal


# ---- chunks / wrapping ---------------------------------------------------------------------------
def words(*items: tuple[str, float, float]) -> list[Word]:
    return [Word(s, e, t) for t, s, e in items]


def test_join_tokens() -> None:
    assert (
        join_tokens(["Hello", "world"]) == "Hello world"
        and join_tokens(["\u4eca", "\u5929", "ok"]) == "\u4eca\u5929ok"
        and join_tokens([]) == ""
    )


def test_make_chunks_split_on_length_pause_punctuation_and_glue() -> None:
    cfg = CaptionsCfg(max_chars=12, max_seconds=2.0)
    chunks = make_chunks(
        words(("Hello", 0, 0.4), ("big", 0.4, 0.8), ("world.", 0.8, 1.2), ("Next", 1.3, 1.6), ("one", 1.6, 2.0), ("here", 4.0, 4.1)), cfg
    )
    assert [c.text for c in chunks] == ["Hello big", "world.", "Next one", "here"]  # length, sentence end, pause
    assert chunks[0].end <= chunks[1].start and chunks[3].end == pytest.approx(4.3)  # at least 0.3 s on screen
    comma = make_chunks(words(("alpha,", 0, 0.3), ("beta,", 0.3, 0.6), ("gamma", 0.6, 0.9)), CaptionsCfg(max_chars=14))
    assert [c.text for c in comma] == ["alpha, beta,", "gamma"]
    punct = make_chunks(words(("Hi.", 0, 0.5), ("!!", 0.5, 0.8)), CaptionsCfg())
    assert len(punct) == 1 and punct[0].text == "Hi.!!"
    slow = make_chunks(words(("a", 0, 1.5), ("b", 1.5, 3.0)), CaptionsCfg(max_seconds=2.0))
    assert len(slow) == 2
    assert make_chunks([], CaptionsCfg()) == []


def test_can_highlight() -> None:
    assert can_highlight("two words") and can_highlight("\u043f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440")
    assert (
        not can_highlight("single")
        and not can_highlight("\u0645\u0631\u062d\u0628\u0627 \u0628\u0643\u0645")
        and not can_highlight("\u4eca\u5929 \u597d")
    )


def test_balanced_lines() -> None:
    assert balanced_lines([50.0], 10.0, 400.0) == [[0]]
    assert balanced_lines([100, 100, 100, 40], 10.0, 350.0) == [[0, 1], [2, 3]]  # greedy would orphan the last word
    assert balanced_lines([300, 10, 10], 5.0, 100.0) == [[0], [1, 2]]  # an over-long word gets its own line
    assert balanced_lines([40.0] * 6, 10.0, 130.0) == [[0, 1], [2, 3], [4, 5]]


# ---- rendering -----------------------------------------------------------------------------------
STYLE = CaptionsCfg(bg_opacity=0.0)


def opaque(arr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return arr[arr[..., 3] > 250][:, :3]


def test_render_plain_and_highlight(real_font: Path) -> None:
    plain = render_text_image("one two three", real_font, 36, STYLE, 600)
    assert plain.ndim == 3 and plain.shape[2] == 4 and plain.dtype == np.uint8
    assert not ((opaque(plain)[:, 0] > 240) & (opaque(plain)[:, 2] < 40)).any()  # no yellow without highlight
    marked = render_text_image("one two three", real_font, 36, STYLE, 600, active=1)
    px = opaque(marked)
    assert ((px[:, 0] > 240) & (px[:, 2] < 40)).any() and (px > 240).all(axis=1).any()  # yellow word + white words


def test_render_box_stroke_and_wrapping(real_font: Path) -> None:
    boxed = render_text_image("hi", real_font, 30, CaptionsCfg(bg_opacity=0.5), 400)
    assert boxed[0, 0, 3] == 0 or boxed[boxed.shape[0] // 2, 2, 3] == 127  # translucent box, rounded corners
    stroked = render_text_image(
        "hi there", real_font, 30, CaptionsCfg(bg_opacity=0, stroke_width=3, stroke_color="#FF0000", highlight=True), 400, active=0
    )
    assert ((stroked[..., 0] > 200) & (stroked[..., 1] < 60) & (stroked[..., 3] > 200)).any()  # red outline present
    wide = render_text_image("word " * 12, real_font, 30, STYLE, 300)
    assert wide.shape[0] > 2 * render_text_image("word", real_font, 30, STYLE, 300).shape[0] and wide.shape[1] <= 300 + 60


def test_render_without_spaces_and_with_builtin_font() -> None:
    cjk = render_text_image(
        "\u4eca\u5929\u6211\u4eec\u6765\u804a\u804a\u5982\u4f55\u521b\u4e1a\u554a\u554a\u554a\u554a", None, 40, STYLE, 200
    )
    assert cjk.shape[0] > 40 and cjk.shape[1] <= 260  # wrapped character by character with the built-in Latin font


def test_rtl_without_raqm_is_reshaped_in_python(real_font: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(features, "check", lambda _name: False)
    face = Face(real_font, 30, "\u0645\u0631\u062d\u0628\u0627")
    assert face.reshape and face.direction is None and face.prepare("\u0645\u0631\u062d\u0628\u0627") != "\u0645\u0631\u062d\u0628\u0627"
    assert face.width("\u0645\u0631\u062d\u0628\u0627") > 0
    assert (
        render_text_image(
            "\u0645\u0631\u062d\u0628\u0627 \u0628\u0643\u0645 \u0641\u064a \u0647\u0630\u0627", real_font, 30, STYLE, 300
        ).size
        > 0
    )
    assert not Face(real_font, 30, "hello").reshape


@pytest.mark.skipif(not features.check("raqm"), reason="Pillow without libraqm (typical on Windows/macOS wheels)")
def test_rtl_with_raqm_uses_native_direction(real_font: Path) -> None:
    face = Face(real_font, 30, "\u0645\u0631\u062d\u0628\u0627")
    assert face.direction == "rtl" and not face.reshape and face.width("\u0645\u0631\u062d\u0628\u0627") > 0


@pytest.mark.skipif(not features.check("raqm"), reason="Pillow without libraqm (typical on Windows/macOS wheels)")
def test_rtl_with_builtin_font_asks_for_no_direction() -> None:
    """Pillow's built-in font is BASIC layout and rejects a text direction (regression)."""
    face = Face(None, 30, "\u0645\u0631\u062d\u0628\u0627")
    assert face.direction is None and face.reshape
    assert face.width("\u0645\u0631\u062d\u0628\u0627") > 0
    assert render_text_image("\u0645\u0631\u062d\u0628\u0627 \u0628\u0643\u0645", None, 30, STYLE, 300).size > 0


def test_missing_raqm_warns_once_for_complex_scripts(
    real_font: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(features, "check", lambda _name: False)
    captions._warn_no_raqm.cache_clear()
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        Face(real_font, 30, "\u0928\u092e\u0938\u094d\u0924\u0947")
        Face(real_font, 30, "\u0928\u092e\u0938\u094d\u0924\u0947")
    assert caplog.text.count("lacks libraqm") == 1
    captions._warn_no_raqm.cache_clear()


# ---- track ---------------------------------------------------------------------------------------
@pytest.fixture
def cache(tmp_path: Path):  # type: ignore[no-untyped-def]
    store = Store(tmp_path / "c.db")
    c = Cache(store, RESOURCES)
    yield c
    c.close()
    store.close()


def solid(h: int, w: int, rgba: tuple[int, int, int, int]) -> NDArray[np.uint8]:
    return np.full((h, w, 4), rgba, np.uint8)


def test_track_blits_the_current_sprite_only(cache: Cache) -> None:
    sprites = [
        Sprite(1.0, 2.0, "a", lambda: solid(10, 20, (255, 0, 0, 255))),
        Sprite(2.0, 3.0, "b", lambda: solid(40, 200, (0, 255, 0, 128))),
    ]
    track = CaptionTrack(cache, sprites[::-1], width=100, band_h=30)  # unsorted input is sorted
    clip = track.clip(4.0)
    assert clip.get_frame(0.5).shape == (30, 100, 3) and clip.mask.get_frame(0.5).max() == 0  # nothing yet
    red = clip.get_frame(1.5)
    assert tuple(red[15, 50]) == (255, 0, 0) and clip.mask.get_frame(1.5)[15, 50] == 1.0 and clip.mask.get_frame(1.5)[0, 0] == 0
    green_mask = clip.mask.get_frame(2.5)  # sprite larger than the band: cropped, not an error
    assert green_mask.shape == (30, 100) and green_mask[15, 50] == pytest.approx(128 / 255)
    assert clip.mask.get_frame(3.5).max() == 0  # after the last sprite
    assert track._compose(1.5)[0] is track._compose(1.6)[0]  # unchanged sprite is not recomposed


def test_safe_top_keeps_captions_out_of_app_ui() -> None:
    cfg = CaptionsCfg()
    assert safe_top(0.68, 1920, 1080, cfg, 100) == int(0.68 * 1920 - 50)  # inside the zone: untouched
    assert safe_top(0.95, 1920, 1080, cfg, 100) == int(0.8 * 1920 - 100)  # pushed up above the bottom UI
    assert safe_top(0.0, 1920, 1080, cfg, 100) == int(0.10 * 1920)  # pushed down below the top UI
    assert safe_top(0.95, 1080, 1920, cfg, 100) == int(0.97 * 1080 - 100)  # landscape: thin margins only
    assert safe_top(0.5, 100, 50, CaptionsCfg(safe_top=0.9, safe_bottom=0.9), 80) == 90  # impossible zone: top wins


def test_sprite_key_is_stable_and_sensitive() -> None:
    assert sprite_key("a", 1) == sprite_key("a", 1) and sprite_key("a", 1) != sprite_key("a", 2) and sprite_key("a").startswith("cap:")


def test_without_fillers_closes_the_gap() -> None:
    ws = [Word(0, 1, "so"), Word(1, 1.4, "um,"), Word(1.8, 3.0, "yes"), Word(3, 4, "uh", pseudo=True), Word(9, 10, "end")]
    out = without_fillers(ws, {"um", "uh"})
    assert [w.text for w in out] == ["so", "yes", "uh", "end"]  # pseudo-timed words are never treated as fillers
    assert out[0].end == 1.8  # extended over the removed filler
    assert out[2].end == 4 and out[2].pseudo
    far = without_fillers([Word(0, 1, "a"), Word(1, 2, "um"), Word(30, 31, "b")], {"um"})
    assert far[0].end == 1  # a long real gap is kept


# ---- build_captions ------------------------------------------------------------------------------
def config(**captions_cfg: Any) -> Config:
    return parse(Config, {"captions": captions_cfg})


def sentence() -> list[Word]:
    return words(
        ("Welcome", 0.0, 0.5),
        ("to", 0.5, 0.8),
        ("the", 0.8, 1.0),
        ("show.", 1.0, 1.6),
        ("Um,", 1.6, 1.9),
        ("Next", 2.0, 2.5),
        ("up", 2.5, 3.0),
    )


def test_build_captions_highlights_each_word(cache: Cache, real_font: Path) -> None:
    clip, lines = build_captions(sentence(), Timeline([(0.0, 3.0)]), config(), cache, (360, 640), 3.0)
    assert clip is not None and [ln.text for ln in lines] == ["Welcome to the show.", "Next up"]
    assert all(isinstance(ln, Span) and ln.end > ln.start for ln in lines)
    first = clip.mask.get_frame(0.2)
    assert first.max() > 0 and first.shape[1] == 360
    assert clip.pos(0)[1] >= int(0.10 * 640)  # inside the safe zone


def test_build_captions_plain_uppercase_and_skips(cache: Cache, real_font: Path) -> None:
    cjk = [Word(0.0, 0.5, "\u4eca"), Word(0.5, 1.0, "\u5929"), Word(1.0, 1.5, "\u597d")]
    clip, lines = build_captions(cjk, Timeline([(0.0, 2.0)]), config(uppercase=True), cache, (360, 640), 2.0)
    assert clip is not None and [ln.text for ln in lines] == ["\u4eca\u5929\u597d"]
    shouty, shout_lines = build_captions(
        words(("hello", 0, 1), ("world", 1, 2)), Timeline([(0.0, 2.0)]), config(uppercase=True, highlight=False), cache, (360, 640), 2.0
    )
    assert shouty is not None and shout_lines[0].text == "HELLO WORLD"
    removed, none_lines = build_captions(sentence(), Timeline([(5.0, 6.0)]), config(), cache, (360, 640), 1.0)
    assert removed is None and none_lines == []  # every caption lies in a removed part of the clip
    tiny, _ = build_captions(words(("blink", 0.0, 0.01)), Timeline([(0.0, 0.05)]), config(), cache, (360, 640), 0.05)
    assert tiny is None  # shown for under 80 ms: skipped
    keep_fillers, kf_lines = build_captions(
        sentence(), Timeline([(0.0, 3.0)]), parse(Config, {"silence": {"remove_fillers": False}}), cache, (360, 640), 3.0
    )
    assert keep_fillers is not None and any("Um," in ln.text for ln in kf_lines)


def test_chunk_dataclass_defaults() -> None:
    assert Chunk(0, 1, "x").words == []


def test_plain_text_with_outline_and_skipped_highlight_sprites(cache: Cache, real_font: Path) -> None:
    outlined = render_text_image("hi there", real_font, 30, CaptionsCfg(bg_opacity=0, stroke_width=3, stroke_color="#FF0000"), 400)
    assert ((outlined[..., 0] > 200) & (outlined[..., 1] < 60) & (outlined[..., 3] > 200)).any()  # outline without a highlighted word
    ws = words(("alpha", 0.0, 0.5), ("beta", 0.5, 1.0), ("gamma", 1.0, 1.5))
    clip, lines = build_captions(ws, Timeline([(0.0, 0.5), (1.0, 1.5)]), config(), cache, (360, 640), 1.0)  # "beta" lies in a removed part
    assert clip is not None and len(lines) == 1
