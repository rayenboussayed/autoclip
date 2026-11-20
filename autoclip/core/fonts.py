"""Finding a font that can draw a caption, on Linux, macOS and Windows alike.

System font folders are indexed once per process; per script a list of well-known bold fonts is tried in
order and the first one that really covers the text's glyphs wins (checked with fontTools).
"""

from __future__ import annotations

import functools
import logging
import os
import platform
import unicodedata
from collections.abc import Mapping
from pathlib import Path

from fontTools.ttLib import TTFont

log = logging.getLogger("autoclip")

#: Fonts covering Latin, Cyrillic and Greek (and often Arabic / Hebrew), bold first.
UNIVERSAL = [
    "arialbd.ttf", "Arial Bold.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf",
    "NotoSans-Bold.ttf", "FreeSansBold.ttf", "Arial Unicode.ttf", "Helvetica.ttc",
]  # fmt: skip
CJK = [
    "msyhbd.ttc", "msyh.ttc", "PingFang.ttc", "STHeiti Medium.ttc", "NotoSansCJK-Bold.ttc", "NotoSansCJKsc-Bold.otf",
    "YuGothB.ttc", "meiryob.ttc", "Hiragino Sans GB.ttc", "SourceHanSansSC-Bold.otf", "wqy-microhei.ttc",
]  # fmt: skip
SCRIPT_FONTS: dict[str, list[str]] = {
    "ARABIC": ["NotoSansArabic-Bold.ttf", "tahomabd.ttf", "GeezaPro.ttc"],
    "DEVANAGARI": ["Nirmalab.ttf", "Nirmala.ttf", "Devanagari Sangam MN.ttc", "Lohit-Devanagari.ttf"],
    "THAI": ["LeelawUI.ttf", "leelawad.ttf", "Ayuthaya.ttf", "tahoma.ttf", "Waree-Bold.ttf"],
    "HANGUL": ["malgunbd.ttf", "malgun.ttf", "AppleSDGothicNeo.ttc", "NanumGothicBold.ttf", *CJK],
    "CJK": CJK,
    "HIRAGANA": CJK,
}


def font_dirs(system: str, env: Mapping[str, str], home: Path) -> list[Path]:
    """The folders where ``system`` ("Linux", "Darwin" or "Windows") keeps fonts."""
    if system == "Windows":
        windir = Path(env.get("WINDIR", r"C:\Windows"))
        local = Path(env.get("LOCALAPPDATA", str(home / "AppData" / "Local")))
        return [windir / "Fonts", local / "Microsoft" / "Windows" / "Fonts"]
    if system == "Darwin":
        return [Path("/System/Library/Fonts"), Path("/Library/Fonts"), home / "Library" / "Fonts"]
    folders = [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), home / ".fonts", home / ".local" / "share" / "fonts"]
    if "SNAP" in env:  # strict confinement: stage-packages land under $SNAP, not the base snap's /usr
        folders.append(Path(env["SNAP"]) / "usr/share/fonts")
    return folders


@functools.cache
def font_index() -> dict[str, Path]:
    """Lower-cased file name -> path for every font file in the system font folders."""
    index: dict[str, Path] = {}
    for folder in font_dirs(platform.system(), os.environ, Path.home()):
        for root, _dirs, files in os.walk(folder):
            index.update({f.lower(): Path(root, f) for f in files if f.lower().endswith((".ttf", ".otf", ".ttc"))})
    return index


def scripts_in(text: str) -> list[str]:
    """Every Unicode script with letters in ``text``, most frequent first (``LATIN`` when there are none).

    A caption can mix scripts; looking at only the dominant one misses a dual-script font that is
    installed under a name we do not know, which is why font lookup uses all of them.
    """
    counts: dict[str, int] = {}
    for ch in filter(str.isalpha, text):
        script = unicodedata.name(ch, "LATIN").split()[0]
        counts[script] = counts.get(script, 0) + 1
    kana = counts.pop("HIRAGANA", 0) + counts.pop("KATAKANA", 0)
    ordered = [name for name, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    return (["HIRAGANA"] if kana else []) + ordered or ["LATIN"]


def dominant_script(text: str) -> str:
    """The Unicode script most letters of ``text`` belong to (``LATIN`` when there are none)."""
    return scripts_in(text)[0]


def has_rtl(text: str) -> bool:
    """Does ``text`` contain right-to-left characters (Arabic, Hebrew ...)?"""
    return any(unicodedata.bidirectional(c) in ("R", "AL") for c in text)


@functools.lru_cache(maxsize=256)
def _codepoints(path: Path) -> frozenset[int]:
    """Code points a font file can draw (first face of a collection)."""
    with TTFont(path, fontNumber=0, lazy=True) as font:
        return frozenset(font.getBestCmap())


def _coverage(path: Path, text: str) -> tuple[int, int]:
    """Number of the text's letters/digits present in ``path``, and how many there are."""
    wanted = [c for c in text if unicodedata.category(c)[0] in "LMN"]
    try:
        have = _codepoints(path)
    except Exception:  # unreadable / unsupported font file: treat as not covering
        return 0, len(wanted)
    return sum(ord(c) in have for c in wanted), len(wanted)


def covers(path: Path, text: str) -> bool:
    """Does the font contain at least 95% of the letters and digits of ``text``?"""
    present, total = _coverage(path, text)
    return present >= 0.95 * total  # a punctuation-only string is trivially covered


def covers_all(path: Path, text: str) -> bool:
    """Does the font contain every non-space glyph of ``text`` (the caption renderer's stricter check)?"""
    wanted = [c for c in text if not c.isspace()]
    try:
        have = _codepoints(path)
    except Exception:
        return False
    return bool(wanted) and all(ord(c) in have for c in wanted)


def _scan_fonts(text: str) -> Path | None:
    """Best font from the whole index, ranked by glyph coverage then by a stable preference order."""
    best: tuple[float, int, int, str] | None = None
    chosen: Path | None = None
    index = font_index()
    for name in sorted(index):
        path = index[name]
        present, total = _coverage(path, text)
        if not total or present < 0.95 * total:
            continue
        key = (-present / total, 0 if name.startswith("noto") else 1, 0 if "bold" in name or name.endswith("bd.ttf") else 1, name)
        if best is None or key < best:
            best, chosen = key, path
    return chosen


def find_font(text: str, preferred: str = "auto") -> Path | None:
    """A font file that can draw ``text``: ``preferred`` (path or name) if it covers it, else a system font.

    Candidates come from *every* script in the text (a mixed Arabic+Latin caption needs a font for both),
    then from any installed font, deterministically ranked. Returns ``None`` when nothing suitable exists.
    """
    index = font_index()
    if preferred != "auto":
        direct = Path(preferred).expanduser()
        named = index.get(preferred.lower()) or index.get(f"{preferred.lower()}.ttf")
        for candidate in (direct if direct.is_file() else None, named):
            if candidate and covers(candidate, text):
                return candidate
        log.warning("font '%s' not found or lacks the glyphs for this text; using a system font", preferred)
    names: list[str] = []
    for script in scripts_in(text):
        names += [*SCRIPT_FONTS.get(script, []), f"NotoSans{script.title()}-Bold.ttf", f"NotoSans{script.title()}-Regular.ttf"]
    names += [*UNIVERSAL, "arialuni.ttf", "tahoma.ttf", "NotoSansArabic-Bold.ttf", "NotoSansArabic-Regular.ttf"]
    for name in names:
        path = index.get(name.lower())
        if path and covers(path, text):
            return path
    chosen = _scan_fonts(text)
    if chosen is not None:
        return chosen
    log.warning("no installed font covers %s text (install a Noto font); captions may show empty boxes", dominant_script(text).title())
    return None
