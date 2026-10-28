"""The yt-dlp downloader: file naming, format reports, the download flow, sidecars and ranges.

Everything here is offline: ``yt_dlp`` is replaced by a scripted fake that writes real files, and
the media it writes is the real fixture video, so probing and trimming really run.  The
``network``-marked tests at the bottom talk to the internet and are skipped unless selected.
"""

from __future__ import annotations

import copy
import logging
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from autoclip.core import download
from autoclip.core.config import DownloadCfg, JsRuntimeCfg, parse
from autoclip.core.download import Downloader, FormatReport, FormatRow, format_report_text
from autoclip.core.errors import AutoclipError, DownloadError
from autoclip.core.formats import Choice
from autoclip.core.media import ffmpeg_exe, probe, trim_video

from ..support import Media, make_cfg

URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


# --------------------------------------------------------------------------- helpers


def video_format(
    format_id: str,
    width: int,
    height: int,
    *,
    vcodec: str = "avc1.640028",
    acodec: str = "none",
    ext: str = "mp4",
    fps: float = 30,
    tbr: float = 1000,
    **extra: Any,
) -> dict[str, Any]:
    """One synthetic video format."""
    return {
        "format_id": format_id,
        "width": width,
        "height": height,
        "vcodec": vcodec,
        "acodec": acodec,
        "ext": ext,
        "fps": fps,
        "tbr": tbr,
        **extra,
    }


def info(*formats: dict[str, Any], title: str = "A video", video_id: str = "abc", **extra: Any) -> dict[str, Any]:
    """A full extraction result with the given formats."""
    return {"_type": "video", "id": video_id, "title": title, "formats": list(formats), **extra}


class FakeSession:
    """One ``YoutubeDL`` instance as seen by the downloader (a context manager)."""

    def __init__(self, owner: FakeYtDlp, params: dict[str, Any]) -> None:
        self.owner = owner
        self.params = params
        self.download = False  # the ``download`` argument of the last ``process_ie_result``

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def extract_info(self, url: str, download: bool = False) -> Any:
        """``YoutubeDL.extract_info``."""
        self.owner.calls.append(("extract_info", self, url))
        return self.owner.handler("extract_info", self, url)

    def process_ie_result(self, result: dict[str, Any], download: bool = False) -> Any:
        """``YoutubeDL.process_ie_result``."""
        self.owner.calls.append(("process_ie_result", self, result))
        self.download = download
        return self.owner.handler("process_ie_result", self, result)


class FakeYtDlp:
    """A scripted stand-in for ``yt_dlp.YoutubeDL`` (records every call)."""

    def __init__(self, handler: Any) -> None:
        self.handler = handler
        self.sessions: list[FakeSession] = []
        self.calls: list[tuple[str, FakeSession, Any]] = []

    def __call__(self, params: dict[str, Any]) -> FakeSession:
        session = FakeSession(self, params)
        self.sessions.append(session)
        return session

    @property
    def media_calls(self) -> list[FakeSession]:
        """Sessions that downloaded media (a format was chosen)."""
        return [session for session in self.sessions if "format" in session.params and not session.params.get("skip_download")]

    @property
    def sidecar_calls(self) -> list[FakeSession]:
        """Sessions that only fetched subtitles/metadata."""
        return [session for session in self.sessions if session.params.get("skip_download")]

    @property
    def extraction_calls(self) -> list[FakeSession]:
        """Sessions that only listed the entries of a URL."""
        return [session for session in self.sessions if "extract_flat" in session.params]


def prefix_of(session: FakeSession) -> str:
    """The file-name prefix of a session (read from its output template)."""
    template = str(session.params["outtmpl"]["temp"])
    marker = ".media.%(ext)s"
    assert template.endswith(marker), template
    return template[: -len(marker)]


def write_media(session: FakeSession, source: Path, ext: str = "mp4") -> Path:
    """Write the fixture video where yt-dlp would have left the media (the temporary folder)."""
    target = Path(session.params["paths"]["temp"]) / f"{prefix_of(session)}.media.{ext}"
    shutil.copyfile(source, target)
    return target


def write_home(session: FakeSession, name: str, text: str = "") -> Path:
    """Write a sidecar into the final folder (as yt-dlp's move step would)."""
    target = Path(session.params["paths"]["home"]) / name
    target.write_text(text, encoding="utf-8")
    return target


def simple_handler(media: Media, full: dict[str, Any], *, ranged: Path | None = None) -> FakeYtDlp:
    """A handler that returns ``full`` and saves the fixture video for every media pass."""

    def handle(method: str, session: FakeSession, argument: Any) -> Any:
        if method == "extract_info":
            return full
        if not session.params.get("skip_download"):
            write_media(session, ranged or media.video)
        return argument

    return FakeYtDlp(handle)


def srt(start: float, end: float, text: str = "hello") -> str:
    """A one-cue SRT."""
    return f"1\n{download._srt_time(start)} --> {download._srt_time(end)}\n{text}\n"


def vtt(start: float, end: float, text: str = "hello") -> str:
    """A one-cue WebVTT."""

    def stamp(value: float) -> str:
        millis = round(value * 1000)
        hours, millis = divmod(millis, 3_600_000)
        minutes, millis = divmod(millis, 60_000)
        seconds, millis = divmod(millis, 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

    return f"WEBVTT\n\n{stamp(start)} --> {stamp(end)}\n{text}\n"


@pytest.fixture
def dl_factory(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """``dl_factory(**download_sections)`` -> a Downloader with a deterministic JS runtime."""

    def no_runtime(self: Downloader, cfg: JsRuntimeCfg) -> dict[str, dict[str, str]]:
        return {}

    monkeypatch.setattr(Downloader, "_runtime_config", no_runtime)

    def factory(**sections: Any) -> Downloader:
        base: dict[str, Any] = {"download": {"metadata": {"info_json": False}}}
        for name, value in sections.items():
            base[name] = value
        return Downloader(make_cfg(media, tmp_path / "out", **base))

    return factory


# ------------------------------------------------------------------------ pure helpers


def test_clean_error_strips_every_error_prefix() -> None:
    assert download._clean_error("  ERROR: ERROR: nope  ") == "nope"
    assert download._clean_error("plain") == "plain"


@pytest.mark.parametrize(
    "url, ext",
    [
        ("https://x.test/a/b.en.vtt?token=1", "vtt"),
        ("https://x.test/a/b.SRT", "srt"),
        ("https://x.test/a/b", ""),
        ("", ""),
    ],
)
def test_ext_from_url(url: str, ext: str) -> None:
    assert download._ext_from_url(url) == ext


def test_entry_url_prefers_http_urls_then_recovers_an_id() -> None:
    assert download._entry_url({"url": "https://a.test/1", "webpage_url": "https://a.test/0"}) == "https://a.test/0"
    assert download._entry_url({"original_url": "https://a.test/2"}) == "https://a.test/2"
    assert download._entry_url({"id": "abc", "ie_key": "Youtube"}) == "https://www.youtube.com/watch?v=abc"
    assert download._entry_url({"id": "abc", "url": "not-http", "ie_key": "Vimeo"}) == "not-http"
    assert download._entry_url({"id": "abc"}) is None
    assert download._entry_url({"id": "", "ie_key": "Youtube"}) is None


def test_safe_title_is_portable_and_bounded() -> None:
    assert download._safe_title('a/b\\c:d*e?f"g<h>i|j') == "a_b_c_d_e_f_g_h_i_j"
    assert download._safe_title("  ..hidden--  ") == "hidden"
    assert download._safe_title("") == "video"
    assert download._safe_title("CON") == "_CON"
    assert download._safe_title("com1.txt") == "_com1.txt"
    assert download._safe_title("x" * 200, limit=10) == "x" * 10
    # a multi-byte character is never cut in half
    assert download._safe_title("é" * 100, limit=5) == "éé"


def test_safe_title_handles_dots_and_unicode_limits() -> None:
    assert download._safe_title("..") == "video"
    assert download._safe_title("... name") == "name"
    assert download._safe_title("a" * 79 + "é", limit=80) == "a" * 79  # the é does not fit in 80 bytes


def test_safe_id_is_empty_without_an_id() -> None:
    assert download._safe_id("") == ""
    assert download._safe_id("  ") == ""
    assert download._safe_id("a/b") == "a_b"
    assert download._safe_id("x" * 60) == "x" * 40


def test_range_and_suffix() -> None:
    settings = DownloadCfg(time_range="01:00-02:30")
    assert download._range(settings) == (60.0, 150.0)
    assert download._range(DownloadCfg()) is None
    assert download._range_suffix((60.0, 90.5)) == " [60-90.5s]"
    assert download._range_suffix(None) == ""


def test_prefix_contains_format_container_and_range() -> None:
    choice = Choice(format_id="137")
    assert download._prefix("A [id]", choice, DownloadCfg(), "") == "A [id] [137]"
    assert download._prefix("A [id]", choice, DownloadCfg(container="webm"), "") == "A [id] [137] [webm]"
    assert download._prefix("A [id]", choice, DownloadCfg(), " [0-10s]") == "A [id] [137] [0-10s]"


def test_media_in_finds_the_final_merged_or_single_stream(tmp_path: Path) -> None:
    assert download._media_in(tmp_path, "p") is None
    (tmp_path / "p.mp4").write_bytes(b"")
    assert download._media_in(tmp_path, "p") == tmp_path / "p.mp4"
    (tmp_path / "p.mp4").unlink()
    (tmp_path / "p.media.webm").write_bytes(b"")
    assert download._media_in(tmp_path, "p") == tmp_path / "p.media.webm"
    (tmp_path / "p.media.webm").unlink()
    (tmp_path / "p.media.f137.mp4").write_bytes(b"")
    assert download._media_in(tmp_path, "p") == tmp_path / "p.media.f137.mp4"
    (tmp_path / "p.media.f140.m4a").write_bytes(b"")
    assert download._media_in(tmp_path, "p") is None  # two separate streams were never merged
    (tmp_path / "p.media.f140.m4a").unlink()
    (tmp_path / "p.media.f137.mp4.part").write_bytes(b"")
    (tmp_path / "p.media.f137.mp4").unlink()
    assert download._media_in(tmp_path, "p") is None


def test_canonical_media_renames_and_survives_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    merged = tmp_path / "p.media.MP4"
    merged.write_bytes(b"x")
    assert download._canonical_media(tmp_path, "p", merged) == tmp_path / "p.mp4"
    assert (tmp_path / "p.mp4").read_bytes() == b"x"

    other = tmp_path / "q.media.webm"
    other.write_bytes(b"x")

    def boom(self: Path, target: Path) -> Path:
        raise OSError("read-only")

    monkeypatch.setattr(Path, "replace", boom)
    assert download._canonical_media(tmp_path, "q", other) == other


def test_srt_time_and_text_round_trip() -> None:
    assert download._srt_time(0) == "00:00:00,000"
    assert download._srt_time(3661.5) == "01:01:01,500"
    assert download._srt_time(-3) == "00:00:00,000"
    text = download._srt_text([{"start": 1.0, "end": 2.5, "text": " hello\n"}])
    assert text == "1\n00:00:01,000 --> 00:00:02,500\nhello\n"


def test_shift_srt_moves_and_drops_cues() -> None:
    text = "\n".join(
        [
            "1",
            "00:00:01,000 --> 00:00:02,000",
            "before",
            "",
            "2",
            "00:00:06,000 --> 00:00:08,000",
            "inside",
            "",
            "3",
            "00:00:09,000 --> 00:00:12,000",
            "clipped",
            "",
        ]
    )
    shifted = download._shift_srt(text, (5.0, 10.0), 5.0)
    assert "before" not in shifted
    assert "00:00:01,000 --> 00:00:03,000" in shifted
    assert "00:00:04,000 --> 00:00:05,000" in shifted
    with pytest.raises(DownloadError, match="do not cover"):
        download._shift_srt(text, (30.0, 40.0), 10.0)


def test_write_srt_passes_an_srt_through(tmp_path: Path) -> None:
    source = tmp_path / "a.srt"
    source.write_text(srt(1, 2), encoding="utf-8")
    target = tmp_path / "out.srt"
    download._write_srt(source, target, None, lambda data: "")
    assert target.read_text(encoding="utf-8") == srt(1, 2)
    assert not (tmp_path / "out.srt.tmp.srt").exists()


def test_write_srt_uses_the_ttml_converter(tmp_path: Path) -> None:
    source = tmp_path / "a.ttml"
    source.write_bytes(b"<tt/>")
    target = tmp_path / "out.srt"
    seen: list[bytes] = []

    def convert(data: bytes) -> str:
        seen.append(data)
        return srt(1, 2)

    download._write_srt(source, target, None, convert)
    assert seen == [b"<tt/>"]
    assert target.read_text(encoding="utf-8") == srt(1, 2)


def test_write_srt_converts_other_formats_with_ffmpeg(tmp_path: Path) -> None:
    source = tmp_path / "a.vtt"
    source.write_text(vtt(1, 2), encoding="utf-8")
    target = tmp_path / "out.srt"
    download._write_srt(source, target, None, lambda data: "")
    text = target.read_text(encoding="utf-8")
    assert "hello" in text and "-->" in text


def test_write_srt_rejects_empty_and_unconvertible_files(tmp_path: Path) -> None:
    empty = tmp_path / "a.srt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(DownloadError, match="empty"):
        download._write_srt(empty, tmp_path / "out.srt", None, lambda data: "")
    junk = tmp_path / "b.ass"
    junk.write_text("not a subtitle", encoding="utf-8")
    with pytest.raises(DownloadError, match="cannot convert"):
        download._write_srt(junk, tmp_path / "out2.srt", None, lambda data: "")
    assert not (tmp_path / "out2.srt.tmp.srt").exists()


def test_write_srt_shifts_a_range(tmp_path: Path) -> None:
    source = tmp_path / "a.srt"
    source.write_text(srt(6, 8), encoding="utf-8")
    target = tmp_path / "out.srt"
    download._write_srt(source, target, (5.0, 10.0), lambda data: "")
    assert "00:00:01,000 --> 00:00:03,000" in target.read_text(encoding="utf-8")


def test_pick_variant_skips_what_it_cannot_convert() -> None:
    assert download._pick_variant("nope") is None
    variants = [
        {"ext": "json3", "url": "https://x/1"},
        {"ext": "vtt", "url": "https://x/2", "has_drm": True},
        {"ext": "vtt", "url": "https://x/3", "kind": "fonts"},
        {"url": "https://x/4.en.vtt"},
        {"url": "https://x/5", "ext": "mhtml"},
        {"url": "https://x/6.en.srt", "kind": "captions"},
    ]
    assert download._pick_variant(variants) == {"url": "https://x/6.en.srt", "kind": "captions", "ext": "srt"}
    assert download._pick_variant([{"url": "https://x/7"}]) is None
    # a lone unusable variant is skipped before the loop ends
    assert download._pick_variant([{"url": "https://x/8", "has_drm": True}]) is None
    assert download._pick_variant([{"url": "https://x/9", "kind": "fonts"}]) is None


def test_subtitle_file_finds_each_converted_extension(tmp_path: Path) -> None:
    assert download._subtitle_file(tmp_path, "p", "en") is None
    for ext in ("vtt", "srt"):
        path = tmp_path / f"p.subtitle.en.{ext}"
        path.write_text(srt(1, 2), encoding="utf-8")
    assert download._subtitle_file(tmp_path, "p", "en") == tmp_path / "p.subtitle.en.srt"


def test_pick_sub_prefers_exact_then_prefix_and_manual_store_order() -> None:
    store = {
        "en-US": [{"ext": "vtt", "url": "https://x/en-US.vtt"}],
        "en": [{"ext": "json3", "url": "https://x/en.json3"}],
        "fr": [{"ext": "vtt", "url": "https://x/fr.vtt"}],
        "de": "not a list",
    }
    assert download._pick_sub(store, "en") == ("en-US", {"ext": "vtt", "url": "https://x/en-US.vtt"})
    assert download._pick_sub(store, "FR") == ("fr", {"ext": "vtt", "url": "https://x/fr.vtt"})
    assert download._pick_sub(store, "de") is None
    assert download._pick_sub(store, " ") is None
    assert download._pick_sub("nope", "en") is None


def test_row_describes_video_audio_and_storyboards() -> None:
    assert download._row(video_format("137", 1920, 1080)) == FormatRow(
        "137", "mp4", "1920x1080", "avc1.640028", "30", "1000k", "video only"
    )
    assert download._row({"format_id": "140", "ext": "m4a", "vcodec": "none", "acodec": "mp4a", "abr": 128}).quality == "audio"
    assert "audio only" in download._row({"format_id": "140", "ext": "m4a", "vcodec": "none", "acodec": "mp4a"}).notes
    assert "DRM" in download._row(video_format("1", 640, 360, has_drm=True)).notes
    assert "storyboard" in download._row({"format_id": "sb", "ext": "mhtml", "vcodec": "none", "acodec": "none"}).notes
    row = download._row({"format_id": None, "vcodec": None, "acodec": None})
    assert row.format_id == "?" and row.quality == "?" and row.codec == "?" and row.fps == "-" and row.bitrate == "-"


def test_clean_info_removes_the_bookkeeping_of_a_previous_pass() -> None:
    full = {key: "x" for key in download.STALE_KEYS}
    full["title"] = "keep"
    cleaned = download._clean_info(full)
    assert cleaned == {"title": "keep"}
    assert full["title"] == "keep" and "requested_formats" in full  # the original is untouched


def test_format_report_text_lists_rows_and_warnings() -> None:
    report = FormatReport(
        url=URL,
        title="Title",
        platform="youtube",
        chosen="137+140",
        rows=(FormatRow("137", "mp4", "1920x1080", "avc1", "30", "1000k", "video only"),),
        warnings=("careful",),
    )
    text = format_report_text(report)
    assert "title:    Title" in text
    assert "chosen:   137+140" in text
    assert "137" in text and "video only" in text
    assert "! careful" in text
    plain = format_report_text(FormatReport(url=URL, title="Plain", platform="youtube", chosen="137", rows=(), warnings=()))
    assert "Plain" in plain and "!" not in plain


def test_format_report_truncates_a_long_codec_instead_of_shifting_the_columns() -> None:
    row = FormatRow("602", "mp4", "192x144", "vp09.00.20.08.01.05.01.06.0015", "15", "75k", "video only")
    text = format_report_text(FormatReport(url=URL, title="T", platform="youtube", chosen="602", rows=(row,), warnings=()))
    line = next(line for line in text.splitlines() if line.startswith("602"))
    assert line[63:] == "video only"  # NOTES starts after ID+EXT+QUALITY+CODEC+FPS+BITRATE
    assert "…" in line


def test_empty_removes_files_and_directories_but_survives_a_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "temp"
    (directory / "nested").mkdir(parents=True)
    (directory / "a.srt").write_text("x", encoding="utf-8")
    (directory / "nested" / "b.srt").write_text("x", encoding="utf-8")
    download._empty(directory)
    assert list(directory.iterdir()) == []

    locked = directory / "locked.srt"
    locked.write_text("x", encoding="utf-8")

    def refuse(self: Path, missing_ok: bool = False) -> None:
        raise OSError("in use")

    monkeypatch.setattr(Path, "unlink", refuse)
    download._empty(directory)  # best effort: one locked file must not raise
    assert locked.is_file()


def test_ydl_logger_routes_to_the_autoclip_logger(caplog: pytest.LogCaptureFixture) -> None:
    logger = download._YdlLogger()
    with caplog.at_level(logging.DEBUG, logger="autoclip"):
        logger.debug("[debug] detail")
        logger.info("info")
        logger.warning("warn")
        logger.error("bad")
    messages = [record.getMessage() for record in caplog.records]
    assert "yt-dlp: detail" in messages
    assert "yt-dlp: info" in messages
    assert "yt-dlp: warn" in messages
    assert "yt-dlp: bad" in messages


# ---------------------------------------------------------------------- construction


def test_downloader_imports_yt_dlp_and_reports_the_policy(dl_factory: Any) -> None:
    dl = dl_factory()
    assert dl.platform(URL) == "youtube"
    assert dl.permitted("YouTube")
    assert dl.describe(URL) == "youtube: allowed"
    text = dl.policy_text()
    assert "allow: @popular" in text
    assert "deny: -" in text
    assert "permitted: " in text


def test_describe_and_policy_text_name_groups_and_settings(dl_factory: Any) -> None:
    dl = dl_factory(
        download={
            "metadata": {"info_json": False},
            "platforms": {
                "allow": ["@popular", "@mine"],
                "deny": ["tiktok"],
                "groups": {"@mine": ["vimeo"]},
                "settings": {"vimeo": {"quality": 720}},
            },
        }
    )
    assert dl.describe("https://vimeo.com/1").endswith("vimeo: allowed")
    assert "blocked" in dl.describe("https://www.tiktok.com/@a/video/1")
    text = dl.policy_text()
    assert "groups: @mine = vimeo" in text
    assert "settings: vimeo" in text


def test_allowed_extractors_follow_the_allow_list(dl_factory: Any) -> None:
    popular = dl_factory()
    assert "end" in popular.allowed_extractors
    assert "-generic" not in popular.allowed_extractors

    without_generic = dl_factory(download={"metadata": {"info_json": False}, "platforms": {"allow": ["@all"]}})
    assert without_generic.allowed_extractors == ["default", "-generic", "-html5", "end"]

    everything = dl_factory(download={"metadata": {"info_json": False}, "platforms": {"allow": ["@all", "generic"]}})
    assert everything.allowed_extractors == ["default", "end"]


def test_permitted_is_false_for_a_family_outside_the_allow_list(dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}, "platforms": {"allow": ["vimeo"]}})
    assert dl.permitted("vimeo")
    assert not dl.permitted("youtube")
    assert dl._require("https://vimeo.com/1") == "vimeo"
    with pytest.raises(DownloadError, match="youtube: blocked"):
        dl._require(URL)


def test_missing_yt_dlp_is_an_actionable_error(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    cfg = make_cfg(media, tmp_path / "out")
    monkeypatch.setitem(sys.modules, "yt_dlp", None)
    with pytest.raises(DownloadError, match='pip install "yt-dlp\\[default\\]"'):
        Downloader(cfg)


def test_runtime_config_picks_the_first_available_runtime() -> None:
    class Missing:
        def __init__(self, path: str | None = None) -> None:
            self.info = None

    class Found:
        def __init__(self, path: str | None = None) -> None:
            self.info = {"version": path or "1"}

    dl = object.__new__(Downloader)
    dl._runtime_classes = {"deno": Found, "node": Found, "bun": Missing, "quickjs": Missing}
    assert dl._runtime_config(JsRuntimeCfg(name="auto")) == {"deno": {}}

    dl._runtime_classes = {"deno": Missing, "node": Found}
    assert dl._runtime_config(JsRuntimeCfg(name="auto")) == {"node": {}}

    dl._runtime_classes = {"deno": Missing, "node": Missing, "bun": Missing, "quickjs": Missing}
    assert dl._runtime_config(JsRuntimeCfg(name="auto")) == {}


def test_runtime_config_named_runtime_path_and_failures(tmp_path: Path) -> None:
    class Missing:
        def __init__(self, path: str | None = None) -> None:
            self.info = None

    class Found:
        def __init__(self, path: str | None = None) -> None:
            self.info = {"version": path or "1"}

    binary = tmp_path / "node"
    binary.write_text("", encoding="utf-8")
    dl = object.__new__(Downloader)
    dl._runtime_classes = {"node": Missing, "bun": Found}
    assert dl._runtime_config(JsRuntimeCfg(name="node", path=str(binary))) == {"node": {"path": str(binary)}}
    assert dl._runtime_config(JsRuntimeCfg(name="node")) == {}
    assert dl._runtime_config(JsRuntimeCfg(name="bun")) == {"bun": {}}
    assert dl._runtime_config(JsRuntimeCfg(name="quickjs")) == {}  # not in the table at all
    with pytest.raises(DownloadError, match="is not a file"):
        dl._runtime_config(JsRuntimeCfg(name="node", path=str(tmp_path / "absent")))


# ------------------------------------------------------------------------- settings


def test_per_platform_settings_are_merged_and_cached(dl_factory: Any) -> None:
    dl = dl_factory(
        download={
            "metadata": {"info_json": False},
            "subtitles": {"languages": ["en", "ar"]},
            "platforms": {
                "allow": ["@popular"],
                "settings": {"youtube": {"quality": 720, "subtitles": {"languages": ["ar"]}}},
            },
        }
    )
    youtube = dl._settings("YouTube")
    assert youtube.quality == 720
    assert youtube.subtitles.languages == ["ar"]
    assert dl._settings("youtube") is youtube
    tiktok = dl._settings("tiktok")
    assert tiktok.quality == 1080
    assert tiktok.subtitles.languages == ["en", "ar"]


def test_base_params_carry_network_cookies_and_playlist_policy(dl_factory: Any) -> None:
    dl = dl_factory(
        download={
            "metadata": {"info_json": False},
            "playlist": "all",
            "max_videos": 3,
            "cookies": {"from_browser": "firefox"},
            "network": {"proxy": "http://127.0.0.1:8080", "rate_limit_mbps": 8, "retries": 5, "socket_timeout": 7},
        }
    )
    settings = dl._settings("youtube")
    params = dl._base_params(settings)
    assert params["proxy"] == "http://127.0.0.1:8080"
    assert params["ratelimit"] == 8 * 1024 * 1024 / 8
    assert params["retries"] == 5 and params["fragment_retries"] == 5
    assert params["socket_timeout"] == 7
    assert params["cookiesfrombrowser"] == ("firefox", None, None, None)
    assert params["noplaylist"] is False and params["playlistend"] == 3
    assert params["allowed_extractors"] == dl.allowed_extractors
    assert params["js_runtimes"] == {}
    assert params["remote_components"] == []
    assert params["ffmpeg_location"] == ffmpeg_exe()


def test_base_params_single_playlist_and_cookie_file(dl_factory: Any, tmp_path: Path) -> None:
    cookie = tmp_path / "cookies.txt"
    dl = dl_factory(
        download={"metadata": {"info_json": False}, "cookies": {"file": str(cookie)}},
    )
    params = dl._base_params(dl._settings("youtube"))
    assert params["cookiefile"] == str(cookie)
    assert "cookiesfrombrowser" not in params
    assert params["noplaylist"] is True and params["playlistend"] == 1
    assert "proxy" not in params and "ratelimit" not in params


def test_outtmpl_escapes_percent_and_uses_every_sidecar_key() -> None:
    templates = Downloader._outtmpl("100% [id]")
    assert templates == {
        "default": "100%% [id].%(ext)s",
        "temp": "100%% [id].media.%(ext)s",
        "subtitle": "100%% [id].subtitle.%(ext)s",
        "thumbnail": "100%% [id].thumbnail.%(ext)s",
        "description": "100%% [id]",
        "infojson": "100%% [id]",
    }


def test_directory_groups_by_platform(tmp_path: Path, media: Media, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Downloader, "_runtime_config", lambda self, cfg: {})
    cfg = make_cfg(media, tmp_path / "out", download={"metadata": {"info_json": False}, "platforms": {"group_by_platform": True}})
    dl = Downloader(cfg)
    directory = dl._directory("youtube")
    assert directory == tmp_path / "out" / ".work" / "downloads" / "youtube"
    assert directory.is_dir()


# -------------------------------------------------------------------------- entries


def test_entries_of_a_single_video_and_of_a_playlist(dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    one = info(video_format("137", 1920, 1080), video_id="one")
    fake = FakeYtDlp(lambda method, session, argument: one)
    dl._YoutubeDL = fake
    assert dl._entries(URL, dl._settings("youtube"), limit=None) == [one]
    assert fake.calls[0][1].params["extract_flat"] == "in_playlist"

    playlist: dict[str, Any] = {
        "_type": "playlist",
        "entries": [
            info(video_format("137", 1920, 1080), video_id="one"),
            "not a dict",
            info(video_format("137", 1920, 1080), video_id="two"),
        ],
    }
    fake = FakeYtDlp(lambda method, session, argument: playlist)
    dl._YoutubeDL = fake
    entries = dl._entries(URL, dl._settings("youtube"), limit=None)
    assert [entry["id"] for entry in entries] == ["one"]
    assert dl._entries(URL, dl._settings("youtube"), limit=2) == playlist["entries"][:1] + playlist["entries"][2:]
    assert fake.sessions[-1].params["playlistend"] == 2

    dl.cfg.download.playlist = "all"
    settings = parse(DownloadCfg, {**asdict(dl.cfg.download)})
    assert len(dl._entries(URL, settings, limit=None)) == 2


def test_entries_reject_an_empty_playlist_and_a_non_dict(dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    settings = dl._settings("youtube")
    dl._YoutubeDL = FakeYtDlp(lambda method, session, argument: {"_type": "playlist", "entries": []})
    with pytest.raises(DownloadError, match="no downloadable entries"):
        dl._entries(URL, settings, limit=None)
    dl._YoutubeDL = FakeYtDlp(lambda method, session, argument: "no")
    with pytest.raises(DownloadError, match="no information"):
        dl._entries(URL, settings, limit=None)


def test_full_info_reuses_a_resolved_entry_and_resolves_a_flat_one(dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    resolved = info(video_format("137", 1920, 1080))
    assert dl._full_info(resolved, None, dl._settings("youtube")) is resolved

    seen: list[dict[str, Any]] = []

    def handler(method: str, session: FakeSession, argument: Any) -> Any:
        seen.append(copy.deepcopy(argument))
        return resolved

    dl._YoutubeDL = FakeYtDlp(handler)
    flat = {"_type": "url", "id": "abc", "url": URL, "ie_key": "Youtube"}
    assert dl._full_info(flat, URL, dl._settings("youtube")) == resolved
    assert seen[0]["_type"] == "url" and seen[0]["url"] == URL

    with pytest.raises(DownloadError, match="has no URL"):
        dl._full_info({"_type": "url"}, None, dl._settings("youtube"))


def test_full_info_rejects_a_non_dict_result(dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    dl._YoutubeDL = FakeYtDlp(lambda method, session, argument: None)
    with pytest.raises(DownloadError, match="no information"):
        dl._full_info({"_type": "url", "id": "x"}, URL, dl._settings("youtube"))


def test_entry_family_reads_the_extractor_its_key_or_the_url(dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    assert dl._entry_family({"extractor": "youtube:tab"}, "generic", None) == "youtube"
    assert dl._entry_family({"extractor": "x"}, "generic", None) == "twitter"
    assert dl._entry_family({"extractor_key": "Youtube"}, "generic", None) == "youtube"
    assert dl._entry_family({"ie_key": "Vimeo"}, "generic", None) == "vimeo"
    assert dl._entry_family({"ie_key": "NoSuchExtractor"}, "youtube", URL) == "youtube"
    assert dl._entry_family({}, "youtube", "https://vimeo.com/1") == "vimeo"
    assert dl._entry_family({"extractor": "generic"}, "youtube", URL) == "youtube"
    assert dl._entry_family({"extractor": "generic"}, "generic", "https://example.com/a.mp4") == "generic"
    assert dl._entry_family({}, "youtube", None) == "youtube"


# ------------------------------------------------------------------------ preparing


def test_prepare_deduplicates_names_and_reuses_the_cache(dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    full = info(video_format("137", 1920, 1080), title="Same", video_id="id")
    settings = dl._settings("youtube")
    name, directory, prefix, cached = dl._prepare(full, "youtube", settings, Choice("137"), None)
    assert (name, prefix, cached) == ("Same [id]", "Same [id] [137]", None)
    media = directory / f"{prefix}.mp4"
    media.write_bytes(b"x")
    # the same format is in the cache: the same name is returned, not a numbered one
    assert dl._prepare(full, "youtube", settings, Choice("137"), None)[3] == media

    # another format of the same video caches under another prefix but must not reuse the name:
    # the name is the output/work folder, which would otherwise be shared by two clips
    other = info(video_format("22", 1280, 720), title="Same", video_id="id")
    other_name, _, other_prefix, other_cached = dl._prepare(other, "youtube", settings, Choice("22"), None)
    assert (other_name, other_prefix, other_cached) == ("Same [id] (2)", "Same [id] (2) [22]", None)

    duplicate = info(video_format("137", 1920, 1080), title="Same", video_id="id")
    media.unlink()  # no cache hit any more, so the name is taken and has to be numbered
    dl2 = dl_factory(download={"metadata": {"info_json": False}})
    dl2._used = dl._used
    dl2.root = dl.root
    assert dl2._prepare(duplicate, "youtube", settings, Choice("137"), None)[0] == "Same [id] (3)"

    # a different id is a different name
    other = info(video_format("137", 1920, 1080), title="Same", video_id="other")
    assert dl._prepare(other, "youtube", settings, Choice("137"), None)[0] == "Same [other]"


def test_prepare_adds_the_container_and_range_to_the_prefix(dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    full = info(video_format("137", 1920, 1080), title="T", video_id="i")
    settings = parse(DownloadCfg, {"container": "webm", "time_range": "00:10-00:20"})
    name, directory, prefix, _ = dl._prepare(full, "youtube", settings, Choice("137"), (10.0, 20.0))
    assert name == "T [i] [10-20s]"
    assert prefix == "T [i] [137] [webm] [10-20s]"
    assert directory == dl.root


def test_prepare_without_an_id_does_not_add_brackets(dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    full = {"title": "Only title", "formats": [video_format("137", 1920, 1080)]}
    name, _, prefix, _ = dl._prepare(full, "youtube", dl._settings("youtube"), Choice("137"), None)
    assert name == "Only title"
    assert prefix == "Only title [137]"


def test_duration_ok_uses_a_generous_tolerance(media: Media, tmp_path: Path, dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    assert dl._duration_ok(media.video, (0.0, 40.0))
    assert dl._duration_ok(media.video, (0.0, 42.0))  # 40 s against 42 s is within max(3 s, 10 %)
    assert not dl._duration_ok(media.video, (0.0, 30.0))  # 10 s of difference is too much
    assert not dl._duration_ok(media.video, (0.0, 48.0))
    garbage = tmp_path / "bad.mp4"
    garbage.write_bytes(b"not a video")
    assert not dl._duration_ok(garbage, (0.0, 10.0))


def test_collect_moves_from_the_temporary_folder_and_reports_failures(
    media: Media, tmp_path: Path, dl_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    directory = tmp_path / "final"
    temporary = tmp_path / "temp"
    directory.mkdir()
    temporary.mkdir()
    assert dl._collect(directory, temporary, "p") is None
    shutil.copyfile(media.video, temporary / "p.media.mp4")
    collected = dl._collect(directory, temporary, "p")
    assert collected == directory / "p.mp4" and collected.is_file()

    shutil.copyfile(media.video, temporary / "q.media.mp4")

    def broken(source: str, target: str) -> None:
        raise OSError("cross-device")

    monkeypatch.setattr(shutil, "move", broken)
    with pytest.raises(DownloadError, match="cannot move"):
        dl._collect(directory, temporary, "q")


def test_collect_uses_media_already_moved_into_the_final_folder(media: Media, tmp_path: Path, dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    directory = tmp_path / "final"
    temporary = tmp_path / "temp"
    directory.mkdir()
    temporary.mkdir()
    shutil.copyfile(media.video, directory / "p.media.mp4")
    assert dl._collect(directory, temporary, "p") == directory / "p.mp4"
    assert (directory / "p.mp4").is_file()


# ------------------------------------------------------------------------ download


def playlist(*entries: dict[str, Any]) -> dict[str, Any]:
    """A flat playlist result."""
    return {"_type": "playlist", "entries": list(entries)}


def flat_entry(video_id: str, url: str = "", **extra: Any) -> dict[str, Any]:
    """One entry of a flat playlist."""
    entry: dict[str, Any] = {"_type": "url", "id": video_id, "title": f"Entry {video_id}"}
    if url:
        entry["url"] = url
    entry.update(extra)
    return entry


def flow_fake(
    media: Media,
    flat: dict[str, Any],
    resolved: dict[str, dict[str, Any]] | None = None,
    *,
    media_source: Path | None = None,
    ranged_source: Path | None = None,
    sidecar_files: dict[str, str] | None = None,
    media_errors: tuple[Exception, ...] = (),
    sidecar_error: Exception | None = None,
) -> FakeYtDlp:
    """A yt-dlp stand-in for a whole download flow: flat entries, resolution and real files."""
    table = resolved or {}
    pending = list(media_errors)

    def handle(method: str, session: FakeSession, argument: Any) -> Any:
        if method == "extract_info":
            return flat
        if not session.download:
            key = str(argument.get("id") or "")
            assert key in table, f"unexpected resolution of {argument}"
            return table[key]
        if session.params.get("skip_download"):
            if sidecar_error is not None:
                raise sidecar_error
            for name, text in (sidecar_files or {}).items():
                write_home(session, name, text)
            return argument
        if pending:
            raise pending.pop(0)
        ranged = session.params.get("download_ranges") is not None
        source = ranged_source if ranged and ranged_source is not None else media_source or media.video
        write_media(session, source)
        return argument

    return FakeYtDlp(handle)


def ten_seconds(media: Media, tmp_path: Path) -> Path:
    """A real 10 s clip cut out of the fixture (the result of a range download)."""
    target = tmp_path / "ten.mp4"
    trim_video(media.video, target, 10.0, 20.0)
    return target


def test_download_saves_the_media_in_the_cache(media: Media, dl_factory: Any, caplog: pytest.LogCaptureFixture) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    full = info(video_format("137", 1920, 1080), title="Nice", video_id="v1")
    fake = flow_fake(media, full)
    dl._YoutubeDL = fake
    with caplog.at_level(logging.INFO, logger="autoclip"):
        result = dl.download(URL)

    assert [(item.name, item.platform, item.subtitle, item.subtitle_manual) for item in result] == [("Nice [v1]", "youtube", None, False)]
    item = result[0]
    expected = dl.root / "Nice [v1] [137].mp4"
    assert item.path == expected and expected.is_file()
    assert "downloaded Nice [v1] (youtube)" in [record.getMessage() for record in caplog.records]

    assert len(fake.extraction_calls) == 1
    assert fake.extraction_calls[0].params["extract_flat"] == "in_playlist"
    assert len(fake.media_calls) == 1
    params = fake.media_calls[0].params
    assert params["format"] == "137"
    assert params["paths"]["home"] == str(dl.root)
    assert params["outtmpl"]["default"] == "Nice [v1] [137].%(ext)s"
    assert fake.calls[1][0] == "process_ie_result" and fake.calls[1][1].download is True
    assert not fake.sidecar_calls
    assert [path.name for path in dl.root.iterdir()] == ["Nice [v1] [137].mp4"]


def test_download_reuses_the_cached_original_without_a_media_pass(media: Media, dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    full = info(video_format("137", 1920, 1080), title="Nice", video_id="v1")
    dl._YoutubeDL = flow_fake(media, full)
    saved = dl.download(URL)[0]

    second = flow_fake(media, full)
    dl._YoutubeDL = second
    again = dl.download(URL)[0]
    assert again.path == saved.path and again.name == saved.name
    assert len(second.extraction_calls) == 1
    assert not second.media_calls and not second.sidecar_calls


def test_a_cache_hit_fetches_only_the_missing_sidecars(media: Media, dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": True}})
    full = info(video_format("137", 1920, 1080), title="Nice", video_id="v1")
    prefix = "Nice [v1] [137]"
    dl._YoutubeDL = flow_fake(media, full, sidecar_files={f"{prefix}.info.json": "{}"})
    dl.download(URL)
    assert (dl.root / f"{prefix}.info.json").is_file()
    assert len(dl._YoutubeDL.sidecar_calls) == 1

    dl._YoutubeDL = flow_fake(media, full, sidecar_files={f"{prefix}.info.json": "{}"})
    dl.download(URL)
    assert not dl._YoutubeDL.media_calls and not dl._YoutubeDL.sidecar_calls  # nothing is missing

    (dl.root / f"{prefix}.info.json").unlink()
    dl._YoutubeDL = flow_fake(media, full, sidecar_files={f"{prefix}.info.json": '{"again": true}'})
    dl.download(URL)
    assert len(dl._YoutubeDL.sidecar_calls) == 1
    assert (dl.root / f"{prefix}.info.json").read_text(encoding="utf-8") == '{"again": true}'


def test_a_playlist_downloads_every_entry_and_skips_a_failing_one(media: Media, dl_factory: Any, caplog: pytest.LogCaptureFixture) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}, "playlist": "all"})
    entries = [flat_entry("a", URL), flat_entry("b", URL)]
    resolved = {
        "a": info(video_format("137", 1920, 1080), title="A", video_id="a"),
        "b": info(video_format("137", 1920, 1080), title="B", video_id="b"),
    }

    def handle(method: str, session: FakeSession, argument: Any) -> Any:
        if method == "extract_info":
            return playlist(*entries)
        if not session.download:
            return resolved[str(argument["id"])]
        if argument["id"] == "b":
            raise DownloadError("b is gone")
        write_media(session, media.video)
        return argument

    dl._YoutubeDL = FakeYtDlp(handle)
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        result = dl.download(URL)
    assert [item.name for item in result] == ["A [a]"]
    assert "skipping a playlist entry" in [record.getMessage() for record in caplog.records][0]
    assert (dl.root / "A [a] [137].mp4").is_file()


def test_a_playlist_where_every_entry_fails_raises(media: Media, dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}, "playlist": "all"})
    entries = [flat_entry("a", URL), flat_entry("b", URL)]
    resolved = {
        "a": info(video_format("137", 1920, 1080), title="A", video_id="a"),
        "b": info(video_format("137", 1920, 1080), title="B", video_id="b"),
    }

    def handle(method: str, session: FakeSession, argument: Any) -> Any:
        if method == "extract_info":
            return playlist(*entries)
        if not session.download:
            return resolved[str(argument["id"])]
        raise DownloadError("gone")

    dl._YoutubeDL = FakeYtDlp(handle)
    with pytest.raises(DownloadError, match="nothing could be downloaded"):
        dl.download(URL)


def test_a_single_video_failure_is_raised_as_it_is(media: Media, dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    full = info(video_format("137", 1920, 1080), title="Nice", video_id="v1")
    dl._YoutubeDL = flow_fake(media, full, media_errors=(DownloadError("gone"),))
    with pytest.raises(DownloadError, match="gone"):
        dl.download(URL)


def test_a_playlist_sleeps_between_entries(
    media: Media, dl_factory: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}, "playlist": "all", "network": {"sleep_between_seconds": 2.5}})
    entries = [flat_entry("a", URL), flat_entry("b", URL)]
    resolved = {
        "a": info(video_format("137", 1920, 1080), title="A", video_id="a"),
        "b": info(video_format("137", 1920, 1080), title="B", video_id="b"),
    }

    def handle(method: str, session: FakeSession, argument: Any) -> Any:
        if method == "extract_info":
            return playlist(*entries)
        if not session.download:
            return resolved[str(argument["id"])]
        write_media(session, media.video)
        return argument

    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    dl._YoutubeDL = FakeYtDlp(handle)
    with caplog.at_level(logging.INFO, logger="autoclip"):
        dl.download(URL)
    assert slept == [2.5]  # never before the first entry


def test_a_blocked_platform_fails_before_any_request(dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}, "platforms": {"allow": ["vimeo"]}})

    def no_request(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("a blocked URL must not reach yt-dlp")

    fake = FakeYtDlp(no_request)
    dl._YoutubeDL = fake
    with pytest.raises(DownloadError, match="youtube: blocked"):
        dl.download(URL)
    assert fake.sessions == []


def test_a_blocked_playlist_entry_is_reported(dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}, "platforms": {"allow": ["youtube"]}})
    entries = playlist(flat_entry("t", "https://www.tiktok.com/@a/video/1"))
    dl._YoutubeDL = FakeYtDlp(lambda method, session, argument: entries)
    with pytest.raises(DownloadError, match="'tiktok' is blocked"):
        dl.download(URL)


def test_a_full_extraction_that_is_a_playlist_downloads_each_entry(media: Media, dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    first = info(video_format("137", 1920, 1080), title="First", video_id="f1")
    second = info(video_format("137", 1920, 1080), title="Second", video_id="s2")
    outer = {"_type": "url", "id": "outer", "title": "Outer", "url": URL}

    def handle(method: str, session: FakeSession, argument: Any) -> Any:
        if method == "extract_info":
            return outer
        if not session.download:
            return {"_type": "playlist", "entries": [first, "junk", second]}
        write_media(session, media.video)
        return argument

    dl._YoutubeDL = FakeYtDlp(handle)
    assert [item.name for item in dl.download(URL)] == ["First [f1]", "Second [s2]"]


def test_no_entries_is_a_download_error(dl_factory: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    monkeypatch.setattr(dl, "_entries", lambda *args, **kwargs: [])
    with pytest.raises(DownloadError, match="no downloadable entries"):
        dl.formats(URL)
    with pytest.raises(DownloadError, match="no downloadable entries"):
        dl.download(URL)


def test_a_playlist_entry_of_another_platform_uses_its_own_settings(media: Media, dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}, "platforms": {"settings": {"vimeo": {"quality": 720}}}})
    entries = [flat_entry("v", "https://vimeo.com/1")]
    full = info(video_format("v1080", 1920, 1080), video_format("v720", 1280, 720), title="V", video_id="v")
    fake = flow_fake(media, playlist(*entries), {"v": full})
    dl._YoutubeDL = fake

    item = dl.download(URL)[0]
    assert (item.name, item.platform) == ("V [v]", "vimeo")
    assert fake.media_calls[0].params["format"] == "v720"
    assert item.path == dl.root / "V [v] [v720].mp4"


def test_an_audio_only_url_is_rejected(media: Media, dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    full = info({"format_id": "140", "vcodec": "none", "acodec": "mp4a", "ext": "m4a", "abr": 128})
    dl._YoutubeDL = flow_fake(media, full)
    with pytest.raises(DownloadError, match="only offers audio"):
        dl.download(URL)


# --------------------------------------------------------------------------- ranges


def test_a_range_is_downloaded_by_yt_dlp_when_ffmpeg_can_cut(media: Media, tmp_path: Path, dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}, "time_range": "00:10-00:20"})
    dl._range_supported = True
    full = info(video_format("137", 1920, 1080), title="Nice", video_id="v1")
    fake = flow_fake(media, full, ranged_source=ten_seconds(media, tmp_path))
    dl._YoutubeDL = fake

    item = dl.download(URL)[0]
    assert item.name == "Nice [v1] [10-20s]"
    assert item.path == dl.root / "Nice [v1] [137] [10-20s].mp4"
    assert probe(item.path).duration == pytest.approx(10, abs=0.5)
    assert len(fake.media_calls) == 1
    assert fake.media_calls[0].params["force_keyframes_at_cuts"] is True


def test_a_wrong_length_range_falls_back_to_a_local_cut(media: Media, dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}, "time_range": "00:10-00:20"})
    dl._range_supported = True
    full = info(video_format("137", 1920, 1080), title="Nice", video_id="v1")
    fake = flow_fake(media, full, ranged_source=media.video)  # the whole video, not the range
    dl._YoutubeDL = fake

    item = dl.download(URL)[0]
    assert item.path == dl.root / "Nice [v1] [137] [10-20s].mkv"
    assert probe(item.path).duration == pytest.approx(10, abs=0.5)
    assert len(fake.media_calls) == 2
    assert "download_ranges" in fake.media_calls[0].params
    assert "download_ranges" not in fake.media_calls[1].params
    assert not (dl.root / "Nice [v1] [137] [10-20s].mp4").exists()


def test_a_failed_range_download_retries_the_whole_video(media: Media, dl_factory: Any, caplog: pytest.LogCaptureFixture) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}, "time_range": "00:10-00:20"})
    dl._range_supported = True
    full = info(video_format("137", 1920, 1080), title="Nice", video_id="v1")
    fake = flow_fake(media, full, media_errors=(DownloadError("range refused"),))
    dl._YoutubeDL = fake

    with caplog.at_level(logging.WARNING, logger="autoclip"):
        item = dl.download(URL)[0]
    assert item.path.suffix == ".mkv"
    assert len(fake.media_calls) == 2
    assert "range download failed" in [record.getMessage() for record in caplog.records][0]


def test_a_range_attempt_that_saves_nothing_falls_back_to_the_whole_video(media: Media, dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}, "time_range": "00:10-00:20"})
    dl._range_supported = True
    full = info(video_format("137", 1920, 1080), title="Nice", video_id="v1")

    def handle(method: str, session: FakeSession, argument: Any) -> Any:
        if method == "extract_info" or not session.download:
            return full if method == "extract_info" else argument
        if session.params.get("download_ranges") is None:
            write_media(session, media.video)
        return argument

    dl._YoutubeDL = FakeYtDlp(handle)
    item = dl.download(URL)[0]
    assert item.path.suffix == ".mkv"
    assert probe(item.path).duration == pytest.approx(10, abs=0.5)


def test_a_range_needs_ffprobe_so_it_falls_back_to_a_local_cut(media: Media, dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}, "time_range": "00:10-00:20"})
    dl._range_supported = False
    full = info(video_format("137", 1920, 1080), title="Nice", video_id="v1")
    fake = flow_fake(media, full)
    dl._YoutubeDL = fake

    item = dl.download(URL)[0]
    assert item.path.suffix == ".mkv"
    assert len(fake.media_calls) == 1 and "download_ranges" not in fake.media_calls[0].params


def test_a_failing_local_cut_is_an_error_and_removes_the_download(media: Media, dl_factory: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}, "time_range": "00:10-00:20"})
    dl._range_supported = False
    full = info(video_format("137", 1920, 1080), title="Nice", video_id="v1")
    dl._YoutubeDL = flow_fake(media, full)

    def broken(source: Path, target: Path, start: float, end: float) -> None:
        raise AutoclipError("no encoder")

    monkeypatch.setattr(download, "trim_video", broken)
    with pytest.raises(DownloadError, match="cannot cut 00:10-00:20"):
        dl.download(URL)
    assert not (dl.root / "Nice [v1] [137] [10-20s].mp4").exists()


def test_a_media_pass_that_saves_nothing_is_an_error(media: Media, dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    full = info(video_format("137", 1920, 1080), title="Nice", video_id="v1")

    def handle(method: str, session: FakeSession, argument: Any) -> Any:
        return full if method == "extract_info" else argument

    dl._YoutubeDL = FakeYtDlp(handle)
    with pytest.raises(DownloadError, match="did not save a video file"):
        dl.download(URL)


# -------------------------------------------------------------------------- formats


def test_formats_reports_the_choice_and_notes_a_playlist(dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    full = info(video_format("137", 1920, 1080), title="Nice", video_id="v1")
    full["formats"].append({"format_id": "140", "vcodec": "none", "acodec": "mp4a", "ext": "m4a", "abr": 128})
    dl._YoutubeDL = FakeYtDlp(lambda method, session, argument: playlist(full, full))

    report = dl.formats(URL)
    assert (report.platform, report.title, report.chosen) == ("youtube", "Nice", "137+140")
    assert [row.format_id for row in report.rows] == ["137", "140"]
    assert "playlist" in report.warnings[-1]


def test_formats_warns_about_an_audio_only_url(dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    full = info({"format_id": "140", "vcodec": "none", "acodec": "mp4a", "ext": "m4a", "abr": 128})
    dl._YoutubeDL = FakeYtDlp(lambda method, session, argument: full)
    report = dl.formats(URL)
    assert report.chosen == "140"
    assert "only offers audio" in " ".join(report.warnings)


def test_formats_rejects_a_blocked_playlist_entry(dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}, "platforms": {"allow": ["youtube"]}})
    entries = playlist(flat_entry("t", "https://www.tiktok.com/@a/video/1"))
    dl._YoutubeDL = FakeYtDlp(lambda method, session, argument: entries)
    with pytest.raises(DownloadError, match="tiktok"):
        dl.formats(URL)


def test_call_translates_every_yt_dlp_error() -> None:
    dl = object.__new__(Downloader)
    from yt_dlp.utils import YoutubeDLError

    dl._ytdl_error = YoutubeDLError
    assert dl._call(lambda value: value + 1, 1) == 2

    def fail() -> None:
        raise YoutubeDLError("ERROR: nope")

    with pytest.raises(DownloadError, match="nope"):
        dl._call(fail)


# ------------------------------------------------------------------------- sidecars


def sub_settings(**extra: Any) -> DownloadCfg:
    """Download settings with subtitles enabled for ``en`` (and whatever ``extra`` says)."""
    data: dict[str, Any] = {"subtitles": {"enabled": True, "languages": ["en"], "use_as_transcript": "always"}}
    data.update(extra)
    return parse(DownloadCfg, data)


def test_plan_picks_manual_then_automatic_and_skips_unusable_tracks(dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    full = {
        "title": "T",
        "subtitles": {
            "en": [{"ext": "json3", "url": "https://x/en.json3"}],
            "fr": [{"ext": "vtt", "url": "https://x/fr.vtt"}],
        },
        "automatic_captions": {
            "en": [{"ext": "vtt", "url": "https://x/en.vtt"}],
            "de": [{"ext": "srv1", "url": "https://x/de.srv1"}],
        },
    }
    settings = sub_settings(subtitles={"enabled": True, "languages": ["en", "de", "fr", "es"], "use_as_transcript": "never"})
    plan = dl._plan(full, settings)
    assert [(item.language, item.key, item.manual) for item in plan] == [("en", "en", False), ("fr", "fr", True)]


def test_apply_plan_leaves_only_the_chosen_tracks_and_variants(dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    chosen = {"ext": "vtt", "url": "https://x/en-1.vtt"}
    info_dict = {
        "subtitles": {"en": [{"ext": "json3", "url": "https://x/en.json3"}, chosen], "de": [{"ext": "vtt", "url": "https://x/de.vtt"}]},
        "automatic_captions": {"en": [{"ext": "vtt", "url": "https://x/en-auto.vtt"}]},
    }
    dl._apply_plan(info_dict, [download._ChosenSub(language="en", key="en", manual=True, variant=chosen)])
    assert info_dict["subtitles"] == {"en": [chosen]}
    assert info_dict["automatic_captions"] == {}


def test_convert_subtitles_skips_missing_and_broken_files(tmp_path: Path, dl_factory: Any, caplog: pytest.LogCaptureFixture) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    directory = tmp_path / "subs"
    directory.mkdir()
    prefix = "p"
    plan = [
        download._ChosenSub(language="en", key="en", manual=False, variant={"url": "https://x/en.vtt"}),
        download._ChosenSub(language="de", key="de", manual=True, variant={"url": "https://x/de.vtt"}),
        download._ChosenSub(language="fr", key="fr", manual=True, variant={"url": "https://x/fr.vtt"}),
    ]
    (directory / f"{prefix}.subtitle.de.vtt").write_text(vtt(1.0, 2.0), encoding="utf-8")
    (directory / f"{prefix}.subtitle.fr.vtt").write_text("", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="autoclip"):
        dl._convert_subtitles(directory, prefix, plan, None)
        dl._convert_subtitles(directory, prefix, [], None)  # nothing requested is nothing to do

    assert not (directory / f"{prefix}.auto.en.srt").exists()
    converted = directory / f"{prefix}.manual.de.srt"
    assert converted.is_file() and "hello" in converted.read_text(encoding="utf-8")
    assert not (directory / f"{prefix}.subtitle.de.vtt").exists()
    assert not (directory / f"{prefix}.manual.fr.srt").exists()
    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "no subtitle file was saved for language en",
        "the fr subtitles cannot be used (DownloadError: the subtitle file is empty)",
    ]


def test_sidecars_save_every_requested_file_and_convert_the_subtitles(media: Media, dl_factory: Any) -> None:
    dl = dl_factory(
        download={
            "metadata": {"info_json": True, "thumbnail": True, "description": True},
            "subtitles": {"enabled": True, "languages": ["en"], "use_as_transcript": "always"},
        }
    )
    full = info(
        video_format("137", 1920, 1080),
        title="Nice",
        video_id="v1",
        automatic_captions={"en": [{"ext": "vtt", "url": "https://x.test/en.vtt"}]},
    )
    prefix = "Nice [v1] [137]"
    files = {
        f"{prefix}.subtitle.en.vtt": vtt(1.0, 2.0),
        f"{prefix}.info.json": "{}",
        f"{prefix}.thumbnail.webp": "image",
        f"{prefix}.description": "about",
    }
    fake = flow_fake(media, full, sidecar_files=files)
    dl._YoutubeDL = fake

    item = dl.download(URL)[0]
    assert item.subtitle == dl.root / f"{prefix}.auto.en.srt"
    assert item.subtitle_manual is False
    assert "hello" in item.subtitle.read_text(encoding="utf-8")
    assert not (dl.root / f"{prefix}.subtitle.en.vtt").exists()
    for name in ("mp4", "info.json", "thumbnail.webp", "description", "auto.en.srt"):
        assert (dl.root / f"{prefix}.{name}").is_file(), name
    assert dl._complete(dl.root, prefix, dl._settings("youtube"))

    params = fake.sidecar_calls[0].params
    assert params["skip_download"] is True
    assert params["writesubtitles"] is False and params["writeautomaticsub"] is True
    assert params["subtitleslangs"] == ["en"]
    assert params["writeinfojson"] is True and params["writethumbnail"] is True and params["writedescription"] is True


def test_a_sidecar_failure_keeps_the_downloaded_media(media: Media, dl_factory: Any, caplog: pytest.LogCaptureFixture) -> None:
    dl = dl_factory(download={"metadata": {"info_json": True}})
    full = info(video_format("137", 1920, 1080), title="Nice", video_id="v1")
    dl._YoutubeDL = flow_fake(media, full, sidecar_error=RuntimeError("boom"))

    with caplog.at_level(logging.WARNING, logger="autoclip"):
        item = dl.download(URL)[0]
    assert item.path.is_file()
    assert "could not save every extra file" in [record.getMessage() for record in caplog.records][0]


def test_sweep_moves_only_sidecars_and_never_overwrites(tmp_path: Path, dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    directory = tmp_path / "final"
    temporary = tmp_path / "temp"
    directory.mkdir()
    temporary.mkdir()
    prefix = "p"
    names = (
        f"{prefix}.media.en.vtt",
        f"{prefix}.media.webp",
        f"{prefix}.media.f137.mp4",
        f"{prefix}.media.mp4",
        f"{prefix}.media.mp4.part",
    )
    for name in names:
        (temporary / name).write_text("x", encoding="utf-8")
    (temporary / f"{prefix}.media.fr.srt").write_text("x", encoding="utf-8")
    (directory / f"{prefix}.subtitle.fr.srt").write_text("keep", encoding="utf-8")
    (temporary / f"{prefix}.media.dir").mkdir()

    dl._sweep(temporary, directory, prefix)
    assert (directory / f"{prefix}.subtitle.en.vtt").is_file()
    assert (directory / f"{prefix}.thumbnail.webp").is_file()
    assert (directory / f"{prefix}.subtitle.fr.srt").read_text(encoding="utf-8") == "keep"
    assert not (directory / f"{prefix}.subtitle.mp4").exists()
    assert (temporary / f"{prefix}.media.f137.mp4").is_file()  # a video stream is not a sidecar
    assert (temporary / f"{prefix}.media.mp4.part").is_file()


def test_sweep_survives_a_file_it_cannot_move(
    tmp_path: Path, dl_factory: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    directory = tmp_path / "final"
    temporary = tmp_path / "temp"
    directory.mkdir()
    temporary.mkdir()
    (temporary / "p.media.en.vtt").write_text("x", encoding="utf-8")

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise OSError("cross-device")

    monkeypatch.setattr(shutil, "move", refuse)
    with caplog.at_level(logging.DEBUG, logger="autoclip"):
        dl._sweep(temporary, directory, "p")  # best effort: a failed move must not raise
    assert "could not move" in caplog.records[0].getMessage()
    assert not (directory / "p.subtitle.en.vtt").exists()


def test_complete_requires_every_requested_sidecar(tmp_path: Path, dl_factory: Any) -> None:
    dl = dl_factory(
        download={
            "metadata": {"info_json": True, "thumbnail": True, "description": True},
            "subtitles": {"enabled": True, "languages": ["en"], "use_as_transcript": "never"},
        }
    )
    directory = tmp_path / "files"
    directory.mkdir()
    settings = dl._settings("youtube")
    prefix = "p"
    assert not dl._complete(directory, prefix, settings)
    (directory / f"{prefix}.info.json").write_text("{}", encoding="utf-8")
    assert not dl._complete(directory, prefix, settings)  # the description is still missing
    (directory / f"{prefix}.description").write_text("x", encoding="utf-8")
    assert not dl._complete(directory, prefix, settings)  # the thumbnail is still missing
    (directory / f"{prefix}.thumbnail.jpg").write_bytes(b"x")
    assert not dl._complete(directory, prefix, settings)  # the subtitles are still missing
    (directory / f"{prefix}.auto.en.srt").write_text(srt(1, 2), encoding="utf-8")
    assert dl._complete(directory, prefix, settings)
    plain = parse(DownloadCfg, {"metadata": {"info_json": False}})
    assert dl._complete(directory, "nothing", plain)


def test_subtitle_for_follows_the_policy_and_the_transcript_mode(tmp_path: Path, dl_factory: Any) -> None:
    dl = dl_factory(download={"metadata": {"info_json": False}})
    directory = tmp_path / "subs"
    directory.mkdir()
    prefix = "p"
    manual = directory / f"{prefix}.manual.en.srt"
    auto = directory / f"{prefix}.auto.en.srt"
    manual.write_text(srt(1, 2), encoding="utf-8")
    auto.write_text(srt(3, 4), encoding="utf-8")

    def settings(policy: str, languages: list[str] | None = None) -> DownloadCfg:
        return parse(
            DownloadCfg,
            {"subtitles": {"enabled": True, "languages": languages or ["en"], "use_as_transcript": policy}},
        )

    assert dl._subtitle_for(directory, prefix, parse(DownloadCfg, {})) == (None, False)  # disabled
    assert dl._subtitle_for(directory, prefix, settings("never")) == (None, False)
    assert dl._subtitle_for(directory, prefix, settings("manual")) == (manual, True)
    assert dl._subtitle_for(directory, prefix, settings("always")) == (manual, True)

    manual.unlink()
    assert dl._subtitle_for(directory, prefix, settings("manual")) == (None, False)  # auto needs "always"
    assert dl._subtitle_for(directory, prefix, settings("always")) == (auto, False)

    arabic = directory / f"{prefix}.auto.ar.srt"
    arabic.write_text(srt(1, 2), encoding="utf-8")
    assert dl._subtitle_for(directory, prefix, settings("always", ["ar", "en"])) == (arabic, False)

    dl.cfg.transcript.mode = "whisper"
    assert dl._subtitle_for(directory, prefix, settings("manual")) == (None, False)
    assert dl._subtitle_for(directory, prefix, settings("always")) == (auto, False)

    auto.write_text("not an srt", encoding="utf-8")
    assert dl._subtitle_for(directory, prefix, settings("always")) == (None, False)


def test_valid_srt_checks_the_content(tmp_path: Path) -> None:
    good = tmp_path / "a.srt"
    good.write_text(srt(1, 2), encoding="utf-8")
    assert download._valid_srt(good)
    empty = tmp_path / "b.srt"
    empty.write_text("", encoding="utf-8")
    assert not download._valid_srt(empty)
    assert not download._valid_srt(tmp_path / "missing.srt")
