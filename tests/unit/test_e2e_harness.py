"""Offline tests for the end-to-end harness (``tests/e2e/harness.py``).

The harness is the part of the opt-in suite that decides where results go, runs the command line and
records what happened. These tests exercise it without the network, so a broken results folder, timeout
or summary is caught by the normal ``pytest`` run instead of a weekly workflow.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from autoclip import __version__

from ..e2e import harness
from ..e2e.harness import CaseResult, E2ERun
from ..support import ManifestEntry


def make_case(tmp_path: Path, **overrides: Any) -> CaseResult:
    """A successful case whose files are not necessarily on disk."""
    values: dict[str, Any] = {
        "name": "case",
        "directory": tmp_path,
        "config_path": tmp_path / "config.json",
        "command": ["python", "-m", "autoclip"],
        "started_at": "2026-01-01T00:00:00+00:00",
        "duration": 1.23456,
        "returncode": 0,
        "timed_out": False,
        "stdout": "out",
        "stderr": "",
        "stdout_path": tmp_path / "stdout.txt",
        "stderr_path": tmp_path / "stderr.txt",
        "log_path": None,
    }
    values.update(overrides)
    return CaseResult(**values)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("formats_XV_B96fAzKs", "formats_XV_B96fAzKs"),
        ("pipeline highlights BBC", "pipeline_highlights_BBC"),
        ("a/b:c*d?e", "a_b_c_d_e"),
        ("__hello__", "hello"),
        ("...", "case"),
        ("??", "case"),
        ("a" * 200, "a" * 80),
        ("MiXeD.Case-1_ok", "MiXeD.Case-1_ok"),
    ],
)
def test_slug_keeps_folders_readable_and_safe(text: str, expected: str) -> None:
    assert harness._slug(text) == expected


def test_tail_keeps_the_last_lines() -> None:
    text = "\n".join(f"line {index}" for index in range(100))
    assert harness._tail(text) == "\n".join(f"line {index}" for index in range(60, 100))
    assert harness._tail(text, lines=2) == "line 98\nline 99"
    assert harness._tail("one\ntwo", lines=10) == "one\ntwo"
    assert harness._tail("") == ""


def test_new_run_directory_creates_the_root_and_avoids_a_second_collision(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "results"
    first = harness.new_run_directory(root, stamp="20260101_120000")
    second = harness.new_run_directory(root, stamp="20260101_120000")
    third = harness.new_run_directory(root, stamp="20260101_120000")
    assert first == root / "20260101_120000" and first.is_dir()
    assert second == root / "20260101_120000_2" and second.is_dir()
    assert third == root / "20260101_120000_3" and third.is_dir()


def test_choose_bbc_entry_prefers_the_bbc_clip_then_falls_back_to_the_last() -> None:
    manifest = [ManifestEntry("Joe Rogan", "https://youtu.be/1"), ManifestEntry("BBC News at six", "https://youtu.be/2")]
    assert harness.choose_bbc_entry(manifest).title == "BBC News at six"
    assert harness.choose_bbc_entry([ManifestEntry("Only", "https://youtu.be/3")]).url == "https://youtu.be/3"
    assert harness.choose_bbc_entry(manifest, "https://youtu.be/custom") == ManifestEntry("custom", "https://youtu.be/custom")
    assert harness.choose_bbc_entry(manifest, "").title == "BBC News at six"


def test_download_config_requests_subtitles_metadata_and_an_optional_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTOCLIP_E2E_COOKIES", raising=False)
    config = harness.download_config(tmp_path / "out", "https://youtu.be/x")
    assert config["input"] == ["https://youtu.be/x"]
    assert config["output_dir"] == str(tmp_path / "out")
    assert config["download"]["quality"] == "best"
    assert config["download"]["subtitles"] == {"enabled": True, "languages": ["en"], "use_as_transcript": "always"}
    assert config["download"]["metadata"] == {"info_json": True}
    assert "time_range" not in config["download"] and "cookies" not in config["download"]
    assert "original_dir" not in config["download"]
    assert config["logging"]["enabled"] is True and config["resources"]["max_workers"] == 1

    monkeypatch.setenv("AUTOCLIP_E2E_COOKIES", "/tmp/cookies.txt")
    ranged = harness.download_config(
        tmp_path / "out", "https://youtu.be/x", time_range="00:01:00-00:02:00", original_dir=tmp_path / "cache"
    )
    assert ranged["download"]["time_range"] == "00:01:00-00:02:00"
    assert ranged["download"]["cookies"] == {"file": "/tmp/cookies.txt"}
    assert ranged["download"]["original_dir"] == str(tmp_path / "cache")


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        (None, 1),
        ({}, 1),
        ({"AUTOCLIP_E2E_LEVEL": ""}, 1),
        ({"AUTOCLIP_E2E_LEVEL": " 3 "}, 3),
        ({"AUTOCLIP_E2E_LEVEL": "30"}, 30),
    ],
)
def test_e2e_level_reads_a_positive_whole_number(env: dict[str, str] | None, expected: int) -> None:
    assert harness.e2e_level(env) == expected


@pytest.mark.parametrize(
    ("raw", "message"), [("two", "positive whole number"), ("1.5", "positive whole number"), ("0", "at least 1"), ("-2", "at least 1")]
)
def test_e2e_level_rejects_anything_that_is_not_a_ladder_level(raw: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        harness.e2e_level({"AUTOCLIP_E2E_LEVEL": raw})


def test_e2e_cache_defaults_to_the_repository_cache_and_honours_the_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AUTOCLIP_E2E_CACHE", raising=False)
    assert harness.e2e_cache() == harness.DEFAULT_CACHE
    assert harness.e2e_cache({}) == harness.DEFAULT_CACHE
    monkeypatch.setenv("AUTOCLIP_E2E_CACHE", str(tmp_path / "cache"))
    assert harness.e2e_cache() == (tmp_path / "cache").resolve()
    assert harness.e2e_cache({"AUTOCLIP_E2E_CACHE": "~/e2e"}) == Path("~/e2e").expanduser().resolve()


def test_level_range_and_level_seconds_follow_the_six_minute_ladder() -> None:
    long = ManifestEntry("Long", "https://youtu.be/l", 10730)
    assert harness._hms(0) == "00:00:00"
    assert harness._hms(3661.6) == "01:01:02"
    assert harness.level_range(long, 1) == "00:00:00-00:06:00"
    assert harness.level_range(long, 2) == "00:00:00-00:12:00"
    assert harness.level_seconds(long, 1) == 360
    assert harness.level_seconds(long, 2) == 720

    short = ManifestEntry("Short", "https://youtu.be/s", 350)
    assert harness.level_range(short, 1) is None
    assert harness.level_seconds(short, 1) == 350
    assert harness.level_range(short, 2) is None

    unknown = ManifestEntry("Unknown", "https://youtu.be/u")
    assert harness.level_range(unknown, 1) == "00:00:00-00:06:00"
    assert harness.level_seconds(unknown, 1) is None


def test_cache_folder_separates_the_videos_and_every_level(tmp_path: Path) -> None:
    url = "https://www.youtube.com/watch?v=XV_B96fAzKs"
    assert harness.cache_folder(tmp_path, url, 2) == tmp_path / "XV_B96fAzKs" / "level_2"
    assert harness.cache_folder(tmp_path, url, "range_1m") == tmp_path / "XV_B96fAzKs" / "range_1m"


def test_ledger_round_trip_compares_each_level_with_the_previous_one(tmp_path: Path) -> None:
    assert harness.read_ledger(tmp_path) == {}
    url = "https://www.youtube.com/watch?v=XV_B96fAzKs"
    first = {
        "source_seconds": 360.0,
        "short": {"clips": 1, "width": 144, "height": 256, "trimmed": [20.0]},
        "long": {"clips": 1, "width": 256, "height": 144, "trimmed": [200.0]},
    }
    assert harness.record_level(tmp_path, url, 1, first) == []
    assert harness.read_ledger(tmp_path) == {"XV_B96fAzKs": {"1": first}}

    second = {
        "source_seconds": 720.0,
        "short": {"clips": 1, "width": 144, "height": 256, "trimmed": [22.0]},
        "long": {"clips": 1, "width": 256, "height": 144, "trimmed": [240.0]},
    }
    assert harness.record_level(tmp_path, url, 2, second) == []
    assert harness.read_ledger(tmp_path)["XV_B96fAzKs"]["2"] == second

    shrunk = {
        "source_seconds": 700.0,
        "short": {"clips": 2, "width": 128, "height": 256, "trimmed": [20.0, 20.0]},
        "long": {"clips": 1, "width": 256, "height": 144, "trimmed": [240.0]},
    }
    problems = harness.record_level(tmp_path, url, 3, shrunk)
    assert any("source shrank" in problem for problem in problems)
    assert any("short.clips changed" in problem for problem in problems)
    assert any("short.width changed" in problem for problem in problems)
    assert not any("long" in problem for problem in problems)
    assert (tmp_path / harness.LEDGER_NAME).is_file()
    assert not (tmp_path / (harness.LEDGER_NAME + ".tmp")).exists()


def test_read_ledger_tolerates_other_json_shapes_and_record_level_ignores_foreign_keys(tmp_path: Path) -> None:
    (tmp_path / harness.LEDGER_NAME).write_text(json.dumps(["not", "a", "ledger"]), encoding="utf-8")
    assert harness.read_ledger(tmp_path) == {}
    (tmp_path / harness.LEDGER_NAME).write_text(json.dumps({"id": "not an object", "1": {"old": True}}), encoding="utf-8")
    assert harness.read_ledger(tmp_path) == {"1": {"old": True}}
    summary = {"source_seconds": 360.0, "short": {"clips": 1}, "long": {"clips": 1}}
    assert harness.record_level(tmp_path, "https://youtu.be/x", 2, summary) == []
    assert harness.read_ledger(tmp_path)["https_youtu_be_x"] == {"2": summary}


def test_level_summary_describes_both_kinds_and_tolerates_a_missing_one() -> None:
    clips = {
        "short": [{"width": 144, "height": 256, "trimmed": 21.23456}],
        "long": [{"width": 256, "height": 144, "trimmed": 200.0}],
    }
    assert harness.level_summary(clips, 360.0) == {
        "source_seconds": 360.0,
        "short": {"clips": 1, "width": 144, "height": 256, "trimmed": [21.235]},
        "long": {"clips": 1, "width": 256, "height": 144, "trimmed": [200.0]},
    }
    assert harness.level_summary({"short": []}, 60.0)["long"] == {"clips": 0, "width": 0, "height": 0, "trimmed": []}


def test_compare_levels_ignores_a_kind_that_was_not_recorded() -> None:
    assert (
        harness.compare_levels({"short": {"clips": 1, "width": 144, "height": 256}}, {"short": {"clips": 1, "width": 144, "height": 256}})
        == []
    )


def test_pipeline_config_merges_clip_overrides_over_its_fast_defaults(tmp_path: Path) -> None:
    config = harness.pipeline_config(tmp_path / "out", Path("video.mp4"), Path("subs.srt"))
    assert config["input"] == ["video.mp4"]
    assert config["transcript"] == {"mode": "file", "file": "subs.srt"}
    assert config["clips"]["kinds"] == ["short"] and config["clips"]["scenes"] == ["best"]
    assert config["clips"]["short"] == {"count": 1, "min_seconds": 15, "target_seconds": 25, "max_seconds": 45}
    assert config["video"] == {"short_side": 144, "preset": "ultrafast", "gpu": "off", "thumbnail": True}
    assert config["compute"] == {"device": "cpu"} and config["resources"] == {"max_workers": 1}

    ranged = harness.pipeline_config(
        tmp_path / "out", Path("video.mp4"), Path("subs.srt"), {"mode": "ranges", "short": {"ranges": ["00:00:10-00:00:30"]}}
    )
    assert ranged["clips"]["mode"] == "ranges"
    assert ranged["clips"]["short"]["count"] == 1
    assert ranged["clips"]["short"]["ranges"] == ["00:00:10-00:00:30"]


def test_downloaded_files_keeps_only_media_in_the_originals_cache(tmp_path: Path) -> None:
    case = make_case(tmp_path)
    config_path = case.config_path
    assert config_path is not None
    folder = case.directory / "out" / ".work" / "downloads"
    folder.mkdir(parents=True)
    for name in ("a.mp4", "b.WEBM", "c.srt", "d.info.json", "e.part"):
        (folder / name).write_text("x", encoding="utf-8")
    (folder / "a-directory.mp4").mkdir()
    assert {path.name for path in harness.downloaded_files(case)} == {"a.mp4", "b.WEBM"}

    # An explicit originals cache outside the result folder is resolved from the config.
    other = tmp_path / "cache"
    other.mkdir()
    (other / "cached.mp4").write_text("x", encoding="utf-8")
    (other / "cached.webm").write_text("x", encoding="utf-8")
    (other / "readme.txt").write_text("x", encoding="utf-8")
    config_path.write_text(json.dumps({"download": {"original_dir": str(other)}}), encoding="utf-8")
    assert {path.name for path in harness.downloaded_files(case)} == {"cached.mp4", "cached.webm"}

    # A relative cache directory is resolved against the case directory.
    (case.directory / "relative").mkdir()
    (case.directory / "relative" / "inner.mp4").write_bytes(b"x")
    config_path.write_text(json.dumps({"download": {"original_dir": "./relative"}}), encoding="utf-8")
    assert {path.name for path in harness.downloaded_files(case)} == {"inner.mp4"}

    # A missing cache folder yields nothing even when the config names it.
    config_path.write_text(json.dumps({"download": {"original_dir": str(tmp_path / "nope")}}), encoding="utf-8")
    assert list(harness.downloaded_files(case)) == []
    # A non-object config is not a real config file and keeps the default folder (also empty here after cleanup).
    for entry in (case.directory / "out" / ".work" / "downloads").iterdir():
        (entry if entry.is_file() else entry).unlink() if entry.is_file() else __import__("shutil").rmtree(entry)
    # A non-object config file yields nothing.
    config_path.write_text("[]", encoding="utf-8")
    assert list(harness.downloaded_files(case)) == []

    # A case without a config file resolves to the (still empty) default cache.
    config_path.unlink()
    assert list(harness.downloaded_files(case)) == []


def test_downloaded_files_prefers_the_files_the_run_saved(tmp_path: Path) -> None:
    case = make_case(tmp_path)
    folder = case.directory / "out" / ".work" / "downloads"
    folder.mkdir(parents=True)
    stale = folder / "video [id] [270+140] [0-360s].mp4"
    fresh = folder / "video [id] [137+140] [0-360s].mp4"
    stale.write_text("x", encoding="utf-8")
    fresh.write_text("x", encoding="utf-8")
    (folder / "video [id] [137+140] [0-360s].auto.en.srt").write_text("x", encoding="utf-8")
    case.stdout = f"08:25:12 | saved {fresh} (youtube)\n08:25:12 | finished: 1 original(s) downloaded, 0 URL(s) failed\n"
    assert harness.downloaded_files(case) == [fresh]

    # A saved path that no longer exists (or is not media) falls back to the cache folder.
    case.stdout = "08:00:00 | saved missing.mp4 (youtube)\n"
    assert {path.name for path in harness.downloaded_files(case)} == {stale.name, fresh.name}


def test_english_subtitle_matches_the_original_itself_in_a_shared_cache(tmp_path: Path) -> None:
    first = tmp_path / "video [id] [137+140].mp4"
    second = tmp_path / "video [id] [270+140].mp4"
    first.write_text("x", encoding="utf-8")
    second.write_text("x", encoding="utf-8")
    first.with_suffix(".auto.en.srt").write_text("first", encoding="utf-8")
    second.with_suffix(".auto.en.srt").write_text("second", encoding="utf-8")
    assert harness.english_subtitle(second).read_text(encoding="utf-8") == "second"


def test_english_subtitle_prefers_a_manual_file_and_ignores_other_languages(tmp_path: Path) -> None:
    media = tmp_path / "video [id] [270+140].mp4"
    media.write_text("x", encoding="utf-8")
    (tmp_path / "video [id] [270+140].auto.en.srt").write_text("auto", encoding="utf-8")
    (tmp_path / "video [id] [270+140].auto.fr.srt").write_text("fr", encoding="utf-8")
    assert harness.english_subtitle(media).name.endswith(".auto.en.srt")
    (tmp_path / "video [id] [270+140].manual.en.srt").write_text("manual", encoding="utf-8")
    assert harness.english_subtitle(media).name.endswith(".manual.en.srt")


def test_english_subtitle_fails_with_the_folder_listing(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    media.write_text("x", encoding="utf-8")
    (tmp_path / "video.fr.srt").write_text("fr", encoding="utf-8")
    with pytest.raises(AssertionError, match="no English subtitle next to video.mp4"):
        harness.english_subtitle(media)


def test_video_id_reads_the_youtube_id_and_falls_back_to_the_url() -> None:
    assert harness.video_id("https://www.youtube.com/watch?v=XV_B96fAzKs") == "XV_B96fAzKs"
    assert harness.video_id("https://www.youtube.com/watch?feature=x&v=abc-123_XYZ") == "abc-123_XYZ"
    assert harness.video_id("https://youtu.be/abc") == "https_youtu_be_abc"
    assert harness.video_id("https://www.youtube.com/watch?v=") == "https_www_youtube_com_watch_v_"
    assert harness.video_id("https://example.com/" + "x" * 50) == "x" * 40


def test_read_clips_returns_the_objects_and_rejects_other_json(tmp_path: Path) -> None:
    folder = tmp_path / "kind"
    folder.mkdir()
    (folder / "clips.json").write_text(json.dumps([{"file": "a.mp4"}]), encoding="utf-8")
    assert harness.read_clips(folder) == [{"file": "a.mp4"}]
    (folder / "clips.json").write_text(json.dumps({"file": "a.mp4"}), encoding="utf-8")
    with pytest.raises(AssertionError, match="not a JSON list"):
        harness.read_clips(folder)
    (folder / "clips.json").write_text(json.dumps(["a.mp4"]), encoding="utf-8")
    with pytest.raises(AssertionError, match="non-object"):
        harness.read_clips(folder)


def test_clip_paths_lists_only_the_top_level_mp4_files(tmp_path: Path) -> None:
    for name in ("b.mp4", "a.mp4", "c.srt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "d.mp4").write_text("x", encoding="utf-8")
    assert [path.name for path in harness.clip_paths(tmp_path)] == ["a.mp4", "b.mp4"]


def add_clip(
    case: CaseResult,
    kind: str,
    video: str,
    name: str,
    *,
    metadata: dict[str, Any] | None = None,
    sidecars: tuple[str, ...] = (".srt", ".jpg"),
) -> Path:
    """Write a rendered clip (and its sidecars) where the pipeline would have put it."""
    folder = case.directory / "out" / kind / video
    folder.mkdir(parents=True, exist_ok=True)
    clip = folder / name
    clip.write_bytes(b"clip")
    for suffix in sidecars:
        clip.with_suffix(suffix).write_text("sidecar", encoding="utf-8")
    if metadata is not None:
        (folder / "clips.json").write_text(json.dumps([metadata]), encoding="utf-8")
    return clip


def test_rendered_clips_finds_both_kinds_and_ignores_downloaded_originals(tmp_path: Path) -> None:
    case = make_case(tmp_path)
    short = add_clip(case, "short", "video", "video_best_01_10s.mp4")
    long = add_clip(case, "long", "video", "video_best_01_10s.mp4")
    originals = case.directory / "out" / ".work" / "downloads"
    originals.mkdir(parents=True)
    (originals / "original.mp4").write_bytes(b"original")
    (case.directory / "out" / "notes.txt").write_text("x", encoding="utf-8")
    (case.directory / "out" / "short" / "video" / "directory.mp4").mkdir()
    assert harness.rendered_clips(case) == [long, short]


def test_rendered_clips_is_empty_without_an_output_folder(tmp_path: Path) -> None:
    assert harness.rendered_clips(make_case(tmp_path / "missing")) == []


def test_clip_metadata_matches_the_sibling_clips_json_by_file_name(tmp_path: Path) -> None:
    clip = add_clip(make_case(tmp_path), "short", "video", "video_best_01_10s.mp4")
    rows = [
        {"file": "other.mp4", "scene": "funny"},
        {"file": f"nested/{clip.name}", "scene": "best", "trimmed": 21.5},
    ]
    (clip.parent / "clips.json").write_text(json.dumps(rows), encoding="utf-8")
    assert harness.clip_metadata(clip) == {"file": f"nested/{clip.name}", "scene": "best", "trimmed": 21.5}

    (clip.parent / "clips.json").write_text(json.dumps([{"file": "other.mp4"}]), encoding="utf-8")
    assert harness.clip_metadata(clip) == {}


@pytest.mark.parametrize("content", [None, "not json", "{}", '["a.mp4"]'])
def test_clip_metadata_tolerates_a_missing_or_broken_clips_json(tmp_path: Path, content: str | None) -> None:
    clip = add_clip(make_case(tmp_path), "short", "video", "video_best_01_10s.mp4")
    if content is not None:
        (clip.parent / "clips.json").write_text(content, encoding="utf-8")
    assert harness.clip_metadata(clip) == {}


def test_collect_clips_copies_the_clip_sidecars_and_gallery(tmp_path: Path) -> None:
    run = E2ERun(tmp_path, "now")
    case = make_case(tmp_path / "case one", name="case one")
    case.directory.mkdir()
    add_clip(
        case,
        "short",
        "my video",
        "my video_best_01_10s.mp4",
        metadata={"file": "my video_best_01_10s.mp4", "scene": "best", "title": "<Funny & <b>", "text": "a < b & c", "trimmed": 21.5},
    )
    run.cases.append(case)

    entries = harness.collect_clips(run)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["case"] == "case one"
    assert entry["kind"] == "short" and entry["video"] == "my video"
    assert entry["file"] == "case_one/short/my video/my video_best_01_10s.mp4"
    assert entry["size_bytes"] == len(b"clip")
    assert entry["source"] == str(case.directory / "out" / "short" / "my video" / "my video_best_01_10s.mp4")
    assert entry["sidecars"] == [
        "case_one/short/my video/my video_best_01_10s.srt",
        "case_one/short/my video/my video_best_01_10s.jpg",
    ]
    assert entry["metadata"]["scene"] == "best"
    visible = run.directory / "clips" / entry["file"]
    assert visible.read_bytes() == b"clip"
    assert visible.with_suffix(".srt").read_text(encoding="utf-8") == "sidecar"
    assert visible.with_suffix(".jpg").read_text(encoding="utf-8") == "sidecar"
    assert json.loads((run.directory / "clips.json").read_text(encoding="utf-8")) == entries

    page = (run.directory / "clips" / "index.html").read_text(encoding="utf-8")
    assert "1 clip(s) from 1 case(s)" in page
    assert "case_one/short/my video/my video_best_01_10s.mp4" in page.replace("%20", " ")
    assert 'poster="case_one/short/my video/my video_best_01_10s.jpg"' in page.replace("%20", " ")
    assert "&lt;Funny &amp; &lt;b&gt;" in page
    assert "a &lt; b &amp; c" in page
    assert "21.5s" in page


def test_collect_clips_quotes_urls_and_escapes_the_case_name(tmp_path: Path) -> None:
    run = E2ERun(tmp_path, "now")
    case = make_case(tmp_path / "case")
    case.directory.mkdir()
    add_clip(case, "short", "video", "clip with space.mp4", sidecars=())
    run.cases.append(case)
    entries = harness.collect_clips(run)
    page = (run.directory / "clips" / "index.html").read_text(encoding="utf-8")
    assert "clip%20with%20space.mp4" in page
    assert "poster=" not in page
    assert entries[0]["sidecars"] == []
    assert "&" not in page.split("<main>")[1].split("</main>")[0].replace("&middot;", "")


def test_collect_clips_rebuilds_the_folder_and_handles_an_empty_run(tmp_path: Path) -> None:
    run = E2ERun(tmp_path, "now")
    first = harness.collect_clips(run)
    assert first == []
    assert json.loads((tmp_path / "clips.json").read_text(encoding="utf-8")) == []
    page = (tmp_path / "clips" / "index.html").read_text(encoding="utf-8")
    assert "No clip was rendered in this run." in page

    stale = tmp_path / "clips" / "stale.mp4"
    stale.write_bytes(b"stale")
    case = make_case(tmp_path / "case")
    case.directory.mkdir()
    add_clip(case, "long", "video", "video_custom_01_60s.mp4")
    run.cases.append(case)
    entries = harness.collect_clips(run)
    assert not stale.exists()
    assert [entry["file"] for entry in entries] == ["case/long/video/video_custom_01_60s.mp4"]
    assert harness.collect_clips(run) == entries


def test_execute_rejects_a_case_name_that_would_collide_with_the_clip_gallery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise AssertionError("the command must not run for a reserved case name")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(ValueError, match="collide with the run-level clips/"):
        E2ERun(tmp_path, "now").execute("clips", {})


def test_assert_database_done_accepts_only_finished_rows(tmp_path: Path) -> None:
    database = tmp_path / ".work" / "video" / "autoclip.db"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE clips (status TEXT)")
        connection.execute("INSERT INTO clips VALUES ('done')")
    harness.assert_database_done(tmp_path, "video")

    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO clips VALUES ('pending')")
    with pytest.raises(AssertionError, match="pending"):
        harness.assert_database_done(tmp_path, "video")

    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM clips")
    with pytest.raises(AssertionError):
        harness.assert_database_done(tmp_path, "video")

    with pytest.raises(AssertionError, match="missing"):
        harness.assert_database_done(tmp_path, "other")


def test_require_ok_returns_the_case_on_success(tmp_path: Path) -> None:
    case = make_case(tmp_path)
    assert case.require_ok() is case


def test_require_ok_reports_a_timeout_with_the_results_folder(tmp_path: Path) -> None:
    case = make_case(tmp_path, timed_out=True, returncode=None, duration=12.6)
    with pytest.raises(AssertionError) as error:
        case.require_ok()
    assert "timed out after 13s; see " + str(tmp_path) in str(error.value)


def test_require_ok_reports_the_command_stderr_tail_and_log(tmp_path: Path) -> None:
    stderr = "\n".join(f"error {index}" for index in range(50))
    log = tmp_path / "logs" / "autoclip.log"
    case = make_case(tmp_path, returncode=1, stderr=stderr, log_path=log)
    with pytest.raises(AssertionError) as error:
        case.require_ok()
    message = str(error.value)
    assert "case exited 1; command: python -m autoclip" in message
    assert "error 49" in message and "error 0" not in message
    assert f"log: {log}" in message and f"case: {tmp_path}" in message


def test_require_ok_shows_a_dash_when_the_run_wrote_no_log(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="log: -"):
        make_case(tmp_path, returncode=2).require_ok()


def test_to_json_rounds_the_duration_and_keeps_the_optional_paths(tmp_path: Path) -> None:
    case = make_case(tmp_path, config_path=None, log_path=None, duration=1.23456)
    data = case.to_json()
    assert data["duration_seconds"] == 1.235
    assert data["config"] is None and data["log"] is None
    assert data["command"] == ["python", "-m", "autoclip"]
    assert data["stdout"] == str(tmp_path / "stdout.txt")
    assert data["stderr"] == str(tmp_path / "stderr.txt")

    log = tmp_path / "autoclip.log"
    config = tmp_path / "config.json"
    case = make_case(tmp_path, config_path=config, log_path=log)
    assert case.to_json()["config"] == str(config)
    assert case.to_json()["log"] == str(log)


def test_git_commit_handles_a_checkout_a_non_repository_and_a_missing_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 0, stdout="abc123\n", stderr=""))
    assert harness._git_commit() == "abc123"

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 128, stdout="", stderr="fatal"))
    assert harness._git_commit() is None

    def missing(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise OSError("git is not installed")

    monkeypatch.setattr(subprocess, "run", missing)
    assert harness._git_commit() is None


def test_environment_records_versions_arguments_and_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness, "_git_commit", lambda: "deadbeef")
    monkeypatch.setattr(sys, "argv", ["pytest", "-m", "e2e", "--no-cov"])
    for name in (
        "AUTOCLIP_E2E_TESTS",
        "AUTOCLIP_NETWORK_TESTS",
        "AUTOCLIP_E2E_RESULTS",
        "AUTOCLIP_E2E_CACHE",
        "AUTOCLIP_E2E_LEVEL",
        "AUTOCLIP_E2E_URL",
        "AUTOCLIP_E2E_COOKIES",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AUTOCLIP_E2E_TESTS", "1")
    monkeypatch.setenv("AUTOCLIP_E2E_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("AUTOCLIP_E2E_LEVEL", "4")
    data = harness._environment(tmp_path)
    assert data["autoclip_version"] == __version__
    assert data["python"] == sys.version and data["executable"] == sys.executable
    assert data["git_commit"] == "deadbeef"
    assert data["pytest_arguments"] == ["-m", "e2e", "--no-cov"]
    assert data["directory"] == str(tmp_path)
    assert data["flags"] == {
        "AUTOCLIP_E2E_TESTS": "1",
        "AUTOCLIP_NETWORK_TESTS": None,
        "AUTOCLIP_E2E_RESULTS": None,
        "AUTOCLIP_E2E_CACHE": str(tmp_path / "cache"),
        "AUTOCLIP_E2E_LEVEL": "4",
        "AUTOCLIP_E2E_URL": None,
        "AUTOCLIP_E2E_COOKIES": None,
    }
    assert harness._now().startswith(str(__import__("datetime").datetime.now().year))


def test_execute_writes_the_config_runs_the_cli_and_records_the_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        cwd = Path(str(kwargs["cwd"]))
        (cwd / "logs").mkdir()
        (cwd / "logs" / "autoclip.log").write_text("log", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="hello", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "/existing")
    run = E2ERun(tmp_path, "2026-01-01T00:00:00+00:00")
    config = {"input": ["a.mp4"], "logging": {"file": "logs/autoclip.log"}}
    result = run.execute("my case", config, ["--platforms", "https://youtu.be/x"], timeout=12)

    assert result.directory == tmp_path / "my_case"
    assert result.config_path is not None
    assert json.loads(result.config_path.read_text(encoding="utf-8")) == config
    assert result.stdout == "hello" and result.stderr == ""
    assert result.stdout_path.read_text(encoding="utf-8") == "hello"
    assert result.stderr_path.read_text(encoding="utf-8") == ""
    assert result.log_path == result.directory / "logs" / "autoclip.log"
    assert result.returncode == 0 and result.timed_out is False
    assert run.cases == [result]
    case_json = json.loads((result.directory / "case.json").read_text(encoding="utf-8"))
    assert case_json["name"] == "my case" and case_json["returncode"] == 0

    command, kwargs = calls[0]
    assert command == [sys.executable, "-m", "autoclip", "--platforms", "https://youtu.be/x", str(result.config_path)]
    assert kwargs["cwd"] == result.directory
    assert kwargs["timeout"] == 12
    assert kwargs["encoding"] == "utf-8" and kwargs["errors"] == "replace" and kwargs["check"] is False
    env = kwargs["env"]
    assert isinstance(env, dict)
    assert env["PYTHONPATH"] == os.pathsep.join([str(harness.ROOT), "/existing"])
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_execute_keeps_an_explicit_pythonunbuffered_and_ignores_a_non_object_logging_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONUNBUFFERED", "0")
    result = E2ERun(tmp_path, "now").execute("case", {"logging": "off"})
    env = captured["env"]
    assert isinstance(env, dict) and env["PYTHONUNBUFFERED"] == "0"
    assert result.log_path is None


@pytest.mark.parametrize(("output", "stderr"), [(None, None), ("partial out", "partial err"), (b"bytes out", b"bytes err")])
def test_execute_records_a_timeout_and_decodes_either_stream_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: Any, stderr: Any
) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, 1, output=output, stderr=stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = E2ERun(tmp_path, "now").execute("slow", {}, timeout=1)
    assert result.timed_out is True and result.returncode is None
    assert result.stdout_path.read_text(encoding="utf-8") == (output.decode() if isinstance(output, bytes) else output or "")
    assert result.stderr_path.read_text(encoding="utf-8") == (stderr.decode() if isinstance(stderr, bytes) else stderr or "")
    with pytest.raises(AssertionError, match="slow timed out"):
        result.require_ok()


def test_execute_rejects_a_second_case_with_the_same_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""))
    run = E2ERun(tmp_path, "now")
    run.execute("same case", {})
    with pytest.raises(ValueError, match="already ran"):
        run.execute("same case", {})


def test_execute_runs_the_real_command_line_offline(tmp_path: Path) -> None:
    url = "https://www.youtube.com/watch?v=XV_B96fAzKs"
    result = E2ERun(tmp_path, "now").execute("platforms", {"input": [url]}, ["--platforms", url], timeout=120)
    result.require_ok()
    assert "allow:" in result.stdout and "permitted:" in result.stdout
    assert result.log_path is None


def test_record_and_write_summary_keep_every_phase_and_case(tmp_path: Path) -> None:
    run = E2ERun(tmp_path, "2026-01-01T00:00:00+00:00")
    run._begin = time.time() - 1.0
    run.record("test_a", "setup", "passed")
    run.record("test_a", "call", "failed")
    run.record("test_a", "call", "passed")
    run.record("test_b", "teardown", "failed")
    run.write_summary()
    data = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert data["started_at"] == "2026-01-01T00:00:00+00:00"
    assert data["finished_at"]
    assert data["duration_seconds"] >= 0.9
    assert data["directory"] == str(tmp_path)
    assert data["cases"] == []
    assert data["outcomes"] == {"test_a": {"setup": "passed", "call": "passed"}, "test_b": {"teardown": "failed"}}
    assert data["clips"] == []
    assert (tmp_path / "clips" / "index.html").is_file()


def test_write_summary_collects_every_rendered_clip(tmp_path: Path) -> None:
    run = E2ERun(tmp_path, "now")
    run._begin = time.time()
    case = make_case(tmp_path / "case")
    case.directory.mkdir()
    add_clip(case, "short", "video", "video_best_01_10s.mp4", metadata={"file": "video_best_01_10s.mp4", "scene": "best"})
    run.cases.append(case)
    run.write_summary()
    data = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert [entry["file"] for entry in data["clips"]] == ["case/short/video/video_best_01_10s.mp4"]
    assert data["clips"][0]["metadata"]["scene"] == "best"
    assert (tmp_path / "clips" / "case" / "short" / "video" / "video_best_01_10s.mp4").is_file()


def test_runs_do_not_share_their_case_or_outcome_lists(tmp_path: Path) -> None:
    first = E2ERun(tmp_path / "first", "now")
    second = E2ERun(tmp_path / "second", "now")
    first.cases.append(make_case(tmp_path))
    first.record("test", "call", "passed")
    assert second.cases == [] and second.outcomes == {}
