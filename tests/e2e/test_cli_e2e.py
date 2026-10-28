"""End-to-end checks of the command line against real YouTube videos.

Opt in with ``AUTOCLIP_E2E_TESTS=1 pytest tests/e2e -m e2e --no-cov``. Every case keeps its config,
stdout, stderr, log and outputs under ``tests/results/<timestamp>/``; ``run.json`` summarises the run.
The suite proves the real yt-dlp API, the download cache, subtitle shifting and the render pipeline -
things the offline fake cannot prove.
"""

from __future__ import annotations

import json
import re

import pytest

from autoclip.core.media import probe
from autoclip.core.transcript import parse_srt

from ..support import ManifestEntry, load_manifest
from .harness import (
    FORMATS_TIMEOUT,
    DownloadedMedia,
    E2ERun,
    assert_database_done,
    clip_paths,
    collect_clips,
    download_config,
    downloaded_files,
    pipeline_config,
    read_clips,
    rendered_clips,
    video_id,
)

pytestmark = [pytest.mark.e2e, pytest.mark.network]

MANIFEST = load_manifest()


@pytest.mark.parametrize("entry", MANIFEST, ids=[video_id(entry.url) for entry in MANIFEST])
def test_formats_lists_every_manifest_video(e2e_run: E2ERun, entry: ManifestEntry) -> None:
    """``--formats`` really extracts each manifest URL and chooses a downloadable format."""
    case_dir = e2e_run.directory / f"formats_{video_id(entry.url)}"
    result = e2e_run.execute(
        f"formats_{video_id(entry.url)}",
        download_config(case_dir / "out", entry.url),
        ["--formats", entry.url],
        timeout=FORMATS_TIMEOUT,
    ).require_ok()
    fields: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {"title", "url", "platform", "chosen"}:
            fields[key.strip()] = value.strip()
    assert fields.get("url") == entry.url
    assert fields.get("platform") == "youtube"
    assert fields.get("title") and fields["title"] != "-", result.stdout[:2000]
    chosen = fields.get("chosen", "")
    assert chosen and chosen != "-", result.stdout[:2000]
    assert re.fullmatch(r"[A-Za-z0-9_.+-]+", chosen), chosen
    assert "ID" in result.stdout and "EXT" in result.stdout, result.stdout[:2000]


def test_download_only_saves_the_original_subtitles_and_metadata(downloaded_bbc: DownloadedMedia) -> None:
    """A real ``--download-only`` run leaves the media, an English SRT and the info sidecar in the cache."""
    result = downloaded_bbc.result.require_ok()
    assert downloaded_bbc.path.is_file() and downloaded_bbc.path.stat().st_size > 1_000_000
    assert "[" in downloaded_bbc.name and "]" in downloaded_bbc.name
    info = probe(downloaded_bbc.path)
    assert 300 <= info.duration <= 420, info.duration
    assert (info.width, info.height) == (854, 480), (info.width, info.height)
    assert downloaded_bbc.subtitle.is_file()
    assert downloaded_bbc.subtitle.name.endswith((".auto.en.srt", ".manual.en.srt"))
    cues = parse_srt(downloaded_bbc.subtitle.read_text(encoding="utf-8-sig"))
    assert len(cues) > 20, len(cues)
    assert 0 <= cues[0]["start"] <= 5, cues[0]
    # Auto captions often run a couple of seconds past the stream (a subtitle may be longer than its video).
    assert cues[-1]["end"] <= info.duration + 5
    metadata = list(result.directory.rglob("*.info.json"))
    assert len(metadata) == 1, [path.name for path in metadata]
    assert len(downloaded_files(result)) == 1
    assert result.log_path is not None and "saved " in result.log_path.read_text(encoding="utf-8")


def test_pipeline_renders_a_highlight_from_the_downloaded_subtitle(e2e_run: E2ERun, downloaded_bbc: DownloadedMedia) -> None:
    """The highlights pipeline turns the real download and its SRT into one finished short clip."""
    case_dir = e2e_run.directory / "pipeline_highlights_bbc"
    config = pipeline_config(case_dir / "out", downloaded_bbc.path, downloaded_bbc.subtitle)
    result = e2e_run.execute("pipeline_highlights_bbc", config).require_ok()
    folder = case_dir / "out" / "short" / downloaded_bbc.path.stem
    clips = read_clips(folder)
    assert len(clips) == 1, clips
    clip = clips[0]
    assert clip["scene"] == "best"
    assert (clip["width"], clip["height"]) == (144, 256)
    assert 0 <= float(clip["start"]) < float(clip["end"]), clip
    span = float(clip["end"]) - float(clip["start"])
    trimmed = float(clip["trimmed"])
    # ``trimmed`` is the final length after the pause/filler cuts: never longer than the source window.
    assert 0 < trimmed <= span + 0.01, clip
    path = folder / str(clip["file"])
    assert re.fullmatch(rf"{re.escape(downloaded_bbc.path.stem)}_best_01_\d+s\.mp4", path.name), path.name
    assert path.is_file() and path.stat().st_size > 10_000
    duration = probe(path).duration
    assert 14.5 <= duration <= 45.5, duration
    assert abs(duration - trimmed) <= 0.15, (duration, trimmed)
    assert path.with_suffix(".srt").is_file()
    assert path.with_suffix(".jpg").is_file()
    assert len(clip_paths(folder)) == 1
    assert rendered_clips(result) == clip_paths(folder)
    visible = {entry["file"] for entry in collect_clips(e2e_run)}
    assert f"pipeline_highlights_bbc/short/{downloaded_bbc.path.stem}/{path.name}" in visible
    assert_database_done(case_dir / "out", downloaded_bbc.path.stem)
    assert result.log_path is not None
    assert "finished: 1 clip(s) done, 0 failed" in result.log_path.read_text(encoding="utf-8")


def test_pipeline_renders_explicit_ranges(e2e_run: E2ERun, downloaded_bbc: DownloadedMedia) -> None:
    """``clips.mode: ranges`` cuts exactly the requested spans from the real download."""
    case_dir = e2e_run.directory / "pipeline_ranges_bbc"
    config = pipeline_config(
        case_dir / "out",
        downloaded_bbc.path,
        downloaded_bbc.subtitle,
        clips={"mode": "ranges", "short": {"ranges": ["00:00:10-00:00:30", "00:01:00-00:01:20"]}},
    )
    result = e2e_run.execute("pipeline_ranges_bbc", config).require_ok()
    folder = case_dir / "out" / "short" / downloaded_bbc.path.stem
    clips = read_clips(folder)
    assert len(clips) == 2, clips
    spans = sorted((float(clip["start"]), float(clip["end"])) for clip in clips)
    assert spans == [(10.0, 30.0), (60.0, 80.0)]
    assert {clip["scene"] for clip in clips} == {"custom"}
    assert all(clip["file"] for clip in clips), clips
    names = {path.name for path in clip_paths(folder)}
    assert names == {
        f"{downloaded_bbc.path.stem}_custom_01_10s.mp4",
        f"{downloaded_bbc.path.stem}_custom_02_60s.mp4",
    }
    for path in clip_paths(folder):
        duration = probe(path).duration
        assert 14.5 <= duration <= 20.5, (path.name, duration)
        assert path.with_suffix(".srt").is_file()
        assert path.with_suffix(".jpg").is_file()
    assert rendered_clips(result) == clip_paths(folder)
    assert_database_done(case_dir / "out", downloaded_bbc.path.stem)


def test_time_range_download_trims_the_video_and_shifts_the_subtitles(e2e_run: E2ERun, bbc_entry: ManifestEntry) -> None:
    """A range download starts at zero, lasts the requested minute and moves the SRT with it."""
    case_dir = e2e_run.directory / "download_range_bbc"
    config = download_config(case_dir / "out", bbc_entry.url, time_range="00:01:00-00:02:00")
    result = e2e_run.execute("download_range_bbc", config, ["--download-only"]).require_ok()
    files = downloaded_files(result)
    assert len(files) == 1, [path.name for path in files]
    original = files[0]
    assert "[60-120s]" in original.name, original.name
    assert original.stat().st_size > 1_000_000
    info = probe(original)
    assert 58 <= info.duration <= 62, info.duration
    folder = original.parent
    subtitles = sorted(folder.glob("*.manual.en.srt")) or sorted(folder.glob("*.auto.en.srt"))
    assert len(subtitles) == 1, [path.name for path in folder.iterdir()]
    cues = parse_srt(subtitles[0].read_text(encoding="utf-8-sig"))
    assert len(cues) > 10, len(cues)
    assert cues[0]["start"] <= 1.0, cues[0]
    assert all(cue["start"] >= 0 and cue["end"] <= 61 for cue in cues)
    assert cues[-1]["end"] >= 50, cues[-1]
    assert list(folder.glob("*.info.json"))
    assert result.log_path is not None and "saved " in result.log_path.read_text(encoding="utf-8")


def test_rendered_clips_are_collected_in_the_results_folder(e2e_run: E2ERun) -> None:
    """What the run produced is browsable next to ``run.json``: copied clips and a playable gallery."""
    expected = [(case, clip) for case in e2e_run.cases for clip in rendered_clips(case)]
    if not expected:
        pytest.skip("no clip-producing case ran")
    entries = collect_clips(e2e_run)
    assert len(entries) == len(expected)
    for entry in entries:
        assert (e2e_run.directory / "clips" / entry["file"]).is_file()
        assert entry["metadata"], entry
    page = (e2e_run.directory / "clips" / "index.html").read_text(encoding="utf-8")
    assert page.count("<video controls") == len(entries)
    assert json.loads((e2e_run.directory / "clips.json").read_text(encoding="utf-8")) == entries
