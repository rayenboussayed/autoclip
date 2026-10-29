"""End-to-end checks of the command line against real YouTube videos.

Opt in with ``AUTOCLIP_E2E_TESTS=1 pytest tests/e2e -m e2e --no-cov``. Every case keeps its config,
stdout, stderr, log and outputs under ``tests/results/<timestamp>/``; ``run.json`` summarises the run.
The suite proves the real yt-dlp API, the download cache, subtitle shifting and the render pipeline -
things the offline fake cannot prove.

The manifest cases walk a download ladder: ``AUTOCLIP_E2E_LEVEL`` (default 1) keeps the first six
minutes of every video, level 2 the first twelve, and so on up to the full length. The originals live
in a persistent cache shared by runs, and ``levels.json`` compares each level with the previous one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from autoclip.core.media import probe
from autoclip.core.transcript import parse_srt

from ..support import ManifestEntry, load_manifest
from .harness import (
    FORMATS_TIMEOUT,
    LADDER_CLIPS,
    LADDER_TIMEOUT,
    DownloadedMedia,
    E2ERun,
    assert_database_done,
    cache_folder,
    clip_paths,
    collect_clips,
    download_config,
    downloaded_files,
    level_range,
    level_seconds,
    level_summary,
    pipeline_config,
    read_clips,
    record_level,
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


@pytest.mark.parametrize("entry", MANIFEST, ids=[video_id(entry.url) for entry in MANIFEST])
def test_manifest_video_renders_short_and_long_clips(
    e2e_run: E2ERun, entry: ManifestEntry, download_cache: Path, ladder_level: int
) -> None:
    """One real download of the ladder, then both clip kinds, compared with the previous level."""
    vid = video_id(entry.url)
    originals = cache_folder(download_cache, entry.url, ladder_level)
    download_dir = e2e_run.directory / f"download_{vid}_level{ladder_level}"
    download = e2e_run.execute(
        f"download_{vid}_level{ladder_level}",
        download_config(
            download_dir / "out",
            entry.url,
            time_range=level_range(entry, ladder_level),
            original_dir=originals,
        ),
        ["--download-only"],
        timeout=LADDER_TIMEOUT,
    ).require_ok()
    files = downloaded_files(download)
    assert len(files) == 1, [path.name for path in files]
    source = files[0]
    info = probe(source)
    expected = level_seconds(entry, ladder_level)
    if expected is not None:
        assert abs(info.duration - expected) <= 5, (info.duration, expected, source.name)
    else:
        assert 60 <= info.duration <= ladder_level * 360 + 5, (info.duration, source.name)
    # ``quality: best`` really picks a high-resolution format (the old default capped it at 480p).
    assert max(info.width, info.height) >= 720, (info.width, info.height, source.name)
    subtitle = next(iter(sorted(source.parent.glob("*.manual.en.srt")) or sorted(source.parent.glob("*.auto.en.srt"))))
    cues = parse_srt(subtitle.read_text(encoding="utf-8-sig"))
    assert len(cues) > 20, len(cues)
    assert 0 <= cues[0]["start"] <= 5, cues[0]
    assert cues[-1]["end"] <= info.duration + 5, (cues[-1], info.duration)
    assert list(source.parent.glob("*.info.json"))

    pipeline_dir = e2e_run.directory / f"pipeline_{vid}_level{ladder_level}"
    config = pipeline_config(pipeline_dir / "out", source, subtitle, LADDER_CLIPS)
    result = e2e_run.execute(f"pipeline_{vid}_level{ladder_level}", config, timeout=LADDER_TIMEOUT).require_ok()
    clips = {kind: read_clips(pipeline_dir / "out" / kind / source.stem) for kind in ("short", "long")}
    assert [len(rows) for rows in clips.values()] == [1, 1], clips
    for kind, rows in clips.items():
        clip = rows[0]
        assert clip["scene"] == "best", clip
        start, end, trimmed = float(clip["start"]), float(clip["end"]), float(clip["trimmed"])
        assert 0 <= start < end <= info.duration + 1, (kind, clip, info.duration)
        assert 0 < trimmed <= end - start + 0.01, (kind, clip)
    short, long = clips["short"][0], clips["long"][0]
    assert (short["width"], short["height"]) == (144, 256), short
    assert (long["width"], long["height"]) == (256, 144), long
    assert 14.5 <= float(short["trimmed"]) <= 45.5, short
    assert 179.5 <= float(long["trimmed"]) <= 600.5, long
    for kind, clip in (("short", short), ("long", long)):
        path = pipeline_dir / "out" / kind / source.stem / str(clip["file"])
        assert path.is_file() and path.stat().st_size > 10_000, path
        assert abs(probe(path).duration - float(clip["trimmed"])) <= 0.15, (kind, path)
        assert path.with_suffix(".srt").is_file() and path.with_suffix(".jpg").is_file(), path
        assert len(clip_paths(path.parent)) == 1, path.parent
        assert_database_done(pipeline_dir / "out", source.stem)
    assert rendered_clips(result) == sorted(
        path for kind in ("short", "long") for path in clip_paths(pipeline_dir / "out" / kind / source.stem)
    )

    summary = level_summary(clips, info.duration)
    problems = record_level(download_cache, entry.url, ladder_level, summary)
    assert not problems, f"level {ladder_level} of {vid} differs from the previous level: {problems}"
    assert result.log_path is not None
    assert "finished: 2 clip(s) done, 0 failed" in result.log_path.read_text(encoding="utf-8")


def test_download_only_saves_the_original_subtitles_and_metadata(downloaded_bbc: DownloadedMedia, bbc_entry: ManifestEntry) -> None:
    """A real ``--download-only`` run leaves the media, an English SRT and the info sidecar in the cache."""
    result = downloaded_bbc.result.require_ok()
    assert downloaded_bbc.path.is_file() and downloaded_bbc.path.stat().st_size > 1_000_000
    assert "[" in downloaded_bbc.name and "]" in downloaded_bbc.name
    info = probe(downloaded_bbc.path)
    expected = level_seconds(bbc_entry, 1)
    if expected is not None:
        assert abs(info.duration - expected) <= 5, (info.duration, expected)
    else:
        assert 60 <= info.duration <= 365, info.duration
    # ``quality: best`` keeps the source resolution instead of the old 480p cap.
    assert max(info.width, info.height) >= 1080, (info.width, info.height)
    assert downloaded_bbc.subtitle.is_file()
    assert downloaded_bbc.subtitle.name.endswith((".auto.en.srt", ".manual.en.srt"))
    cues = parse_srt(downloaded_bbc.subtitle.read_text(encoding="utf-8-sig"))
    assert len(cues) > 20, len(cues)
    assert 0 <= cues[0]["start"] <= 5, cues[0]
    # Auto captions often run a couple of seconds past the stream (a subtitle may be longer than its video).
    assert cues[-1]["end"] <= info.duration + 5
    metadata = list(downloaded_bbc.path.parent.glob("*.info.json"))
    assert len(metadata) == 1, [path.name for path in metadata]
    assert len(downloaded_files(result)) == 1
    assert result.log_path is not None and "saved " in result.log_path.read_text(encoding="utf-8")


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


def test_time_range_download_trims_the_video_and_shifts_the_subtitles(
    e2e_run: E2ERun, bbc_entry: ManifestEntry, download_cache: Path
) -> None:
    """A range download starts at zero, lasts the requested minute and moves the SRT with it."""
    case_dir = e2e_run.directory / "download_range_bbc"
    config = download_config(
        case_dir / "out",
        bbc_entry.url,
        time_range="00:01:00-00:02:00",
        original_dir=cache_folder(download_cache, bbc_entry.url, "range_1m"),
    )
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
