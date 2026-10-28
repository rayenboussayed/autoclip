"""The pure helpers of ``tests/support.py``: manifests, config builders and the Whisper stand-in."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autoclip.core.config import Config, load_config, parse
from autoclip.core.errors import ConfigError

from ..support import (
    BASE_CONFIG,
    FIXTURE_SAMPLES,
    MANIFEST_PATH,
    SENTENCES,
    SR,
    ManifestEntry,
    Media,
    _segments,
    _srt_time,
    install_stub_whisper,
    load_manifest,
    make_cfg,
    merge,
    write_config,
)


def test_merge_recurses_without_mutating_the_inputs() -> None:
    base: dict[str, Any] = {"clips": {"kinds": ["short"], "short": {"count": 1, "min_seconds": 6}}, "output_dir": "out"}
    over: dict[str, Any] = {"clips": {"short": {"count": 3}}, "logging": {"enabled": False}}
    merged = merge(base, over)
    assert merged == {
        "clips": {"kinds": ["short"], "short": {"count": 3, "min_seconds": 6}},
        "output_dir": "out",
        "logging": {"enabled": False},
    }
    assert base["clips"]["short"]["count"] == 1 and "logging" not in base
    assert over["clips"]["short"] == {"count": 3}


def test_merge_replaces_a_scalar_with_an_object_and_the_reverse() -> None:
    assert merge({"value": 1}, {"value": {"nested": 2}}) == {"value": {"nested": 2}}
    assert merge({"value": {"nested": 2}}, {"value": 1}) == {"value": 1}
    assert merge({"a": 1}, {}) == {"a": 1}


def test_make_cfg_uses_fast_defaults_and_merges_nested_sections(media: Media, tmp_path: Path) -> None:
    cfg = make_cfg(media, tmp_path / "out", clips={"short": {"count": 3}})
    assert cfg.input == [str(media.video)]
    assert cfg.output_dir == str(tmp_path / "out")
    assert cfg.clips.kinds == ["short"]
    assert (cfg.clips.short.count, cfg.clips.short.min_seconds, cfg.clips.short.max_seconds) == (3, 6, 12)
    assert cfg.transcript.mode == "file" and cfg.transcript.file == str(media.transcript)
    assert (cfg.video.short_side, cfg.video.preset, cfg.video.thumbnail, cfg.video.gpu) == (144, "ultrafast", False, "off")
    assert cfg.logging.enabled is False and cfg.logging.console_level == "ERROR"
    assert cfg.resources.max_workers == 1


def test_make_cfg_accepts_another_source(media: Media, tmp_path: Path) -> None:
    cfg = make_cfg(media, tmp_path / "out", media.silent)
    assert cfg.input == [str(media.silent)]


def test_make_cfg_rejects_an_unknown_option(media: Media, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not_an_option"):
        make_cfg(media, tmp_path / "out", clips={"not_an_option": True})


def test_write_config_writes_the_three_subtitle_files_and_a_loadable_config(media: Media, tmp_path: Path) -> None:
    path = write_config(tmp_path, media, clips={"kinds": ["short", "long"]})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["input"] == [str(media.video), str(media.silent)]
    assert data["output_dir"] == str(tmp_path / "out")
    assert data["clips"]["kinds"] == ["short", "long"]
    assert data["clips"]["short"]["count"] == BASE_CONFIG["clips"]["short"]["count"]
    for stem in ("talk", "silent", "broken"):
        assert (tmp_path / "subs" / f"{stem}.json").read_text(encoding="utf-8") == media.transcript.read_text(encoding="utf-8")
    loaded = load_config(path)
    assert loaded.input == [str(media.video.resolve()), str(media.silent.resolve())]
    assert loaded.output_dir == str((tmp_path / "out").resolve())


def test_segments_cover_the_fixture_with_word_timings() -> None:
    segments = _segments()
    assert len(segments) == len(SENTENCES) == 10
    for index, (segment, sentence) in enumerate(zip(segments, SENTENCES, strict=True)):
        assert segment["start"] == index * 4.0
        assert segment["end"] == segment["start"] + 3.0
        assert segment["text"] == sentence
        words = segment["words"]
        assert [word["word"] for word in words] == sentence.split()
        assert words[0]["start"] == segment["start"]
        assert words[-1]["end"] == pytest.approx(segment["end"])
        assert all(word["start"] < word["end"] for word in words)
        assert all(left["end"] <= right["start"] for left, right in zip(words, words[1:], strict=False))


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00:00,000"),
        (3.25, "00:00:03,250"),
        (1.9999, "00:00:02,000"),
        (59.9996, "00:01:00,000"),
        (3661.5, "01:01:01,500"),
    ],
)
def test_srt_time_carries_rounding_into_the_next_unit(seconds: float, expected: str) -> None:
    assert _srt_time(seconds) == expected


def test_manifest_path_and_constants_describe_the_fixture() -> None:
    assert MANIFEST_PATH.is_file() and MANIFEST_PATH.name == "youtube.json"
    assert SR == 22050 and FIXTURE_SAMPLES == 40 * 16000
    assert len(SENTENCES) == 10


def test_load_manifest_returns_every_unique_youtube_url() -> None:
    entries = load_manifest()
    assert len(entries) == 10
    assert all(isinstance(entry, ManifestEntry) for entry in entries)
    assert all(entry.title.strip() and entry.url.startswith("https://") for entry in entries)
    assert len({entry.url for entry in entries}) == len(entries)


def test_load_manifest_accepts_a_custom_file(tmp_path: Path) -> None:
    path = tmp_path / "custom.json"
    path.write_text(json.dumps([{"title": "A video", "video_link": "http://example.com/v"}]), encoding="utf-8")
    assert load_manifest(path) == [ManifestEntry("A video", "http://example.com/v")]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "expected a non-empty JSON list"),
        ({"title": "x"}, "expected a non-empty JSON list"),
        ([1], "entry 1 is not an object"),
        ([{"video_link": "https://example.com/v"}], "entry 1 has no title"),
        ([{"title": "  ", "video_link": "https://example.com/v"}], "entry 1 has no title"),
        ([{"title": "x", "video_link": "ftp://example.com/v"}], "entry 1 has no http\\(s\\) video_link"),
        ([{"title": "x"}], "entry 1 has no http\\(s\\) video_link"),
    ],
)
def test_load_manifest_rejects_a_malformed_manifest(tmp_path: Path, payload: Any, message: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_manifest(path)


def test_load_manifest_propagates_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_manifest(path)


def test_install_stub_whisper_matches_the_real_call_signatures(media: Media, monkeypatch: pytest.MonkeyPatch) -> None:
    record: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, record, cuda=2)
    import ctranslate2
    import faster_whisper

    assert ctranslate2.get_cuda_device_count() == 2
    model = faster_whisper.WhisperModel("tiny", "cuda", "float16", 4)
    assert record["init"] == ("tiny", "cuda", "float16", 4)

    audio = np.zeros(FIXTURE_SAMPLES, dtype=np.float32)
    segments, info = model.transcribe(audio, language="en")
    assert record["samples"] == FIXTURE_SAMPLES
    assert record["audio_sum"] == 0.0
    assert [segment.text for segment in segments] == [f" {sentence}" for sentence in SENTENCES]
    assert (info.language, info.language_probability, info.duration) == ("en", 0.99, 40.0)
    assert record["options"] == {"language": "en"}


def test_install_stub_whisper_tells_a_chunk_from_the_whole_file(media: Media, monkeypatch: pytest.MonkeyPatch) -> None:
    record: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, record)
    import faster_whisper

    model = faster_whisper.WhisperModel("tiny", "cpu", "int8", 1)
    threshold = FIXTURE_SAMPLES - 16000
    segments, _ = model.transcribe(np.zeros(threshold, dtype=np.float32), language="en")
    assert len(list(segments)) == len(SENTENCES) and "chunks" not in record  # exactly the threshold is the whole file

    segments, _ = model.transcribe(np.zeros(threshold - 1, dtype=np.float32), language="ar")
    chunk = list(segments)
    assert record["chunks"] == [(threshold - 1, "ar")]
    assert len(chunk) == 1 and chunk[0].text == " chunk"
    assert [(word.word, word.start, word.end) for word in chunk[0].words] == [(" chunk", 0.5, 1.5)]
    assert record["init"] == ("tiny", "cpu", "int8", 1)


def test_parse_and_config_are_reachable_from_the_helpers() -> None:
    # ``make_cfg``/``write_config`` build plain dictionaries; the same strict parser validates both.
    assert parse(Config, {"input": ["a.mp4"]}).clips.kinds == ["short", "long"]
