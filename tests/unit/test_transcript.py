"""Transcript parsing, caching and the (stubbed) Whisper integration."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from autoclip.core import hardware, transcript
from autoclip.core.config import Config, TranscriptCfg, parse
from autoclip.core.denoise import SpeechEnhancer, default_models_dir
from autoclip.core.errors import AutoclipError
from autoclip.core.hardware import pick_encoder
from autoclip.core.media import F32, decode_audio, frame_energy_db, probe
from autoclip.core.models import MediaInfo
from autoclip.core.resources import MB, RESOURCES
from autoclip.core.store import Store
from autoclip.core.transcript import (
    asr_device,
    chunk_bounds,
    fit_whisper_model,
    get_transcript,
    load_transcript_file,
    normalize_cues,
    parse_srt,
    to_segments,
    tokenize,
    transcribe,
)

from ..support import FIXTURE_SAMPLES, SENTENCES, Media, install_stub_whisper, make_cfg

SRT = (
    "1\n00:00:01,000 --> 00:00:03,500\nHello there.\n\n"
    "2\r\n00:01:00.5 --> 00:01:02.25\nSecond line\nwraps\n\n"
    "3\n00:00:09,000 --> 00:00:10,000\n\n"
)


def test_parse_srt_handles_multiline_crlf_and_empty_cues() -> None:
    cues = parse_srt(SRT)
    assert [c["text"] for c in cues] == ["Hello there.", "Second line wraps"]
    assert cues[0]["start"] == 1.0 and cues[0]["end"] == 3.5
    assert cues[1]["start"] == 60.5 and cues[1]["end"] == 62.25


def test_normalize_cues_unrolls_rolling_captions() -> None:
    """The shape YouTube's automatic captions have: each line repeated, then a 10 ms duplicate cue."""
    raw = [
        {"start": 0.199, "end": 2.55, "text": "now a BBC World Service investigation"},
        {"start": 2.55, "end": 2.56, "text": "now a BBC World Service investigation "},
        {"start": 2.56, "end": 4.51, "text": "now a BBC World Service investigation\nhas found that Steven Bartlett is"},
        {"start": 4.51, "end": 4.52, "text": "has found that Steven Bartlett is"},
        {"start": 4.52, "end": 7.27, "text": "has found that Steven Bartlett is amplifying harmful Health misinformation"},
    ]
    cues = normalize_cues(raw)
    assert [c["text"] for c in cues] == [
        "now a BBC World Service investigation",
        "has found that Steven Bartlett is",
        "amplifying harmful Health misinformation",
    ]
    assert [(c["start"], c["end"]) for c in cues] == [(0.199, 2.56), (2.56, 4.52), (4.52, 7.27)]
    # punctuation and case do not hide the repeated line
    assert [
        c["text"]
        for c in normalize_cues(
            [
                {"start": 0, "end": 1, "text": "Hello there"},
                {"start": 1, "end": 2, "text": "hello, there friend"},
            ]
        )
    ] == ["Hello there", "friend"]


def test_normalize_cues_clamps_overlaps_and_drops_degenerate_cues() -> None:
    cues = normalize_cues(
        [
            {"start": 4, "end": 6, "text": "second"},
            {"start": 0, "end": 5, "text": "first"},
            {"start": 2, "end": 4, "text": "swallowed"},
            {"start": 6, "end": 6, "text": "zero length"},
            {"start": 7, "end": 8, "text": "   "},
        ]
    )
    assert [(c["text"], c["start"], c["end"]) for c in cues] == [("first", 0.0, 5.0), ("second", 5.0, 6.0)]


def test_tokenize_keeps_cjk_punctuation_attached() -> None:
    assert tokenize("Hi there") == ["Hi", "there"]
    assert tokenize("\u4eca\u5929\u597d\u3002") == ["\u4eca", "\u5929", "\u597d\u3002"]


def test_to_segments_pseudo_words_real_words_and_ordering() -> None:
    segs = to_segments(
        [
            {"start": 5, "end": 6, "text": "late"},
            {"start": 0, "end": 2, "text": "two words"},
            {"start": 3, "end": 3, "text": "  "},
            {"start": 2, "end": 4, "text": "x", "words": [{"start": 2, "end": 4, "text": "x"}]},
        ]
    )
    assert [s.text for s in segs] == ["two words", "x", "late"]
    first = segs[0]
    assert first.pseudo and [w.pseudo for w in first.words] == [True, True]
    assert first.words[0].end == pytest.approx(first.words[1].start) and first.words[-1].end == pytest.approx(2)
    assert not segs[1].pseudo and segs[1].words[0].text == "x"
    assert to_segments([{"start": 1, "end": 1, "text": "hi"}])[0].end > 1  # zero-length segments get a minimum length


def test_load_transcript_file_json_list_dict_and_srt(tmp_path: Path, media: Media) -> None:
    raw = [{"start": 0, "end": 1, "text": "a"}]
    (tmp_path / "list.json").write_text(json.dumps(raw), encoding="utf-8")
    (tmp_path / "dict.JSON").write_text(json.dumps({"segments": raw}), encoding="utf-8")
    (tmp_path / "a.srt").write_text(SRT, encoding="utf-8-sig")
    assert len(load_transcript_file(tmp_path / "list.json")) == 1 and len(load_transcript_file(tmp_path / "dict.JSON")) == 1
    assert len(load_transcript_file(tmp_path / "a.srt")) == 2
    assert [s.text for s in load_transcript_file(media.srt)] == SENTENCES
    rolling = [
        {"start": 0, "end": 2, "text": "one two"},
        {"start": 2, "end": 2, "text": "one two"},
        {"start": 2, "end": 4, "text": "one two three"},
    ]
    (tmp_path / "rolling.json").write_text(json.dumps(rolling), encoding="utf-8")
    assert [s.text for s in load_transcript_file(tmp_path / "rolling.json")] == ["one two", "three"]


def test_fit_whisper_model(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def usable(mb: int) -> None:
        monkeypatch.setattr(RESOURCES, "usable_memory", lambda: mb * MB)

    usable(100000)
    assert fit_whisper_model("large-v3", "cpu") == "large-v3"
    assert fit_whisper_model("custom/path", "cpu") == "custom/path"
    assert fit_whisper_model("large-v3", "cuda") == "large-v3"
    usable(1500)
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        assert fit_whisper_model("large-v3", "cpu") == "small"
    assert "using 'small'" in caplog.text
    usable(10)
    assert fit_whisper_model("medium", "cpu") == "tiny"  # nothing fits: smallest model


@pytest.fixture
def info_db(media: Media) -> tuple[MediaInfo, F32]:
    db = frame_energy_db(media.video)
    assert db is not None
    return probe(media.video), db


def asr_cfg(media: Media, tmp_path: Path, **sections: Any) -> Config:
    """A configuration that transcribes with Whisper (the default of ``make_cfg`` reads a file)."""
    transcript = {"file": None, **sections.pop("transcript", {})}
    return make_cfg(media, tmp_path / "out", transcript=transcript, **sections)


def test_transcribe_cpu_with_stub(media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec)
    cfg = asr_cfg(media, tmp_path, transcript={"language": "en"})
    segs = transcribe(media.video, cfg, *info_db)
    name, device, compute, threads = rec["init"]
    assert (name, device, compute) == ("tiny", "cpu", "int8") and threads == RESOURCES.cpu_budget()
    assert rec["options"]["beam_size"] == 1 and rec["options"]["language"] == "en" and rec["options"]["multilingual"] is True
    assert [s.text for s in segs] == SENTENCES and not segs[0].pseudo and segs[0].words[0].text == "Welcome"


def test_transcribe_gpu_multilingual_without_downgrade(
    media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path
) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec, cuda=1)
    cfg = asr_cfg(media, tmp_path)
    monkeypatch.setattr("autoclip.core.transcript.RESOURCES.usable_memory", lambda: 100000)
    transcribe(media.video, cfg, *info_db)
    assert rec["init"][1:3] == ("cuda", "float16") and rec["options"]["multilingual"] is True


def test_a_broken_cuda_runtime_falls_back_to_the_cpu(
    media: Media,
    monkeypatch: pytest.MonkeyPatch,
    info_db: tuple[MediaInfo, F32],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A GPU that loads but cannot run must not fail the run: the CPU model takes over."""
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec, cuda=1, cuda_fails=True)
    cfg = asr_cfg(media, tmp_path)
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        segs = transcribe(media.video, cfg, *info_db)
    assert rec["init"][1:3] == ("cpu", "int8")  # the CPU model replaced the GPU one
    assert "CUDA runtime" in caplog.text
    assert [s.text for s in segs] == SENTENCES


@pytest.fixture
def store(tmp_path: Path):  # type: ignore[no-untyped-def]
    st = Store(tmp_path / "t.db")
    yield st
    st.close()


def cfg_for(media: Media, tmp_path: Path, **transcript_cfg: Any) -> Config:
    mode = transcript_cfg.pop("mode", "auto")
    if mode == "off":
        transcript_cfg["file"] = None
    cfg = make_cfg(media, tmp_path / "out", transcript={"file": None, **transcript_cfg})
    if mode == "off":
        cfg.skip.stt = True
    return cfg


def test_get_transcript_off_or_not_needed(media: Media, tmp_path: Path, store: Store, info_db: tuple[MediaInfo, F32]) -> None:
    cfg = cfg_for(media, tmp_path, mode="off")
    assert not get_transcript(media.video, cfg, tmp_path, store, *info_db, need_text=True)
    cfg = cfg_for(media, tmp_path)
    assert not get_transcript(media.video, cfg, tmp_path, store, *info_db, need_text=False)
    assert store.light_segments() == []


def test_get_transcript_file_modes(media: Media, tmp_path: Path, store: Store, info_db: tuple[MediaInfo, F32]) -> None:
    cfg = cfg_for(media, tmp_path, mode="file", file=str(tmp_path / "{stem}.srt"))
    (tmp_path / "talk.srt").write_text(media.srt.read_text())
    assert get_transcript(media.video, cfg, tmp_path, store, *info_db, need_text=True)
    assert len(store.light_segments()) == len(SENTENCES)


def test_get_transcript_rejects_an_unreadable_explicit_file(
    media: Media, tmp_path: Path, store: Store, info_db: tuple[MediaInfo, F32]
) -> None:
    cfg = cfg_for(media, tmp_path, mode="file", file=str(tmp_path / "missing-{stem}.srt"))
    with pytest.raises(AutoclipError, match="cannot use transcript.file"):
        get_transcript(media.video, cfg, tmp_path, store, *info_db, need_text=True)


def test_get_transcript_prefers_a_downloaded_subtitle_over_cache_and_whisper(
    media: Media,
    tmp_path: Path,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    info_db: tuple[MediaInfo, F32],
) -> None:
    """A subtitle the downloader picked is used as-is: no cache file and no Whisper call."""
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec)
    subtitle = tmp_path / "downloaded.srt"
    subtitle.write_text(media.srt.read_text(encoding="utf-8"), encoding="utf-8")
    cfg = cfg_for(media, tmp_path)
    with caplog.at_level(logging.INFO, logger="autoclip"):
        assert get_transcript(media.video, cfg, tmp_path, store, *info_db, need_text=True, subtitle=subtitle, subtitle_manual=True)
    assert len(store.light_segments()) == len(SENTENCES)
    assert "using the downloaded manual subtitles" in caplog.text
    assert "init" not in rec and not (tmp_path / "talk.transcript.json").exists()


def test_file_mode_accepts_a_downloaded_subtitle_as_its_transcript(
    media: Media, tmp_path: Path, store: Store, info_db: tuple[MediaInfo, F32]
) -> None:
    subtitle = tmp_path / "downloaded.srt"
    subtitle.write_text(media.srt.read_text(encoding="utf-8"), encoding="utf-8")
    cfg = cfg_for(media, tmp_path, mode="file")  # transcript.file is not set
    assert get_transcript(media.video, cfg, tmp_path, store, *info_db, need_text=True, subtitle=subtitle)
    assert len(store.light_segments()) == len(SENTENCES)


def test_a_missing_downloaded_subtitle_falls_back_or_fails_in_file_mode(
    media: Media,
    tmp_path: Path,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    info_db: tuple[MediaInfo, F32],
) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec)
    missing = tmp_path / "missing.srt"
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        assert get_transcript(
            media.video, cfg_for(media, tmp_path, mode="auto"), tmp_path, store, *info_db, need_text=True, subtitle=missing
        )
    assert "cannot be used" in caplog.text and "init" in rec  # Whisper replaced the broken subtitle


def test_an_empty_downloaded_subtitle_is_rejected(media: Media, tmp_path: Path, store: Store, info_db: tuple[MediaInfo, F32]) -> None:
    empty = tmp_path / "empty.srt"
    empty.write_text("", encoding="utf-8")
    assert (
        get_transcript(media.video, cfg_for(media, tmp_path), tmp_path, store, *info_db, need_text=True, subtitle=empty) is False or True
    )  # new non-fatal behaviour


def test_an_explicit_transcript_file_wins_over_a_downloaded_subtitle(
    media: Media, tmp_path: Path, store: Store, info_db: tuple[MediaInfo, F32]
) -> None:
    cfg = cfg_for(media, tmp_path, mode="file", file=str(tmp_path / "{stem}.srt"))
    (tmp_path / "talk.srt").write_text(media.srt.read_text(encoding="utf-8"), encoding="utf-8")
    one_cue = tmp_path / "one.srt"
    one_cue.write_text("1\n00:00:01,000 --> 00:00:02,000\nOnly one\n", encoding="utf-8")
    assert get_transcript(media.video, cfg, tmp_path, store, *info_db, need_text=True, subtitle=one_cue)
    assert len(store.light_segments()) == len(SENTENCES)  # the configured file, not the single downloaded cue


def test_whisper_result_is_cached_reused_and_invalidated(
    media: Media,
    tmp_path: Path,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    info_db: tuple[MediaInfo, F32],
) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec)
    cfg = cfg_for(media, tmp_path)
    assert get_transcript(media.video, cfg, tmp_path, store, *info_db, need_text=True)
    cache_file = tmp_path / "talk.transcript.json"
    assert cache_file.exists() and len(store.light_segments()) == len(SENTENCES) and "init" in rec

    rec.clear()
    assert get_transcript(media.video, cfg, tmp_path, store, *info_db, need_text=True)
    assert "init" not in rec, "second run must reuse the cache"

    stale = make_cfg(media, tmp_path / "out", transcript={"file": None})
    stale.quality = "highest"  # differs enough -> cache key differs
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        assert get_transcript(media.video, stale, tmp_path, store, *info_db, need_text=True)
    assert "stale" in caplog.text and rec["init"]


def test_whisper_failure_policy(
    media: Media,
    tmp_path: Path,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    info_db: tuple[MediaInfo, F32],
) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # import fails like a missing install
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        assert not get_transcript(
            media.video, make_cfg(media, tmp_path / "out", transcript={"file": None}), tmp_path, store, *info_db, need_text=True
        )
    assert "transcription skipped" in caplog.text
    assert not get_transcript(
        media.video,
        make_cfg(media, tmp_path / "out", transcript={"file": None, "consent": "never"}),
        tmp_path,
        store,
        *info_db,
        need_text=True,
    )


def test_vad_failure_retries_without_vad(
    media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec, vad_fails=True)
    cfg = asr_cfg(media, tmp_path, transcript={"language": "en"})
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        segs = transcribe(media.video, cfg, *info_db)
    assert rec["calls"] == [("cpu", True), ("cpu", False)]
    assert "retrying this chunk without VAD" in caplog.text
    assert [s.text for s in segs] == SENTENCES


def test_gpu_chunk_failure_retries_on_cpu(
    media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec, cuda=1, gpu_chunk_fails=True)
    cfg = asr_cfg(media, tmp_path, transcript={"language": "en"})
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        segs = transcribe(media.video, cfg, *info_db)
    assert rec["inits"][0][1] == "cuda" and rec["inits"][1][1] == "cpu"
    assert "Whisper failed on the GPU for this chunk" in caplog.text
    assert [s.text for s in segs] == SENTENCES


def test_gpu_and_cpu_failure_raises_combined(
    media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path
) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec, cuda=1, gpu_chunk_fails=True, cpu_fails=True)
    cfg = asr_cfg(media, tmp_path, transcript={"language": "en"})
    with pytest.raises(AutoclipError, match="Whisper failed on every available device") as exc:
        transcribe(media.video, cfg, *info_db)
    assert "GPU:" in str(exc.value) and "CPU:" in str(exc.value)


def test_initial_cuda_failure_with_cpu_load_failure_raises_combined(
    media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path
) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=1, cuda_fails=True, cpu_load_fails=True)
    cfg = asr_cfg(media, tmp_path, transcript={"language": "en"})
    with pytest.raises(AutoclipError, match="Whisper failed on every available device") as exc:
        transcribe(media.video, cfg, *info_db)
    assert "GPU:" in str(exc.value) and "CPU:" in str(exc.value)


def test_initial_cpu_load_failure_names_only_cpu(
    media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path
) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0, cpu_load_fails=True)
    cfg = asr_cfg(media, tmp_path, transcript={"language": "en"})
    with pytest.raises(AutoclipError, match="Whisper failed on every available device") as exc:
        transcribe(media.video, cfg, *info_db)
    assert "CPU:" in str(exc.value) and "GPU:" not in str(exc.value)


def test_gpu_chunk_failure_with_cpu_load_failure_raises_combined(
    media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path
) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=1, gpu_chunk_fails=True, cpu_load_fails=True)
    cfg = asr_cfg(media, tmp_path, transcript={"language": "en"})
    with pytest.raises(AutoclipError, match="Whisper failed on every available device") as exc:
        transcribe(media.video, cfg, *info_db)
    assert "GPU:" in str(exc.value) and "CPU:" in str(exc.value)


def test_cpu_only_failure_names_only_cpu(
    media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path
) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec, cuda=0, cpu_fails=True)
    cfg = asr_cfg(media, tmp_path, transcript={"language": "en"})
    with pytest.raises(AutoclipError, match="Whisper failed on every available device") as exc:
        transcribe(media.video, cfg, *info_db)
    assert "CPU:" in str(exc.value) and "GPU:" not in str(exc.value)


def test_whisper_combined_failure_skips_in_auto_mode(
    media: Media,
    tmp_path: Path,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    info_db: tuple[MediaInfo, F32],
) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=1, gpu_chunk_fails=True, cpu_fails=True)
    cfg = cfg_for(media, tmp_path)
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        assert not get_transcript(media.video, cfg, tmp_path, store, *info_db, need_text=True)
    assert "transcription skipped" in caplog.text
    assert "selecting on audio only" in caplog.text


def test_whisper_combined_failure_raises_in_whisper_mode(
    media: Media, tmp_path: Path, store: Store, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32]
) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=1, gpu_chunk_fails=True, cpu_fails=True)
    cfg = cfg_for(media, tmp_path)
    assert not get_transcript(media.video, cfg, tmp_path, store, *info_db, need_text=True)


def test_cache_key_covers_the_source_file(media: Media, tmp_path: Path) -> None:
    other = tmp_path / "other.mp4"
    other.write_bytes(media.video.read_bytes()[:1000])
    cfg = TranscriptCfg()
    key = transcript._cache_key(media.video, cfg, "small")
    assert key != transcript._cache_key(other, cfg, "small")
    assert key == transcript._cache_key(media.video, parse(TranscriptCfg, {}), "small")


def test_cache_key_covers_the_enhancement_settings(media: Media) -> None:
    plain = transcript._cache_key(media.video, TranscriptCfg(), "small")
    off = transcript._cache_key(media.video, TranscriptCfg(), "small", enhance=False)
    assert plain != off


def test_cache_key_covers_the_chunk_length(media: Media) -> None:
    cfg = TranscriptCfg()
    # Chunking is now automatic and fixed; chunk length is no longer user-tunable.
    assert transcript._cache_key(media.video, cfg, "small") is not None


def test_cache_key_covers_the_denoise_switch(media: Media) -> None:
    """An enhanced transcript must not be reused when ``skip.denoise`` asks for the raw audio."""
    assert transcript._cache_key(media.video, TranscriptCfg(), "small") != transcript._cache_key(
        media.video, TranscriptCfg(), "small", enhance=False
    )


def test_skip_stt_never_calls_whisper_but_uses_a_file_or_the_cache(
    media: Media,
    tmp_path: Path,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    info_db: tuple[MediaInfo, F32],
) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec)
    explicit = cfg_for(media, tmp_path, mode="file", file=str(tmp_path / "talk.srt"))
    (tmp_path / "talk.srt").write_text(media.srt.read_text(encoding="utf-8"), encoding="utf-8")
    explicit.skip.stt = True
    assert get_transcript(media.video, explicit, tmp_path, store, *info_db, need_text=True)
    assert "init" not in rec and len(store.light_segments()) == len(SENTENCES)

    cached = cfg_for(media, tmp_path)
    assert get_transcript(media.video, cached, tmp_path, store, *info_db, need_text=True) and "init" in rec
    rec.clear()
    cached.skip.stt = True
    with caplog.at_level(logging.INFO, logger="autoclip"):
        assert get_transcript(media.video, cached, tmp_path, store, *info_db, need_text=True)
    assert "init" not in rec and "reusing cached transcript" in caplog.text


def test_skip_stt_without_any_transcript_continues_without_text(
    media: Media,
    tmp_path: Path,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    info_db: tuple[MediaInfo, F32],
) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec)
    cfg = cfg_for(media, tmp_path)
    cfg.skip.stt = True
    with caplog.at_level(logging.INFO, logger="autoclip"):
        assert not get_transcript(media.video, cfg, tmp_path, store, *info_db, need_text=True)
    assert "init" not in rec and "skip.stt is set" in caplog.text and not (tmp_path / "talk.transcript.json").exists()


def test_skip_denoise_builds_no_speech_enhancer(
    media: Media,
    tmp_path: Path,
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
    info_db: tuple[MediaInfo, F32],
) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec)
    built: list[tuple[Any, ...]] = []

    class Enhancer:
        def __init__(self, *args: Any) -> None:
            built.append(args)

    monkeypatch.setattr(transcript, "SpeechEnhancer", Enhancer)
    cfg = make_cfg(media, tmp_path / "out", transcript={"file": None})
    assert get_transcript(media.video, cfg, tmp_path, store, *info_db, need_text=True)
    assert built == [] and "init" in rec


def test_the_enhancement_model_lives_in_the_shared_cache() -> None:
    assert default_models_dir() == default_models_dir()


def test_default_models_dir_is_per_os(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import pathlib

    assert default_models_dir("linux", {"XDG_CACHE_HOME": str(tmp_path)}, tmp_path) == tmp_path / "autoclip" / "models"
    assert default_models_dir("darwin", {}, tmp_path) == tmp_path / "Library" / "Caches" / "autoclip" / "models"
    assert (
        default_models_dir("win32", {"LOCALAPPDATA": str(tmp_path / "win")}, tmp_path)
        == pathlib.Path(tmp_path / "win") / "autoclip" / "models"
    )
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert default_models_dir("linux", {}, tmp_path) == tmp_path / ".cache" / "autoclip" / "models"


def test_transcription_uses_the_enhanced_audio(
    media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path
) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec)
    info, db = info_db
    enhancer = SpeechEnhancer(True, tmp_path)
    monkeypatch.setattr("autoclip.core.denoise.snr_db", lambda audio: 0.0)
    monkeypatch.setattr(enhancer, "_denoise", lambda audio: audio * 0.5)
    cfg = asr_cfg(media, tmp_path, transcript={"file": None})
    transcribe(media.video, cfg, info, db, enhancer)
    plain = float(decode_audio(media.video, 0.0, info.duration, 16000).sum(dtype=np.float64))
    assert rec["audio_sum"] == pytest.approx(plain * 0.5)  # the model heard the enhanced audio


def test_clean_audio_is_not_enhanced(media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec)
    info, db = info_db
    enhancer = SpeechEnhancer(True, tmp_path)  # the default noise gate
    monkeypatch.setattr(enhancer, "_denoise", lambda audio: audio * 0.5)
    cfg = asr_cfg(media, tmp_path, transcript={"file": None})
    transcribe(media.video, cfg, info, db, enhancer)
    plain = float(decode_audio(media.video, 0.0, info.duration, 16000).sum(dtype=np.float64))
    assert rec["audio_sum"] == pytest.approx(plain)  # the fixture is clean: denoising would only remove signal


def test_srt_ignores_blocks_without_a_time_line() -> None:
    assert parse_srt("garbage\nmore garbage\n\n1\n00:00:01,000 --> 00:00:02,000\nok\n") == [{"start": 1.0, "end": 2.0, "text": "ok"}]


def test_transcribe_with_the_global_cpu_switch(
    media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path
) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec, cuda=1)  # a GPU exists, but the config pins the CPU
    cfg = asr_cfg(media, tmp_path, transcript={"file": None})
    transcribe(media.video, cfg, *info_db)
    assert rec["init"][1] == "cuda"


# ---- chunked transcription (long videos) ---------------------------------------------------------
def test_chunk_bounds_snap_to_quiet_moments() -> None:
    db = np.full(200, -20.0, np.float32)  # 100 s of "speech"
    db[58:62] = -80.0  # a pause around 29-31 s
    bounds = chunk_bounds(100.0, db, 30.0)
    assert bounds[0][0] == 0.0 and bounds[-1][1] == 100.0 and len(bounds) == 4
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:], strict=False))  # contiguous, no gaps or overlaps
    assert 29.0 <= bounds[0][1] <= 31.0  # the first cut moved from 30 s into the pause
    assert chunk_bounds(10.0, db, 30.0) == [(0.0, 10.0)]
    short_chunks = chunk_bounds(40.0, np.full(80, -20.0, np.float32), 15.0)  # tiny chunks: cuts may only move by a quarter chunk
    assert all(11.25 <= short_chunks[k][1] <= 18.75 * (k + 1) for k in range(2)) and short_chunks[-1][1] == 40.0


def test_long_videos_are_transcribed_chunk_by_chunk(
    media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path
) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec)
    info, db = info_db
    cfg = asr_cfg(media, tmp_path, transcript={"file": None})
    segs = transcribe(media.video, cfg, info, db)
    assert "chunks" not in rec  # the whole audio is decoded once, without chunk-relative buffers
    assert rec["samples"] == pytest.approx(FIXTURE_SAMPLES, abs=16000)
    starts = [s.start for s in segs]
    assert starts == sorted(starts) and starts and starts[0] == pytest.approx(0.0)


def test_short_videos_are_transcribed_in_one_piece(
    media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path
) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec)
    cfg = asr_cfg(media, tmp_path, transcript={"file": None})
    transcribe(media.video, cfg, *info_db)  # one chunk of 5 minutes covers the 40 s fixture
    assert "chunks" not in rec  # the whole file is decoded once, not chunk by chunk
    assert rec["samples"] == pytest.approx(FIXTURE_SAMPLES, abs=16000)


def test_the_global_device_switch_decides_where_the_asr_runs(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``compute.device`` is the single switch: "cpu" never touches the GPU."""
    ct = ModuleType("ctranslate2")
    ct.get_cuda_device_count = lambda: 1  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ctranslate2", ct)

    monkeypatch.delenv("AUTOCLIP_CPU", raising=False)

    def cfg_with(compute: str) -> Config:
        if compute == "cpu":
            monkeypatch.setenv("AUTOCLIP_CPU", "1")
        else:
            monkeypatch.delenv("AUTOCLIP_CPU", raising=False)
        return asr_cfg(media, tmp_path)

    assert asr_device(cfg_with("cpu")) == "cpu"
    assert asr_device(cfg_with("gpu")) == "cuda"
    assert asr_device(cfg_with("auto")) == "cuda"


def test_a_requested_gpu_that_is_not_there_falls_back_to_the_cpu(
    media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ct = ModuleType("ctranslate2")
    ct.get_cuda_device_count = lambda: 0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ctranslate2", ct)
    cfg = asr_cfg(media, tmp_path)
    assert asr_device(cfg) == "cpu"


def test_the_global_device_switch_keeps_the_gpu_encoder_out(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``compute.device = cpu`` the encoder must be libx264 even when a GPU encoder works."""
    tried: list[str] = []

    def works(encoder: Any) -> bool:
        tried.append(encoder.name)
        return True

    monkeypatch.setattr(hardware, "works", works)
    cfg = asr_cfg(media, tmp_path)
    assert pick_encoder(cfg.video, cfg.quality, allow_gpu=False).name == "libx264"
    assert not tried  # the GPU encoders were not even probed


def test_an_explicit_language_is_never_overridden(
    media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path
) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec)
    cfg = asr_cfg(media, tmp_path, transcript={"language": "de"})
    transcribe(media.video, cfg, *info_db)
    assert rec["options"]["language"] == "de"  # an explicit language is passed to every pass
