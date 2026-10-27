"""Transcript parsing, caching and the (stubbed) Whisper integration."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from autoclip.core import hardware, transcript
from autoclip.core.config import Config, EnhanceCfg, ProcessingCfg, TranscriptCfg, parse
from autoclip.core.denoise import SpeechEnhancer
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
    parse_srt,
    to_segments,
    tokenize,
    transcribe,
)

from .conftest import SENTENCES, Media, make_cfg

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


FIXTURE_SAMPLES = 40 * 16000  # the fixture video is 40 s long


def asr_cfg(media: Media, tmp_path: Path, **sections: Any) -> Config:
    """A configuration that transcribes with Whisper (the default of ``make_cfg`` reads a file)."""
    transcript = {"mode": "auto", **sections.pop("transcript", {})}
    return make_cfg(media, tmp_path / "out", transcript=transcript, **sections)


def install_stub_whisper(
    monkeypatch: pytest.MonkeyPatch, media: Media, record: dict[str, Any], cuda: int = 0, full_samples: int = FIXTURE_SAMPLES
) -> None:
    """A stand-in for faster-whisper / ctranslate2 with the real call signatures (the model host is offline in CI)."""
    segments = json.loads(media.transcript.read_text())

    class WhisperModel:
        def __init__(self, name: str, device: str, compute_type: str, cpu_threads: int) -> None:
            record["init"] = (name, device, compute_type, cpu_threads)

        def transcribe(self, audio: Any, **options: Any) -> tuple[Any, Any]:
            record["options"] = options
            if isinstance(audio, np.ndarray) and len(audio) < full_samples - 16000:  # a chunk, not the whole file
                record.setdefault("chunks", []).append((len(audio), options["language"]))
                seg = SimpleNamespace(start=0.5, end=2.5, text=" chunk", words=[SimpleNamespace(start=0.5, end=1.5, word=" chunk")])
                return iter([seg]), SimpleNamespace(language="en", language_probability=0.9, duration=len(audio) / 16000)
            record["samples"] = len(audio)
            record["audio_sum"] = float(np.asarray(audio, dtype=np.float64).sum())
            gen = (
                SimpleNamespace(
                    start=s["start"],
                    end=s["end"],
                    text=f" {s['text']}",
                    words=[SimpleNamespace(start=w["start"], end=w["end"], word=f" {w['word']}") for w in s["words"]],
                )
                for s in segments
            )
            return gen, SimpleNamespace(language="en", language_probability=0.99, duration=40.0)

    fw, ct = ModuleType("faster_whisper"), ModuleType("ctranslate2")
    fw.WhisperModel = WhisperModel  # type: ignore[attr-defined]
    ct.get_cuda_device_count = lambda: cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fw)
    monkeypatch.setitem(sys.modules, "ctranslate2", ct)


def test_transcribe_cpu_with_stub(media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec)
    cfg = asr_cfg(media, tmp_path, transcript={"model": "tiny", "beam_size": 1, "language": "en", "multilingual": False})
    segs = transcribe(media.video, cfg, *info_db)
    name, device, compute, threads = rec["init"]
    assert (name, device, compute) == ("tiny", "cpu", "int8") and threads == RESOURCES.cpu_budget()
    assert rec["options"]["beam_size"] == 1 and rec["options"]["language"] == "en" and "multilingual" not in rec["options"]
    assert [s.text for s in segs] == SENTENCES and not segs[0].pseudo and segs[0].words[0].text == "Welcome"


def test_transcribe_gpu_multilingual_without_downgrade(
    media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path
) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec, cuda=1)
    cfg = asr_cfg(media, tmp_path, transcript={"model": "large-v3", "multilingual": True, "auto_downgrade": False})
    transcribe(media.video, cfg, *info_db)
    assert rec["init"][:3] == ("large-v3", "cuda", "float16") and rec["options"]["multilingual"] is True


@pytest.fixture
def store(tmp_path: Path):  # type: ignore[no-untyped-def]
    st = Store(tmp_path / "t.db")
    yield st
    st.close()


def cfg_for(media: Media, tmp_path: Path, **transcript_cfg: Any) -> Config:
    return make_cfg(media, tmp_path / "out", transcript={"mode": "auto", "file": None, **transcript_cfg})


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
    with pytest.raises(AutoclipError, match="neither transcript.file nor a downloaded subtitle"):
        get_transcript(media.video, cfg_for(media, tmp_path, mode="file"), tmp_path, store, *info_db, need_text=True)


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
    cfg = cfg_for(media, tmp_path, mode="auto", model="tiny")
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
    with pytest.raises(AutoclipError, match="cannot use the downloaded subtitles"):
        get_transcript(media.video, cfg_for(media, tmp_path, mode="file"), tmp_path, store, *info_db, need_text=True, subtitle=missing)


def test_an_empty_downloaded_subtitle_is_rejected(media: Media, tmp_path: Path, store: Store, info_db: tuple[MediaInfo, F32]) -> None:
    empty = tmp_path / "empty.srt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(AutoclipError, match="it has no cues"):
        get_transcript(media.video, cfg_for(media, tmp_path, mode="file"), tmp_path, store, *info_db, need_text=True, subtitle=empty)


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
    cfg = cfg_for(media, tmp_path, mode="auto", model="tiny")
    assert get_transcript(media.video, cfg, tmp_path, store, *info_db, need_text=True)
    cache_file = tmp_path / "talk.transcript.json"
    assert cache_file.exists() and len(store.light_segments()) == len(SENTENCES) and "init" in rec

    rec.clear()
    assert get_transcript(media.video, cfg, tmp_path, store, *info_db, need_text=True)
    assert "init" not in rec, "second run must reuse the cache"

    stale = cfg_for(media, tmp_path, mode="auto", model="base")  # model changed -> cache key differs
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        assert get_transcript(media.video, stale, tmp_path, store, *info_db, need_text=True)
    assert "stale" in caplog.text and rec["init"][0] == "base"

    rec.clear()
    no_cache = cfg_for(media, tmp_path, cache=False, model="base")
    assert get_transcript(media.video, no_cache, tmp_path, store, *info_db, need_text=True) and "init" in rec


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
        assert not get_transcript(media.video, cfg_for(media, tmp_path, mode="auto"), tmp_path, store, *info_db, need_text=True)
    assert "transcription unavailable" in caplog.text
    with pytest.raises(AutoclipError, match="transcription failed"):
        get_transcript(media.video, cfg_for(media, tmp_path, mode="whisper"), tmp_path, store, *info_db, need_text=True)


def test_cache_key_covers_the_source_file(media: Media, tmp_path: Path) -> None:
    other = tmp_path / "other.mp4"
    other.write_bytes(media.video.read_bytes()[:1000])
    cfg = TranscriptCfg()
    key = transcript._cache_key(media.video, cfg, ProcessingCfg())
    assert key != transcript._cache_key(other, cfg, ProcessingCfg())
    assert key == transcript._cache_key(media.video, parse(TranscriptCfg, {}), ProcessingCfg())


def test_cache_key_covers_the_enhancement_settings(media: Media) -> None:
    proc = ProcessingCfg()
    plain = transcript._cache_key(media.video, TranscriptCfg(), proc)
    off = transcript._cache_key(media.video, TranscriptCfg(enhance=EnhanceCfg(enabled=False)), proc)
    other = transcript._cache_key(media.video, TranscriptCfg(enhance=EnhanceCfg(model="dpdfnet8")), proc)
    assert plain != off and plain != other and off != other


def test_cache_key_covers_the_chunk_length(media: Media) -> None:
    cfg = TranscriptCfg()
    assert transcript._cache_key(media.video, cfg, ProcessingCfg()) != transcript._cache_key(
        media.video, cfg, ProcessingCfg(chunk_minutes=2)
    )


def test_the_enhancement_model_lives_beside_the_work_directory(tmp_path: Path) -> None:
    work = tmp_path / ".work" / "video"
    assert transcript._model_dir(TranscriptCfg(), work) == tmp_path / ".work" / "models"
    configured = transcript._model_dir(TranscriptCfg(enhance=EnhanceCfg(model_dir=str(tmp_path / "m"))), work)
    assert configured == tmp_path / "m"


def test_transcription_uses_the_enhanced_audio(
    media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path
) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec)
    info, db = info_db
    enhancer = SpeechEnhancer(EnhanceCfg(backend="noisereduce", snr_threshold_db=None), tmp_path)
    monkeypatch.setattr(enhancer, "_denoise", lambda audio: audio * 0.5)
    cfg = asr_cfg(media, tmp_path, transcript={"model": "tiny"})
    transcribe(media.video, cfg, info, db, enhancer)
    plain = float(decode_audio(media.video, 0.0, info.duration, 16000).sum(dtype=np.float64))
    assert rec["audio_sum"] == pytest.approx(plain * 0.5)  # the model heard the enhanced audio


def test_clean_audio_is_not_enhanced(media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec)
    info, db = info_db
    enhancer = SpeechEnhancer(EnhanceCfg(backend="noisereduce"), tmp_path)  # the default noise gate
    monkeypatch.setattr(enhancer, "_denoise", lambda audio: audio * 0.5)
    cfg = asr_cfg(media, tmp_path, transcript={"model": "tiny"})
    transcribe(media.video, cfg, info, db, enhancer)
    plain = float(decode_audio(media.video, 0.0, info.duration, 16000).sum(dtype=np.float64))
    assert rec["audio_sum"] == pytest.approx(plain)  # the fixture is clean: denoising would only remove signal


def test_srt_ignores_blocks_without_a_time_line() -> None:
    assert parse_srt("garbage\nmore garbage\n\n1\n00:00:01,000 --> 00:00:02,000\nok\n") == [{"start": 1.0, "end": 2.0, "text": "ok"}]


def test_transcribe_with_an_explicit_device(
    media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path
) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec, cuda=1)  # a GPU exists, but the config pins the CPU
    cfg = asr_cfg(media, tmp_path, transcript={"model": "tiny", "device": "cpu"})
    transcribe(media.video, cfg, *info_db)
    assert rec["init"][1] == "cpu"


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
    cfg = asr_cfg(media, tmp_path, transcript={"model": "tiny"}, processing={"chunk_threshold_minutes": 0.5, "chunk_minutes": 0.25})
    segs = transcribe(media.video, cfg, info, db)
    sizes = [n for n, _ in rec["chunks"]]
    assert len(sizes) == 3 and sum(sizes) == pytest.approx(40 * 16000, abs=3)  # 40 s in 15 s chunks, every sample used once
    assert max(sizes) <= 16.5 * 16000  # never more than one chunk's audio at a time
    assert [lang for _, lang in rec["chunks"]] == [None, "en", "en"]  # detected once, then reused
    starts = [s.start for s in segs]
    assert starts == sorted(starts) and len(set(starts)) == 3 and starts[0] == pytest.approx(0.5)  # chunk-relative times were shifted
    assert starts[1] > 10 and starts[2] > 25 and segs[1].words[0].start == pytest.approx(starts[1])


def test_short_videos_are_transcribed_in_one_piece(
    media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path
) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec)
    cfg = asr_cfg(media, tmp_path, transcript={"model": "tiny"})
    transcribe(media.video, cfg, *info_db)  # one chunk of 5 minutes covers the 40 s fixture
    assert "chunks" not in rec  # the whole file is decoded once, not chunk by chunk
    assert rec["samples"] == pytest.approx(FIXTURE_SAMPLES, abs=16000)


def test_the_global_device_switch_decides_where_the_asr_runs(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``compute.device`` wins over ``transcript.device``; "cpu" never touches the GPU."""
    ct = ModuleType("ctranslate2")
    ct.get_cuda_device_count = lambda: 1  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ctranslate2", ct)

    def cfg_with(compute: str, device: str = "auto") -> Config:
        return asr_cfg(media, tmp_path, compute={"device": compute}, transcript={"device": device})

    assert asr_device(cfg_with("cpu")) == "cpu"
    assert asr_device(cfg_with("cpu", "cuda")) == "cpu"  # the global switch wins
    assert asr_device(cfg_with("gpu")) == "cuda"
    assert asr_device(cfg_with("auto")) == "cuda"
    assert asr_device(cfg_with("auto", "cpu")) == "cpu"  # the more specific setting wins
    assert asr_device(cfg_with("auto", "cuda")) == "cuda"


def test_a_requested_gpu_that_is_not_there_falls_back_to_the_cpu(
    media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ct = ModuleType("ctranslate2")
    ct.get_cuda_device_count = lambda: 0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ctranslate2", ct)
    cfg = asr_cfg(media, tmp_path, compute={"device": "gpu"})
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        assert asr_device(cfg) == "cpu"
    assert "no CUDA device" in caplog.text


def test_the_global_device_switch_keeps_the_gpu_encoder_out(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``compute.device = cpu`` the encoder must be libx264 even when a GPU encoder works."""
    tried: list[str] = []

    def works(encoder: Any) -> bool:
        tried.append(encoder.name)
        return True

    monkeypatch.setattr(hardware, "works", works)
    cfg = asr_cfg(media, tmp_path, compute={"device": "cpu"})
    assert pick_encoder(cfg.video, allow_gpu=cfg.compute.device != "cpu").name == "libx264"
    assert not tried  # the GPU encoders were not even probed


def test_an_explicit_language_is_never_overridden(
    media: Media, monkeypatch: pytest.MonkeyPatch, info_db: tuple[MediaInfo, F32], tmp_path: Path
) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec)
    cfg = asr_cfg(
        media, tmp_path, transcript={"model": "tiny", "language": "de"}, processing={"chunk_threshold_minutes": 0.1, "chunk_minutes": 0.5}
    )
    transcribe(media.video, cfg, *info_db)
    assert {lang for _, lang in rec["chunks"]} == {"de"}  # an explicit language is never overridden
