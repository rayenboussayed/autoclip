"""Whisper fused into the silence-removal assembly, one bounded chunk at a time."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import autoclip.core.optimizer as optimizer
import autoclip.core.transcript as transcript
from autoclip.core.config import Config
from autoclip.core.cutmap import CutMap
from autoclip.core.derive.constants import CHUNK_MINUTES
from autoclip.core.events import Cancelled, CancelToken
from autoclip.core.hardware import cpu_encoder
from autoclip.core.media import probe
from autoclip.core.optimizer import prepare_source
from autoclip.core.pipeline import process_video
from autoclip.core.transcript import FusedStt, get_transcript

from ..support import Media, bare_media, install_stub_whisper, make_cfg


def _fused(src: Path, cfg: Config, work: Path, stem: str, *, want: bool = True, subtitle: Any = None, model: str = "tiny") -> FusedStt:
    return FusedStt(src, cfg, work, stem, want, model, subtitle=subtitle)


def test_prepare_source_transcribes_each_chunk_during_assembly(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fresh assembly drives Whisper once per kept chunk and leaves a resumable partial."""
    src = bare_media(media, tmp_path)
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec, cuda=0)
    monkeypatch.setattr(optimizer, "UNSILENCE_MIN_TOTAL_SECONDS", 0.0)
    cfg = make_cfg(media, tmp_path / "out", source=src)
    work = tmp_path / "work"
    info = probe(src)
    fused = _fused(src, cfg, work, src.stem)
    prepared = prepare_source(src, cfg, info, work, cancel=CancelToken(), stt_factory=fused.plan)
    assert prepared.audio_ready and not prepared.stt_failed
    assert "init" in rec and rec["chunks"]  # Whisper really ran inside the assembly
    partial = work / f"{src.stem}.transcript.partial.jsonl"
    lines = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["type"] == "meta" and lines[0]["format"] == transcript.TRANSCRIPT_CACHE_FORMAT
    assert [row["done"] for row in lines[1:]] == list(range(1, len(lines)))


def test_fused_stt_failure_does_not_abort_assembly(
    media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A Whisper failure is isolated: the optimized file is committed and the run selects on audio only."""
    src = bare_media(media, tmp_path)
    monkeypatch.setattr(optimizer, "UNSILENCE_MIN_TOTAL_SECONDS", 0.0)
    cfg = make_cfg(media, tmp_path / "out", source=src)
    work = tmp_path / "work"
    info = probe(src)
    calls: list[int] = []

    def failing(self: Any, k: int, pieces: Any, cancel: Any = None) -> None:
        calls.append(k)
        raise RuntimeError("backend gone")

    monkeypatch.setattr(transcript.WhisperSession, "transcribe_chunk", failing)
    prepared = prepare_source(src, cfg, info, work, stt_factory=_fused(src, cfg, work, src.stem).plan)
    assert prepared.audio_ready and prepared.stt_failed and calls == [1]
    from autoclip.core.store import Store

    store = Store(work / "select.db", fresh=True)
    try:
        with caplog.at_level("WARNING", logger="autoclip"):
            assert not get_transcript(src, cfg, work, store, info, stt_failed=prepared.stt_failed)
    finally:
        store.close()
    assert "disabled after a failure while assembling" in caplog.text


def test_fused_cancellation_propagates(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelled is never converted into a transcription failure or an AutoclipError."""
    src = bare_media(media, tmp_path)
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec, cuda=0)
    monkeypatch.setattr(optimizer, "UNSILENCE_MIN_TOTAL_SECONDS", 0.0)
    monkeypatch.setattr(transcript, "_map_local_time", lambda *a: (_ for _ in ()).throw(Cancelled("cancelled")))
    cfg = make_cfg(media, tmp_path / "out", source=src)
    work = tmp_path / "work"
    info = probe(src)
    with pytest.raises(Cancelled):
        prepare_source(src, cfg, info, work, stt_factory=_fused(src, cfg, work, src.stem).plan)


def test_complete_partial_replays_without_loading_the_model(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The standalone path short-circuits when the fused pass already wrote every chunk row."""
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec, cuda=0)
    cfg = make_cfg(media, tmp_path / "out", source=media.video)
    info = probe(media.video)
    cut = CutMap.identity(info.duration)
    monkeypatch.setattr(transcript, "transcript_model", lambda: "tiny")
    key = transcript._cache_key(media.video, "tiny", cut)
    partial = tmp_path / "done.partial.jsonl"
    rows = [{"start": 0.5, "end": 2.5, "text": " hello", "source_start": 0.5, "source_end": 2.5, "words": []}]
    partial.write_text(
        json.dumps({"type": "meta", "meta": key, "format": transcript.TRANSCRIPT_CACHE_FORMAT})
        + "\n"
        + json.dumps({"type": "chunk", "done": 1, "language": "en", "rows": rows})
        + "\n",
        encoding="utf-8",
    )
    chunks = cut.chunked_pieces(CHUNK_MINUTES * 60.0)
    assert len(chunks) == 1
    segments = transcript.transcribe(media.video, cfg, info, chunks=chunks, partial_path=partial, partial_meta=key)
    assert [s.text for s in segments] == ["hello"]
    assert "init" not in rec  # no model load, no decode


def test_fused_plan_disarms_when_a_cheaper_source_or_no_text_wins(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fused pass never runs when the standalone pass would not reach Whisper."""
    cfg = make_cfg(media, tmp_path / "out", source=media.video)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(transcript, "transcript_model", lambda: "tiny")
    info = probe(media.video)
    cut = CutMap.identity(info.duration)
    assert _fused(media.video, cfg, work, "x", want=False).plan(cut) is None
    assert _fused(media.video, cfg, work, "x", subtitle=media.srt).plan(cut) is None
    assert _fused(media.srt, cfg, work, "x").plan(cut) is None  # a sidecar transcript exists
    # A full cache hit disarms the fused pass too.
    src = bare_media(media, tmp_path, "cached.mp4")
    key = transcript._cache_key(src, "tiny", cut)
    (work / "cached.transcript.json").write_text(
        json.dumps(
            {"meta": key, "format": transcript.TRANSCRIPT_CACHE_FORMAT, "segments": [{"start": 0.5, "end": 2.5, "text": "hi", "words": []}]}
        ),
        encoding="utf-8",
    )
    assert _fused(src, cfg, work, "cached").plan(cut) is None


def test_get_transcript_propagates_cancellation(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelled from the transcription stage reaches the caller (it is not a transcription failure)."""
    src = bare_media(media, tmp_path, "cancel.mp4")
    cfg = make_cfg(media, tmp_path / "out", source=src)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(transcript, "transcribe", lambda *a, **k: (_ for _ in ()).throw(Cancelled("cancelled")))
    from autoclip.core.store import Store

    store = Store(work / "select.db", fresh=True)
    try:
        with pytest.raises(Cancelled):
            get_transcript(src, cfg, work, store, probe(src))
    finally:
        store.close()


def test_process_video_fuses_stt_in_the_assembly(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end: the assembly stage produces the transcript and the store sees its mapped rows."""
    src = bare_media(media, tmp_path, "e2e.mp4")
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec, cuda=0)
    monkeypatch.setattr("autoclip.core.optimizer.UNSILENCE_MIN_TOTAL_SECONDS", 0.0)
    out = tmp_path / "out"
    cfg = make_cfg(media, out, source=src, clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    cfg.captions.enabled = False
    cfg.skip.title = True
    done, failed = process_video(src, cfg, None, cpu_encoder(cfg.video, cfg.quality))
    assert failed == 0 and done >= 1
    assert rec["chunks"] and "samples" not in rec  # chunk calls only, never a whole-file replay
    partial = out / ".work" / src.stem / f"{src.stem}.transcript.partial.jsonl"
    assert not partial.exists()  # get_transcript wrote the full cache and removed the partial
    cache = out / ".work" / src.stem / f"{src.stem}.transcript.json"
    assert cache.is_file()
    summary = json.loads((out / "short" / src.stem / "clips.json").read_text(encoding="utf-8"))
    assert summary and all("source_start" in row for row in summary)
    from autoclip.core.store import Store

    store = Store(out / ".work" / src.stem / "autoclip.db")
    try:
        assert store.light_segments()  # the fused transcript reached the store
    finally:
        store.close()


def test_fused_failure_stops_transcribing_later_chunks(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After a fused failure the assembly continues, but STT stops (so `done` stays a prefix)."""
    src = bare_media(media, tmp_path, "many.mp4")
    monkeypatch.setattr(optimizer, "UNSILENCE_MIN_TOTAL_SECONDS", 0.0)
    monkeypatch.setattr(optimizer, "CHUNK_MINUTES", 0.05)
    monkeypatch.setattr(transcript, "_CHUNK_MINUTES", 0.05)
    cfg = make_cfg(media, tmp_path / "out", source=src)
    work = tmp_path / "work"
    info = probe(src)
    calls: list[int] = []

    def failing(self: Any, k: int, pieces: Any, cancel: Any = None) -> None:
        calls.append(k)
        raise RuntimeError("backend gone")

    monkeypatch.setattr(transcript.WhisperSession, "transcribe_chunk", failing)
    prepared = prepare_source(src, cfg, info, work, stt_factory=_fused(src, cfg, work, src.stem).plan)
    assert prepared.stt_failed and calls == [1]


def test_fused_plan_disarms_on_a_complete_partial_and_skip_denoise(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = bare_media(media, tmp_path, "complete.mp4")
    cfg = make_cfg(media, tmp_path / "out", source=src)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(transcript, "transcript_model", lambda: "tiny")
    cut = CutMap.identity(probe(src).duration)
    chunks = cut.chunked_pieces(CHUNK_MINUTES * 60.0)
    assert len(chunks) == 1
    key = transcript._cache_key(src, "tiny", cut)
    partial = work / f"{src.stem}.transcript.partial.jsonl"
    partial.write_text(
        json.dumps({"type": "meta", "meta": key, "format": transcript.TRANSCRIPT_CACHE_FORMAT})
        + "\n"
        + json.dumps({"type": "chunk", "done": 1, "language": "en", "rows": []})
        + "\n",
        encoding="utf-8",
    )
    assert _fused(src, cfg, work, src.stem).plan(cut) is None  # complete partial: no model load
    partial.unlink()
    cfg.skip.denoise = True
    session = _fused(src, cfg, work, src.stem).plan(cut)
    assert session is not None and session.enhancer is None


def test_fused_factory_failure_does_not_abort_assembly(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = bare_media(media, tmp_path, "factory.mp4")
    monkeypatch.setattr(optimizer, "UNSILENCE_MIN_TOTAL_SECONDS", 0.0)
    cfg = make_cfg(media, tmp_path / "out", source=src)
    work = tmp_path / "work"
    info = probe(src)

    def broken(cut: Any) -> Any:
        raise RuntimeError("cannot arm")

    prepared = prepare_source(src, cfg, info, work, stt_factory=broken)
    assert prepared.audio_ready and prepared.stt_failed


def test_partial_torn_tail_is_repaired_on_the_next_append(tmp_path: Path) -> None:
    partial = tmp_path / "partial.jsonl"
    meta: dict[str, Any] = {"model": "tiny"}
    header = {"type": "meta", "meta": meta, "format": transcript.TRANSCRIPT_CACHE_FORMAT}
    segment = {"start": 0.5, "end": 2.5, "text": " hi", "source_start": 0.5, "source_end": 2.5, "words": []}
    good = {"type": "chunk", "done": 1, "language": "en", "rows": [segment]}

    def line(obj: dict[str, Any]) -> str:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    partial.write_text(line(header) + "\n" + line(good) + "\n" + '{"type": "chunk", "done": 2,', encoding="utf-8")
    transcript._partial_append(partial, header, {"type": "chunk", "done": 2, "language": "en", "rows": []})
    loaded = transcript._partial_load(partial, meta, 2)
    assert loaded[2] == 2 and [row["text"] for row in loaded[0]] == [" hi"]  # torn line removed


def test_partial_bad_row_is_truncated_on_append(tmp_path: Path) -> None:
    partial = tmp_path / "partial.jsonl"
    meta: dict[str, Any] = {"model": "tiny"}
    header = {"type": "meta", "meta": meta, "format": transcript.TRANSCRIPT_CACHE_FORMAT}
    segment = {"start": 0.5, "end": 2.5, "text": " hi", "source_start": 0.5, "source_end": 2.5, "words": []}
    good = {"type": "chunk", "done": 1, "language": "en", "rows": [segment]}

    def line(obj: dict[str, Any]) -> str:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    partial.write_text(line(header) + "\n" + line(good) + "\n[]\n", encoding="utf-8")
    transcript._partial_append(partial, header, {"type": "chunk", "done": 2, "language": "en", "rows": []})
    loaded = transcript._partial_load(partial, meta, 2)
    assert loaded[2] == 2 and [row["text"] for row in loaded[0]] == [" hi"]
