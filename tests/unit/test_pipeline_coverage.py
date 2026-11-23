from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autoclip.core.cutmap import CutMap, MeasuredTimeline
from autoclip.core.media import probe
from autoclip.core.optimizer import PreparedSource
from autoclip.core.pipeline import process_video
from tests.support import install_stub_whisper, make_cfg


def test_process_video_uses_the_optimized_source(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    import shutil

    out = tmp_path / "out"
    out.mkdir()
    cfg = make_cfg(media, out, clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    cfg.captions.enabled = False
    cfg.skip.title = True

    destination = out / ".work" / media.video.stem / f"optimized_{media.video.stem}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(media.video, destination)
    monkeypatch.setattr(
        "autoclip.core.pipeline.prepare_source",
        lambda src, cfg, info, work, **k: PreparedSource(
            destination,
            probe(destination),
            CutMap.identity(info.duration),
            MeasuredTimeline.from_cutmap(CutMap.identity(info.duration)),
            __import__("numpy").zeros(1, dtype=np.float32),
            False,
        ),
    )

    rc = process_video(
        media.video, cfg, None, __import__("autoclip.core.hardware", fromlist=["cpu_encoder"]).cpu_encoder(cfg.video, cfg.quality)
    )
    assert rc[0] >= 0


def test_process_video_optimizer_exception_is_not_swallowed(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    out = tmp_path / "out2"
    out.mkdir()
    cfg = make_cfg(media, out, clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    cfg.captions.enabled = False
    cfg.skip.title = True

    def boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("autoclip.core.pipeline.prepare_source", boom)
    with pytest.raises(RuntimeError, match="boom"):
        process_video(
            media.video, cfg, None, __import__("autoclip.core.hardware", fromlist=["cpu_encoder"]).cpu_encoder(cfg.video, cfg.quality)
        )


def test_process_video_optimizer_identity_returns_the_original(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    out = tmp_path / "out4"
    out.mkdir()
    cfg = make_cfg(media, out, clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    cfg.captions.enabled = False
    cfg.skip.title = True
    monkeypatch.setattr(
        "autoclip.core.pipeline.prepare_source",
        lambda src, cfg, info, work, **k: PreparedSource(
            src,
            info,
            CutMap.identity(info.duration),
            MeasuredTimeline.from_cutmap(CutMap.identity(info.duration)),
            __import__("numpy").zeros(1, dtype=np.float32),
            False,
        ),
    )
    from autoclip.core.hardware import cpu_encoder as ce

    rc = process_video(media.video, cfg, None, ce(cfg.video, cfg.quality))
    assert rc[0] >= 0


def test_process_video_writes_ai_caption_and_sd_thumbnail(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    from autoclip.core import ai as core_ai
    from autoclip.core.hardware import cpu_encoder as ce

    monkeypatch.setattr(core_ai, "llm_available", lambda *a, **k: True)
    monkeypatch.setattr(core_ai, "generate_title", lambda *a, **k: "AI title")
    monkeypatch.setattr(core_ai, "generate_caption", lambda *a, **k: "AI caption")
    monkeypatch.setattr(core_ai, "sd_available", lambda: True)
    monkeypatch.setattr(core_ai, "generate_sd_thumbnail", lambda src, dst, *a, **k: dst.write_bytes(b"x") or True)
    out = tmp_path / "out-sd"
    out.mkdir()
    cfg = make_cfg(
        media,
        out,
        clips={
            "mode": "ranges",
            "kinds": ["short"],
            "short": {"count": 1, "min_seconds": 5, "max_seconds": 10, "ranges": ["00:00:00-00:00:05"]},
        },
    )
    rc = process_video(media.video, cfg, None, ce(cfg.video, cfg.quality))
    assert rc[0] == 2
    shorts = out / "short" / "talk"
    assert any(p.suffix == ".txt" and p.read_text(encoding="utf-8") == "AI caption" for p in shorts.iterdir())


def test_process_video_sd_thumbnail_failure_is_logged(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    from autoclip.core import ai as core_ai
    from autoclip.core.hardware import cpu_encoder as ce

    def broken_sd() -> bool:
        raise RuntimeError("no diffusers")

    monkeypatch.setattr(core_ai, "sd_available", broken_sd)
    out = tmp_path / "out-sd-fail"
    out.mkdir()
    cfg = make_cfg(
        media,
        out,
        clips={
            "mode": "ranges",
            "kinds": ["short"],
            "short": {"count": 1, "min_seconds": 5, "max_seconds": 10, "ranges": ["00:00:00-00:00:05"]},
        },
    )
    rc = process_video(media.video, cfg, None, ce(cfg.video, cfg.quality))
    assert rc[0] == 2  # the clip is still produced; only the SD pass failed


def test_process_video_sd_pass_skips_failed_clips(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    from autoclip.core import ai as core_ai
    from autoclip.core import pipeline as pipeline_mod
    from autoclip.core.hardware import cpu_encoder as ce
    from autoclip.core.store import Store

    def failing_render(jobs: list[Any], *a: Any, **k: Any) -> None:
        for job in jobs:
            store = Store(job.db_path)
            store.finish_clip(job.clip_id, ok=False, error="boom")
            store.close()

    monkeypatch.setattr(pipeline_mod, "render_all", failing_render)
    monkeypatch.setattr(core_ai, "sd_available", lambda: True)
    monkeypatch.setattr(core_ai, "generate_sd_thumbnail", lambda *a, **k: True)
    out = tmp_path / "out-sd-skip"
    out.mkdir()
    cfg = make_cfg(
        media,
        out,
        clips={
            "mode": "ranges",
            "kinds": ["short"],
            "short": {"count": 1, "min_seconds": 5, "max_seconds": 10, "ranges": ["00:00:00-00:00:05"]},
        },
    )
    rc = process_video(media.video, cfg, None, ce(cfg.video, cfg.quality))
    assert rc[1] == 2


def test_ranges_moments_accepts_explicit_fillers(media: Any, tmp_path: Path) -> None:
    from autoclip.core.config import Config, KindCfg
    from autoclip.core.config import parse as parse_cfg
    from autoclip.core.media import probe
    from autoclip.core.pipeline import ranges_moments
    from autoclip.core.store import Store

    cfg = parse_cfg(Config, {"output_dir": str(tmp_path)})
    store = Store(tmp_path / "ranges.db", fresh=True)
    try:
        moments = ranges_moments(store, KindCfg(ranges=["0:01-0:03"]), probe(media.video), cfg, False, fillers=frozenset({"um"}))
    finally:
        store.close()
    assert moments and moments[0].start == 1.0


def test_process_video_sd_pass_skips_a_non_video_output(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    from autoclip.core import ai as core_ai
    from autoclip.core import pipeline as pipeline_mod
    from autoclip.core.hardware import cpu_encoder as ce
    from autoclip.core.store import Store

    def fake_render(jobs: list[Any], *a: Any, **k: Any) -> None:
        for job in jobs:
            store = Store(job.db_path)
            store.finish_clip(job.clip_id, ok=True, file="missing.mp4")  # not a readable video
            store.close()

    monkeypatch.setattr(pipeline_mod, "render_all", fake_render)
    monkeypatch.setattr(core_ai, "sd_available", lambda: True)
    monkeypatch.setattr(core_ai, "generate_sd_thumbnail", lambda *a, **k: True)
    out = tmp_path / "out-sd-nonvideo"
    out.mkdir()
    cfg = make_cfg(
        media,
        out,
        clips={
            "mode": "ranges",
            "kinds": ["short"],
            "short": {"count": 1, "min_seconds": 5, "max_seconds": 10, "ranges": ["00:00:00-00:00:05"]},
        },
    )
    rc = process_video(media.video, cfg, None, ce(cfg.video, cfg.quality))
    assert rc[0] == 2  # the SD pass skipped the unreadable frame without failing the clip


def test_process_video_skips_the_thumbnail_pass(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    from autoclip.core import ai as core_ai
    from autoclip.core.hardware import cpu_encoder as ce

    calls: list[int] = []

    def sd_available() -> bool:
        calls.append(1)
        return True

    monkeypatch.setattr(core_ai, "sd_available", sd_available)
    out = tmp_path / "out-no-thumb"
    out.mkdir()
    cfg = make_cfg(
        media,
        out,
        clips={
            "mode": "ranges",
            "kinds": ["short"],
            "short": {"count": 1, "min_seconds": 5, "max_seconds": 10, "ranges": ["00:00:00-00:00:05"]},
        },
    )
    cfg.skip.thumbnail = True
    rc = process_video(media.video, cfg, None, ce(cfg.video, cfg.quality))
    assert rc[0] == 2 and not calls


def test_process_video_ai_failure_keeps_the_heuristic_title(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    from autoclip.core import ai as core_ai
    from autoclip.core.hardware import cpu_encoder as ce

    monkeypatch.setattr(core_ai, "llm_available", lambda *a, **k: True)
    monkeypatch.setattr(core_ai, "generate_title", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(core_ai, "generate_caption", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = tmp_path / "out-ai-fail"
    out.mkdir()
    cfg = make_cfg(
        media,
        out,
        clips={
            "mode": "ranges",
            "kinds": ["short"],
            "short": {"count": 1, "min_seconds": 5, "max_seconds": 10, "ranges": ["00:00:00-00:00:05"]},
        },
    )
    rc = process_video(media.video, cfg, None, ce(cfg.video, cfg.quality))
    assert rc[0] == 2


def test_offline_run_never_calls_ai(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    from autoclip.core import ai as core_ai
    from autoclip.core.hardware import cpu_encoder as ce

    monkeypatch.setattr(core_ai, "llm_available", lambda *a, **k: pytest.fail("an offline run must not probe the LLM"))
    monkeypatch.setattr(core_ai, "sd_available", lambda: pytest.fail("an offline run must not use Stable Diffusion"))
    out = tmp_path / "out-offline"
    out.mkdir()
    cfg = make_cfg(
        media,
        out,
        clips={
            "mode": "ranges",
            "kinds": ["short"],
            "short": {"count": 1, "min_seconds": 5, "max_seconds": 10, "ranges": ["00:00:00-00:00:05"]},
        },
    )
    cfg.skip.llm = True
    rc = process_video(media.video, cfg, None, ce(cfg.video, cfg.quality))
    assert rc[0] == 2


def test_process_video_reports_a_failed_clip(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    from autoclip.core import pipeline as pipeline_mod
    from autoclip.core.hardware import cpu_encoder as ce
    from autoclip.core.store import Store

    def failing_render(jobs: list[Any], *a: Any, **k: Any) -> None:
        for job in jobs:
            store = Store(job.db_path)
            store.finish_clip(job.clip_id, ok=False, error="boom")
            store.close()

    monkeypatch.setattr(pipeline_mod, "render_all", failing_render)
    out = tmp_path / "out-failed"
    out.mkdir()
    cfg = make_cfg(
        media,
        out,
        clips={
            "mode": "ranges",
            "kinds": ["short"],
            "short": {"count": 1, "min_seconds": 5, "max_seconds": 10, "ranges": ["00:00:00-00:00:05"]},
        },
    )
    rc = process_video(media.video, cfg, None, ce(cfg.video, cfg.quality))
    assert rc[1] == 2  # the short and the always-on raw clip both failed


def test_process_video_no_transcript_needed(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    out = tmp_path / "out3"
    out.mkdir()
    cfg = make_cfg(
        media,
        out,
        clips={
            "mode": "ranges",
            "kinds": ["short"],
            "short": {"count": 1, "min_seconds": 5, "max_seconds": 10, "ranges": ["00:00:00-00:00:05"]},
        },
    )
    cfg.captions.enabled = False
    cfg.skip.title = True
    from autoclip.core.pipeline import needs_text as _needs_text

    assert _needs_text(cfg) is False
    from autoclip.core.hardware import cpu_encoder as ce

    rc = process_video(media.video, cfg, None, ce(cfg.video, cfg.quality))
    assert rc[0] >= 0


def test_process_video_cancel_writes_the_partial_index(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancelled run keeps the clips that finished: clips.json holds exactly those rows."""
    import json
    import sqlite3

    import autoclip.core.pipeline as pipeline_mod
    from autoclip.core.events import Cancelled
    from autoclip.core.hardware import cpu_encoder as ce

    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    out = tmp_path / "out-cancel"
    out.mkdir()
    cfg = make_cfg(
        media,
        out,
        clips={
            "mode": "ranges",
            "kinds": ["short"],
            "short": {"count": 1, "min_seconds": 5, "max_seconds": 10, "ranges": ["00:00:00-00:00:05"]},
        },
    )

    def fake_render_all(jobs: list[Any], needs: Any, *, report: Any = None, cancel: Any = None) -> None:
        with sqlite3.connect(jobs[0].db_path) as conn:
            conn.execute(
                "UPDATE clips SET status='done', file=?, width=320, height=180, trimmed=5.0, wall=1.0 WHERE id=?",
                ("kept_clip.mp4", jobs[0].clip_id),
            )
        raise Cancelled("user stopped the run")

    monkeypatch.setattr(pipeline_mod, "render_all", fake_render_all)
    with pytest.raises(Cancelled):
        process_video(media.video, cfg, None, ce(cfg.video, cfg.quality))
    clips_json = out / "short" / "talk" / "clips.json"
    assert clips_json.is_file()
    rows = json.loads(clips_json.read_text(encoding="utf-8"))
    assert [r["file"] for r in rows] == ["kept_clip.mp4"]


def test_cancel_during_an_unreported_render_pass_still_writes_the_partial_index(
    media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancel observed only by the final report must not lose clips.json (render_all does not raise)."""
    import sqlite3

    import autoclip.core.pipeline as pipeline_mod
    from autoclip.core.events import Cancelled, CancelToken, Reporter
    from autoclip.core.hardware import cpu_encoder as ce

    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    out = tmp_path / "out-cancel-late"
    out.mkdir()
    cfg = make_cfg(
        media,
        out,
        clips={
            "mode": "ranges",
            "kinds": ["short"],
            "short": {"count": 1, "min_seconds": 5, "max_seconds": 10, "ranges": ["00:00:00-00:00:05"]},
        },
    )
    token = CancelToken()

    def fake_render_all(jobs: list[Any], needs: Any, *, report: Any = None, cancel: Any = None) -> None:
        if report is None:  # the short/long pass is unreported
            with sqlite3.connect(jobs[0].db_path) as conn:
                conn.execute(
                    "UPDATE clips SET status='done', file=?, width=320, height=180, trimmed=5.0, wall=1.0 WHERE id=?",
                    ("kept_clip.mp4", jobs[0].clip_id),
                )
        cancel.cancel()  # only the final reporter sees the cancellation
        if report is not None:
            report("render", 1.0, "clip 1/1")

    monkeypatch.setattr(pipeline_mod, "render_all", fake_render_all)
    with pytest.raises(Cancelled):
        process_video(media.video, cfg, None, ce(cfg.video, cfg.quality), report=Reporter(cancel=token), cancel=token)
    rows = json.loads((out / "short" / "talk" / "clips.json").read_text(encoding="utf-8"))
    assert [r["file"] for r in rows] == ["kept_clip.mp4"]


def test_ranges_inside_a_removed_span_are_skipped(media: Any, tmp_path: Path) -> None:
    from autoclip.core.cutmap import CutMap
    from autoclip.core.pipeline import ranges_moments
    from autoclip.core.store import Store

    store = Store(tmp_path / "t.db")
    cfg = make_cfg(media, tmp_path, None, clips={"short": {"ranges": ["00:00:02-00:00:04"], "count": 1}})
    info = probe(media.video)
    cut = CutMap.from_intervals(info.duration, [(10.0, info.duration)])
    assert ranges_moments(store, cfg.clips.short, info, cfg, False, cutmap=cut) == []
    store.close()


def test_raw_moment_spans_the_measured_source(media: Any, tmp_path: Path) -> None:
    from autoclip.core.pipeline import raw_moment
    from autoclip.core.store import Store

    info = probe(media.video)
    cut = CutMap.from_intervals(info.duration, [(0.0, 3.0), (4.0, info.duration)])
    timeline = MeasuredTimeline.from_durations(cut.intervals, (3.3, info.duration - 1.0 + 0.4))
    store = Store(tmp_path / "raw.db", fresh=True)
    try:
        m = raw_moment(store, cut, timeline, False)
        assert (m.start, m.end) == (0.0, timeline.last)
        assert (m.source_start, m.source_end) == (0.0, cut.duration)
        assert m.scene == "raw" and m.text == ""
    finally:
        store.close()


def test_export_uncaptioned_raw_copies_the_source_with_a_full_transcript_srt(tmp_path: Path) -> None:
    from autoclip.core.models import Segment
    from autoclip.core.pipeline import _export_uncaptioned_raw
    from autoclip.core.store import Store

    source = tmp_path / "optimized.mp4"
    source.write_bytes(b"video")
    store = Store(tmp_path / "raw.db", fresh=True)
    try:
        store.put_segments([Segment(0.0, 1.25, "hello"), Segment(2.0, 3.5, "world")])
        dst = _export_uncaptioned_raw(tmp_path / "out", "talk", source, store)
    finally:
        store.close()
    assert dst == tmp_path / "out" / "raw" / "talk" / "talk_raw_01_0s_uncaptioned.mp4"
    assert dst.read_bytes() == b"video"
    assert dst.with_suffix(".srt").read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,250\nhello\n\n2\n00:00:02,000 --> 00:00:03,500\nworld\n"
    )


def test_process_video_always_exports_both_raw_variants(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core import pipeline as pipeline_mod
    from autoclip.core.hardware import cpu_encoder as ce
    from autoclip.core.store import Store

    install_stub_whisper(monkeypatch, media, {}, cuda=0)

    def fake_render(jobs: list[Any], *a: Any, **k: Any) -> None:
        for job in jobs:
            job.dst.write_bytes(b"video")
            store = Store(job.db_path)
            store.finish_clip(job.clip_id, ok=True, file=job.dst.name)
            store.close()

    monkeypatch.setattr(pipeline_mod, "render_all", fake_render)
    out = tmp_path / "always-raw"
    cfg = make_cfg(media, out, clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    rc = process_video(media.video, cfg, None, ce(cfg.video, cfg.quality))
    assert rc == (2, 0)
    raw = out / "raw" / media.video.stem
    stem = media.video.stem
    assert (raw / f"{stem}_raw_01_0s.mp4").is_file()  # the captioned render
    uncaptioned = raw / f"{stem}_raw_01_0s_uncaptioned.mp4"
    assert uncaptioned.is_file()
    full = uncaptioned.with_suffix(".srt")
    text = full.read_text(encoding="utf-8")
    assert text.startswith("1\n00:00:") and text.count("\n\n") == 9  # the fixture has ten transcript segments
    assert len(list((out / "short" / stem).glob("*.mp4"))) == 1
    rows = json.loads((raw / "clips.json").read_text(encoding="utf-8"))
    assert [row["file"] for row in rows] == [f"{stem}_raw_01_0s.mp4"]  # the copy is a derivative, not a row


def test_process_video_raw_produces_one_full_length_job(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    from autoclip.core import pipeline as pipeline_mod
    from autoclip.core.hardware import cpu_encoder

    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    out = tmp_path / "raw-out"
    cfg = make_cfg(media, out, clips={"kinds": ["raw"]})
    cfg.captions.enabled = False
    cfg.skip.title = True
    destination = out / ".work" / media.video.stem / f"optimized_{media.video.stem}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(media.video, destination)
    cut = CutMap.identity(probe(media.video).duration)
    monkeypatch.setattr(
        "autoclip.core.pipeline.prepare_source",
        lambda src, cfg, info, work, **k: PreparedSource(
            destination, probe(destination), cut, MeasuredTimeline.from_cutmap(cut), np.zeros(1, dtype=np.float32), False
        ),
    )
    from autoclip.core.store import Store

    batches: list[list[Any]] = []

    def capture(jobs: list[Any], *a: Any, **k: Any) -> None:
        batches.append(jobs)
        for job in jobs:
            store = Store(job.db_path)
            store.finish_clip(job.clip_id, ok=True, file=job.dst.name, size=(320, 180), trimmed=job.moment.end)
            store.close()

    monkeypatch.setattr(pipeline_mod, "render_all", capture)
    process_video(media.video, cfg, None, cpu_encoder(cfg.video, cfg.quality))
    assert len(batches) == 1 and len(batches[0]) == 1
    job = batches[0][0]
    assert job.kind == "raw"
    assert (job.moment.start, job.moment.end) == (0.0, probe(destination).duration)
    assert (job.moment.source_start, job.moment.source_end) == (0.0, cut.duration)
    assert job.dst == out / "raw" / media.video.stem / f"{media.video.stem}_raw_01_0s.mp4"
    assert not (out / "short").exists() and not (out / "long").exists()


def test_process_video_mixed_kinds_render_raw_last(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    from autoclip.core import pipeline as pipeline_mod
    from autoclip.core.hardware import cpu_encoder
    from autoclip.core.store import Store

    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    out = tmp_path / "mixed-out"
    cfg = make_cfg(
        media,
        out,
        clips={"kinds": ["short", "raw"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}},
    )
    cfg.captions.enabled = False
    cfg.skip.title = True
    destination = out / ".work" / media.video.stem / f"optimized_{media.video.stem}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(media.video, destination)
    cut = CutMap.identity(probe(media.video).duration)
    monkeypatch.setattr(
        "autoclip.core.pipeline.prepare_source",
        lambda src, cfg, info, work, **k: PreparedSource(
            destination, probe(destination), cut, MeasuredTimeline.from_cutmap(cut), np.zeros(1, dtype=np.float32), False
        ),
    )
    batches: list[list[Any]] = []

    def capture(jobs: list[Any], *a: Any, **k: Any) -> None:
        batches.append(jobs)
        for job in jobs:
            store = Store(job.db_path)
            store.finish_clip(job.clip_id, ok=True, file=job.dst.name, size=(320, 180), trimmed=job.moment.end)
            store.close()

    monkeypatch.setattr(pipeline_mod, "render_all", capture)
    process_video(media.video, cfg, None, cpu_encoder(cfg.video, cfg.quality))
    assert len(batches) == 2
    assert batches[0] and all(job.kind == "short" for job in batches[0])
    assert [job.kind for job in batches[1]] == ["raw"]


def test_process_video_persists_both_timelines_in_clips_json(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json as _json

    from autoclip.core.hardware import cpu_encoder

    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    monkeypatch.setattr("autoclip.core.optimizer.UNSILENCE_MIN_TOTAL_SECONDS", 0.0)
    out = tmp_path / "out"
    cfg = make_cfg(media, out, source=media.video, clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    cfg.captions.enabled = False
    cfg.skip.title = True
    process_video(media.video, cfg, None, cpu_encoder(cfg.video, cfg.quality))
    rows = list(_json.loads((out / "short" / media.video.stem / "clips.json").read_text(encoding="utf-8")))
    assert rows and all("source_start" in row and "source_end" in row for row in rows)
    assert rows[0]["source_end"] >= rows[0]["end"]
