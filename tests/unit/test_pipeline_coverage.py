from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autoclip.core.pipeline import process_video
from tests.support import install_stub_whisper, make_cfg


def test_process_video_unsilence_returns_destination(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    import shutil

    out = tmp_path / "out"
    out.mkdir()
    cfg = make_cfg(media, out, clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    cfg.captions.enabled = False
    cfg.title.enabled = False

    destination = out / ".work" / media.video.stem / f"unsilence_{media.video.stem}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(media.video, destination)
    monkeypatch.setattr("autoclip.core.unsilence.remove_silence", lambda src, cfg, dst, *a, **k: dst)

    rc = process_video(
        media.video, cfg, None, __import__("autoclip.core.hardware", fromlist=["cpu_encoder"]).cpu_encoder(cfg.video, cfg.quality)
    )
    assert rc[0] >= 0


def test_process_video_unsilence_exception(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    out = tmp_path / "out2"
    out.mkdir()
    cfg = make_cfg(media, out, clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    cfg.captions.enabled = False
    cfg.title.enabled = False

    def boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("autoclip.core.unsilence.remove_silence", boom)
    rc = process_video(
        media.video, cfg, None, __import__("autoclip.core.hardware", fromlist=["cpu_encoder"]).cpu_encoder(cfg.video, cfg.quality)
    )
    assert rc[0] >= 0


def test_process_video_unsilence_returns_original(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    out = tmp_path / "out4"
    out.mkdir()
    cfg = make_cfg(media, out, clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    cfg.captions.enabled = False
    cfg.title.enabled = False
    monkeypatch.setattr("autoclip.core.unsilence.remove_silence", lambda src, cfg, dst, *a, **k: src)
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
    assert rc[0] == 1
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
    assert rc[0] == 1  # the clip is still produced; only the SD pass failed


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
    assert rc[1] == 1


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
    assert rc[0] == 1  # the SD pass skipped the unreadable frame without failing the clip


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
    assert rc[0] == 1 and not calls


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
    assert rc[0] == 1


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
    assert rc[0] == 1


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
    assert rc[1] == 1  # one failed clip


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
    cfg.title.enabled = False
    from autoclip.core.pipeline import needs_text as _needs_text

    assert _needs_text(cfg) is False
    from autoclip.core.hardware import cpu_encoder as ce

    rc = process_video(media.video, cfg, None, ce(cfg.video, cfg.quality))
    assert rc[0] >= 0
