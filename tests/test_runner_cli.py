"""Job runner (sequential, simulated and real process pools) and the command line (multi-video batches)."""

from __future__ import annotations

import json
import logging
import os
import re
import runpy
import sys
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from autoclip import runner
from autoclip.cli import main, parse_hms
from autoclip.config import Config
from autoclip.hardware import HW_MAX_WORKERS, cpu_encoder
from autoclip.models import Encoder, Moment, RenderJob
from autoclip.resources import RESOURCES, WorkerPlan
from autoclip.runner import _srt_time, render_all, render_job
from autoclip.store import Store
from autoclip.transcript import load_transcript_file

from .conftest import Media, make_cfg, merge
from .test_transcript import install_stub_whisper


@pytest.fixture(autouse=True)
def _reset_monitor() -> Iterator[None]:
    yield
    RESOURCES.configure(Config())


def make_jobs(
    media: Media,
    tmp_path: Path,
    n: int = 2,
    log_file: Path | None = None,
    parent_pid: int | None = None,
    encoder: Encoder | None = None,
    **sections: dict[str, Any],
) -> tuple[list[RenderJob], Store, Config]:
    cfg = make_cfg(media, tmp_path, None, **sections)
    cfg.video.short_side = 96
    store = Store(tmp_path / "jobs.db", fresh=True)
    store.put_segments(load_transcript_file(media.transcript))
    jobs = []
    for k in range(1, n + 1):
        m = Moment(4.0 * k, 4.0 * k + 6, "", title="t", scene="best")
        clip_id = store.save_moment("short", "best", k, m)
        enc = encoder or cpu_encoder(cfg.video)
        jobs.append(
            RenderJob(
                clip_id, m, media.video, tmp_path / f"c{k}.mp4", "9:16", enc, True, cfg, store.path, log_file, parent_pid or os.getpid()
            )
        )
    return jobs, store, cfg


def statuses(store: Store) -> list[str]:
    return [r["status"] for r in store.clips()]


def test_srt_time_and_hms() -> None:
    assert _srt_time(3723.5) == "01:02:03,500" and _srt_time(-1) == "00:00:00,000" and _srt_time(59.999) == "00:00:59,999"
    assert parse_hms("01:02:03.5") == 3723.5 and parse_hms("90") == 90


def test_render_job_success_writes_outputs_and_database(media: Media, tmp_path: Path) -> None:
    (job,), store, _ = make_jobs(media, tmp_path, n=1)
    assert render_job(job) is True
    row = store.clips()[0]
    assert row["status"] == "done" and row["file"] == "c1.mp4" and (row["width"], row["height"]) == (96, 170) and row["trimmed"] > 3
    assert (tmp_path / "c1.mp4").exists() and (tmp_path / "c1.srt").read_text(encoding="utf-8").startswith("1\n00:00:")
    store.close()


def test_render_job_failure_is_recorded_not_raised(media: Media, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (job,), store, _ = make_jobs(media, tmp_path, n=1)
    broken = replace(job, src=tmp_path / "missing.mp4")
    with caplog.at_level(logging.ERROR, logger="autoclip"):
        assert render_job(broken) is False
    row = store.clips()[0]
    assert row["status"] == "failed" and row["error"] and "render failed" in caplog.text
    store.close()


def test_render_job_in_a_fresh_worker_rebuilds_logging_and_monitor(media: Media, tmp_path: Path) -> None:
    log_file = tmp_path / "worker.log"
    (job,), store, _ = make_jobs(media, tmp_path, n=1, log_file=log_file, parent_pid=-1)  # pid differs: behaves like a spawned worker
    assert render_job(job) and store.clips()[0]["status"] == "done"
    assert "rendering c1.mp4" in log_file.read_text(encoding="utf-8")
    store.close()


def test_render_job_without_captions_writes_no_srt(media: Media, tmp_path: Path) -> None:
    (job,), store, _ = make_jobs(media, tmp_path, n=1, captions={"enabled": False})
    assert render_job(job) and not (tmp_path / "c1.srt").exists()
    store.close()


def test_render_all_sequential_and_empty(media: Media, tmp_path: Path) -> None:
    render_all([], 1 << 20)  # nothing to do
    jobs, store, cfg = make_jobs(media, tmp_path)
    RESOURCES.configure(cfg)
    render_all(jobs, 100 << 20)
    assert statuses(store) == ["done", "done"]
    store.close()


class ThreadPool(ThreadPoolExecutor):
    """Stands in for ProcessPoolExecutor so the scheduling logic runs (and is measured) in-process."""

    def __init__(self, max_workers: int, mp_context: Any = None) -> None:
        super().__init__(max_workers=max_workers)


def parallel_jobs(media: Media, tmp_path: Path, n: int, cpus: int = 2, **kw: Any) -> tuple[list[RenderJob], Store, Config]:
    jobs, store, cfg = make_jobs(media, tmp_path, n=n, resources={"cpus": cpus, "max_workers": cpus}, **kw)
    RESOURCES.configure(cfg)
    return jobs, store, cfg


def test_render_all_parallel_scheduler(
    media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    jobs, store, _ = parallel_jobs(media, tmp_path, 3)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", ThreadPool)
    with caplog.at_level(logging.INFO, logger="autoclip"):
        render_all(jobs, 100 << 20)
    assert statuses(store) == ["done"] * 3 and caplog.text.count("submit clip") == 3
    store.close()


def test_gpu_encoding_never_uses_more_workers_than_a_gpu_allows(
    media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    gpu = Encoder("libx264", True, "medium", ("-preset", "ultrafast", "-crf", "30"))  # software stands in for the GPU
    jobs, store, _ = parallel_jobs(media, tmp_path, 5, cpus=8, encoder=gpu)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", ThreadPool)
    with caplog.at_level(logging.INFO, logger="autoclip"):
        render_all(jobs, 10 << 20)
    assert f"up to {HW_MAX_WORKERS} worker(s)" in caplog.text and statuses(store) == ["done"] * 5
    assert all(int(m.group(1)) <= HW_MAX_WORKERS for m in re.finditer(r"allowed=(\d+)", caplog.text))
    store.close()


def test_the_cpu_encoder_is_not_capped_like_a_gpu(
    media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    jobs, store, _ = parallel_jobs(media, tmp_path, 4, cpus=8)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", ThreadPool)
    with caplog.at_level(logging.INFO, logger="autoclip"):
        render_all(jobs, 10 << 20)
    assert "up to 4 worker(s)" in caplog.text
    store.close()


def test_render_all_retries_a_killed_worker_alone(
    media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    jobs, store, _ = parallel_jobs(media, tmp_path, 2)
    calls: list[int] = []

    class OomPool(ThreadPool):
        def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
            calls.append(args[0].clip_id)
            if len(calls) == 1:  # the OS kills the first worker
                dead: Future[Any] = Future()
                dead.set_exception(BrokenProcessPool("killed"))
                return dead
            return super().submit(fn, *args, **kwargs)

    monkeypatch.setattr(runner, "ProcessPoolExecutor", OomPool)
    monkeypatch.setattr(RESOURCES, "plan_workers", lambda _b, _n, _limit=0: WorkerPlan(1, 1))  # memory is tight: one at a time
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        render_all(jobs, 100 << 20)
    assert "retrying it alone" in caplog.text and statuses(store) == ["done", "done"]  # job 1 retried, job 2 was still pending
    store.close()


def test_render_all_with_real_spawned_workers(media: Media, tmp_path: Path) -> None:
    log_file = tmp_path / "pool.log"
    jobs, store, cfg = make_jobs(media, tmp_path, n=2, log_file=log_file, resources={"cpus": 2, "max_workers": 2})
    RESOURCES.configure(cfg)
    render_all(jobs, 100 << 20)
    assert statuses(store) == ["done", "done"]
    text = log_file.read_text(encoding="utf-8")
    pids = {line.split("pid=")[1].split()[0] for line in text.splitlines() if "rendering c" in line}
    assert "rendering c1.mp4" in text and "rendering c2.mp4" in text
    assert str(os.getpid()) not in pids and len(pids) >= 1  # rendered by spawned workers, not by this process
    store.close()


# ---- command line --------------------------------------------------------------------------------
BASE_CONFIG: dict[str, Any] = {
    "clips": {
        "kinds": ["short"],
        "short": {"count": 1, "min_seconds": 6, "max_seconds": 12},
        "long": {"count": 1, "min_seconds": 12, "max_seconds": 30},
    },
    "video": {"short_side": 96, "preset": "ultrafast", "gpu": "off"},
    "audio": {"denoise": "none"},
    "logging": {"console_level": "ERROR"},
    "resources": {"max_workers": 1},
}


def write_config(tmp_path: Path, media: Media, **overrides: Any) -> Path:
    (tmp_path / "subs").mkdir(parents=True, exist_ok=True)
    for stem in ("talk", "silent", "broken"):
        (tmp_path / "subs" / f"{stem}.json").write_text(media.transcript.read_text(encoding="utf-8"), encoding="utf-8")
    data = merge(
        {**BASE_CONFIG, "input": [str(media.video), str(media.silent)], "output_dir": str(tmp_path / "out"),
         "transcript": {"mode": "file", "file": str(tmp_path / "subs" / "{stem}.json")}},
        overrides,
    )  # fmt: skip
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def clips_json(tmp_path: Path, kind: str, stem: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = json.loads((tmp_path / "out" / kind / stem / "clips.json").read_text(encoding="utf-8"))
    return result


def test_main_processes_several_videos_into_the_short_long_video_structure(media: Media, tmp_path: Path) -> None:
    assert main([str(write_config(tmp_path, media))]) == 0
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == [".work", "logs", "short"]  # only the requested kind gets a folder
    for stem in ("talk", "silent"):
        summary = clips_json(tmp_path, "short", stem)
        assert len(summary) == 1 and (out / "short" / stem / summary[0]["file"]).exists()
        assert summary[0]["file"].startswith(f"{stem}_best_01_") and summary[0]["scene"] == "best" and summary[0]["width"] == 96
        assert isinstance(summary[0]["start"], float) and set(summary[0]) >= {"title", "score", "trimmed"}
        assert (out / ".work" / stem / "autoclip.db").exists()  # shared work files stay out of the short/long folders
    (log_file,) = (out / "logs").glob("autoclip_*.log")
    text = log_file.read_text(encoding="utf-8")
    assert "video 1/2" in text and "video 2/2" in text and "finished: 2 clip(s) done, 0 failed" in text and "START render_all" in text


def test_main_short_and_long_clips_for_several_scene_types(media: Media, tmp_path: Path) -> None:
    cfg = write_config(
        tmp_path,
        media,
        input=[str(media.video)],
        clips={"kinds": ["short", "long"], "scenes": ["funny", "sad"], "long": {"aspect": "16:9"}},
    )
    assert main([str(cfg)]) == 0
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir() if not p.name.startswith(".") and p.name != "logs") == ["long", "short"]
    for kind, size in (("short", (96, 170)), ("long", (170, 96))):
        rows = clips_json(tmp_path, kind, "talk")
        assert {r["scene"] for r in rows} == {"funny", "sad"} or len(rows) >= 1  # a short video may leave little room for the 2nd scene
        assert all((r["width"], r["height"]) == size for r in rows)
        for r in rows:
            assert re.fullmatch(r"talk_(funny|sad)_\d\d_\d+s\.mp4", r["file"]) and (out / kind / "talk" / r["file"]).exists()
        assert max(r["trimmed"] for r in rows) <= (13 if kind == "short" else 31)
    longest = max(r["trimmed"] for r in clips_json(tmp_path, "long", "talk"))
    assert longest > max(r["trimmed"] for r in clips_json(tmp_path, "short", "talk"))  # long-form clips really are longer


def test_a_kind_without_room_is_skipped_but_the_rest_is_produced(media: Media, tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    clips = {"kinds": ["short", "long"], "long": {"min_seconds": 100, "max_seconds": 200}}
    cfg = write_config(tmp_path, media, input=[str(media.video)], clips=clips, logging={"file": str(log_file)})
    assert main([str(cfg)]) == 0
    assert "no long clips for talk.mp4" in log_file.read_text(encoding="utf-8")
    assert (tmp_path / "out" / "short" / "talk").is_dir() and not (tmp_path / "out" / "long").exists()


def test_main_no_clips_for_any_kind_fails_the_video(media: Media, tmp_path: Path) -> None:
    cfg = write_config(tmp_path, media, input=[str(media.video)], clips={"short": {"min_seconds": 90, "max_seconds": 120}})
    assert main([str(cfg)]) == 1


def test_main_one_broken_video_does_not_stop_the_batch(media: Media, tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_text("this is not a video")
    assert main([str(write_config(tmp_path, media, input=[str(broken), str(media.video)]))]) == 1
    assert len(clips_json(tmp_path, "short", "talk")) == 1


def test_main_reports_clips_that_failed_to_render(media: Media, tmp_path: Path) -> None:
    missing = str(tmp_path / "missing.png")
    cfg = write_config(tmp_path, media, input=[str(media.video)], video={"background": {"type": "image", "image": missing}})
    assert main([str(cfg)]) == 1
    assert clips_json(tmp_path, "short", "talk") == []


def test_main_ranges_are_given_per_clip_kind(media: Media, tmp_path: Path) -> None:
    clips = {
        "mode": "ranges",
        "kinds": ["short", "long"],
        "short": {"ranges": ["00:00:04-00:00:10"]},
        "long": {"ranges": ["0:20-0:45"], "aspect": "original"},
    }
    assert main([str(write_config(tmp_path, media, input=[str(media.video)], clips=clips))]) == 0
    short, long = clips_json(tmp_path, "short", "talk"), clips_json(tmp_path, "long", "talk")
    assert [(c["start"], c["end"], c["scene"], c["file"]) for c in short] == [(4.0, 10.0, "custom", "talk_custom_01_4s.mp4")]
    assert [(c["start"], c["end"]) for c in long] == [(20.0, 40.0)]  # the range is clamped to the video's end
    assert (long[0]["width"], long[0]["height"]) == (320, 180)  # aspect "original" keeps the source shape


def test_main_ranges_skip_kinds_without_ranges_and_fail_without_any(media: Media, tmp_path: Path) -> None:
    only_short = {"mode": "ranges", "kinds": ["short", "long"], "short": {"ranges": ["0:04-0:10"]}}
    assert main([str(write_config(tmp_path / "a", media, input=[str(media.video)], clips=only_short))]) == 0
    assert (tmp_path / "a" / "out" / "short").is_dir() and not (tmp_path / "a" / "out" / "long").exists()
    assert main([str(write_config(tmp_path / "b", media, input=[str(media.video)], clips={"mode": "ranges"}))]) == 1


def test_main_split_mode_cuts_equal_parts_into_the_long_folder(media: Media, tmp_path: Path) -> None:
    assert main([str(write_config(tmp_path, media, input=[str(media.video)], clips={"mode": "split", "split_seconds": 15}))]) == 0
    folder = tmp_path / "out" / "long" / "talk"
    assert json.loads((folder / "clips.json").read_text(encoding="utf-8")) == ["talk_part001.mp4", "talk_part002.mp4", "talk_part003.mp4"]
    assert all((folder / f"talk_part{k:03d}.mp4").stat().st_size > 1000 for k in (1, 2, 3))


def test_main_chunks_long_videos_for_transcription(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec)
    cfg = write_config(tmp_path, media, input=[str(media.video)], transcript={"mode": "whisper", "file": None, "model": "tiny"},
                       processing={"chunk_threshold_minutes": 0.5, "chunk_minutes": 0.25})  # fmt: skip
    assert main([str(cfg)]) == 0
    assert len(rec["chunks"]) == 3 and len(clips_json(tmp_path, "short", "talk")) == 1  # three chunk calls, clip still produced
    assert (tmp_path / "out" / ".work" / "talk" / "talk.transcript.json").exists()


def test_main_logs_which_video_encoder_was_chosen(media: Media, tmp_path: Path) -> None:
    cfg = write_config(
        tmp_path,
        media,
        input=[str(media.video)],
        video={"gpu": "auto"},
        logging={"file": str(tmp_path / "gpu.log"), "console_level": "ERROR"},
    )
    assert main([str(cfg)]) == 0
    text = (tmp_path / "gpu.log").read_text(encoding="utf-8")
    assert "GPU video encoder:" in text or "no usable GPU video encoder: using libx264 on the CPU" in text  # either, by machine
    assert "libx264" in text or "h264_" in text


def test_main_config_errors_and_init(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path / "nope.json")]) == 2 and "cannot read config" in capsys.readouterr().err
    target = tmp_path / "new.json"
    assert main(["--init", str(target)]) == 0
    assert json.loads(target.read_text(encoding="utf-8"))["clips"]["scenes"] == ["best"]
    assert main(["--init", str(target)]) == 2 and "already exists" in capsys.readouterr().err
    full = tmp_path / "full.json"
    assert main(["--init-full", str(full)]) == 0
    data = json.loads(full.read_text(encoding="utf-8"))
    assert data["resources"]["memory_fraction"] == 0.75 and data["clips"]["long"]["aspect"] == "16:9" and data["video"]["gpu"] == "auto"
    assert data["processing"] == {"chunk_threshold_minutes": 0, "chunk_minutes": 5}
    assert data["transcript"]["model"] == "large-v3" and data["transcript"]["enhance"]["enabled"] is True


def test_generated_configs_load_without_errors(tmp_path: Path) -> None:
    from autoclip.config import load_config

    (tmp_path / "video.mp4").write_bytes(b"x")  # the starter config names video.mp4
    for flag in ("--init", "--init-full"):
        target = tmp_path / f"{flag.strip('-')}.json"
        assert main([flag, str(target)]) == 0
        cfg = load_config(target)
        assert cfg.clips.kinds == ["short", "long"] and Path(cfg.input[0]).name == "video.mp4"


def test_main_logging_can_be_disabled_and_directed(media: Media, tmp_path: Path) -> None:
    quiet = write_config(tmp_path, media, input=[str(media.video)], logging={"enabled": False})
    assert main([str(quiet)]) == 0 and not (tmp_path / "out" / "logs").exists()
    custom = write_config(tmp_path, media, input=[str(media.video)], logging={"file": str(tmp_path / "my.log"), "enabled": True})
    assert main([str(custom)]) == 0 and "autoclip start" in (tmp_path / "my.log").read_text(encoding="utf-8")


def test_module_entry_point(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["autoclip", "--init", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("autoclip", run_name="__main__")
    assert exit_info.value.code == 0 and (tmp_path / "x.json").exists()
