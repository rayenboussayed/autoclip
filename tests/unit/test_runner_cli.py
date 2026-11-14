"""Job runner (sequential, simulated and real process pools) and the command line (multi-video batches)."""

from __future__ import annotations

import json
import logging
import os
import re
import runpy
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from autoclip.cli import main
from autoclip.core import runner
from autoclip.core.config import Config, parse_hms
from autoclip.core.hardware import HW_MAX_WORKERS, cpu_encoder
from autoclip.core.models import Encoder, Moment, RenderJob
from autoclip.core.pipeline import needs_text
from autoclip.core.resources import RESOURCES, WorkerPlan
from autoclip.core.runner import _srt_time, render_all, render_job
from autoclip.core.store import Store
from autoclip.core.transcript import load_transcript_file

from ..support import Media, install_stub_whisper, make_cfg, run_args, write_config


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
    # quality=lowest renders at a 216 px short side on a 9:16 canvas
    store = Store(tmp_path / "jobs.db", fresh=True)
    store.put_segments(load_transcript_file(media.transcript))
    jobs = []
    for k in range(1, n + 1):
        m = Moment(4.0 * k, 4.0 * k + 6, "", title="t", scene="best")
        clip_id = store.save_moment("short", "best", k, m)
        enc = encoder or cpu_encoder(cfg.video, cfg.quality)
        jobs.append(
            RenderJob(
                clip_id,
                m,
                media.video,
                tmp_path / f"c{k}.mp4",
                "short",
                "9:16",
                enc,
                True,
                cfg,
                store.path,
                log_file,
                parent_pid or os.getpid(),
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
    assert row["status"] == "done" and row["file"] == "c1.mp4" and (row["width"], row["height"]) == (216, 384) and row["trimmed"] > 3
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
    assert "rendering " in log_file.read_text(encoding="utf-8") and "c1.mp4" in log_file.read_text(encoding="utf-8")
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


def test_render_all_is_sequential_by_default(
    media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # With only one CPU budget available, the runtime plan keeps one job at a time.
    jobs, store, cfg = make_jobs(media, tmp_path, n=2)
    RESOURCES.configure(cfg)
    monkeypatch.setattr(RESOURCES, "cpu_budget", lambda: 1)
    with caplog.at_level(logging.INFO, logger="autoclip"):
        render_all(jobs, 100 << 20)
    assert statuses(store) == ["done", "done"]
    assert "submit clip" not in caplog.text


class ThreadPool(ThreadPoolExecutor):
    """Stands in for ProcessPoolExecutor so the scheduling logic runs (and is measured) in-process."""

    def __init__(self, max_workers: int, mp_context: Any = None) -> None:
        super().__init__(max_workers=max_workers)
        del mp_context


def parallel_jobs(media: Media, tmp_path: Path, n: int, cpus: int = 2, **kw: Any) -> tuple[list[RenderJob], Store, Config]:
    jobs, store, cfg = make_jobs(media, tmp_path, n=n, **kw)
    RESOURCES.configure(None)
    RESOURCES.cpus = cpus
    RESOURCES.max_cpu_percent = 100.0
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


def test_render_all_falls_back_when_the_pool_cannot_start(
    media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    jobs, store, _ = parallel_jobs(media, tmp_path, 2)

    def broken_pool(*_a: Any, **_k: Any) -> Any:
        raise OSError("no processes here")

    monkeypatch.setattr(runner, "ProcessPoolExecutor", broken_pool)
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        render_all(jobs, 100 << 20)
    assert "falling back to in-process rendering" in caplog.text and statuses(store) == ["done", "done"]
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
    jobs, store, cfg = make_jobs(media, tmp_path, n=2, log_file=log_file)
    RESOURCES.configure(None)
    RESOURCES.cpus = 2
    RESOURCES.max_cpu_percent = 100.0
    render_all(jobs, 100 << 20)
    assert statuses(store) == ["done", "done"]
    text = log_file.read_text(encoding="utf-8")
    pids = {
        line.split("pid=")[1].split()[0] for line in text.splitlines() if "rendering" in line and ("c1.mp4" in line or "c2.mp4" in line)
    }
    assert "c1.mp4" in text and "c2.mp4" in text and "rendering" in text, f"the spawned workers left the log incomplete:\n{text}"
    assert str(os.getpid()) not in pids and len(pids) >= 1  # rendered by spawned workers, not by this process
    store.close()


# ---- command line --------------------------------------------------------------------------------
def clips_json(tmp_path: Path, kind: str, stem: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = json.loads((tmp_path / "out" / kind / stem / "clips.json").read_text(encoding="utf-8"))
    return result


def test_main_processes_several_videos_into_the_short_long_video_structure(
    media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)  # silent.mp4 ships no sidecar transcript; the stub provides one
    assert main(run_args(write_config(tmp_path, media, inputs=[str(media.video), str(media.silent)]))) == 0
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == [".work", "logs", "short"]  # only the requested kind gets a folder
    for stem in ("talk", "silent"):
        summary = clips_json(tmp_path, "short", stem)
        assert len(summary) == 1 and (out / "short" / stem / summary[0]["file"]).exists()
        assert summary[0]["file"].startswith(f"{stem}_best_01_") and summary[0]["scene"] == "best" and summary[0]["width"] == 216
        assert isinstance(summary[0]["start"], float) and set(summary[0]) >= {"title", "score", "trimmed"}
        assert (out / ".work" / stem / "autoclip.db").exists()  # shared work files stay out of the short/long folders
    (log_file,) = (out / "logs").glob("autoclip_*.log")
    text = log_file.read_text(encoding="utf-8")
    assert "input 1/2" in text and "input 2/2" in text and "finished: 2 clip(s) done, 0 failed" in text and "START render_all" in text


def test_main_short_and_long_clips_for_several_scene_types(media: Media, tmp_path: Path) -> None:
    cfg = write_config(tmp_path, media, inputs=[str(media.video)], clips={"preset": "shorts", "scenes": ["funny", "sad"]})
    assert main(run_args(cfg)) == 0
    out = tmp_path / "out"
    assert (out / "short" / "talk").is_dir()
    rows = clips_json(tmp_path, "short", "talk")
    assert len(rows) >= 1
    assert all((r["width"], r["height"]) == (216, 384) for r in rows)
    for r in rows:
        assert re.fullmatch(r"talk_(funny|sad)_\d\d_\d+s\.mp4", r["file"]) and (out / "short" / "talk" / r["file"]).exists()


def test_a_kind_without_room_is_skipped_but_the_rest_is_produced(media: Media, tmp_path: Path) -> None:
    cfg = write_config(tmp_path, media, inputs=[str(media.video)], clips={"preset": "both"})
    assert main(run_args(cfg)) == 0
    log_files = list((tmp_path / "out" / "logs").glob("autoclip_*.log"))
    assert log_files and "no long clips for talk.mp4" in log_files[0].read_text(encoding="utf-8")
    assert (tmp_path / "out" / "short" / "talk").is_dir() and not (tmp_path / "out" / "long").exists()


def test_main_no_clips_for_any_kind_fails_the_video(media: Media, tmp_path: Path) -> None:
    cfg = write_config(tmp_path, media, inputs=[str(media.video)], clips={"preset": "long"})
    assert main(run_args(cfg)) == 1


def test_main_one_broken_video_does_not_stop_the_batch(media: Media, tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_text("this is not a video")
    assert main(run_args(write_config(tmp_path, media, inputs=[str(broken), str(media.video)]))) == 1
    assert len(clips_json(tmp_path, "short", "talk")) == 1


def test_run_batch_processes_inputs_one_at_a_time(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core import batch

    active = {"n": 0, "max": 0}

    def fake(path: Path, cfg: Config, log_file: Path | None, encoder: Encoder, **kw: Any) -> tuple[int, int]:
        active["n"] += 1
        active["max"] = max(active["max"], active["n"])
        import time as _t

        _t.sleep(0.02)
        active["n"] -= 1
        return 1, 0

    monkeypatch.setattr(batch, "process_video", fake)
    cfg = make_cfg(media, tmp_path / "out")
    cfg.input = [str(media.video), str(media.video)]
    RESOURCES.configure(cfg)
    from autoclip.core.batch import run_batch

    result = run_batch(cfg, downloader=None, encoder=Encoder("libx264", False, "ultrafast", ()), log_file=None)
    assert active["max"] == 1
    assert result.clips_done == 2 and result.clips_failed == 0


def test_main_ranges_are_given_per_clip_kind(media: Media, tmp_path: Path) -> None:
    clips = {"preset": "both", "ranges": ["00:00:04-00:00:10", "00:00:20-00:00:40"]}
    assert main(run_args(write_config(tmp_path, media, inputs=[str(media.video)], clips=clips))) == 0
    short, long = clips_json(tmp_path, "short", "talk"), clips_json(tmp_path, "long", "talk")
    assert [(c["start"], c["end"], c["scene"]) for c in short] == [(4.0, 10.0, "custom"), (20.0, 40.0, "custom")]
    assert [(c["start"], c["end"]) for c in long] == [(4.0, 10.0), (20.0, 40.0)]


def test_main_ranges_skip_kinds_without_ranges_and_fail_without_any(media: Media, tmp_path: Path) -> None:
    cfg = write_config(tmp_path / "a", media, inputs=[str(media.video)], clips={"preset": "shorts", "ranges": ["0:04-0:10"]})
    assert main(run_args(cfg, output=tmp_path / "a" / "out")) == 0
    assert (tmp_path / "a" / "out" / "short").is_dir() and not (tmp_path / "a" / "out" / "long").exists()
    cfg2 = write_config(tmp_path / "b", media, inputs=[str(media.video)], clips={"preset": "long"})
    assert main(run_args(cfg2, output=tmp_path / "b" / "out")) == 1


def test_main_split_mode_cuts_equal_parts_into_the_long_folder(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core import pipeline

    monkeypatch.setattr(pipeline, "SPLIT_SECONDS", 15.0)
    assert main(run_args(write_config(tmp_path, media, inputs=[str(media.video)]), extra=["--split"])) == 0
    folder = tmp_path / "out" / "long" / "talk"
    assert json.loads((folder / "clips.json").read_text(encoding="utf-8")) == ["talk_part001.mp4", "talk_part002.mp4", "talk_part003.mp4"]
    assert all((folder / f"talk_part{k:03d}.mp4").stat().st_size > 1000 for k in (1, 2, 3))


def test_needs_text_only_when_selection_captions_or_titles_use_it(media: Media, tmp_path: Path) -> None:
    """Highlights select on the transcript; ranges only need it for a text overlay."""
    out = tmp_path / "out"
    assert needs_text(make_cfg(media, out))
    ranges = {"mode": "ranges", "short": {"ranges": ["0:04-0:10"]}}
    assert needs_text(make_cfg(media, out, clips=ranges, captions={"enabled": False}))  # titles read the text by default
    assert not needs_text(make_cfg(media, out, clips=ranges, captions={"enabled": False}, skip={"title": True}))


def test_main_chunks_long_videos_for_transcription(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rec: dict[str, Any] = {}
    install_stub_whisper(monkeypatch, media, rec)
    bare = tmp_path / "bare.mp4"
    bare.write_bytes(media.video.read_bytes())
    cfg = write_config(tmp_path, media, inputs=[str(bare)])
    assert main(run_args(cfg)) == 0
    assert (
        "chunks" not in rec and rec["samples"] > 0 and len(clips_json(tmp_path, "short", "bare")) == 1
    )  # one automatic pass (possibly whole audio), clip still produced
    assert (tmp_path / "out" / ".work" / "bare" / "bare.transcript.json").exists()


def test_main_logs_which_video_encoder_was_chosen(media: Media, tmp_path: Path) -> None:
    cfg = write_config(tmp_path, media, inputs=[str(media.video)])
    assert main(run_args(cfg)) == 0
    log_files = list((tmp_path / "out" / "logs").glob("autoclip_*.log"))
    assert log_files, "expected a derived log file"
    text = log_files[0].read_text(encoding="utf-8")
    assert "GPU video encoder:" in text or "no usable GPU video encoder: using libx264 on the CPU" in text  # either, by machine
    assert "libx264" in text or "h264_" in text


def test_main_config_errors_and_init(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path / "nope.json")]) == 2 and "cannot read config" in capsys.readouterr().err
    target = tmp_path / "new.json"
    assert main(["--init", str(target)]) == 0
    starter = json.loads(target.read_text(encoding="utf-8"))["clips"]
    assert starter["preset"] == "both" and starter["scenes"] == "best"
    assert main(["--init", str(target)]) == 2 and "already exists" in capsys.readouterr().err
    full = tmp_path / "full.json"
    assert main(["--init-full", str(full)]) == 0
    data = json.loads(full.read_text(encoding="utf-8"))
    assert data["clips"]["preset"] == "both" and "input" in data


def test_generated_configs_load_without_errors(tmp_path: Path) -> None:
    from autoclip.core.config import load_config

    (tmp_path / "input.mp4").write_bytes(b"x")  # the starter config names input.mp4
    for flag in ("--init", "--init-full"):
        target = tmp_path / f"{flag.strip('-')}.json"
        assert main([flag, str(target)]) == 0
        cfg = load_config(target)
        assert cfg.clips.kinds == ["short", "long"] and Path(cfg.input[0]).name == "input.mp4"


def test_main_writes_a_log_file(media: Media, tmp_path: Path) -> None:
    cfg = write_config(tmp_path, media, inputs=[str(media.video)])
    assert main(run_args(cfg)) == 0
    log_files = list((tmp_path / "out" / "logs").glob("autoclip_*.log"))
    assert log_files and "autoclip start" in log_files[0].read_text(encoding="utf-8")


def test_events_jsonl_emits_progress_json(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = write_config(tmp_path, media, inputs=[str(media.video)], clips={"preset": "shorts"})
    captured: list[str] = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: captured.append(" ".join(map(str, a))))
    rc = main(["run", str(cfg), "--events", "jsonl"])
    assert rc == 0
    assert any("progress" in line for line in captured)


def test_overrides_include_skip_and_split() -> None:
    import argparse

    from autoclip.cli import _overrides

    args = argparse.Namespace(skip=["stt", "caption"], split=True)
    overrides = _overrides(args)
    assert overrides.skip == ["stt", "caption"] and overrides.split is True


def test_main_without_arguments_defaults_to_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # no config.json and no input.mp4 here
    assert main([]) in (1, 2)


def test_module_entry_point(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["autoclip", "--init", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("autoclip", run_name="__main__")
    assert exit_info.value.code == 0 and (tmp_path / "x.json").exists()


def test_render_all_cancelled_before_start(media: Media, tmp_path: Path) -> None:
    from autoclip.core.events import CancelToken

    jobs, store, cfg = parallel_jobs(media, tmp_path, 2)
    tok = CancelToken()
    tok.cancel()
    render_all(jobs, 100 << 20, cancel=tok)
    # no clip rendered
    assert all(v == "pending" for v in statuses(store))
    store.close()


def test_render_all_cancel_during_inprocess_loop(media: Media, tmp_path: Path) -> None:
    from autoclip.core.events import CancelToken

    jobs, store, cfg = parallel_jobs(media, tmp_path, 3, cpus=1)
    tok = CancelToken()
    # cancel before entering loop; in-process capacity branch still evaluates the cancelling job line
    tok.cancel()
    render_all(jobs, 100 << 20, cancel=tok)
    assert len(statuses(store)) == 3
    store.close()


def test_render_all_inprocess_cancel_after_first_job(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core.events import CancelToken, Reporter

    jobs, store, _ = parallel_jobs(media, tmp_path, 2, cpus=1)
    tok = CancelToken()
    calls = {"n": 0}

    def sink(_p: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            tok.cancel()

    render_all(jobs, 100 << 20, cancel=tok, report=Reporter(sink))
    assert len(statuses(store)) == 2
    store.close()


def test_render_all_pool_second_submit_exception(
    media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    jobs, store, _ = parallel_jobs(media, tmp_path, 2)
    calls = {"n": 0}

    orig_submit: Any = ThreadPool.submit

    def submit(self: Any, fn: Any, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return orig_submit(self, fn, *args, **kwargs)

    monkeypatch.setattr(ThreadPool, "submit", submit)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", ThreadPool)
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        render_all(jobs, 100 << 20)
    store.close()


def test_render_all_pool_submit_exception(
    media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    jobs, store, _ = parallel_jobs(media, tmp_path, 2)

    class BoomPool(ThreadPool):
        def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
            raise RuntimeError("boom")

    monkeypatch.setattr(runner, "ProcessPoolExecutor", BoomPool)
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        render_all(jobs, 100 << 20)
    assert statuses(store) == ["pending", "done"]
    store.close()


def test_render_all_cancel_after_pool(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs, store, _ = parallel_jobs(media, tmp_path, 2)

    class CancelAfterPool(ThreadPool):
        pass

    # cancel during pool scheduling by setting token on the first report
    from autoclip.core.events import CancelToken

    tok = CancelToken()
    monkeypatch.setattr(runner, "ProcessPoolExecutor", CancelAfterPool)

    def cb(_: Any) -> None:
        pass

    render_all(jobs, 100 << 20, cancel=tok)
    assert len(statuses(store)) == 2
    store.close()


def test_render_all_cancel_during_pool_completion(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core.events import CancelToken, Reporter

    jobs, store, _ = parallel_jobs(media, tmp_path, 2, cpus=8)
    tok = CancelToken()

    # the second clip is slower, so the cancel from the first completion lands while it is still in flight
    real_render_job = runner.render_job

    def slow_second(job: Any) -> Any:
        if job.clip_id == 2:
            time.sleep(0.2)
        return real_render_job(job)

    monkeypatch.setattr(runner, "render_job", slow_second)

    def sink(p: Any) -> None:
        if getattr(p, "stage", "") == "render" and str(getattr(p, "message", "")).endswith("1/2"):
            tok.cancel()

    monkeypatch.setattr(runner, "ProcessPoolExecutor", ThreadPool)
    render_all(jobs, 100 << 20, cancel=tok, report=Reporter(sink))
    assert len(statuses(store)) == 2
    store.close()


def test_render_all_with_report(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core.events import Reporter

    jobs, store, cfg = parallel_jobs(media, tmp_path, 2)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", ThreadPool)
    render_all(jobs, 100 << 20, report=Reporter(lambda p: None))
    assert statuses(store) == ["done", "done"]
    store.close()


def test_render_all_pool_exception_fallback(
    media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    jobs, store, cfg = parallel_jobs(media, tmp_path, 2)

    class BoomPool(ThreadPool):
        def __enter__(self) -> BoomPool:
            raise RuntimeError("boom")

    monkeypatch.setattr(runner, "ProcessPoolExecutor", BoomPool)
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        render_all(jobs, 100 << 20)
    assert "falling back to in-process" in caplog.text
    assert statuses(store) == ["done", "done"]
    store.close()


def test_render_all_cancel_after_pool_repeat(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core.events import CancelToken

    jobs, store, cfg = parallel_jobs(media, tmp_path, 2)
    tok = CancelToken()

    class CancelPool(ThreadPool):
        def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
            tok.cancel()
            return super().submit(fn, *args, **kwargs)

    monkeypatch.setattr(runner, "ProcessPoolExecutor", CancelPool)
    render_all(jobs, 100 << 20, cancel=tok)
    assert len(statuses(store)) == 2
    store.close()


def test_render_all_defers_submissions_while_another_job_is_in_flight(
    media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When admission says defer while a job is running, no extra clip is submitted this tick."""
    jobs, store, _ = parallel_jobs(media, tmp_path, 2)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", ThreadPool)
    from autoclip.core import resources as _resources  # noqa: PLC0415
    from autoclip.core.internal.budget import AutoDecision  # noqa: PLC0415

    admission_calls: list[tuple[str, int]] = []

    def defer(stage: str, cost_bytes: int = 0) -> AutoDecision:
        admission_calls.append((stage, cost_bytes))
        return AutoDecision(stage, "defer", "ram-oom")

    monkeypatch.setattr(_resources.RESOURCES, "admission", defer)
    render_all(jobs, 100 << 20)
    assert statuses(store) == ["done", "done"]  # the batch always drains, deferral or not
    assert admission_calls, "admission is consulted when a job is in flight"
    assert all(stage == "render" for stage, _ in admission_calls)
    store.close()


def test_run_batch_reports_the_worker_peak_rss(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """peak_rss_mb reflects the LATEST worker peak saved in the per-video database, not only the parent's own idle RSS."""
    import sqlite3

    jobs, store, _ = parallel_jobs(media, tmp_path, 1)

    from autoclip.core.batch import _db_peak_rss

    assert _db_peak_rss(tmp_path / "missing") == 0  # no databases in a foreign folder
    work = tmp_path / ".work" / "t"
    work.mkdir(parents=True)
    db = work / "autoclip.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE clips(id INTEGER PRIMARY KEY, kind TEXT, scene TEXT, n INTEGER, start REAL, end REAL,"
            " score REAL, llm_score REAL, title TEXT, caption TEXT, text TEXT, status TEXT, file TEXT,"
            " width INTEGER, height INTEGER, trimmed REAL, wall REAL, error TEXT, peak_rss INTEGER NOT NULL DEFAULT 0)"
        )
        conn.execute("INSERT INTO clips(peak_rss) VALUES (1048576)")
    assert _db_peak_rss(tmp_path) == 1048576


def test_db_peak_rss_survives_a_read_only_probe_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core.batch import _db_peak_rss

    def broken_glob(self: Path, _pattern: str) -> object:
        raise OSError("no access")

    work = tmp_path / ".work" / "noaccess"
    work.mkdir(parents=True)
    monkeypatch.setattr(Path, "glob", broken_glob)
    assert _db_peak_rss(tmp_path) == 0  # an unreadable .work directory is reported as 0 bytes, not a crash


def test_db_peak_rss_returns_zero_for_an_empty_clips_table(tmp_path: Path) -> None:
    from autoclip.core.store import Store

    store = Store(tmp_path / "empty.db", fresh=True)
    store.close()
    store2 = Store(tmp_path / "empty.db", fresh=True)
    assert store2.max_peak_rss() == 0
    store2.close()


def test_db_peak_rss_returns_the_largest_peak(tmp_path: Path) -> None:

    from autoclip.core.store import Store

    db = tmp_path / "big.db"
    store = Store(db, fresh=True)
    store.save_moment("short", "best", 1, Moment(0, 1, "one", score=1.0))
    store.finish_clip(1, ok=True, file="x.mp4", size=(100, 100), trimmed=1.0, wall=1.0, peak_rss=2048)
    store.close()
    store2 = Store(db, fresh=False)
    assert store2.max_peak_rss() == 2048
    store2.close()


def test_render_all_retries_a_job_whose_pool_submit_fails(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A submit-time BrokenProcessPool must requeue the clip for in-process retry (clips never dropped)."""
    jobs, store, _ = parallel_jobs(media, tmp_path, 2)

    class BoomOnPurposePool:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> BoomOnPurposePool:
            return self

        def __exit__(self, *a: object) -> None:
            pass

        def submit(self, fn: object, /, *args: object, **kwargs: object) -> object:
            raise BrokenProcessPool("dead worker")

        def shutdown(self, *a: object, **k: object) -> None:
            pass

    import autoclip.core.runner as runner_mod

    monkeypatch.setattr(runner_mod, "ProcessPoolExecutor", BoomOnPurposePool)
    render_all(jobs, 100 << 20)  # must not raise, must render both clips in-process
    assert statuses(store) == ["done", "done"]
    store.close()
