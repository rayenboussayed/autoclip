"""Crash-safe silence-removal assembly from real FFmpeg segments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

import autoclip.core.optimizer as optimizer
from autoclip.core import denoise
from autoclip.core.cutmap import CutMap
from autoclip.core.denoise import fetch_model
from autoclip.core.errors import AutoclipError
from autoclip.core.media import probe
from autoclip.core.optimizer import prepare_source

from ..support import Media, make_cfg


def test_prepare_source_keeps_the_original_below_the_gate(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(media, tmp_path, None)
    info = probe(media.video)
    monkeypatch.setattr(optimizer, "UNSILENCE_MIN_TOTAL_SECONDS", 999.0)
    prepared = prepare_source(media.video, cfg, info, tmp_path / "work")
    assert prepared.path == media.video
    assert prepared.cutmap.intervals == ((0.0, info.duration),)
    assert not prepared.audio_ready


def test_prepare_source_encodes_the_kept_intervals(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(media, tmp_path, None)
    info = probe(media.video)
    monkeypatch.setattr(optimizer, "UNSILENCE_MIN_TOTAL_SECONDS", 0.0)
    prepared = prepare_source(media.video, cfg, info, tmp_path / "work")
    assert prepared.audio_ready
    assert prepared.path.is_file() and prepared.path.name == f"optimized_{media.video.stem}.mp4"
    assert prepared.cutmap.total_kept < info.duration
    assert abs(probe(prepared.path).duration - prepared.cutmap.total_kept) < 2.0
    manifest = json.loads((tmp_path / "work" / "optimizer.json").read_text(encoding="utf-8"))
    assert manifest["assembled"] is True


def test_prepare_source_keeps_a_silent_video_as_identity(media: Media, tmp_path: Path) -> None:
    cfg = make_cfg(media, tmp_path, None)
    info = probe(media.silent)
    prepared = prepare_source(media.silent, cfg, info, tmp_path / "work")
    assert prepared.path == media.silent and prepared.cutmap.intervals == ((0.0, info.duration),)


def test_prepare_source_requires_the_file(media: Media, tmp_path: Path) -> None:
    cfg = make_cfg(media, tmp_path, None)
    with pytest.raises((AutoclipError, FileNotFoundError, OSError, RuntimeError)):
        prepare_source(tmp_path / "missing.mp4", cfg, probe(media.video), tmp_path / "work")


def test_assembly_resumes_from_the_last_committed_segment(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(media, tmp_path, None)
    info = probe(media.video)
    cut = CutMap.from_intervals(info.duration, [(0.0, 5.0), (10.0, info.duration)])
    calls: list[str] = []
    real_run = optimizer._run

    def flaky(command: list[str], *args: Any, **kwargs: Any) -> None:
        calls.append(command[-1])
        if len(calls) == 2:
            raise AutoclipError("simulated crash")
        real_run(command)

    monkeypatch.setattr(optimizer, "_run", flaky)
    with pytest.raises(AutoclipError, match="simulated crash"):
        optimizer.assemble_optimized(media.video, info, cfg, tmp_path / "work", cut)
    manifest = json.loads((tmp_path / "work" / "optimizer.json").read_text(encoding="utf-8"))
    assert manifest["chunks"][0]["committed"] is True

    calls.clear()

    def second_run(command: list[str], *args: Any, **kwargs: Any) -> Any:
        calls.append(command[-1])
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(optimizer, "_run", second_run)
    out = optimizer.assemble_optimized(media.video, info, cfg, tmp_path / "work", cut)
    assert out.is_file()
    assert len([call for call in calls if Path(call).suffix == ".mp4" and Path(call).parent.name == "optimizer"]) == 1
    assert calls[0].endswith(".part.mp4") and "000001" in Path(calls[0]).name


def test_segment_command_selects_direct_audio_video_and_silent_shapes(media: Media, tmp_path: Path) -> None:
    cfg = make_cfg(media, tmp_path, None)
    info = probe(media.video)
    silent = probe(media.silent)

    direct = optimizer._segment_command(media.video, None, (0.0, 1.0), info, cfg, tmp_path / "direct.mp4")
    assert "-map" in direct and "0:a:0" in direct and direct[-1] == str(tmp_path / "direct.mp4")
    enhanced = optimizer._segment_command(media.video, tmp_path / "audio.wav", (0.0, 1.0), info, cfg, tmp_path / "enhanced.mp4")
    assert "-i" in enhanced and "1:a:0" in enhanced
    silent_cmd = optimizer._segment_command(media.silent, None, (0.0, 1.0), silent, cfg, tmp_path / "silent.mp4")
    assert "-an" in silent_cmd and "0:a:0" not in silent_cmd


def test_run_raises_when_ffmpeg_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProcess:
        def poll(self) -> int:
            return 1

        def wait(self, timeout: float | None = None) -> int:
            return 1

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

    monkeypatch.setattr("autoclip.core.optimizer.subprocess.Popen", lambda *args, **kwargs: FakeProcess())
    with pytest.raises(AutoclipError, match="ffmpeg failed"):
        optimizer._run(["ffmpeg", "-version"])


def test_assembly_reprobes_a_bad_stream_copy_before_using_it(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(media, tmp_path, None)
    info = probe(media.video)
    cut = CutMap.from_intervals(info.duration, [(0.0, 5.0), (10.0, 15.0)])
    durations = iter([0.0, cut.total_kept])
    monkeypatch.setattr(optimizer, "_probe_duration", lambda path: next(durations))
    out = optimizer.assemble_optimized(media.video, info, cfg, tmp_path / "work", cut)
    assert out.is_file()


def test_assembly_reports_a_concat_failure_after_segmentation(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(media, tmp_path, None)
    info = probe(media.video)
    cut = CutMap.from_intervals(info.duration, [(0.0, 5.0)])
    real_run = optimizer._run

    def fake_run(command: list[str], *args: Any, **kwargs: Any) -> Any:
        if "concat" in command:
            return 1
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(optimizer, "_run", fake_run)
    with pytest.raises(AutoclipError, match="concatenate"):
        optimizer.assemble_optimized(media.video, info, cfg, tmp_path / "work", cut)
    assert not (tmp_path / "work" / "optimized_talk.mp4").exists()


def test_prepare_source_encodes_a_silent_video_when_the_scan_finds_speech(
    media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    cfg = make_cfg(media, tmp_path, None)
    info = probe(media.silent)

    class FakeScan:
        energy_db = np.zeros(1, dtype=np.float32)
        intervals = [(0.0, 2.0)]

    monkeypatch.setattr(optimizer, "scan_audio", lambda *args, **kwargs: FakeScan())
    monkeypatch.setattr(optimizer, "UNSILENCE_MIN_TOTAL_SECONDS", 0.0)
    prepared = prepare_source(media.silent, cfg, info, tmp_path / "work")
    assert prepared.path.is_file() and prepared.path.name == "optimized_silent.mp4"


def test_fetch_model_reverifies_a_planted_model_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def read(self, _n: int = -1) -> bytes:
            return b"the-model"

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    planted = tmp_path / "gtcrn_simple.onnx"
    planted.write_bytes(b"tampered")
    monkeypatch.setattr("autoclip.core.denoise.urllib.request.urlopen", lambda *args, **kwargs: Response())
    monkeypatch.setattr(denoise, "SHERPA_MODEL_SHA256", {"gtcrn_simple": hashlib.sha256(b"the-model").hexdigest()})
    assert fetch_model("gtcrn", tmp_path).read_bytes() == b"the-model"


def test_fetch_model_rejects_an_oversized_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def read(self, _n: int = -1) -> bytes:
            return b"x" * (denoise.MAX_MODEL_BYTES + 1)

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr("autoclip.core.denoise.urllib.request.urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(Exception, match="larger than"):
        fetch_model("gtcrn", tmp_path)


def test_cancel_terminates_an_ffmpeg_child_during_assembly(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core.events import Cancelled, CancelToken

    token = CancelToken()
    token.cancel()

    class FakeProc:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self) -> int | None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            pass

    proc = FakeProc()
    monkeypatch.setattr("autoclip.core.optimizer.subprocess.Popen", lambda *args, **kwargs: proc)
    with pytest.raises(Cancelled):
        optimizer._run(["ffmpeg", "-version"], cancel=token)
    assert proc.terminated


def test_reused_assembly_skips_the_re_encode(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(media, tmp_path, None)
    info = probe(media.video)
    monkeypatch.setattr(optimizer, "UNSILENCE_MIN_TOTAL_SECONDS", 0.0)
    first = prepare_source(media.video, cfg, info, tmp_path / "work")
    assert first.path.is_file()
    before = first.path.stat().st_mtime_ns
    second = prepare_source(media.video, cfg, info, tmp_path / "work")
    assert second.path == first.path
    assert second.path.stat().st_mtime_ns == before  # the optimized file is reused, not re-encoded


def test_source_mtime_change_invalidates_the_reuse(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(media, tmp_path, None)
    info = probe(media.video)
    monkeypatch.setattr(optimizer, "UNSILENCE_MIN_TOTAL_SECONDS", 0.0)
    first = prepare_source(media.video, cfg, info, tmp_path / "work")
    os.utime(media.video, (first.path.stat().st_atime_ns, first.path.stat().st_mtime_ns + 1_000_000))
    assert optimizer._load_reused_assembly(media.video, cfg, tmp_path / "work", info) is None


def test_corrupt_reuse_manifests_rebuild_the_assembly(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(media, tmp_path, None)
    info = probe(media.video)
    monkeypatch.setattr(optimizer, "UNSILENCE_MIN_TOTAL_SECONDS", 0.0)
    prepare_source(media.video, cfg, info, tmp_path / "work")
    payload = json.loads((tmp_path / "work" / "optimizer.json").read_text(encoding="utf-8"))

    payload["path"] = "another.mp4"
    (tmp_path / "work" / "optimizer.json").write_text(json.dumps(payload), encoding="utf-8")
    assert optimizer._load_reused_assembly(media.video, cfg, tmp_path / "work", info) is None

    payload["path"] = "optimized_talk.mp4"
    payload["intervals"] = [[0.0, 1.0]]
    (tmp_path / "work" / "optimizer.json").write_text(json.dumps(payload), encoding="utf-8")
    assert optimizer._load_reused_assembly(media.video, cfg, tmp_path / "work", info) is None

    (tmp_path / "work" / "optimizer.json").write_text("{broken", encoding="utf-8")
    assert optimizer._load_reused_assembly(media.video, cfg, tmp_path / "work", info) is None

    stale = {
        "schema": 1,
        "source_fingerprint": "nope",
        "cut_fingerprint": "y",
        "intervals": [[0.0, 1.0]],
        "path": "optimized_talk.mp4",
        "assembled": True,
    }
    (tmp_path / "work" / "optimizer.json").write_text(json.dumps(stale), encoding="utf-8")
    assert optimizer._load_reused_assembly(media.video, cfg, tmp_path / "work", info) is None


def test_a_corrupt_chunk_record_is_ignored_and_the_assembly_rebuilds(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(media, tmp_path, None)
    info = probe(media.video)
    work = tmp_path / "work"
    work.mkdir()
    (work / "optimizer.json").write_text(
        json.dumps({"schema": 1, "source_fingerprint": "x", "cut_fingerprint": "y", "chunks": [{"index": "junk"}]}), encoding="utf-8"
    )
    cut = CutMap.from_intervals(info.duration, [(0.0, 5.0)])
    out = optimizer.assemble_optimized(media.video, info, cfg, work, cut)
    assert out.is_file()


def test_cancel_escalates_to_kill_when_the_child_ignores_terminate(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from subprocess import TimeoutExpired

    from autoclip.core.events import Cancelled, CancelToken

    token = CancelToken()
    token.cancel()
    waits = {"n": 0}

    class StubbornProc:
        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            pass

        def wait(self, timeout: float | None = None) -> int:
            waits["n"] += 1
            if waits["n"] == 1:
                raise TimeoutExpired("ffmpeg", timeout or 1.0)
            return 0

        def kill(self) -> None:
            waits["killed"] = 1

    monkeypatch.setattr("autoclip.core.optimizer.subprocess.Popen", lambda *args, **kwargs: StubbornProc())
    with pytest.raises(Cancelled):
        optimizer._run(["ffmpeg", "-version"], cancel=token)
    assert waits["killed"] == 1


def test_corrupt_intervals_are_rejected_by_the_reuse_check(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(media, tmp_path, None)
    info = probe(media.video)
    monkeypatch.setattr(optimizer, "UNSILENCE_MIN_TOTAL_SECONDS", 0.0)
    prepare_source(media.video, cfg, info, tmp_path / "work")
    manifest = json.loads((tmp_path / "work" / "optimizer.json").read_text(encoding="utf-8"))
    manifest["intervals"] = [[1]]
    (tmp_path / "work" / "optimizer.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert optimizer._load_reused_assembly(media.video, cfg, tmp_path / "work", info) is None


def test_junk_chunk_records_are_skipped_individually(media: Media, tmp_path: Path) -> None:
    cfg = make_cfg(media, tmp_path, None)
    info = probe(media.video)
    cut = CutMap.from_intervals(info.duration, [(0.0, 5.0)])
    work = tmp_path / "work"
    work.mkdir()
    manifest = {
        "schema": optimizer._SCHEMA,
        "source_fingerprint": optimizer._source_fingerprint(media.video, cfg),
        "cut_fingerprint": cut.fingerprint,
        "chunks": ["junk", {"noindex": 1}, {"index": "not-a-number"}],
    }
    (work / "optimizer.json").write_text(json.dumps(manifest), encoding="utf-8")
    out = optimizer.assemble_optimized(media.video, info, cfg, work, cut)
    assert out.is_file()
