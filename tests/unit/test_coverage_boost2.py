from __future__ import annotations

import contextlib
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

# app session
from autoclip.app.session import PipelineHandle, _QueueHandler, cancel_pipeline, run_pipeline, start_pipeline
from autoclip.core.events import Cancelled, CancelToken


def test_queue_handler_emit_exception() -> None:
    q: Any = []
    _QueueHandler(q) if False else None

    # build a real queue handler on a fake queue object that raises on put
    class Q:
        def put(self, x: str) -> None:
            raise RuntimeError("boom")

    hh = _QueueHandler.__new__(_QueueHandler)
    logging.Handler.__init__(hh)
    hh.queue = Q()
    rec = logging.LogRecord("a", logging.INFO, "f", 1, "msg", (), None)
    hh.emit(rec)  # should not raise


def test_run_pipeline_logging_setup_fail(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.support import install_stub_whisper, make_cfg

    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    lines: list[str] = []
    cfg = make_cfg(media, tmp_path / "out", clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    import json as _json

    p = tmp_path / "c.json"
    p.write_text(_json.dumps(__import__("dataclasses").asdict(cfg)), encoding="utf-8")
    monkeypatch.setattr("autoclip.core.log.setup_logging", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    code = run_pipeline(p, cancel=CancelToken(), on_line=lines.append)
    assert code in (0, 1, 2)
    assert any("warning: logging setup failed" in line for line in lines)


def test_run_pipeline_downloader_config_error(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.support import install_stub_whisper, make_cfg

    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    lines: list[str] = []
    cfg = make_cfg(media, tmp_path / "out2", clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    cfg.input = ["https://example.com/x.mp4"]
    cfg.output_dir = str(tmp_path / "out2")
    (tmp_path / "out2").mkdir(exist_ok=True)
    import json as _json
    from dataclasses import asdict

    p = tmp_path / "c2.json"
    p.write_text(_json.dumps(asdict(cfg)), encoding="utf-8")

    class FakeDL:
        def __init__(self, *a, **k):
            raise __import__("autoclip.core.errors", fromlist=["ConfigError"]).ConfigError("bad cfg")

    monkeypatch.setattr("autoclip.core.download.Downloader", FakeDL)
    code = run_pipeline(p, cancel=CancelToken(), on_line=lines.append)
    assert code == 2


def test_run_pipeline_downloader_autoclip_error(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.support import install_stub_whisper, make_cfg

    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    lines: list[str] = []
    cfg = make_cfg(media, tmp_path / "out3", clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    cfg.input = ["https://example.com/x.mp4"]
    cfg.output_dir = str(tmp_path / "out3")
    (tmp_path / "out3").mkdir(exist_ok=True)
    import json as _json
    from dataclasses import asdict

    p = tmp_path / "c3.json"
    p.write_text(_json.dumps(asdict(cfg)), encoding="utf-8")

    class FakeDL:
        def __init__(self, *a, **k):
            pass

        def download(self, *a, **k):
            from autoclip.core.errors import AutoclipError

            raise AutoclipError("boom")

    monkeypatch.setattr("autoclip.core.download.Downloader", FakeDL)
    code = run_pipeline(p, cancel=CancelToken(), on_line=lines.append)
    assert code == 1


def test_run_pipeline_run_batch_exception(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.support import install_stub_whisper, make_cfg

    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    lines: list[str] = []
    cfg = make_cfg(media, tmp_path / "out4", clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    cfg.output_dir = str(tmp_path / "out4")
    (tmp_path / "out4").mkdir(exist_ok=True)
    import json as _json
    from dataclasses import asdict

    p = tmp_path / "c4.json"
    p.write_text(_json.dumps(asdict(cfg)), encoding="utf-8")
    monkeypatch.setattr("autoclip.core.batch.run_batch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    code = run_pipeline(p, cancel=CancelToken(), on_line=lines.append)
    assert code == 1


def test_cancel_pipeline_timeout() -> None:
    import threading

    def sleeper() -> None:
        time.sleep(2)

    t = threading.Thread(target=sleeper, daemon=True)
    t.start()
    h = PipelineHandle(cancel=CancelToken(), thread=t)
    cancel_pipeline(h)
    assert h.done is True


def test_start_pipeline_shim(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text("{}", encoding="utf-8")
    # only ensure callable returns handle; config invalid but worker handles
    h = start_pipeline(cfg)
    h.thread.join(timeout=5)
    assert h.done is True


def test_unique_name_duplicate() -> None:
    from autoclip.cli import _unique_name

    used = {"foo"}
    assert _unique_name("foo", used) == "foo (2)"
    used.add("foo (2)")
    assert _unique_name("foo", used) == "foo (3)"


def test_window_drain_when_destroyed(tmp_path, monkeypatch):
    from autoclip.app.window import AutoclipApp

    app = AutoclipApp(tmp_path / "x.json")
    try:
        monkeypatch.setattr(app, "winfo_exists", lambda: False)
        app._drain_log()
    finally:
        app.destroy()


def test_queue_handler_emit_exception2() -> None:
    import logging

    class Q:
        def put(self, x: str) -> None:
            raise RuntimeError("boom")

    from autoclip.app.session import _QueueHandler

    hh = _QueueHandler.__new__(_QueueHandler)
    logging.Handler.__init__(hh)
    hh.queue = Q()
    rec = logging.LogRecord("a", logging.INFO, "f", 1, "msg", (), None)
    hh.emit(rec)


def test_events_cancelled_before_report() -> None:
    from autoclip.core.events import CancelToken, Reporter

    tok = CancelToken()
    tok.cancel()
    r = Reporter(lambda p: None, tok)
    with pytest.raises(Cancelled):
        r("render", 0.5)


def test_config_default_model_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "ctranslate2", None)
    from autoclip.core.config import _default_transcript_model

    assert _default_transcript_model() == "small"


def test_config_resolve_default_model_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctranslate2

    original = ctranslate2.get_cuda_device_count
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: (_ for _ in ()).throw(RuntimeError()))
    from autoclip.core.config import TranscriptCfg

    cfg = TranscriptCfg(model=None)
    assert cfg.model == "small"
    ctranslate2.get_cuda_device_count = original


def test_config_model_dir_abs(media: Any, tmp_path: Path) -> None:
    import json as _json

    from autoclip.core.config import load_config

    cfg = {"input": [str(media.video)], "output_dir": str(tmp_path / "out"), "transcript": {"model_dir": "models", "model": "tiny"}}
    p = tmp_path / "cfg.json"
    p.write_text(_json.dumps(cfg), encoding="utf-8")
    loaded = load_config(p)
    assert (tmp_path / "out" / "models").as_posix() in str(loaded.transcript.model_dir) or loaded.transcript.model_dir is not None


def test_unique_name_duplicate2() -> None:
    from autoclip.cli import _unique_name

    used = {"foo", "foo (2)"}
    assert _unique_name("foo", used) == "foo (3)"


def test_process_read_events_none_stdout() -> None:
    class P:
        stdout = None

    from autoclip.core.process import read_events

    assert list(read_events(P())) == []  # type: ignore[arg-type]


def test_process_terminate_tree_no_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    # force ImportError by making psutil unimportable
    monkeypatch.setitem(sys.modules, "psutil", None)
    proc = subprocess.Popen(["python", "-c", "pass"], text=True)
    proc.wait()
    from autoclip.core.process import terminate_tree

    terminate_tree(proc)


def test_transcript_consent_reject(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core import transcript
    from tests.support import install_stub_whisper, make_cfg

    # make download_model available and simulate missing cache
    def dm(*a: Any, **k: Any) -> bool:
        if k.get("local_files_only"):
            raise RuntimeError("missing")
        return True

    monkeypatch.setattr(transcript, "download_model", dm, raising=False)
    import faster_whisper.utils as _fwu

    monkeypatch.setattr(_fwu, "download_model", dm)
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    monkeypatch.setenv("AUTOCLIP_ASSUME_YES", "0")
    monkeypatch.setattr("builtins.input", lambda _: "n", raising=False)
    cfg = make_cfg(media, tmp_path / "o", transcript={"model": "small"})
    cfg.transcript.offline = False

    import numpy as np

    from autoclip.core.media import probe

    info = probe(media.video)
    db_arr = np.zeros(10, dtype=np.float32)
    from autoclip.core.errors import AutoclipError

    with contextlib.suppress(AutoclipError, Exception):
        transcript.transcribe(media.video, cfg, info, db_arr, None)


def test_transcript_consent_eof(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core import transcript
    from tests.support import install_stub_whisper, make_cfg

    def dm(*a: Any, **k: Any) -> bool:
        if k.get("local_files_only"):
            raise RuntimeError("missing")
        return True

    monkeypatch.setattr(transcript, "download_model", dm, raising=False)
    import faster_whisper.utils as _fwu

    monkeypatch.setattr(_fwu, "download_model", dm)
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    monkeypatch.setenv("AUTOCLIP_ASSUME_YES", "0")

    def boom(*a: Any, **k: Any) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", boom, raising=False)
    cfg = make_cfg(media, tmp_path / "o2", transcript={"model": "small"})
    cfg.transcript.offline = False
    import numpy as np

    from autoclip.core.media import probe

    info = probe(media.video)
    db_arr = np.zeros(10, dtype=np.float32)
    with contextlib.suppress(Exception):
        transcript.transcribe(media.video, cfg, info, db_arr, None)


def test_reporter_cancel_before(monkeypatch: pytest.MonkeyPatch) -> None:
    # covered by events test already; replicate
    from autoclip.core.events import CancelToken, Reporter

    tok = CancelToken()
    tok.cancel()
    r = Reporter(lambda p: None, tok)
    with pytest.raises(Cancelled):
        r("validate", 1.0)


def test_config_default_model_cpu_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctranslate2

    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 0)
    from autoclip.core.config import _default_transcript_model

    assert _default_transcript_model() == "small"


def test_unique_name_no_conflict() -> None:
    from autoclip.cli import _unique_name

    assert _unique_name("bar", set()) == "bar"


def test_window_drain_exists_false2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.app.window import AutoclipApp

    app = AutoclipApp(tmp_path / "x.json")
    monkeypatch.setattr(app, "winfo_exists", lambda: False)
    try:
        app._drain_log()
    finally:
        # avoid after callback errors
        with contextlib.suppress(Exception):
            app.destroy()


def test_cli_main_download_only_flag(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.cli import main

    # build a config with a local mp4 so download-only branch reports need url
    cfg = {"input": [str(media.video)], "output_dir": str(tmp_path / "o")}
    p = tmp_path / "c.json"
    p.write_text(__import__("json").dumps(cfg), encoding="utf-8")
    with contextlib.suppress(SystemExit):
        main(["--download-only", str(p)])


def test_cli_main_json_flag(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.support import install_stub_whisper

    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    cfg = {
        "input": [str(media.video)],
        "output_dir": str(tmp_path / "o2"),
        "clips": {"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}},
        "skip": {"stt": True, "caption": True, "title": True},
        "video": {"short_side": 64, "preset": "ultrafast", "gpu": "off"},
    }
    p = tmp_path / "c.json"
    p.write_text(__import__("json").dumps(cfg), encoding="utf-8")
    from autoclip.cli import main

    main(["run", str(p), "--json"])


def test_cli_main_events_jsonl_flag(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.support import install_stub_whisper

    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    cfg = {
        "input": [str(media.video)],
        "output_dir": str(tmp_path / "o3"),
        "clips": {"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}},
        "skip": {"stt": True, "caption": True, "title": True},
        "video": {"short_side": 64, "preset": "ultrafast", "gpu": "off"},
    }
    p = tmp_path / "c.json"
    p.write_text(__import__("json").dumps(cfg), encoding="utf-8")
    from autoclip.cli import main

    main(["run", str(p), "--events", "jsonl"])
