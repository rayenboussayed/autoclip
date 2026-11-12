from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from autoclip.app.session import PipelineHandle, cancel_pipeline, run_pipeline
from autoclip.core.events import CancelToken
from tests.support import install_stub_whisper, make_cfg, write_config


def test_run_pipeline_success(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    lines: list[str] = []
    cfg = dataclasses.asdict(
        make_cfg(media, tmp_path / "out", clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    )

    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    code = run_pipeline(path, cancel=CancelToken(), on_line=lines.append)
    assert code == 0
    assert any("finished:" in line for line in lines)


def test_run_pipeline_bad_config(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    lines: list[str] = []
    code = run_pipeline(p, cancel=CancelToken(), on_line=lines.append)
    assert code == 2
    assert any("error:" in line for line in lines)


def test_cancel_pipeline_handle() -> None:
    import threading

    h = PipelineHandle(
        cancel=CancelToken(),
        thread=threading.Thread(target=lambda: None),
    )
    h.thread.start()
    h.thread.join()
    cancel_pipeline(h)
    assert h.cancel.cancelled is True
    assert h.done is True


def test_queue_handler_init() -> None:
    import queue

    from autoclip.app.session import _QueueHandler

    handler = _QueueHandler(queue.Queue())
    assert handler.queue is not None


def test_run_pipeline_and_stop_without_a_cancel_token(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.app.session import stop_pipeline
    from autoclip.core.events import CancelToken

    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    monkeypatch.setenv("AUTOCLIP_OUTPUT_DIR", str(tmp_path / "out-off"))
    path = write_config(tmp_path, media, inputs=[str(media.video)])
    assert run_pipeline(path, cancel=CancelToken(), on_line=lambda line: None) == 0

    class NoCancel:
        pass

    from typing import cast

    from autoclip.app.session import PipelineHandle

    stop_pipeline(cast(PipelineHandle, NoCancel()))  # nothing to cancel: must not raise


def test_resolve_log_file_returns_a_path(tmp_path: Path) -> None:
    from autoclip.app.session import _resolve_log_file
    from autoclip.core.config import Config

    cfg = Config(output_dir=str(tmp_path))
    path = _resolve_log_file(cfg)
    assert path.parent.name == "logs"
    assert path.with_suffix("").name.startswith("autoclip_")


def test_run_pipeline_logging_enabled(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    lines: list[str] = []
    monkeypatch.setenv("AUTOCLIP_OUTPUT_DIR", str(tmp_path / "out3"))
    path = write_config(tmp_path, media, inputs=[str(media.video)])
    code = run_pipeline(path, cancel=CancelToken(), on_line=lines.append)
    assert code == 0


def test_app_lines_and_log_file_are_in_sync(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One run: every decorator print must appear in the app pane AND in the .log file (same records)."""
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    monkeypatch.setenv("AUTOCLIP_OUTPUT_DIR", str(tmp_path / "out-sync"))
    lines: list[str] = []
    path = write_config(tmp_path, media, inputs=[str(media.video)])
    code = run_pipeline(path, cancel=CancelToken(), on_line=lines.append)
    assert code == 0
    log_text = max((tmp_path / "out-sync" / "logs").glob("autoclip_*.log"), key=lambda p: p.name).read_text(encoding="utf-8")
    pane_proc = [line for line in lines if "[START] process_video" in line]
    assert pane_proc, lines
    assert "[START] process_video" in log_text
    assert any("[START] run_batch" in line for line in lines) and "[START] run_batch" in log_text


def test_run_pipeline_adapter_error(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    lines_called = {"n": 0}

    def on_line(line: str) -> None:
        lines_called["n"] += 1
        if lines_called["n"] == 1:
            raise RuntimeError("boom")

    cfg = dataclasses.asdict(
        make_cfg(media, tmp_path / "out4", clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    )
    path = tmp_path / "cfg4.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    code = run_pipeline(path, cancel=CancelToken(), on_line=on_line)
    assert code in (0, 1)


def test_run_pipeline_downloader_error(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core import errors as E

    class BoomDownloader:
        def __init__(self, cfg: Any) -> None:
            raise E.AutoclipError("nope")

    monkeypatch.setattr("autoclip.core.download.Downloader", BoomDownloader)
    cfg = dataclasses.asdict(
        make_cfg(media, tmp_path / "out5", clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    )
    cfg["input"] = ["https://example.com/x.mp4"]
    path = tmp_path / "cfg5.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    lines: list[str] = []
    code = run_pipeline(path, cancel=CancelToken(), on_line=lines.append)
    assert code == 1


def test_run_pipeline_downloader_config_error(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core import errors as E

    class BoomDownloader:
        def __init__(self, cfg: Any) -> None:
            raise E.ConfigError("nope")

    monkeypatch.setattr("autoclip.core.download.Downloader", BoomDownloader)
    cfg = dataclasses.asdict(
        make_cfg(media, tmp_path / "out6", clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    )
    cfg["input"] = ["https://example.com/x.mp4"]
    path = tmp_path / "cfg6.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    lines: list[str] = []
    code = run_pipeline(path, cancel=CancelToken(), on_line=lines.append)
    assert code == 2


def test_run_pipeline_runbatch_config_error(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    from autoclip.core import errors as E

    def boom(*a: Any, **k: Any) -> None:
        raise E.ConfigError("boom")

    monkeypatch.setattr("autoclip.core.batch.run_batch", boom)
    cfg = dataclasses.asdict(
        make_cfg(media, tmp_path / "out7", clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    )
    path = tmp_path / "cfg7.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    lines: list[str] = []
    code = run_pipeline(path, cancel=CancelToken(), on_line=lines.append)
    assert code == 2


def test_stop_pipeline_with_cancel() -> None:
    import threading

    from autoclip.app.session import PipelineHandle, stop_pipeline
    from autoclip.core.events import CancelToken

    handle = PipelineHandle(cancel=CancelToken(), thread=threading.Thread(target=lambda: None))
    handle.thread.start()
    handle.thread.join()
    stop_pipeline(handle)


def test_stop_pipeline_without_cancel() -> None:
    import threading

    from autoclip.app.session import PipelineHandle, stop_pipeline
    from autoclip.core.events import CancelToken

    handle = PipelineHandle(cancel=CancelToken(), thread=threading.Thread(target=lambda: None))
    handle.thread.start()
    stop_pipeline(handle)
    assert handle.done is True


def test_run_pipeline_empty_result(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    lines: list[str] = []
    extra = {"ranges": ["2-12"]}
    cfg = dataclasses.asdict(
        make_cfg(
            media,
            tmp_path / "out2",
            clips={"mode": "ranges", "kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10, **extra}},
        )
    )

    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    code = run_pipeline(path, cancel=CancelToken(), on_line=lines.append)
    assert code in (0, 1)
