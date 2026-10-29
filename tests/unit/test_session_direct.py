from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from autoclip.app.session import PipelineHandle, cancel_pipeline, run_pipeline
from autoclip.core.events import CancelToken
from tests.support import install_stub_whisper, make_cfg


def test_run_pipeline_success(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    lines: list[str] = []
    cfg = dataclasses.asdict(
        make_cfg(media, tmp_path / "out", clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    )
    import json as _json

    path = tmp_path / "config.json"
    path.write_text(_json.dumps(cfg), encoding="utf-8")
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
    import json as _json

    path = tmp_path / "cfg.json"
    path.write_text(_json.dumps(cfg), encoding="utf-8")
    code = run_pipeline(path, cancel=CancelToken(), on_line=lines.append)
    assert code in (0, 1)
