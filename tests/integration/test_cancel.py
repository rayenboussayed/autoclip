import dataclasses
import threading
from pathlib import Path
from typing import Any

import pytest

from autoclip.server.service import run_job
from autoclip.server.state import Job
from tests.support import install_stub_whisper, make_cfg


def test_cancel_marks_state_cancelled(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOCLIP_SERVER_INPROCESS", "1")
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    custom = dataclasses.asdict(
        make_cfg(media, tmp_path / "out", clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    )
    custom.pop("input", None)
    custom.pop("output_dir", None)
    out = tmp_path / "cancelout"
    out.mkdir()
    ev = threading.Event()
    ev.set()
    job = Job(
        job_id="cancel",
        filename=media.video.name,
        input_path=media.video,
        output_dir=out,
        config={"_custom": custom},
        work_dir=out / ".work" / media.video.stem,
    )
    run_job(job, cancel_event=ev)
    assert job.state.value == "cancelled"
