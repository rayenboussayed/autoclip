"""Integration checks: the CLI pipeline and the server job must agree on clips.json."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from autoclip.server.service import run_job
from autoclip.server.state import Job
from tests.support import install_stub_whisper, make_cfg


@pytest.mark.parametrize("mode", ["highlights", "ranges"])
def test_parity_process_video_vs_server_job(media: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setenv("AUTOCLIP_SERVER_INPROCESS", "1")
    install_stub_whisper(monkeypatch, media, {}, cuda=0)

    from autoclip.core.config import Config, parse
    from autoclip.core.hardware import pick_encoder
    from autoclip.core.pipeline import process_video

    # direct
    out_direct = tmp_path / "direct"
    out_direct.mkdir()
    extra = {"ranges": ["2-12"]} if mode == "ranges" else {}
    cfg_data = dataclasses.asdict(
        make_cfg(
            media, out_direct, clips={"mode": mode, "kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10, **extra}}
        )
    )
    cfg = parse(Config, cfg_data)
    cfg.input = [str(media.video)]
    cfg.output_dir = str(out_direct)
    encoder = pick_encoder(cfg.video, allow_gpu=False)
    process_video(media.video, cfg, None, encoder, name=media.video.stem)

    # server
    out_srv = tmp_path / "srv"
    out_srv.mkdir()
    extra = {"ranges": ["2-12"]} if mode == "ranges" else {}
    custom = dataclasses.asdict(
        make_cfg(
            media, out_srv, clips={"mode": mode, "kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10, **extra}}
        )
    )
    custom.pop("input", None)
    custom.pop("output_dir", None)
    job = Job(
        job_id="p",
        filename=media.video.name,
        input_path=media.video,
        output_dir=out_srv,
        config={"_custom": custom},
        work_dir=out_srv / ".work" / media.video.stem,
    )
    run_job(job)
    assert job.state.value == "completed"

    def rows(root: Path) -> list[dict[str, object]]:
        for cj in root.rglob("clips.json"):
            if ".work" in cj.relative_to(root).parts:
                continue
            parsed = json.loads(cj.read_text())
            if isinstance(parsed, list):
                return parsed
            return []
        return []

    a = rows(out_direct)
    b = rows(out_srv)
    # compare ignoring wall and file path names
    for r in a + b:
        r.pop("wall", None)
        r.pop("file", None)
    assert a == b
