"""Integration checks: the CLI pipeline and the server job must agree on clips.json."""

import json
from pathlib import Path

import pytest

from autoclip.core.config import Config, FileConfig, JsonValue, RuntimeOverrides, parse
from autoclip.core.hardware import pick_encoder
from autoclip.core.pipeline import process_video
from autoclip.server.service import run_job
from autoclip.server.state import Job
from tests.support import Media, install_stub_whisper


@pytest.mark.parametrize("mode", ["highlights", "ranges"])
def test_parity_process_video_vs_server_job(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setenv("AUTOCLIP_SERVER_INPROCESS", "1")
    install_stub_whisper(monkeypatch, media, {}, cuda=0)
    monkeypatch.setattr("autoclip.core.ai.llm_available", lambda *a, **k: False)

    # Both entry points must derive their runtime configuration from the same nine-key file config;
    # shrink what is derived (one cheap clip, no LLM) so the two runs stay fast and deterministic.
    original = FileConfig.to_config

    def fast_to_config(self: FileConfig, overrides: RuntimeOverrides | None = None) -> Config:
        cfg = original(self, overrides)
        cfg.quality = "lowest"
        cfg.clips.short.count = 1
        return cfg

    monkeypatch.setattr(FileConfig, "to_config", fast_to_config)

    clips: dict[str, JsonValue] = {"preset": "shorts", "scenes": "best"}
    if mode == "ranges":
        clips["ranges"] = ["2-12"]
    file_data: dict[str, JsonValue] = {"clips": clips}

    # direct pipeline run
    out_direct = tmp_path / "direct"
    out_direct.mkdir()
    file_cfg = parse(FileConfig, file_data)
    cfg = file_cfg.to_config(RuntimeOverrides(input=[str(media.video)], output_dir=str(out_direct)))
    encoder = pick_encoder(cfg.video, cfg.quality, allow_gpu=False)
    process_video(media.video, cfg, None, encoder, name=media.video.stem)

    # the same file config uploaded to a server job
    out_srv = tmp_path / "srv"
    out_srv.mkdir()
    job = Job(
        job_id="p",
        filename=media.video.name,
        input_path=media.video,
        output_dir=out_srv,
        config={"_custom": file_data},
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
    assert a
    assert a == b
