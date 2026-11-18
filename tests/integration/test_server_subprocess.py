"""The server re-enters `python -m autoclip --events jsonl` as a fresh subprocess.

Every other pipeline integration test sets ``AUTOCLIP_SERVER_INPROCESS=1`` and therefore never
examines the subprocess path: the NDJSON parser in ``run_job``, the ``--output-dir`` override,
the child's own environment and the ``result`` failure accounting in the worker all stayed
unexercised. This test forces the real subprocess and proves a job reaches ``completed`` with
nothing but a hermetic faster-whisper stub on the child's import path.
"""

import dataclasses
from pathlib import Path

import pytest

from autoclip.server.schemas import JobState
from autoclip.server.service import run_job
from autoclip.server.state import Job
from tests.support import Media, bare_media, make_cfg


def _stub_packages(stubs: Path, marker: Path) -> None:
    """Write a byte-compile-free faster-whisper stub that records it was imported.

    ``PYTHONPATH`` outranks site-packages, so the child sees these files instead of the real
    (network-downloading) libraries. No model is ever fetched.
    """
    fw = stubs / "faster_whisper"
    fw.mkdir(parents=True)
    (fw / "__init__.py").write_text(
        "from types import SimpleNamespace\n"
        f"open({str(marker)!r}, 'w').write('stub')\n"
        "class WhisperModel:\n"
        "    def __init__(self, *a, **kw):\n"
        "        pass\n"
        "    def transcribe(self, audio, **options):\n"
        "        seg = SimpleNamespace(start=0.5, end=2.5, text=' chunk',\n"
        "            words=[SimpleNamespace(start=0.5, end=1.5, word=' chunk')])\n"
        "        info = SimpleNamespace(language='en', language_probability=0.99, duration=40.0)\n"
        "        return iter([seg]), info\n",
        encoding="utf-8",
    )
    (fw / "utils.py").write_text("def download_model(*a, **kw):\n    return None\n", encoding="utf-8")
    (stubs / "ctranslate2.py").write_text("def get_cuda_device_count():\n    return 0\n", encoding="utf-8")


def test_run_job_over_the_subprocess_reentry(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTOCLIP_SERVER_INPROCESS", raising=False)
    monkeypatch.setenv("AUTOCLIP_SKIP", "llm")  # keep the child away from any Ollama probe
    stubs = tmp_path / "stubs"
    marker = tmp_path / "stub-was-used"
    _stub_packages(stubs, marker)
    monkeypatch.setenv("PYTHONPATH", str(stubs))
    video = bare_media(media, tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    custom = dataclasses.asdict(
        make_cfg(media, out, source=video, clips={"kinds": ["short"], "short": {"count": 1, "min_seconds": 5, "max_seconds": 10}})
    )
    custom.pop("input", None)
    custom.pop("output_dir", None)
    job = Job(
        job_id="ndjson-e2e",
        filename=video.name,
        input_path=video,
        output_dir=out,
        config={"_custom": custom},
        work_dir=out / ".work" / video.stem,
    )
    run_job(job)

    assert job.state is JobState.completed, f"job failed: {job.error}"
    assert job.error is None
    assert marker.exists(), "the child never imported the stub: it might have used the real model"
    clips = [p for p in out.rglob("*.mp4") if ".work" not in p.relative_to(out).parts]
    assert clips, "no rendered clip was collected from the subprocess run"
    assert any(p.name == "clips.json" for p in out.rglob("clips.json"))
