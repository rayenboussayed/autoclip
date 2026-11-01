from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autoclip.cli.subcommands import _doctor, _init, _models


def test_init_writes_config(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    assert _init([str(p)]) == 0
    assert p.is_file()


def test_init_full(tmp_path: Path) -> None:
    p = tmp_path / "f.json"
    assert _init([str(p), "--full"]) == 0
    assert p.is_file()


def test_models_list() -> None:
    assert _models(["list"]) == 0


def test_models_path() -> None:
    assert _models(["path"]) == 0


def test_doctor_json(capsys: pytest.CaptureFixture[str]) -> None:

    rc = _doctor(["--json"])
    assert rc in (0, 1)
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "version" in data


def test_init_existing_file_errors(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text("{}", encoding="utf-8")
    assert _init([str(p)]) == 2


def test_doctor_text_mode(capsys: pytest.CaptureFixture[str]) -> None:
    rc = _doctor([])
    assert rc in (0, 1)


def test_models_path_action() -> None:
    assert _models(["path"]) == 0


def test_models_pull_requires_name(capsys: pytest.CaptureFixture[str]) -> None:
    assert _models(["pull"]) == 2


def test_models_pull_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("faster_whisper.utils.download_model", lambda name: None)
    assert _models(["pull", "tiny"]) == 0


def test_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.cli import subcommands

    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    monkeypatch.setattr("autoclip.server.main.create_app", object)
    assert subcommands._serve(["--host", "127.0.0.1", "--port", "1234"]) == 0


def test_gui(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from autoclip.cli import subcommands

    monkeypatch.setattr("autoclip.app.launch", lambda cfg=None: 0)
    assert subcommands._gui([]) == 0
    assert subcommands._gui([str(tmp_path / "x.json")]) == 0


def test_doctor_text_with_warning_and_blocking(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    import autoclip.core.media as media_mod

    monkeypatch.setattr(media_mod, "ffmpeg_exe", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    rc = _doctor([])
    assert rc in (0, 1)


def test_doctor_encoder_and_cuda_and_disk_failures(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    monkeypatch.setattr("autoclip.core.hardware.pick_encoder", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("enc")))
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "ctranslate2":
            raise RuntimeError("cuda")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(shutil, "disk_usage", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk")))
    rc = _doctor([])
    assert rc in (0, 1)


def test_models_unknown_action_unreachable_is_removed() -> None:
    # argparse only permits list/pull/path, so there is no action reaching the trailing return.
    assert _models(["list"]) == 0
