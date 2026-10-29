from __future__ import annotations

import json
from pathlib import Path

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
