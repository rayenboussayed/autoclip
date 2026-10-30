"""The frozen build opens the desktop app without arguments and the command line with any.

``python -m autoclip`` keeps its own no-argument rule (``./config.json`` or ``./input.mp4``); only
the one-file build needs this dispatcher.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from autoclip import frozen

ROOT = Path(__file__).resolve().parents[2]


def test_no_arguments_open_the_desktop_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """A double-click (no arguments) opens the app instead of failing as the command line."""
    seen: list[Path | None] = []

    def fake_launch(path: Path | None = None) -> int:
        seen.append(path)
        return 0

    monkeypatch.setattr("autoclip.app.launch", fake_launch)
    assert frozen.main([]) == 0
    assert seen == [None]


def test_no_explicit_arguments_reads_the_command_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """``argv=None`` reads ``sys.argv[1:]``, like the command line does."""
    seen: list[Path | None] = []

    def fake_launch(path: Path | None = None) -> int:
        seen.append(path)
        return 5

    monkeypatch.setattr("autoclip.app.launch", fake_launch)
    monkeypatch.setattr(sys, "argv", ["/path/to/autoclip"])
    assert frozen.main() == 5
    assert seen == [None]


def test_arguments_run_the_command_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any argument keeps the frozen binary a command line (config path, run, --gui, --help...)."""
    seen: list[list[str]] = []

    def fake_cli(argv: list[str] | None = None) -> int:
        seen.append(list(argv or ()))
        return 7

    def must_not_open(path: Path | None = None) -> int:
        pytest.fail("the app must not open")

    monkeypatch.setattr("autoclip.cli.main", fake_cli)
    monkeypatch.setattr("autoclip.app.launch", must_not_open)
    assert frozen.main(["config.json"]) == 7
    assert seen == [["config.json"]]


def test_the_build_entry_point_exits_with_the_dispatcher_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """The PyInstaller script calls the dispatcher and exits with its status."""
    called: list[bool] = []

    def fake_main() -> int:
        called.append(True)
        return 3

    monkeypatch.setattr("autoclip.frozen.main", fake_main)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(ROOT / "tools" / "frozen_entry.py"), run_name="__main__")
    assert exit_info.value.code == 3
    assert called == [True]
