"""The AppImage builder runs appimagetool from the AppDir's parent directory."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools import make_appimage


def test_relative_tool_is_resolved_before_the_working_directory_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The relative tool and AppDir must be absolute inside the child process."""
    monkeypatch.chdir(tmp_path)
    tool = tmp_path / "appimagetool"
    tool.write_text("", encoding="utf-8")
    tool.chmod(0o755)
    appdir = Path("build") / "autoclip.AppDir"
    appdir.mkdir(parents=True)
    seen: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.extend(command)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert make_appimage.build(appdir, tmp_path / "out.AppImage", "./appimagetool") == 1
    assert seen == [str(tool), "--appimage-extract-and-run", str(appdir.resolve())]
