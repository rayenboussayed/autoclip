"""The PyInstaller spec must bundle the server pages and fail loudly when they are required."""

from __future__ import annotations

import builtins
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SERVER_DATA = [("templates/index.html", "autoclip/server/templates"), ("templates/job.html", "autoclip/server/templates")]
FFMPEG_DATA = [("binaries/ffmpeg-linux-x86_64", "imageio_ffmpeg/binaries")]
EJS_DATA = [("yt_dlp_ejs/yt_dlp_ejs.min.js", "yt_dlp_ejs")]


class _Analysis:
    """Stands in for PyInstaller's Analysis: records what the spec asks it to bundle."""

    def __init__(self, scripts: list[str], **kwargs: Any) -> None:
        self.scripts = scripts
        self.kwargs = kwargs
        self.pure: list[Any] = []
        self.binaries: list[Any] = []
        self.datas = kwargs["datas"]


class _Step:
    """Any later PyInstaller stage (PYZ, EXE, COLLECT): it only has to accept the call."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


def _run_spec(monkeypatch: pytest.MonkeyPatch, *, server_files: list[tuple[str, str]] | Exception) -> dict[str, Any]:
    """Execute the real spec with fake PyInstaller hooks and stages."""

    def collect_data_files(package: str) -> list[tuple[str, str]]:
        if package == "autoclip.server":
            if isinstance(server_files, Exception):
                raise server_files
            return server_files
        if package == "imageio_ffmpeg":
            return FFMPEG_DATA
        if package == "yt_dlp_ejs":
            return EJS_DATA
        return []

    hooks = ModuleType("PyInstaller.utils.hooks")
    hooks.collect_data_files = collect_data_files  # type: ignore[attr-defined]
    hooks.collect_submodules = lambda package: []  # type: ignore[attr-defined]
    package = ModuleType("PyInstaller")
    package.__path__ = []
    utils = ModuleType("PyInstaller.utils")
    utils.__path__ = []
    for name, module in {
        "PyInstaller": package,
        "PyInstaller.utils": utils,
        "PyInstaller.utils.hooks": hooks,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    # PyInstaller injects these names into the spec's globals; builtins is the closest stand-in.
    for name, value in {"Analysis": _Analysis, "PYZ": _Step, "EXE": _Step, "COLLECT": _Step, "SPECPATH": str(ROOT)}.items():
        monkeypatch.setattr(builtins, name, value, raising=False)
    return runpy.run_path(str(ROOT / "autoclip.spec"))


def test_spec_bundles_the_server_templates(monkeypatch: pytest.MonkeyPatch) -> None:
    """The server pages are data files: a later ``datas = ...`` must not overwrite them."""
    result = _run_spec(monkeypatch, server_files=SERVER_DATA)
    datas = result["a"].kwargs["datas"]
    assert all(entry in datas for entry in SERVER_DATA)
    # the other data files must survive the merge too
    assert all(entry in datas for entry in FFMPEG_DATA + EJS_DATA)


def test_spec_keeps_the_server_optional_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the server extra the spec still builds a command-line app."""
    monkeypatch.delenv("AUTOCLIP_REQUIRE_SERVER", raising=False)
    result = _run_spec(monkeypatch, server_files=RuntimeError("the server extra is not installed"))
    assert all(entry in result["a"].kwargs["datas"] for entry in FFMPEG_DATA + EJS_DATA)


def test_spec_fails_when_the_server_is_required_but_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The release build sets AUTOCLIP_REQUIRE_SERVER=1: silently shipping without /view/ is a bug."""
    monkeypatch.setenv("AUTOCLIP_REQUIRE_SERVER", "1")
    with pytest.raises(RuntimeError, match="server extra is not installed"):
        _run_spec(monkeypatch, server_files=RuntimeError("the server extra is not installed"))
