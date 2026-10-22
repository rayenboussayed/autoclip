"""The desktop app: a native window around the autoclip pipeline.

Start it with ``python -m autoclip.app`` (or ``python -m autoclip --gui``). The window shows every
option of the config file with an explanation, writes the config you build and runs the same
pipeline the command line runs.

Tkinter is imported lazily, so a machine without a graphical toolkit can still use the command
line and this package can still be imported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def launch(config_path: Path | None = None) -> int:
    """Open the desktop app (returns the exit status of the app)."""
    from .window import launch as _launch

    return _launch(config_path)


def __getattr__(name: str) -> Any:
    """The window class, imported only when it is really needed."""
    if name == "AutoclipApp":
        from .window import AutoclipApp

        return AutoclipApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
