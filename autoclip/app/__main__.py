"""``python -m autoclip.app`` - the desktop app."""

from __future__ import annotations

import multiprocessing

from . import launch

if __name__ == "__main__":
    multiprocessing.freeze_support()  # required for frozen Windows executables
    raise SystemExit(launch())
