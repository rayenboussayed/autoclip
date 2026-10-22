"""The entry point of the frozen build.

``python -m autoclip`` is the normal way to start the program; this script is the one the
PyInstaller build uses (``autoclip.spec``), because a frozen ``__main__`` has no package around it
and can therefore not use a relative import.
"""

from __future__ import annotations

import multiprocessing

from autoclip.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()  # required for frozen Windows executables
    raise SystemExit(main())
