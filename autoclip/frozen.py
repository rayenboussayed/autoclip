"""The entry point of the frozen build: the desktop app without arguments, the command line with any.

A frozen binary is one executable, so unlike ``python -m autoclip`` it cannot be told which entry
point to use. It follows the convention of the AppImage and the snap: no argument opens the
desktop app, an argument runs the command line (so ``autoclip config.json``, ``autoclip run``,
``autoclip --gui``, ``autoclip --help`` and the rest keep working). ``python -m autoclip`` keeps
its own rule - no arguments runs ``./config.json`` or ``./input.mp4`` - because it is the command
line, not the application.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Open the desktop app without arguments, run the command line otherwise.

    Args:
        argv: the arguments after the program name; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The exit status of the app or of the command line.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        from .app import launch

        return launch(None)
    from .cli import main as cli_main

    return cli_main(arguments)
