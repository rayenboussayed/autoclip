"""``python -m autoclip``."""

import multiprocessing

from .cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()  # required for frozen Windows executables
    raise SystemExit(main())
