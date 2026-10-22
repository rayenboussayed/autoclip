"""Logging: one ``autoclip`` logger, a file handler shared by all processes, and the ``traced`` decorator."""

from __future__ import annotations

import functools
import io
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec, TypeVar

from .resources import RESOURCES

P = ParamSpec("P")
R = TypeVar("R")
LOG = logging.getLogger("autoclip")


def setup_logging(log_file: Path | None, file_level: str, console_level: str | None) -> None:
    """(Re)configure the ``autoclip`` logger.

    Args:
        log_file: Appended to (UTF-8); ``None`` disables file logging.
        file_level: Minimum level written to ``log_file``.
        console_level: Minimum level printed to stdout; ``None`` disables console output (worker processes).
    """
    LOG.setLevel(logging.DEBUG)
    LOG.propagate = False
    for handler in list(LOG.handlers):
        LOG.removeHandler(handler)
        handler.close()
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(
            logging.Formatter("%(asctime)s.%(msecs)03d | %(levelname)-7s | pid=%(process)-6d | %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        LOG.addHandler(fh)
    if console_level is not None:
        if isinstance(sys.stdout, io.TextIOWrapper):  # legacy Windows code pages must not crash on titles
            sys.stdout.reconfigure(errors="replace")
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(console_level)
        sh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
        LOG.addHandler(sh)
    if not LOG.handlers:
        LOG.addHandler(logging.NullHandler())


def speed_text(media_seconds: float, wall_seconds: float) -> str:
    """Throughput as ``"3.20x realtime"`` (media seconds processed per wall second)."""
    if media_seconds <= 0 or wall_seconds <= 0:
        return "n/a"
    return f"{media_seconds / wall_seconds:.2f}x realtime"


def traced(label: str | None = None, level: int = logging.INFO) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator: log ``START`` / ``END`` (or ``FAIL``) with wall time and machine state.

    After the operation the resource monitor rebalances (re-sizes caches, collects garbage under memory
    pressure) so RAM is free for the next step.

    Args:
        label: Operation name in the log (default: the function's qualified name).
        level: Level of the START/END lines (FAIL is always ERROR).
    """

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        op = label or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            t0 = time.perf_counter()
            LOG.log(level, "START %s | %s", op, RESOURCES.brief())
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                LOG.error("FAIL  %s | %.3fs | %s: %s", op, time.perf_counter() - t0, type(exc).__name__, exc)
                raise
            LOG.log(level, "END   %s | %.3fs | %s", op, time.perf_counter() - t0, RESOURCES.brief())
            RESOURCES.after_operation(op)
            return result

        return wrapper

    return decorate
