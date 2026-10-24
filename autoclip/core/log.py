"""Logging: one ``autoclip`` logger, a file handler shared by all processes, and the ``traced`` decorator."""

from __future__ import annotations

import contextlib
import functools
import importlib
import io
import logging
import os
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import ParamSpec, Protocol, TypeVar, cast

from .resources import RESOURCES

P = ParamSpec("P")
R = TypeVar("R")
LOG = logging.getLogger("autoclip")


class _SharedFileHandler(logging.FileHandler):
    """Append-only handler for a log file that several processes write to.

    POSIX appends a ``write`` atomically. Windows does not: the C runtime seeks to the end of the file and
    then writes, so two workers logging at the same moment can overwrite each other's line. Every record is
    therefore written under a one-byte lock of the file (see ``_lock_writes``).
    """

    def emit(self, record: logging.LogRecord) -> None:
        stream = self.stream
        if stream is None:  # closed handler: there is no file left to write to
            return
        with _lock_writes(stream):
            super().emit(record)


class _Msvcrt(Protocol):
    """The part of the Windows-only ``msvcrt`` module used here (its stub is empty on other platforms)."""

    LK_LOCK: int
    LK_UNLCK: int

    def locking(self, fd: int, mode: int, nbytes: int, /) -> None: ...


def _appends_need_locking() -> bool:
    """Whether appends to a shared file need an explicit lock on this platform (Windows)."""
    return os.name == "nt"


@contextlib.contextmanager
def _lock_writes(stream: io.TextIOBase) -> Iterator[None]:
    """Serialise one write to ``stream`` where the OS needs the help.

    The lock is a byte of the log file itself (locking past the end of a new file is allowed). It only
    keeps *our* writers apart, which is enough: every process of the run logs through this handler. A
    lock that cannot be taken must not lose the record, so the write then happens unlocked - an
    interleaved line is better than a hole.
    """
    if not _appends_need_locking():
        yield
        return
    msvcrt = cast(_Msvcrt, importlib.import_module("msvcrt"))  # no other platform ever loads it

    fd = stream.fileno()
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
    except OSError:
        yield
        return
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


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
        fh = _SharedFileHandler(log_file, encoding="utf-8")
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
