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
from typing import Any, ParamSpec, Protocol, TypeVar, cast

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
            stream.flush()  # crash (e.g. OOM kill) must not lose buffered lines
            if os.environ.get("AUTOCLIP_LOG_FSYNC") == "1":
                fd = stream.fileno()
                with contextlib.suppress(OSError):
                    os.fsync(fd)


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


class _StdoutTee(io.TextIOBase):
    """Capture ``print`` output so it lands in the ``.log`` file while staying visible on the console.

    ``setup_logging`` installs this as ``sys.stdout`` whenever a log file is configured. Each complete
    line is emitted back to the ``autoclip`` logger (which the file handler persists), and the original
    (un-redirected) stdout is written to as well. The emit call goes through the file handler only, so
    neither this tee nor the logger recurses.
    """

    def __init__(self, mirror: Any, logger: logging.Logger) -> None:
        self._mirror = mirror
        self._logger = logger
        self._buffer = ""

    def write(self, s: object) -> int:
        text = s if isinstance(s, str) else str(s)
        self._buffer += text
        if self._mirror is not None:
            try:
                self._mirror.write(text)
                self._mirror.flush()
            except (OSError, ValueError):
                pass
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line != "":
                self._logger.info(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._logger.info(self._buffer)
            self._buffer = ""
        if self._mirror is not None:
            with contextlib.suppress(OSError, ValueError):
                self._mirror.flush()

    def isatty(self) -> bool:
        if self._mirror is None:
            return False
        try:
            return bool(self._mirror.isatty())
        except (OSError, ValueError, AttributeError):
            return False


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
    orig_stdout = sys.stdout
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = _SharedFileHandler(log_file, encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(
            logging.Formatter("%(asctime)s.%(msecs)03d | %(levelname)-7s | pid=%(process)-6d | %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        LOG.addHandler(fh)
    if console_level is not None:
        if isinstance(orig_stdout, io.TextIOWrapper):  # legacy Windows code pages must not crash on titles
            orig_stdout.reconfigure(errors="replace")
        sh = logging.StreamHandler(orig_stdout)
        sh.setLevel(console_level)
        sh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
        LOG.addHandler(sh)
    if log_file is not None:
        tee = _StdoutTee(None, LOG)
        sys.stdout = tee
    if not LOG.handlers:
        LOG.addHandler(logging.NullHandler())


def speed_text(media_seconds: float, wall_seconds: float) -> str:
    """Throughput as ``"3.20x realtime"`` (media seconds processed per wall second)."""
    if media_seconds <= 0 or wall_seconds <= 0:
        return "n/a"
    return f"{media_seconds / wall_seconds:.2f}x realtime"


def logger_decorator(callback: Callable[P, R]) -> Callable[P, R]:
    """Wrap a function so its start/end and elapsed time are printed (and captured into ``.log``).

    Uses the user-supplied ``print`` form. The prints land in the ``.log`` file because
    :func:`setup_logging` redirects ``sys.stdout`` through a tee whenever a log file is configured, so
    console and file stay in sync. Use it sparingly: every wrapped call pays an extra clock read, string
    format and I/O, which would worsen OOM.
    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        callback_name: str = getattr(callback, "__name__", "<unknown>")
        print(f"[START] {callback_name}")
        start_time = time.perf_counter()
        result: R = callback(*args, **kwargs)
        elapsed_time = time.perf_counter() - start_time
        print(f"[END] {callback_name} - Elapsed time: {elapsed_time:.6f} seconds")
        return result

    return functools.wraps(callback)(wrapper)


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
