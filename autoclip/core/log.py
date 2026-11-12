"""Logging: one ``autoclip`` logger, a file handler shared by all processes, and the ``traced`` decorator."""

from __future__ import annotations

import contextlib
import functools
import importlib
import inspect
import io
import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, ParamSpec, Protocol, TextIO, TypeVar, cast, overload

if TYPE_CHECKING:
    from .resources import ResourceMonitor

P = ParamSpec("P")
R = TypeVar("R")
LOG = logging.getLogger("autoclip")
_TRACE_DEPTH = threading.local()


def _resources() -> ResourceMonitor | None:
    """Try to import the shared resource monitor without breaking bootstrap imports."""
    try:
        from .resources import RESOURCES
    except ImportError:  # pragma: no cover
        return None  # pragma: no cover
    return RESOURCES


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
            stream.flush()
            fd = stream.fileno()  # no buffering in user space: the bytes are on disk before the caller gets back
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

    def __init__(self, mirror: TextIO | None, logger: logging.Logger, console: TextIO | None = None) -> None:
        self._mirror = mirror
        self._logger = logger
        # The stream console log records go to. A reconfiguration (`main` runs twice in one process, the
        # desktop app starts several pipelines) must not hand the previous tee back to a StreamHandler:
        # the tee would emit the record to the logger again and recurse.
        self._console = console if console is not None else mirror
        self._buffer = ""

    def write(self, s: str) -> int:
        text = s
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


def default_log_file(output_dir: str | Path) -> Path:
    """The log file of a run: ``{output_dir}/logs/autoclip_<timestamp>.log`` (one file per run)."""
    return Path(output_dir) / "logs" / f"autoclip_{datetime.now():%Y%m%d_%H%M%S}.log"


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
    if isinstance(orig_stdout, _StdoutTee) and orig_stdout._console is not None:
        orig_stdout = orig_stdout._console
    if file_level == "OFF":
        log_file = None
    if console_level == "OFF":
        console_level = None
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
        tee = _StdoutTee(None, LOG, console=orig_stdout)
        sys.stdout = tee
    if not LOG.handlers:
        LOG.addHandler(logging.NullHandler())


def speed_text(media_seconds: float, wall_seconds: float) -> str:
    """Throughput as ``"3.20x realtime"`` (media seconds processed per wall second)."""
    if media_seconds <= 0 or wall_seconds <= 0:
        return "n/a"
    return f"{media_seconds / wall_seconds:.2f}x realtime"


@overload
def traced(fn: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def traced(label: str | None = None, /, level: int = logging.INFO) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def traced(
    label_or_fn: Callable[P, R] | str | None = None, /, level: int = logging.INFO
) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
    """Universal operation decorator: stdout spec lines, structured log lines and resource balancing.

    Use ``@traced`` directly or ``@traced("label", level)``. Keeps the user-supplied ``[START] name``
    / ``[END] name - Elapsed time: ...`` print form (captured into ``.log`` by :func:`setup_logging`),
    adds ``START/END`` machine state and a ``FAIL`` line, and rebalances CPU/RAM after the call.
    """

    def wrap(fn: Callable[P, R], label: str | None) -> Callable[P, R]:
        op = label or fn.__qualname__
        callback_name = getattr(fn, "__name__", "<unknown>")

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                _TRACE_DEPTH.depth = getattr(_TRACE_DEPTH, "depth", 0) + 1
                t0 = time.perf_counter()
                print(f"[START] {callback_name}")
                LOG.log(level, "START %s | %s", op, health_check(op))
                try:
                    result = await fn(*args, **kwargs)
                except BaseException as exc:
                    elapsed = time.perf_counter() - t0
                    LOG.error("FAIL  %s | %.3fs | %s: %s", op, elapsed, type(exc).__name__, exc)
                    _TRACE_DEPTH.depth -= 1
                    raise
                elapsed = time.perf_counter() - t0
                print(f"[END] {callback_name} - Elapsed time: {elapsed:.6f} seconds")
                monitor = _resources()
                if monitor is None:
                    LOG.log(level, "END   %s | %.3fs | bootstrap", op, elapsed)
                else:
                    LOG.log(level, "END   %s | %.3fs | %s", op, elapsed, monitor.brief())
                    monitor.after_operation(op)
                _TRACE_DEPTH.depth -= 1
                return cast(R, result)

            return cast(Callable[P, R], async_wrapper)

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            _TRACE_DEPTH.depth = getattr(_TRACE_DEPTH, "depth", 0) + 1
            t0 = time.perf_counter()
            print(f"[START] {callback_name}")
            LOG.log(level, "START %s | %s", op, health_check(op))
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                elapsed = time.perf_counter() - t0
                LOG.error("FAIL  %s | %.3fs | %s: %s", op, elapsed, type(exc).__name__, exc)
                _TRACE_DEPTH.depth -= 1
                raise
            elapsed = time.perf_counter() - t0
            print(f"[END] {callback_name} - Elapsed time: {elapsed:.6f} seconds")
            monitor = _resources()
            if monitor is None:
                LOG.log(level, "END   %s | %.3fs | bootstrap", op, elapsed)
            else:
                LOG.log(level, "END   %s | %.3fs | %s", op, elapsed, monitor.brief())
                monitor.after_operation(op)
            _TRACE_DEPTH.depth -= 1
            return result

        return cast(Callable[P, R], wrapper)

    if callable(label_or_fn):
        return wrap(label_or_fn, None)
    label = label_or_fn if isinstance(label_or_fn, str) or label_or_fn is None else str(label_or_fn)

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        return wrap(fn, label)

    return decorate


def redact_url(raw: str) -> str:
    """Return a URL with secret-ish query parameters redacted for logging, never password/full token."""
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    try:
        parsed = urlsplit(str(raw))
    except ValueError:
        return str(raw)
    if parsed.fragment or parsed.query:
        sensitive = {"token", "key", "auth", "sig", "signature", "secret", "password", "pass", "bearer"}
        kept = [
            (k, "<redacted>" if k.split("-", 1)[-1].lower() in sensitive or len(v) > 48 else v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        ]
        safe_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(kept, doseq=True), None))
        return safe_url
    return str(raw)


def safe_path_text(path: Path, *, home: Path | None = None) -> str:
    """Render a path for logs while folding the home folder to ``~`` so it stays readable but not exposing user."""
    source = Path(home) if home is not None else Path.home()
    try:
        rel = Path(path).resolve().relative_to(source.resolve())
    except (OSError, ValueError):
        return str(path)
    return "~/" + ("", rel.as_posix())[bool(rel.parts)]


def system_profile(
    *,
    output_dir: Path | None = None,
    log_path: Path | None = None,
    cache_roots: Sequence[str] | tuple[Path, ...] | None = None,
) -> str:
    """A compact, secret-free system snapshot for logs and app pane, as a single line of key=value pairs."""
    import platform

    try:
        from .resources import MB, RESOURCES, cgroup_memory_available
    except ImportError:  # pragma: no cover
        return "system profile unavailable during bootstrap"  # pragma: no cover

    snap = RESOURCES.snapshot()
    mb_total = snap.mem_total // MB
    available = snap.mem_available // MB
    os_name = platform.system()
    parts = [
        f"os={os_name}",
        f"python={platform.python_version()}",
        f"cpus={RESOURCES.cpus}",
        f"ram_total={mb_total}MB",
        f"ram_available={available}MB",
        f"pressure={snap.pressure}",
        f"cpu_percent={snap.cpu_percent:.2f}",
        f"output_dir={safe_path_text(output_dir)}" if output_dir else "output_dir=<not-set>",
        f"log={safe_path_text(log_path)}" if log_path else "log=<none>",
    ]
    if isinstance(cache_roots, (list, tuple)):
        try:
            for root in cache_roots:
                parts.append(f"cache_root={safe_path_text(Path(str(root)))}")
        except Exception as exc:  # pragma: no cover
            LOG.debug("could not add cache root %s: %s", root, exc)  # pragma: no cover
    cgroup = cgroup_memory_available()
    if cgroup is not None:
        parts.append(f"cgroup_free={cgroup // MB}MB")
    return " | ".join(parts)


def _bootstrap_health(label: str) -> str:
    """A health line for when the resources module cannot be imported yet (best effort)."""
    import psutil  # pragma: no cover

    memory = psutil.virtual_memory()  # pragma: no cover
    cpus = psutil.cpu_count(logical=True) or 1  # pragma: no cover
    return f"health {label}: bootstrap | ram_total={memory.total // (1024 * 1024)}MB | cpus={cpus}"  # pragma: no cover


def health_check(label: str = "step") -> str:
    """A compact, secret-free system health record (before steps/hooks)."""
    try:
        from .resources import MB, RESOURCES, cgroup_memory_available
    except ImportError:  # pragma: no cover
        return _bootstrap_health(label)  # pragma: no cover

    try:
        snap = RESOURCES.snapshot()
        available = max(0, snap.mem_available // MB - RESOURCES.memory_reserve_mb)
        cgroup = cgroup_memory_available()
    except ImportError:  # pragma: no cover
        return _bootstrap_health(label)  # pragma: no cover
    parts = [
        f"health {label}: cpus={RESOURCES.cpus}",
        f"cpu_budget={RESOURCES.cpu_budget()}",
        f"cpu_percent={snap.cpu_percent:.2f}",
        f"ram_total={snap.mem_total // MB}MB",
        f"ram_available={snap.mem_available // MB}MB",
        f"ram_free={available}MB",
        f"rss={snap.rss // MB}MB",
        f"pressure={snap.pressure}",
    ]
    if cgroup is not None:
        parts.append(f"cgroup_free={cgroup // MB}MB")
    return " | ".join(parts)
