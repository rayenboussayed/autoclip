"""The behaviour of the desktop app, without any widgets.

The window draws the form and collects the values; everything that can be tested without a
display lives here: reading and writing the config file, validating it, starting and stopping the
pipeline and opening the output folder.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import queue
import subprocess
import sys
import threading
import webbrowser
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Queue as _Queue
from typing import cast

from ..core.config import Config, FileConfig, JsonValue, RuntimeOverrides, _default_output_dir, is_url, load_config, parse
from ..core.errors import AutoclipError, ConfigError
from ..core.events import CancelToken

#: The folder of the project (the parent of the package), used to run the pipeline from a source checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def flatten(data: dict[str, JsonValue], prefix: str = "") -> dict[str, JsonValue]:
    """A nested config dict as ``{"clips.short.count": 5, ...}`` (lists stay lists).

    Every nested dictionary is expanded into its option paths.
    """
    flat: dict[str, JsonValue] = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def unflatten(values: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """``{"clips.short.count": 5}`` back into a nested config dict."""
    out: dict[str, JsonValue] = {}
    for path, value in values.items():
        parts = path.split(".")
        node = out
        for part in parts[:-1]:
            node = cast(dict[str, JsonValue], node.setdefault(part, {}))
        node[parts[-1]] = value
    return out


def split_csv(text: str) -> list[str]:
    """A comma separated field as a list of trimmed, non-empty items."""
    return [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]


def default_values() -> dict[str, JsonValue]:
    """The values of a fresh config file (every option with its default)."""
    return flatten(asdict(FileConfig()))


def default_output_dir() -> Path:
    """The output folder of a run when the user has not chosen one."""
    return Path(_default_output_dir())


def default_config_path(
    system: str | None = None,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """The config file the desktop app offers on its first run.

    The folder follows the convention of the operating system: ``XDG_CONFIG_HOME`` (or
    ``~/.config``) on Linux, ``Application Support`` on macOS and ``%APPDATA%`` on Windows.

    Args:
        system: the value of ``platform.system()`` (injected by the tests).
        env: the environment to read (defaults to ``os.environ``).
        home: the home folder (defaults to ``Path.home()``).
    """
    system = platform.system() if system is None else system
    env = os.environ if env is None else env
    home = Path.home() if home is None else Path(home)
    if system == "Darwin":
        return home / "Library" / "Application Support" / "autoclip" / "config.json"
    if system == "Windows":
        appdata = env.get("APPDATA")
        base = Path(appdata) if appdata else home / "AppData" / "Roaming"
        return base / "autoclip" / "config.json"
    xdg = env.get("XDG_CONFIG_HOME")
    return (Path(xdg) if xdg else home / ".config") / "autoclip" / "config.json"


def read_config(path: Path) -> dict[str, JsonValue]:
    """The config file as a flat dict of option paths.

    Raises:
        ConfigError: if the file is not a JSON object.
        OSError: if the file cannot be read.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError("the config file must contain a JSON object")
    return flatten(data)


def write_config(path: Path, data: dict[str, JsonValue]) -> None:
    """Write a config file (the folders are created when needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def check_config(data: dict[str, JsonValue]) -> None:
    """Check the data against the schema of the config file.

    Raises:
        ConfigError: if the data does not fit the schema.
    """
    parse(FileConfig, data)


__all__ = [
    "check_config",
    "default_config_path",
    "default_output_dir",
    "default_values",
    "flatten",
    "open_folder",
    "read_config",
    "split_csv",
    "unflatten",
    "write_config",
]


@dataclass
class PipelineHandle:
    """In-memory handle for a run driven directly by the core (3.0)."""

    cancel: CancelToken
    thread: threading.Thread
    result: int = 0
    done: bool = False


class _QueueHandler(logging.Handler):
    """Forward log records to a GUI queue (strings only; no widget access)."""

    def __init__(self, queue_obj: queue.Queue[str]) -> None:
        super().__init__()
        self.queue = queue_obj

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put(self.format(record))
        except Exception:  # never take down the worker because the GUI stalls
            self.handleError(record)


def _resolve_log_file(cfg: Config) -> Path:
    """The log file of a run: one per run inside ``{output_dir}/logs``."""
    from ..core.log import default_log_file

    return default_log_file(cfg.output_dir)


def run_pipeline(
    config_path: Path,
    *,
    cancel: CancelToken,
    on_line: Callable[[str], None],
    output_dir: Path | str | None = None,
) -> int:
    """Run one config file entirely in-process (no CLI subprocess).

    Reports every log line through ``on_line`` and returns the exit code (0 ok, 1 failed, 2 bad config).
    ``output_dir`` is the app's own output-folder choice; it is never written to the config file.
    """
    import logging as _logging

    try:
        overrides = RuntimeOverrides(output_dir=str(output_dir)) if output_dir is not None else None
        cfg = load_config(config_path, overrides)
    except ConfigError as e:
        on_line(f"error: {e}")
        return 2
    log_file = _resolve_log_file(cfg)
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        from ..core.log import default_level as _default_level
        from ..core.log import setup_logging as _setup_logging

        _setup_logging(log_file, _default_level(), None)
    except Exception as e:  # logging must never stop a run
        on_line(f"warning: logging setup failed: {e}")

    from ..core.log import LOG as _LOG

    # Re-route logging to our queue handler by wrapping on_line as a downstream sink of a proxy queue.
    # Instead, attach the handler's emission to on_line via a small adapter class defined above.
    _proxy = on_line  # local alias

    class _Adapter(logging.Handler):
        def emit(self, record: _logging.LogRecord) -> None:
            try:
                _proxy(self.format(record))
            except Exception:
                self.handleError(record)

    adapter = _Adapter()
    adapter.setFormatter(_logging.Formatter("%(message)s"))
    _LOG.addHandler(adapter)

    from ..core.resources import MB as _R_MB
    from ..core.resources import RESOURCES as _RESOURCES

    _RESOURCES.configure(cfg)
    from ..core.internal.decisions import log_decisions as _log_decisions

    _log_decisions(cfg)
    snap = _RESOURCES.snapshot()
    from ..core.log import LOG as log
    from ..core.log import health_check as _health

    log.info("system health before run_pipeline: %s", _health("app_run"))
    log.info(
        "autoclip start | python %s on %s | cpus=%d ram=%.0fMB free=%.0fMB | %d input(s) | log: %s",
        platform.python_version() if platform else "unknown",
        platform.system(),
        _RESOURCES.cpus,
        snap.mem_total / _R_MB,
        snap.mem_available / _R_MB,
        len(cfg.input),
        log_file,
    )

    from ..core.log import LOG as _profile_log
    from ..core.log import system_profile as _sys_profile

    _profile_log.info(
        "system profile: %s",
        _sys_profile(
            output_dir=Path(cfg.output_dir),
            log_path=log_file,
            cache_roots=(Path(cfg.output_dir), Path(cfg.output_dir) / ".work" / "downloads"),
        ),
    )

    from ..core.download import Downloader
    from ..core.hardware import pick_encoder
    from ..core.resources import RESOURCES as _R

    snap = _R.snapshot()
    has_urls = any(is_url(item) for item in cfg.input)
    downloader: Downloader | None = None
    if has_urls:
        try:
            downloader = Downloader(cfg)
        except ConfigError as e:
            on_line(f"error: {e}")
            _cleanup_handlers(_LOG, (adapter,))
            return 2
        except AutoclipError as e:
            log.error("%s", e)
            on_line(str(e))
            _cleanup_handlers(_LOG, (adapter,))
            return 1
    from ..core.events import Reporter

    reporter_sink = on_line
    reporter = Reporter(
        lambda p: reporter_sink(f"{p.stage}: {p.message}"),
        cancel=cancel,
    )

    encoder = pick_encoder(cfg.video, cfg.quality, allow_gpu=True)
    from ..core.batch import run_batch

    try:
        config_dir = config_path.resolve().parent
        result = run_batch(
            cfg,
            config_dir=config_dir,
            downloader=downloader,
            encoder=encoder,
            log_file=log_file,
            report=reporter,
            cancel=cancel,
        )
        on_line(f"finished: {result.clips_done} done, {result.clips_failed} failed")
        _cleanup_handlers(_LOG, (adapter,))
        on_line("")
        return result.exit_code
    except ConfigError as e:
        on_line(f"error: {e}")
        _cleanup_handlers(_LOG, (adapter,))
        on_line("")
        return 2
    except Exception as e:  # a broken run must not take the GUI down
        log.error("%s: %s", type(e).__name__, e, exc_info=True)
        on_line(f"error: {type(e).__name__}: {e}")
        _cleanup_handlers(_LOG, (adapter,))
        on_line("")
        return 1


def _cleanup_handlers(logger: logging.Logger, handlers: tuple[logging.Handler, ...]) -> None:
    import contextlib

    for h in handlers:
        with contextlib.suppress(Exception):
            logger.removeHandler(h)


def cancel_pipeline(handle: PipelineHandle) -> None:
    handle.cancel.cancel()
    if handle.thread.is_alive():
        handle.thread.join(timeout=1.0)
    handle.done = True


def start_pipeline(config_path: Path) -> PipelineHandle:
    """Compatibility shim: spawn a handle whose worker runs ``run_pipeline`` in-process."""
    from ..core.events import CancelToken

    cancel = CancelToken()
    lines: queue.Queue[str] = _Queue()

    def on_line(line: str) -> None:
        lines.put(line)

    def worker() -> None:
        code = run_pipeline(config_path, cancel=cancel, on_line=on_line)
        handle.result = code
        handle.done = True

    import threading as _t

    thread = _t.Thread(target=worker, daemon=True)
    handle = PipelineHandle(cancel=cancel, thread=thread)
    thread.start()
    # keep the queue on the handle for the window to drain (window uses its own _lines queue, so
    # instead attach via a shared closure captured below)
    return handle


def stop_pipeline(process: PipelineHandle) -> None:
    if hasattr(process, "cancel"):
        cancel_pipeline(process)


# Re-exported names expected by window.py / tests
__all__ = [
    "CancelToken",
    "PipelineHandle",
    "cancel_pipeline",
    "open_folder",
    "run_pipeline",
]


def open_folder(path: Path) -> None:
    """Open a folder in the file manager of the system."""
    path.mkdir(parents=True, exist_ok=True)
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)  # noqa: S607
        elif sys.platform == "win32" and hasattr(os, "startfile"):
            os.startfile(str(path))  # noqa: S606
        else:
            subprocess.run(["xdg-open", str(path)], check=False)  # noqa: S607
    except OSError:
        webbrowser.open(path.as_uri())
