"""The behaviour of the desktop app, without any widgets.

The window draws the form and collects the values; everything that can be tested without a
display lives here: reading and writing the config file, validating it, starting and stopping the
pipeline and opening the output folder.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import webbrowser
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..core.config import Config, parse
from ..core.errors import ConfigError

#: The folder of the project (the parent of the package), used to run the pipeline from a source checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """A nested config dict as ``{"clips.short.count": 5, ...}`` (lists stay lists)."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def unflatten(values: dict[str, Any]) -> dict[str, Any]:
    """``{"clips.short.count": 5}`` back into a nested config dict."""
    out: dict[str, Any] = {}
    for path, value in values.items():
        parts = path.split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def split_csv(text: str) -> list[str]:
    """A comma separated field as a list of trimmed, non-empty items."""
    return [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]


def default_values() -> dict[str, Any]:
    """The values of a fresh config file (every option with its default)."""
    return flatten(asdict(Config()))


def read_config(path: Path) -> dict[str, Any]:
    """The config file as a flat dict of option paths.

    Raises:
        ConfigError: if the file is not a JSON object.
        OSError: if the file cannot be read.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError("the config file must contain a JSON object")
    return flatten(data)


def write_config(path: Path, data: dict[str, Any]) -> None:
    """Write a config file (the folders are created when needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def check_config(data: dict[str, Any]) -> None:
    """Check the data against the schema of the config file.

    Raises:
        ConfigError: if the data does not fit the schema.
    """
    parse(Config, data)


def pipeline_command(config_path: Path) -> list[str]:
    """The command that runs the pipeline for a config file.

    A frozen build is the command line itself, so it runs itself; an installed package runs the
    module of the project.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, str(config_path)]
    return [sys.executable, "-m", "autoclip", str(config_path)]


def start_pipeline(config_path: Path) -> subprocess.Popen[str]:
    """Start the pipeline as a child process (its output is the log of the app)."""
    env = dict(os.environ)
    if not getattr(sys, "frozen", False):
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PROJECT_ROOT), env.get("PYTHONPATH", "")]))
    return subprocess.Popen(
        pipeline_command(config_path),
        cwd=str(config_path.parent),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=os.name != "nt",
    )


def stop_pipeline(process: subprocess.Popen[str]) -> None:
    """Stop a running pipeline (the whole process group, so the workers die with it)."""
    if os.name != "nt":
        with contextlib.suppress(OSError):  # no such group any more: fall back to the process itself
            # start_pipeline put the child in its own session, so its pid is the id of the group
            os.kill(-process.pid, signal.SIGTERM)
            return
    with contextlib.suppress(OSError):  # already gone (Windows refuses to kill an exited process)
        process.terminate()


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
