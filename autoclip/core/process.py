"""Process helpers for running the per-video pipeline in a child process.

``core.batch.run_batch`` is the in-process path; this module is what front ends (desktop app,
HTTP server) use to spawn one job per subprocess, giving real cancellation (kill the process
tree) and isolation of native crashes.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

#: The folder of the project (the parent of the package), used to run the pipeline from a source checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def read_events(process: subprocess.Popen[str]) -> Iterator[dict[str, object]]:
    """Yield each NDJSON event printed by a ``--events jsonl`` run.

    Lines that are not valid JSON are skipped silently (they are typically log lines).
    """
    if process.stdout is None:
        return
    for raw in process.stdout:
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            yield obj


def terminate_tree(process: subprocess.Popen[str]) -> None:
    """Kill ``process`` and every descendant (best effort, never raises)."""
    try:
        import psutil
    except ImportError:  # psutil is a core dependency; the fallback keeps terminate_tree usable without it
        psutil = None
    if psutil is not None:
        try:
            proc = psutil.Process(process.pid)
            for child in proc.children(recursive=True):
                with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                    child.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    with contextlib.suppress(OSError, ProcessLookupError):
        process.kill()
    with contextlib.suppress(OSError):
        process.wait(timeout=5)
