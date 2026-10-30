from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import time

import psutil

from autoclip.core.process import read_events, terminate_tree


def test_read_events_yields_json_and_skips_noise() -> None:
    proc = subprocess.Popen(["python", "-c", "print('hi')"], stdout=subprocess.PIPE, text=True)
    # build fake stdout: blank lines, log lines and JSON that is not an object are all skipped
    fake = io.StringIO('{"a":1}\nnot json\n\n42\n[]\n"text"\n{"b":2}\n')
    proc.stdout = fake
    out = list(read_events(proc))
    assert out == [{"a": 1}, {"b": 2}]
    proc.wait()


def test_terminate_tree_handles_dead_process() -> None:
    proc = subprocess.Popen(["python", "-c", "pass"], text=True)
    proc.wait()
    terminate_tree(proc)  # must not raise
    assert proc.poll() is not None


def test_terminate_tree_kills_the_children_too() -> None:
    """The pipeline's workers are children of the process the server kills."""
    script = "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)"
    parent = subprocess.Popen([sys.executable, "-c", script], text=True)
    try:
        root = psutil.Process(parent.pid)
        children: list[psutil.Process] = []
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            children = root.children(recursive=True)
            if children:
                break
            time.sleep(0.05)
        assert children, "the parent never started a child"

        terminate_tree(parent)

        assert parent.poll() is not None
        for child in children:
            child.wait(timeout=10)
            assert not child.is_running()
    finally:
        with contextlib.suppress(OSError):
            parent.kill()
