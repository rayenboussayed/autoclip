from __future__ import annotations

import io
import subprocess

from autoclip.core.process import read_events, terminate_tree


def test_read_events_yields_json_and_skips_noise() -> None:
    proc = subprocess.Popen(["python", "-c", "print('hi')"], stdout=subprocess.PIPE, text=True)
    # build fake stdout
    fake = io.StringIO('{"a":1}\nnot json\n\n{"b":2}\n')
    proc.stdout = fake
    out = list(read_events(proc))
    assert out == [{"a": 1}, {"b": 2}]


def test_terminate_tree_handles_dead_process() -> None:
    proc = subprocess.Popen(["python", "-c", "pass"], text=True)
    proc.wait()
    terminate_tree(proc)  # must not raise
    assert proc.poll() is not None
