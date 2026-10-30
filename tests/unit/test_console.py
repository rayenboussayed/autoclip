"""Hiding the console of a frozen Windows GUI launch (best effort, without detaching it).

The Win32 functions are faked: these tests run on every CI platform, and the point is our logic
(ownership, pointer-sized handles, retries), not Windows itself.
"""

from __future__ import annotations

import ctypes
import sys
import time
import types
from typing import Any

import pytest

from autoclip.app import console


class FakeFunction:
    """A stand-in for a Win32 function that records its calls and yields scripted results."""

    def __init__(self, *results: Any) -> None:
        self.results = list(results) or [0]
        self.calls: list[tuple[Any, ...]] = []
        self.restype: Any = None
        self.argtypes: Any = None

    def __call__(self, *args: Any) -> Any:
        self.calls.append(args)
        return self.results[0] if len(self.results) == 1 else self.results.pop(0)


def fake_windows(
    *, window: int = 1234, owners: int = 1, visible: tuple[int, ...] = (0,)
) -> tuple[types.SimpleNamespace, types.SimpleNamespace, types.SimpleNamespace]:
    """A fake ``ctypes.windll`` with the four functions the helper uses."""
    kernel32 = types.SimpleNamespace(
        GetConsoleWindow=FakeFunction(window),
        GetConsoleProcessList=FakeFunction(owners),
    )
    user32 = types.SimpleNamespace(
        ShowWindow=FakeFunction(1),
        IsWindowVisible=FakeFunction(*visible),
    )
    return types.SimpleNamespace(kernel32=kernel32, user32=user32), kernel32, user32


def on_windows(monkeypatch: pytest.MonkeyPatch, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    """Pretend to be Windows with the fake API installed."""
    windll, kernel32, user32 = fake_windows(*args, **kwargs)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", windll, raising=False)
    return windll, kernel32, user32


def test_a_non_windows_run_does_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """There is no console window to hide outside Windows."""
    monkeypatch.setattr(sys, "platform", "linux")
    assert console.hide_console() is False


def test_a_windows_python_without_windll_does_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Python build without ctypes.windll is left alone instead of crashing."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "windll", None, raising=False)
    assert console.hide_console() is False


def test_a_process_without_a_console_does_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GUI process that never had a console has no window to hide."""
    _, kernel32, user32 = on_windows(monkeypatch, window=0)
    assert console.hide_console() is False
    assert user32.ShowWindow.calls == []
    # The handle is pointer-sized, not a 32-bit int.
    assert kernel32.GetConsoleWindow.restype is ctypes.c_void_p


def test_a_console_shared_with_a_terminal_is_left_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    """Closing the console a shell also uses would close that shell's window."""
    _, kernel32, user32 = on_windows(monkeypatch, owners=2)
    assert console.hide_console() is False
    assert user32.ShowWindow.calls == []
    assert len(kernel32.GetConsoleProcessList.calls) == 1
    buffer, count = kernel32.GetConsoleProcessList.calls[0]
    assert len(buffer) == 2
    assert count == 2


def test_an_owned_console_is_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """A console this process owns is hidden once, with a pointer-sized window handle."""
    _, kernel32, user32 = on_windows(monkeypatch)
    assert console.hide_console() is True
    assert kernel32.GetConsoleWindow.calls == [()]
    assert user32.ShowWindow.calls == [(1234, console.SW_HIDE)]
    assert user32.ShowWindow.argtypes == [ctypes.c_void_p, ctypes.c_int]
    assert user32.IsWindowVisible.argtypes == [ctypes.c_void_p]
    assert user32.IsWindowVisible.restype is ctypes.c_bool


def test_hiding_is_retried_when_the_window_is_still_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    """The window manager may apply the request late; a later attempt succeeds."""
    _, _, user32 = on_windows(monkeypatch, visible=(1, 1, 0))
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    assert console.hide_console() is True
    assert len(user32.ShowWindow.calls) == 3
    assert sleeps == [console._RETRY_SECONDS] * 2


def test_a_console_windows_keeps_visible_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows Terminal can ignore the request; the helper stops after a bounded number of tries."""
    _, _, user32 = on_windows(monkeypatch, visible=(1,))
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    assert console.hide_console() is False
    assert len(user32.ShowWindow.calls) == console._ATTEMPTS
    assert sleeps == [console._RETRY_SECONDS] * (console._ATTEMPTS - 1)
