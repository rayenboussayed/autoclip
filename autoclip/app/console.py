"""Hide the console a frozen Windows GUI launch opened, without detaching from it.

The frozen build is a console program because the command line needs its stdout. A double-click
therefore opens a console before the desktop app can appear. Hiding that window is enough, and it
keeps the console attached: the pipeline and its workers inherit it, so they do not allocate a
new, visible one. Detaching (``FreeConsole``) would close the handles they write to, and Windows
Terminal does not reliably close the tab anyway.

The helper is best effort. A console shared with a shell (a launch from a terminal) is left
alone, and Windows Terminal may keep its window visible; the app still opens. On any other
platform, and without a console, it does nothing.
"""

from __future__ import annotations

import ctypes
import sys
import time
from typing import Any

#: The ``ShowWindow`` command that hides a window.
SW_HIDE = 0

#: How often hiding is retried, and how long to wait between attempts. The window manager applies
#: the request asynchronously, so one call can be too early.
_ATTEMPTS = 5
_RETRY_SECONDS = 0.05


def _api() -> tuple[Any, Any] | None:
    """The Windows ``kernel32`` and ``user32`` libraries, or ``None`` where there is no console."""
    if sys.platform != "win32":
        return None
    windll = getattr(ctypes, "windll", None)
    if windll is None:  # Windows always has it; a guard keeps the module importable and testable.
        return None
    return windll.kernel32, windll.user32


def hide_console() -> bool:
    """Hide the console window of this process when this process is its only user.

    Returns:
        ``True`` when the console is hidden (or was already), ``False`` when there is nothing to
        hide, another process shares the console, or Windows ignored the request.
    """
    api = _api()
    if api is None:
        return False
    kernel32, user32 = api

    get_window = kernel32.GetConsoleWindow
    # The handle is pointer-sized: without this ctypes would truncate it to 32 bits on 64-bit Windows.
    get_window.restype = ctypes.c_void_p
    window = get_window()
    if not window:
        return False

    # GetConsoleProcessList counts this process too: one entry means nobody else (no shell) is attached.
    processes = (ctypes.c_uint * 2)()
    if kernel32.GetConsoleProcessList(processes, 2) != 1:
        return False

    show = user32.ShowWindow
    show.argtypes = [ctypes.c_void_p, ctypes.c_int]
    visible = user32.IsWindowVisible
    visible.argtypes = [ctypes.c_void_p]
    visible.restype = ctypes.c_bool
    for attempt in range(_ATTEMPTS):
        show(window, SW_HIDE)
        if not visible(window):
            return True
        if attempt + 1 < _ATTEMPTS:
            time.sleep(_RETRY_SECONDS)
    return False
