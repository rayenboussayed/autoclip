"""``autoclip.core.events`` - the one place progress and cancellation are defined (stdlib only).

Every front end (CLI, desktop app, HTTP server) consumes :class:`Progress` values and/or a
:class:`CancelToken` through this module, so there is a single definition of stage ranges and
monotonic-percent behaviour across the whole application.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

# Stage -> (start %, end %): the same numbers the HTTP server publishes today, so API clients
# see no change across the 3.0 refactor. The ranges are contiguous from 0 to 100.
STAGES: dict[str, tuple[float, float]] = {
    "validate": (0.0, 5.0),
    "transcribe": (5.0, 40.0),
    "select": (40.0, 55.0),
    "render": (55.0, 90.0),
    "package": (90.0, 100.0),
}


class Cancelled(Exception):
    """Raised at a safe point when a job was cancelled; never swallowed by per-clip isolation."""


@dataclass(frozen=True, slots=True)
class Progress:
    """A single progress notification.

    ``percent`` is ``0..100`` and never decreases within a run.
    """

    stage: str
    percent: float  # 0..100, never decreasing within a run
    message: str = ""


ProgressSink = Callable[[Progress], None]


def ignore(_: Progress) -> None:
    """The default sink: a run with no front end that wants events."""


class CancelToken:
    """A thread-safe cancellation flag.

    One token per run; ``cancel()`` may be called from any thread (e.g. an HTTP handler
    responding to ``DELETE /api/jobs/{id}``) and the run observes it at the next ``check()``.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise Cancelled("cancelled")


class Reporter:
    """Turns ``(stage, fraction)`` into monotonic percentages; one instance per run."""

    def __init__(self, sink: ProgressSink = ignore, cancel: CancelToken | None = None) -> None:
        self._sink = sink
        self._cancel = cancel if cancel is not None else CancelToken()
        self._last = 0.0

    def __call__(self, stage: str, fraction: float = 0.0, message: str = "") -> None:
        # Every report is a cancellation point.
        self._cancel.check()
        lo, hi = STAGES[stage]
        frac = min(max(fraction, 0.0), 1.0)
        pct = max(self._last, lo + (hi - lo) * frac)
        self._last = pct
        self._sink(Progress(stage, round(pct, 1), message))
