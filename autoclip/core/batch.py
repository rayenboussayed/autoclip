"""The per-run batch: download the URLs of ``input``, run the pipeline per source, aggregate results.

Front ends (CLI, desktop app, HTTP server) used to embed this loop themselves; there is now a
single implementation so progress and cancellation behave identically everywhere.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config, is_url
from .download import Downloader
from .errors import AutoclipError, ConfigError
from .events import CancelToken, Reporter
from .models import Encoder
from .pipeline import process_video
from .resources import MB, RESOURCES


@dataclass(slots=True)
class BatchResult:
    """Aggregate outcome of one ``run_batch`` call."""

    clips_done: int
    clips_failed: int
    elapsed_seconds: float
    peak_rss_mb: float

    @property
    def exit_code(self) -> int:
        return 1 if self.clips_failed else 0


def _unique_name(candidate: str, used: set[str]) -> str:
    """A name not clashing with ``used`` (case-insensitive); falls back to "Name (2)" etc."""
    if candidate.casefold() not in used:
        used.add(candidate.casefold())
        return candidate
    n = 2
    while f"{candidate} ({n})".casefold() in used:
        n += 1
    used.add(f"{candidate} ({n})".casefold())
    return f"{candidate} ({n})"


def run_batch(
    cfg: Config,
    *,
    downloader: Downloader | None = None,
    encoder: Encoder,
    log_file: Path | None,
    report: Reporter | None = None,
    cancel: CancelToken | None = None,
) -> BatchResult:
    """Run the configured batch and return its aggregate result.

    ``downloader`` may be ``None`` for a fully local run; URLs in ``input`` then fail fast.
    """
    from .log import LOG as log

    reporter = report if report is not None else Reporter(cancel=cancel)
    used_names: set[str] = {Path(item).stem.casefold() for item in cfg.input if not is_url(item)}
    t0 = time.perf_counter()
    done = 0
    failed = 0
    inputs = list(cfg.input)
    for i, item in enumerate(inputs, 1):
        scoped = reporter.scoped(i, len(inputs)) if inputs else reporter
        scoped("validate", 0.0, f"input {i}/{len(inputs)}")
        label = item if is_url(item) else Path(item).name
        log.info("=== input %d/%d: %s ===", i, len(inputs), label)
        if is_url(item):
            sources: list[tuple[Path, str | None, Path | None, bool]] = []
            if downloader is None:
                log.error("%s failed: no downloader available for a URL", item)
                failed += 1
                continue
            try:
                sources = [
                    (entry.path, _unique_name(entry.name, used_names), entry.subtitle, entry.subtitle_manual)
                    for entry in downloader.download(item)
                ]
            except (AutoclipError, ConfigError) as e:
                log.error("%s failed: %s", item, e)
                failed += 1
                continue
        else:
            sources = [(Path(item), None, None, False)]
        for path, name, subtitle, subtitle_manual in sources:
            try:
                ok, bad = process_video(
                    path,
                    cfg,
                    log_file,
                    encoder,
                    name=name,
                    subtitle=subtitle,
                    subtitle_manual=subtitle_manual,
                    report=scoped,
                    cancel=reporter.cancel if cancel is None else cancel,
                )
            except Exception as e:  # one broken video must not stop the batch
                log.error("%s failed: %s: %s", path.name, type(e).__name__, e, exc_info=not isinstance(e, AutoclipError))
                ok, bad = 0, 1
            done += ok
            failed += bad
    elapsed = time.perf_counter() - t0
    peak = RESOURCES.peak_rss / MB
    log.info("finished: %d clip(s) done, %d failed in %.1fs (peak rss %.0fMB)", done, failed, elapsed, peak)
    reporter("package", 1.0, "batch complete")
    return BatchResult(clips_done=done, clips_failed=failed, elapsed_seconds=elapsed, peak_rss_mb=peak)
