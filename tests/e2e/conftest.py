"""Fixtures and hooks of the opt-in end-to-end suite; the pure logic lives in ``harness.py``."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from ..support import ManifestEntry, load_manifest
from .harness import (
    DEFAULT_RESULTS,
    DEFAULT_TIMEOUT,
    DownloadedMedia,
    E2ERun,
    _environment,
    _now,
    choose_bbc_entry,
    download_config,
    downloaded_files,
    english_subtitle,
    new_run_directory,
)


class _RunHolder:
    """The session's run, so the hook can reach it without a module-level ``global``."""

    run: E2ERun | None = None


_HOLDER = _RunHolder()


@pytest.fixture(scope="session")
def e2e_run() -> Iterator[E2ERun]:
    """The timestamped results folder of this session (``AUTOCLIP_E2E_RESULTS`` overrides its parent)."""
    root = Path(os.environ.get("AUTOCLIP_E2E_RESULTS") or DEFAULT_RESULTS).expanduser().resolve()
    directory = new_run_directory(root)
    (directory / "environment.json").write_text(json.dumps(_environment(directory), indent=2, ensure_ascii=False), encoding="utf-8")
    run = E2ERun(directory=directory, started_at=_now())
    run._begin = time.time()
    _HOLDER.run = run
    try:
        yield run
    finally:
        run.write_summary()
        _HOLDER.run = None


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Record the outcome of every e2e test in ``run.json`` (setup errors and teardown failures included)."""
    if _HOLDER.run is not None:
        _HOLDER.run.record(report.nodeid, report.when, report.outcome)


@pytest.fixture(scope="session")
def manifest() -> list[ManifestEntry]:
    """Every video of ``tests/assets/youtube.json``."""
    return load_manifest()


@pytest.fixture(scope="session")
def bbc_entry(manifest: list[ManifestEntry]) -> ManifestEntry:
    """The short BBC News clip used by the download/pipeline cases (``AUTOCLIP_E2E_URL`` overrides it)."""
    return choose_bbc_entry(manifest, os.environ.get("AUTOCLIP_E2E_URL"))


@pytest.fixture(scope="session")
def downloaded_bbc(e2e_run: E2ERun, bbc_entry: ManifestEntry) -> DownloadedMedia:
    """Download the BBC clip once and reuse the original for both pipeline cases."""
    case_dir = e2e_run.directory / "shared_download_bbc"
    result = e2e_run.execute(
        "shared_download_bbc", download_config(case_dir / "out", bbc_entry.url), ["--download-only"], timeout=DEFAULT_TIMEOUT
    )
    result.require_ok()
    files = downloaded_files(result)
    assert len(files) == 1, f"expected one original, found {[p.name for p in files]}"
    return DownloadedMedia(result, files[0], english_subtitle(files[0]), files[0].stem)
