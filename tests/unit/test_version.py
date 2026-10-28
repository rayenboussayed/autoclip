"""The version of the package is the same in every file that carries it."""

from __future__ import annotations

import re
from pathlib import Path

import autoclip

ROOT = Path(__file__).resolve().parents[2]


def _version(path: str, pattern: str) -> str:
    """Return the version captured by pattern in path."""
    match = re.search(pattern, (ROOT / path).read_text(encoding="utf-8"), re.MULTILINE)
    assert match is not None, f"no version in {path}"
    return match.group(1)


def test_every_version_is_the_package_version() -> None:
    """The build metadata and the snap recipe carry the package version."""
    assert _version("pyproject.toml", r'^version = "([^"]+)"$') == autoclip.__version__
    assert _version("snapcraft.yaml", r'^version: "([^"]+)"$') == autoclip.__version__
