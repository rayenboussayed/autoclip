"""Real network checks for the downloader: a small, long-lived YouTube video.

These talk to the internet, so the suite skips them unless ``AUTOCLIP_NETWORK_TESTS=1``. They are the
only tests that prove the real yt-dlp API (not the scripted fake in ``tests/unit/test_download.py``) still works
with the version installed: the scheduled ``network`` workflow runs them every week.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoclip.core.config import Config, parse
from autoclip.core.download import Downloader

pytestmark = pytest.mark.network

URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo", up since 2005, ~19 s


def make_downloader(tmp_path: Path, **download: object) -> Downloader:
    """A downloader that may only use YouTube, writing under ``tmp_path``."""
    section: dict[str, object] = {"platforms": {"allow": ["youtube"]}, "metadata": {"info_json": True}}
    section.update(download)
    return Downloader(parse(Config, {"output_dir": str(tmp_path), "input": [URL], "download": section}))


def test_formats_of_a_real_video(tmp_path: Path) -> None:
    report = make_downloader(tmp_path).formats(URL)
    assert report.platform == "youtube"
    assert report.title.lower().startswith("me at the zoo")
    assert report.rows and report.chosen


def test_downloading_a_real_video(tmp_path: Path) -> None:
    downloaded = make_downloader(tmp_path).download(URL)
    assert len(downloaded) == 1
    original = downloaded[0]
    assert original.platform == "youtube" and "[jNQXAC9IVRw]" in original.name
    assert original.path.is_file() and original.path.stat().st_size > 0
    assert original.path.parent == tmp_path / ".work" / "downloads"
