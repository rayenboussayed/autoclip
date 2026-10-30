"""The build workflow must ship what the installers download (and check it really works)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")


def _job(name: str) -> str:
    """The text of one top-level job (job keys are indented by two spaces)."""
    match = re.search(rf"^  {name}:\n(.*?)(?=^  [\w-]+:\n|\Z)", WORKFLOW, re.MULTILINE | re.DOTALL)
    assert match, f"the workflow has no {name!r} job"
    return match.group(1)


def test_the_build_installs_and_requires_the_server_extra() -> None:
    """Without the extra the bundle has no FastAPI: the frozen smoke and the templates fail."""
    build = _job("build")
    assert '".[server]"' in build
    assert "AUTOCLIP_REQUIRE_SERVER" in build


def test_the_frozen_binary_serves_the_pages() -> None:
    """A bundle without the server templates answers 500 on /view/; the smoke must catch that."""
    build = _job("build")
    assert " serve " in build
    assert "/view/" in build


def test_the_release_ships_tar_gz_for_unix_and_zip_for_windows() -> None:
    """install.sh downloads autoclip-<target>.tar.gz; install.ps1 downloads the Windows .zip."""
    release = _job("release")
    assert "tar -czf" in release
    assert "zip -qr" in release
    assert "*-windows-*" in release


def test_the_release_publishes_checksums_and_installers() -> None:
    """Both installers verify SHA256SUMS, so the release must contain it and the scripts themselves."""
    release = _job("release")
    assert "SHA256SUMS" in release
    assert "install.sh" in release
    assert "install.ps1" in release


def test_the_release_attests_its_assets() -> None:
    """Build provenance makes a tampered release detectable."""
    release = _job("release")
    assert "attest-build-provenance" in release
    assert "attestations: write" in release
    assert "id-token: write" in release


def test_the_installers_are_run_against_the_real_build() -> None:
    """A checksum that matches a broken archive is still a broken install: run both scripts."""
    installers = _job("installers")
    assert "install.sh" in installers and "install.ps1" in installers
    assert "autoclip-linux-x64" in installers and "autoclip-windows-x64" in installers


def test_the_release_asset_names_match_the_installers() -> None:
    """The workflow names the assets; the scripts hard-code them. Drift breaks every install."""
    artifacts = set(re.findall(r"artifact: (autoclip-[\w-]+)", _job("build")))
    assert {
        "autoclip-linux-x64",
        "autoclip-linux-arm64",
        "autoclip-macos-x64",
        "autoclip-macos-arm64",
        "autoclip-windows-x64",
    } <= artifacts
    assert 'asset="autoclip-$target.tar.gz"' in (ROOT / "install.sh").read_text(encoding="utf-8")
    assert '$asset = "autoclip-windows-x64.zip"' in (ROOT / "install.ps1").read_text(encoding="utf-8")
