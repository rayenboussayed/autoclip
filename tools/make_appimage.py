"""Build the AppImage from the PyInstaller build.

    python -m PyInstaller autoclip.spec --noconfirm
    python tools/make_appimage.py --dist dist/autoclip --out dist/autoclip-x86_64.AppImage

The AppDir is the layout the AppImage specification asks for: an ``AppRun`` entry point, one
desktop file, the icon and the payload in ``usr/lib``.  ``appimagetool`` (from the AppImage project
releases) turns it into the AppImage; it is looked up in ``APPIMAGETOOL`` or on the PATH.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
#: The entry point: without arguments the desktop app opens, with arguments the command line runs.
APPRUN = """#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
if [ $# -eq 0 ]; then
    exec "$HERE/usr/bin/autoclip" --gui
fi
exec "$HERE/usr/bin/autoclip" "$@"
"""
#: The wrapper in ``usr/bin`` (a symlink would break PyInstaller: it looks for its files next to itself).
WRAPPER = """#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/../lib/autoclip/autoclip" "$@"
"""


def appdir(dist: Path, target: Path) -> Path:
    """Build the AppDir of the app from the PyInstaller output."""
    if target.exists():
        shutil.rmtree(target)
    payload = target / "usr" / "lib" / "autoclip"
    payload.parent.mkdir(parents=True)
    shutil.copytree(dist, payload)
    for junk in ("build", "__pycache__"):
        shutil.rmtree(payload / junk, ignore_errors=True)

    run = target / "AppRun"
    run.write_text(APPRUN, encoding="utf-8")
    run.chmod(0o755)
    wrapper = target / "usr" / "bin" / "autoclip"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(WRAPPER, encoding="utf-8")
    wrapper.chmod(0o755)

    shutil.copy(ROOT / "assets" / "autoclip.png", target / "autoclip.png")
    icon = target / ".DirIcon"
    if icon.exists() or icon.is_symlink():
        icon.unlink()
    icon.symlink_to("autoclip.png")
    shutil.copy(ROOT / "assets" / "autoclip.desktop", target / "autoclip.desktop")
    return target


def build(target: Path, out: Path, tool: str) -> int:
    """Turn the AppDir into an AppImage.

    ``appimagetool`` writes the result into the working directory and names it after the AppDir
    (``autoclip.AppDir`` -> ``autoclip-x86_64.AppImage``), so the tool runs next to the AppDir and
    the file it leaves behind is moved to ``out``.
    """
    env = dict(os.environ, ARCH=os.environ.get("ARCH") or platform.machine())
    # the tool runs in the AppDir's parent, so a relative path must be resolved against the caller's
    # working directory before the child gets a different one (shutil.which misses ./tool on Windows)
    if Path(tool).is_file():
        tool = str(Path(tool).resolve())
    else:
        found = shutil.which(tool)
        if found is not None:
            tool = str(Path(found).resolve())
    command = [tool, "--appimage-extract-and-run", str(target.resolve())]
    print(" ".join(command))
    result = subprocess.run(command, env=env, cwd=target.parent, check=False)
    if result.returncode != 0:
        return result.returncode
    built = sorted(target.parent.glob("*.AppImage"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not built:
        print("appimagetool did not write an AppImage")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    if built[0].resolve() != out.resolve():
        if out.exists():
            out.unlink()
        built[0].replace(out)
    out.chmod(0o755)
    print(f"wrote {out} ({out.stat().st_size // (1024 * 1024)} MB)")
    return 0


def main(argv: list[str]) -> int:
    """Build the AppImage."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dist", type=Path, default=Path("dist/autoclip"), help="the PyInstaller build")
    parser.add_argument("--out", type=Path, default=Path("dist/autoclip.AppImage"), help="the AppImage to write")
    parser.add_argument("--appdir", type=Path, default=Path("build/autoclip.AppDir"), help="the AppDir to build")
    parser.add_argument("--tool", default=os.environ.get("APPIMAGETOOL", "appimagetool"), help="appimagetool")
    args = parser.parse_args(argv[1:])
    if not args.dist.is_dir():
        print(f"no PyInstaller build in {args.dist} - run: python -m PyInstaller autoclip.spec --noconfirm")
        return 2
    appdir(args.dist, args.appdir)
    return build(args.appdir, args.out, args.tool)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
