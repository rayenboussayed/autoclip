# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: a standalone autoclip for the platform it is built on.

    pyinstaller autoclip.spec --noconfirm

The binary contains the whole package - the engine, the command line and the desktop app - and
runs without a Python installation.  It is built for the platform of the machine that builds it:
PyInstaller cannot cross-compile, so the CI builds every target on a runner of that architecture.

The optional extras (the speech enhancement, the spectral denoising) are not bundled: they are
optional, the app degrades gracefully without them, and they would make the binary much bigger.
A ``pip install autoclip[enhance]`` adds them again.
"""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

BUILD = os.environ.get("AUTOCLIP_BUILD", "full").strip().lower()
HEADLESS = BUILD == "headless"

#: The modules that are imported inside functions and are therefore invisible to the analysis.
HIDDEN = [
    "autoclip.core.config",
    "autoclip.core.pipeline",
    "autoclip.core.transcript",
    "autoclip.core.moments",
    "autoclip.core.captions",
    "autoclip.core.render",
    "autoclip.core.runner",
    "autoclip.core.denoise",
    "autoclip.core.fonts",
    "autoclip.core.hardware",
    "autoclip.core.media",
    "autoclip.core.store",
    "autoclip.frozen",
    "autoclip.cli",
    "autoclip.cli.subcommands",
    "faster_whisper",
    "ctranslate2",
    "tokenizers",
    "onnxruntime",
    "av",
    "cv2",
    "scenedetect",
    "scenedetect.detectors",
    "scenedetect.scene_manager",
    "yt_dlp",
]
if not HEADLESS:
    HIDDEN += [
        "autoclip.app",
        "autoclip.app.console",
        "autoclip.app.window",
        "autoclip.app.form",
        "autoclip.app.session",
        "tkinter",
        "tkinter.ttk",
        "tkinter.filedialog",
        "tkinter.messagebox",
        "tkinter.scrolledtext",
        "PIL._tkinter_finder",
    ]
# The HTTP server is bundled when available; uvicorn's loaders are dynamic.
try:
    import uvicorn  # noqa: F401

    HIDDEN += [
        "autoclip.server",
        "autoclip.server.main",
        "autoclip.server.config",
        "autoclip.server.state",
        "autoclip.server.service",
        "autoclip.server.progress",
        "autoclip.server.auth",
        "autoclip.server.schemas",
        "autoclip.server.routers.api",
        "autoclip.server.routers.view",
    ]
    HIDDEN += collect_submodules("uvicorn")
except Exception:
    pass

hiddenimports = (
    HIDDEN
    + collect_submodules("moviepy")
    + collect_submodules("scenedetect")
    + collect_submodules("yt_dlp")
    + collect_submodules("yt_dlp_ejs")
)
datas = collect_data_files("imageio_ffmpeg") + collect_data_files("yt_dlp_ejs")
# server templates
try:
    datas = datas + collect_data_files("autoclip.server")
except Exception:
    pass

#: The ffmpeg binary of imageio-ffmpeg (used when there is no ffmpeg on the PATH) and the EJS solver
#: scripts of yt-dlp (the JavaScript challenge solver is data, not a Python module).
datas = collect_data_files("imageio_ffmpeg") + collect_data_files("yt_dlp_ejs")

a = Analysis(  # noqa: F821 - PyInstaller injects these names
    [str(Path(SPECPATH) / "tools" / "frozen_entry.py")],
    pathex=[str(Path(SPECPATH))],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter.test", "test", "pytest"] + (["tkinter", "tkinterdnd2"] if HEADLESS else []),
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="autoclip",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="autoclip",
)
