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

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

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
    "autoclip.cli",
    "autoclip.app",
    "autoclip.app.window",
    "autoclip.app.form",
    "autoclip.app.session",
    "tkinter",
    "tkinter.ttk",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "tkinter.scrolledtext",
    "PIL._tkinter_finder",
    "faster_whisper",
    "ctranslate2",
    "tokenizers",
    "onnxruntime",
    "av",
    "cv2",
    "scenedetect",
    "scenedetect.detectors",
    "scenedetect.scene_manager",
]
hiddenimports = HIDDEN + collect_submodules("moviepy") + collect_submodules("scenedetect")

#: The ffmpeg binary of imageio-ffmpeg (used when there is no ffmpeg on the PATH).
datas = collect_data_files("imageio_ffmpeg")

a = Analysis(  # noqa: F821 - PyInstaller injects these names
    [str(Path(SPECPATH) / "tools" / "frozen_entry.py")],
    pathex=[str(Path(SPECPATH))],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter.test", "test", "pytest"],
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
