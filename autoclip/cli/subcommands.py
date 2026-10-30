"""Subcommand router for the autoclip CLI (WP2, 3.0)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .. import __version__

SUBCOMMANDS = ("run", "serve", "gui", "init", "doctor", "models", "download")
_PASSTHROUGH = {"-h", "--help", "--version"}


def normalize(argv: Sequence[str]) -> list[str]:
    """Keep ``autoclip [flags] config.json`` (the 2.x form) working."""
    args = list(argv)
    if not args or (args[0] not in SUBCOMMANDS and args[0] not in _PASSTHROUGH):
        return ["run", *args]
    return args


def _serve(rest: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="autoclip serve")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(rest)
    import uvicorn

    from ..server.main import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, reload=False)
    return 0


def _gui(rest: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="autoclip gui")
    ap.add_argument("config", nargs="?", default=None, type=Path)
    args = ap.parse_args(rest)
    from ..app import launch

    return launch(args.config) if args.config is not None else launch(None)


def _init(rest: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="autoclip init")
    ap.add_argument("config", nargs="?", default=None, type=Path)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args(rest)
    from ..cli import init_config
    from ..core.errors import ConfigError

    try:
        init_config(args.config or Path("config.json"), args.full)
    except (ConfigError, OSError) as e:
        print(f"error: {e}")
        return 2
    return 0


def _doctor(rest: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="autoclip doctor")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--online", action="store_true")
    args = ap.parse_args(rest)

    report: dict[str, Any] = {}
    report["version"] = __version__
    report["platform"] = sys.platform
    # ffmpeg
    try:
        report["ffmpeg"] = __import__("autoclip.core.media", fromlist=["ffmpeg_exe"]).ffmpeg_exe()
    except Exception as e:
        report["ffmpeg"] = f"error: {e}"
    # encoder probe
    try:
        from ..core import hardware

        enc = hardware.pick_encoder(__import__("autoclip.core.config", fromlist=["VideoCfg"]).VideoCfg(), allow_gpu=False)
        report["encoder"] = {"name": enc.name, "hardware": enc.hardware}
    except Exception as e:
        report["encoder"] = f"error: {e}"
    # cuda
    try:
        import ctranslate2

        report["cuda_devices"] = int(ctranslate2.get_cuda_device_count()) if hasattr(ctranslate2, "get_cuda_device_count") else 0
    except Exception as e:
        report["cuda_devices"] = f"error: {e}"
    # js runtime (only needed for YouTube URLs; a warning, not a blocker)
    js = None
    for name in ("deno", "node", "bun", "qjs", "quickjs"):
        if shutil.which(name):
            js = name
            break
    report["js_runtime"] = js
    warnings: list[str] = []
    blocking: list[str] = []
    if js is None:
        warnings.append("no JavaScript runtime (deno/node/bun/qjs/quickjs) found; needed only for YouTube URLs")
    if "error" in str(report.get("ffmpeg", "")):
        blocking.append("ffmpeg is unavailable")
    # disk
    try:
        report["free_disk_bytes"] = shutil.disk_usage(Path.cwd()).free
    except Exception as e:
        report["free_disk_bytes"] = f"error: {e}"
        blocking.append(f"cannot read free disk: {e}")
    # models cache (best effort, honour the same env vars as the transcription code)
    report["model_cache"] = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")

    ok = not blocking
    report["warnings"] = warnings
    report["blocking"] = blocking

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        for key, value in report.items():
            if key in ("warnings", "blocking"):
                continue
            print(f"{key}: {value}")
        if warnings:
            print("warnings:")
            for w in warnings:
                print(f"  - {w}")
        if blocking:
            print("blocking:")
            for b in blocking:
                print(f"  - {b}")
        print("OK" if ok else "NOT READY")
    return 0 if ok else 1


def _models(rest: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="autoclip models")
    ap.add_argument("action", choices=["list", "pull", "path"])
    ap.add_argument("name", nargs="?")
    args = ap.parse_args(rest)
    from faster_whisper.utils import available_models, download_model

    if args.action == "list":
        for name in available_models():
            print(name)
        return 0
    if args.action == "path":
        print(os.path.expanduser("~/.cache/huggingface"))
        return 0
    if args.action == "pull":
        if not args.name:
            print("error: models pull requires a name", file=sys.stderr)
            return 2
        download_model(args.name)
        return 0
    return 2
