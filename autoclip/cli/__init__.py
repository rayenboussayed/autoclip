"""The command line front end: ``python -m autoclip config.json``.

It parses the arguments, writes a starter config with ``--init`` / ``--init-full`` and then runs
the batch of videos described by the config file. The work itself happens in
:mod:`autoclip.core.pipeline`.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ..core.config import Config, load_config
from ..core.errors import AutoclipError, ConfigError
from ..core.hardware import pick_encoder
from ..core.log import setup_logging
from ..core.pipeline import process_video
from ..core.resources import MB, RESOURCES

log = logging.getLogger("autoclip")


def init_config(path: Path, full: bool) -> None:
    """Write a starter (or complete) config file; never overwrites an existing one."""
    if path.exists():
        raise ConfigError(f"{path} already exists (not overwriting)")
    starter = {
        "input": ["video.mp4"],
        "clips": {
            "kinds": ["short", "long"],
            "scenes": ["best"],
            "short": {"count": 3, "min_seconds": 20, "max_seconds": 60, "aspect": "9:16"},
            "long": {"count": 1, "min_seconds": 180, "max_seconds": 600, "aspect": "16:9"},
        },
    }
    data = asdict(Config()) if full else starter
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Exit status: 0 all clips rendered, 1 some video or clip failed, 2 configuration error."""
    ap = argparse.ArgumentParser(
        prog="autoclip", description="Cut long videos into the best short clips (free, local). All options live in the config file."
    )
    ap.add_argument("config", nargs="?", default="config.json", type=Path, help="path to the JSON config (default: ./config.json)")
    ap.add_argument("--init", action="store_true", help="write a tiny starter config and exit")
    ap.add_argument("--init-full", action="store_true", help="write a config with every option and its default, and exit")
    ap.add_argument("--gui", action="store_true", help="open the desktop app instead of running a config file")
    args = ap.parse_args(argv)
    if args.gui:
        from ..app import launch

        return launch(args.config)
    try:
        if args.init or args.init_full:
            init_config(args.config, args.init_full)
            return 0
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    lg = cfg.logging
    log_file = (
        (Path(lg.file) if lg.file else Path(cfg.output_dir) / "logs" / f"autoclip_{datetime.now():%Y%m%d_%H%M%S}.log")
        if lg.enabled
        else None
    )
    setup_logging(log_file, lg.level, lg.console_level)
    RESOURCES.configure(cfg)
    snap = RESOURCES.snapshot()
    log.info(
        "autoclip start | python %s on %s | cpus=%d ram=%.0fMB free=%.0fMB | %d video(s) | log: %s",
        platform.python_version(),
        platform.system(),
        RESOURCES.cpus,
        snap.mem_total / MB,
        snap.mem_available / MB,
        len(cfg.input),
        log_file,
    )
    encoder = pick_encoder(cfg.video, allow_gpu=cfg.compute.device != "cpu")
    t0, done, failed = time.perf_counter(), 0, 0
    for n, name in enumerate(cfg.input, 1):
        log.info("=== video %d/%d: %s ===", n, len(cfg.input), Path(name).name)
        try:
            ok, bad = process_video(Path(name), cfg, log_file, encoder)
        except Exception as e:  # one broken video must not stop the batch
            log.error("%s failed: %s: %s", Path(name).name, type(e).__name__, e, exc_info=not isinstance(e, AutoclipError))
            ok, bad = 0, 1
        done, failed = done + ok, failed + bad
    log.info(
        "finished: %d clip(s) done, %d failed in %.1fs (peak rss %.0fMB)", done, failed, time.perf_counter() - t0, RESOURCES.peak_rss / MB
    )
    return 1 if failed else 0
