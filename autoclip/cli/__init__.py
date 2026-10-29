"""The command line front end: ``python -m autoclip config.json``.

It parses the arguments, writes a starter config with ``--init`` / ``--init-full`` and then runs
the batch of videos described by the config file. The work itself happens in
:mod:`autoclip.core.pipeline`; URLs in ``input`` are downloaded first by
:class:`autoclip.core.download.Downloader`.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .. import __version__
from ..core.config import Config, PlatformsCfg, is_url, load_config, parse
from ..core.download import Downloader, format_report_text
from ..core.errors import AutoclipError, ConfigError
from ..core.events import Progress, Reporter
from ..core.hardware import pick_encoder
from ..core.log import setup_logging
from ..core.resources import MB, RESOURCES

log = logging.getLogger("autoclip")


def init_config(path: Path, full: bool) -> None:
    """Write a starter (or complete) config file; never overwrites an existing one."""
    if path.exists():
        raise ConfigError(f"{path} already exists (not overwriting)")
    starter = {
        "input": ["input.mp4"],
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


def _csv(values: Sequence[str] | None) -> list[str] | None:
    """The values of a repeatable ``a,b`` option as one list (``None`` when the option was not given)."""
    if values is None:
        return None
    return [name.strip() for value in values for name in value.split(",") if name.strip()]


def _with_policy(cfg: Config, allow: list[str] | None, deny: list[str] | None) -> None:
    """Apply ``--allow`` / ``--deny`` to the loaded platform policy, re-validating it."""
    if allow is None and deny is None:
        return
    data = asdict(cfg.download.platforms)
    if allow is not None:
        data["allow"] = allow
    if deny is not None:
        data["deny"] = deny
    cfg.download.platforms = parse(PlatformsCfg, data)


def _target_urls(cfg: Config, value: str | bool) -> list[str]:
    """The URL of ``--formats URL`` / ``--platforms URL``, or every URL of ``input`` for the bare flag."""
    return [value] if isinstance(value, str) else [item for item in cfg.input if is_url(item)]


def _show_formats(downloader: Downloader, urls: Sequence[str]) -> int:
    """Print the format table of every URL (``--formats``); exit 1 when one cannot be inspected."""
    blocks, status = [], 0
    for url in urls:
        try:
            blocks.append(format_report_text(downloader.formats(url)))
        except AutoclipError as e:
            print(f"error: {url}: {e}", file=sys.stderr)
            status = 1
    if blocks:
        print("\n\n".join(blocks))
    return status


def _show_platforms(downloader: Downloader, urls: Sequence[str]) -> int:
    """Print the allow/deny policy and, when URLs were given, where each one would go (``--platforms``)."""
    lines = [downloader.policy_text()]
    if urls:
        lines.append("")
        lines.extend(f"{url}\n  {downloader.describe(url)}" for url in urls)
    print("\n".join(lines))
    return 0


def _download_only(cfg: Config, downloader: Downloader) -> int:
    """Download every URL of ``input`` and do nothing else (``--download-only``)."""
    urls = [item for item in cfg.input if is_url(item)]
    if not urls:
        print("error: --download-only needs at least one URL in input", file=sys.stderr)
        return 2
    saved, failed = 0, 0
    for url in urls:
        try:
            items = downloader.download(url)
        except AutoclipError as e:
            log.error("%s failed: %s", url, e)
            failed += 1
            continue
        for item in items:
            log.info("saved %s (%s)", item.path, item.platform)
            saved += 1
    log.info("finished: %d original(s) downloaded, %d URL(s) failed", saved, failed)
    return 1 if failed else 0


def _unique_name(name: str, used: set[str]) -> str:
    """``name`` unless the batch already used it; then ``name (2)``, ``name (3)`` ... (case-insensitively)."""
    candidate, n = name, 2
    while candidate.casefold() in used:
        candidate = f"{name} ({n})"
        n += 1
    used.add(candidate.casefold())
    return candidate


def _run_command(argv: Sequence[str] | None = None) -> int:
    """Entry point. Exit status: 0 all clips rendered, 1 some video or clip failed, 2 configuration error."""
    ap = argparse.ArgumentParser(
        prog="autoclip", description="Cut long videos into the best short clips (free, local). All options live in the config file."
    )
    ap.add_argument("--version", action="version", version=f"autoclip {__version__}")
    ap.add_argument("--events", choices=["jsonl"], default=None, help="emit progress events on stdout (jsonl)")
    ap.add_argument("--json", action="store_true", help="emit the final result as JSON on stdout")
    ap.add_argument(
        "config",
        nargs="?",
        default=None,
        type=Path,
        help="path to the JSON config (default: ./config.json when it exists; otherwise the built-in defaults)",
    )
    ap.add_argument("--init", action="store_true", help="write a tiny starter config and exit")
    ap.add_argument("--init-full", action="store_true", help="write a config with every option and its default, and exit")
    ap.add_argument("--gui", action="store_true", help="open the desktop app instead of running a config file")
    ap.add_argument("--download-only", action="store_true", help="download the URLs of input and exit (no clips)")
    ap.add_argument("--formats", nargs="?", const=True, metavar="URL", help="show the formats of URL (or of every URL in input)")
    ap.add_argument("--platforms", nargs="?", const=True, metavar="URL", help="show the platform policy, and where URL would go")
    ap.add_argument("--allow", action="append", metavar="A,B", help="platforms/@groups to permit (overrides download.platforms.allow)")
    ap.add_argument("--deny", action="append", metavar="C", help="platforms/@groups to refuse (overrides download.platforms.deny)")
    args = ap.parse_args(argv)
    if args.gui:
        from ..app import launch

        return launch(args.config)
    try:
        if args.init or args.init_full:
            init_config(args.config or Path("config.json"), args.init_full)
            return 0
        if args.formats and args.platforms:
            raise ConfigError("choose one of --formats and --platforms")
        default_path = Path("config.json")
        if args.config is not None:
            cfg = load_config(args.config)
            built_in = False
        elif default_path.is_file():
            cfg = load_config(default_path)
            built_in = False
        else:
            cfg = Config()  # no file: the built-in defaults, kept in memory (never written)
            built_in = True
        _with_policy(cfg, _csv(args.allow), _csv(args.deny))
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.formats or args.platforms:
        try:
            inspector = Downloader(cfg)
        except ConfigError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        except AutoclipError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        if args.formats:
            urls = _target_urls(cfg, args.formats)
            if not urls:
                print("error: --formats needs a URL (pass one or put URLs in input)", file=sys.stderr)
                return 2
            return _show_formats(inspector, urls)
        return _show_platforms(inspector, _target_urls(cfg, args.platforms))

    if built_in:
        if args.download_only:
            print("error: --download-only needs at least one URL in input", file=sys.stderr)
            return 2
        if not any(not is_url(item) and Path(item).is_file() for item in cfg.input):
            print(
                "error: no config file and no input.mp4 in the current folder\n"
                "hint: run 'python -m autoclip --init config.json', put an input.mp4 here, or pass a config file",
                file=sys.stderr,
            )
            return 2

    lg = cfg.logging
    log_file = (
        (Path(lg.file) if lg.file else Path(cfg.output_dir) / "logs" / f"autoclip_{datetime.now():%Y%m%d_%H%M%S}.log")
        if lg.enabled
        else None
    )
    setup_logging(log_file, lg.level, None if getattr(args, "events", None) == "jsonl" else lg.console_level)
    RESOURCES.configure(cfg)
    snap = RESOURCES.snapshot()
    has_urls = any(is_url(item) for item in cfg.input)
    log.info(
        "autoclip start | python %s on %s | cpus=%d ram=%.0fMB free=%.0fMB | %d input(s) | log: %s",
        platform.python_version(),
        platform.system(),
        RESOURCES.cpus,
        snap.mem_total / MB,
        snap.mem_available / MB,
        len(cfg.input),
        log_file,
    )
    downloader: Downloader | None = None
    if has_urls:
        try:
            downloader = Downloader(cfg)
        except ConfigError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        except AutoclipError as e:
            log.error("%s", e)
            return 1
        if args.download_only:
            return _download_only(cfg, downloader)
    if args.download_only:
        print("error: --download-only needs at least one URL in input", file=sys.stderr)
        return 2

    encoder = pick_encoder(cfg.video, allow_gpu=cfg.compute.device != "cpu")
    from ..core.batch import run_batch

    events = getattr(args, "events", None)

    def _emit(p: Progress) -> None:
        if events == "jsonl":
            print(json.dumps({"event": "progress", "stage": p.stage, "percent": p.percent, "message": p.message}), flush=True)

    reporter = Reporter(_emit) if events == "jsonl" else Reporter()

    result = run_batch(cfg, downloader=downloader, encoder=encoder, log_file=log_file, report=reporter)
    if events == "jsonl":
        print(json.dumps({"event": "result", "done": result.clips_done, "failed": result.clips_failed}), flush=True)
    elif getattr(args, "json", False):
        print(json.dumps({"done": result.clips_done, "failed": result.clips_failed}), flush=True)
    return result.exit_code


def main(argv: Sequence[str] | None = None) -> int:
    import sys

    from .. import __version__
    from . import subcommands

    raw = sys.argv[1:] if argv is None else list(argv)
    argv_list = subcommands.normalize(raw)
    if not argv_list:
        argv_list = ["run"]
    cmd = argv_list[0]
    rest = argv_list[1:]

    if cmd == "run":
        return _run_command(rest)
    if cmd == "serve":
        return subcommands._serve(rest)
    if cmd == "gui":
        return subcommands._gui(rest)
    if cmd == "init":
        return subcommands._init(rest)
    if cmd == "doctor":
        return subcommands._doctor(rest)
    if cmd == "models":
        return subcommands._models(rest)
    if cmd == "download":
        return _run_command(["--download-only", *rest])
    if cmd in ("-h", "--help"):
        print("autoclip: run|serve|gui|init|doctor|models|download")
        return 0
    if cmd == "--version":
        print(f"autoclip {__version__}")
        return 0
    return _run_command(argv_list)
