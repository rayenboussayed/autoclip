"""Pure harness logic of the opt-in end-to-end suite (fixtures and hooks live in ``conftest.py``).

Every case writes its config, stdout, stderr, logs and outputs to its own folder under
``tests/results/<timestamp>/`` (override the root with ``AUTOCLIP_E2E_RESULTS``). The run folder also
gets ``environment.json`` (what ran this) and ``run.json`` (what happened). Nothing here talks to the
network by itself, so the whole module is unit-tested offline in ``tests/unit/test_e2e_harness.py``.
"""

from __future__ import annotations

import html
import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from autoclip import __version__

from ..support import ManifestEntry, merge

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIMEOUT = 900  # a download plus a render on a slow CI runner
FORMATS_TIMEOUT = 240  # extraction only, but YouTube can be slow
LADDER_TIMEOUT = 2400  # a best-quality slice download plus a short and a long clip
DEFAULT_RESULTS = Path(__file__).resolve().parent.parent / "results"
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / ".cache" / "e2e"
#: Case folder names that would collide with the run-level files and folders.
RESERVED_CASE_NAMES = frozenset({"clips"})
#: The download ladder: every extra level adds this much source video (up to the full length).
CHUNK_SECONDS = 360
DEFAULT_LEVEL = 1
LEDGER_NAME = "levels.json"
#: The manifest cases always render one short and one long clip; the long kind uses the production range.
LADDER_CLIPS: dict[str, Any] = {
    "kinds": ["short", "long"],
    "scenes": ["best"],
    "short": {"count": 1, "min_seconds": 15, "target_seconds": 25, "max_seconds": 45, "aspect": "9:16"},
    "long": {"count": 1, "min_seconds": 180, "max_seconds": 600, "aspect": "16:9"},
}


def _now() -> str:
    """An ISO-8601 timestamp in local time (the same clock as the folder name)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _slug(text: str) -> str:
    """A folder-safe version of ``text`` (keeps it readable in the results tree)."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_.-")[:80] or "case"


def _tail(text: str, lines: int = 40) -> str:
    """The last ``lines`` lines of a stream (enough to diagnose a failed subprocess)."""
    return "\n".join(text.splitlines()[-lines:])


def new_run_directory(root: Path, *, stamp: str | None = None) -> Path:
    """Create a timestamped results folder, adding ``_2``, ``_3``... when that second already has one."""
    root.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = root / stamp
    suffix = 2
    while directory.exists():
        directory = root / f"{stamp}_{suffix}"
        suffix += 1
    directory.mkdir(parents=True)
    return directory


def choose_bbc_entry(manifest: Sequence[ManifestEntry], override: str | None = None) -> ManifestEntry:
    """The short BBC clip used by the download/pipeline cases (``AUTOCLIP_E2E_URL`` overrides it)."""
    if override:
        return ManifestEntry("custom", override)
    return next((entry for entry in manifest if "BBC News" in entry.title), manifest[-1])


def e2e_level(env: Mapping[str, str] | None = None) -> int:
    """The download-ladder level of this run (``AUTOCLIP_E2E_LEVEL``, default 1 = the first six minutes)."""
    raw = (os.environ if env is None else env).get("AUTOCLIP_E2E_LEVEL", "").strip()
    if not raw:
        return DEFAULT_LEVEL
    try:
        level = int(raw)
    except ValueError:
        raise ValueError(f"AUTOCLIP_E2E_LEVEL must be a positive whole number, not {raw!r}") from None
    if level < 1:
        raise ValueError(f"AUTOCLIP_E2E_LEVEL must be at least 1, not {level}")
    return level


def e2e_cache(env: Mapping[str, str] | None = None) -> Path:
    """The persistent download cache shared by runs (``AUTOCLIP_E2E_CACHE`` overrides it)."""
    raw = (os.environ if env is None else env).get("AUTOCLIP_E2E_CACHE", "").strip()
    return Path(raw).expanduser().resolve() if raw else DEFAULT_CACHE


def _hms(seconds: float) -> str:
    """A whole number of seconds as ``HH:MM:SS``."""
    hours, rest = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def level_range(entry: ManifestEntry, level: int) -> str | None:
    """The ``download.time_range`` of a ladder level (``None`` means the whole video).

    Each level keeps the first ``level * CHUNK_SECONDS`` seconds, capped at the manifest duration.
    The last level of an entry is the full download, so no range (and no local trimming) is needed.
    """
    seconds = float(level * CHUNK_SECONDS)
    if entry.duration is not None and seconds >= entry.duration:
        return None
    return f"00:00:00-{_hms(seconds)}"


def level_seconds(entry: ManifestEntry, level: int) -> float | None:
    """How many seconds of source a ladder level is expected to provide (``None`` when unknown)."""
    if entry.duration is None:
        return None
    return min(float(level * CHUNK_SECONDS), entry.duration)


def cache_folder(cache: Path, url: str, level: int | str) -> Path:
    """The persistent originals folder of one video and one ladder level (or another label)."""
    return cache / video_id(url) / f"level_{level}" if isinstance(level, int) else cache / video_id(url) / str(level)


def read_ledger(cache: Path) -> dict[str, dict[str, Any]]:
    """The recorded output of every ladder level that already ran (empty when there is none yet)."""
    path = cache / LEDGER_NAME
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return {str(key): value for key, value in data.items() if isinstance(value, dict)}


def level_summary(clips: Mapping[str, Sequence[Mapping[str, Any]]], source_seconds: float) -> dict[str, Any]:
    """What one ladder level produced, in a shape the next level can compare against."""
    summary: dict[str, Any] = {"source_seconds": round(source_seconds, 3)}
    for kind in ("short", "long"):
        rows = clips.get(kind, [])
        summary[kind] = {
            "clips": len(rows),
            "width": int(rows[0].get("width", 0)) if rows else 0,
            "height": int(rows[0].get("height", 0)) if rows else 0,
            "trimmed": [round(float(row.get("trimmed", 0)), 3) for row in rows],
        }
    return summary


def compare_levels(previous: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """Differences that a longer ladder level must not introduce (the clips keep their shape and count)."""
    problems: list[str] = []
    old_source = float(previous.get("source_seconds") or 0)
    new_source = float(current.get("source_seconds") or 0)
    if new_source + 0.5 < old_source:
        problems.append(f"the source shrank from {old_source:.1f}s to {new_source:.1f}s")
    for kind in ("short", "long"):
        old, new = previous.get(kind), current.get(kind)
        if not isinstance(old, Mapping) or not isinstance(new, Mapping):
            continue
        for key in ("clips", "width", "height"):
            if old.get(key) != new.get(key):
                problems.append(f"{kind}.{key} changed from {old.get(key)!r} to {new.get(key)!r}")
    return problems


def record_level(cache: Path, url: str, level: int, summary: dict[str, Any]) -> list[str]:
    """Compare a level with the closest lower one already recorded, then remember it in the cache.

    Returns the differences (empty when the level is consistent with the previous one). The ledger lives
    next to the downloaded originals, so a deeper level run days later still compares with level 1.
    """
    ledger = read_ledger(cache)
    levels = ledger.setdefault(video_id(url), {})
    lower = [int(key) for key in levels if key.isdigit() and int(key) < level]
    problems = compare_levels(levels[str(max(lower))], summary) if lower else []
    levels[str(level)] = summary
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / LEDGER_NAME
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return problems


@dataclass
class CaseResult:
    """One command-line invocation and the files it produced."""

    name: str
    directory: Path
    config_path: Path | None
    command: list[str]
    started_at: str
    duration: float
    returncode: int | None
    timed_out: bool
    stdout: str
    stderr: str
    stdout_path: Path
    stderr_path: Path
    log_path: Path | None

    def require_ok(self) -> CaseResult:
        """Fail the test with the command, its stderr tail and the log path when the run failed."""
        assert not self.timed_out, f"{self.name} timed out after {self.duration:.0f}s; see {self.directory}"
        assert self.returncode == 0, (
            f"{self.name} exited {self.returncode}; command: {' '.join(self.command)}\n"
            f"stderr:\n{_tail(self.stderr)}\nlog: {self.log_path or '-'}\ncase: {self.directory}"
        )
        return self

    def to_json(self) -> dict[str, Any]:
        """The machine-readable record written next to the case's files."""
        return {
            "name": self.name,
            "directory": str(self.directory),
            "config": str(self.config_path) if self.config_path else None,
            "command": self.command,
            "started_at": self.started_at,
            "duration_seconds": round(self.duration, 3),
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "stdout": str(self.stdout_path),
            "stderr": str(self.stderr_path),
            "log": str(self.log_path) if self.log_path else None,
        }


@dataclass
class E2ERun:
    """The results folder of one ``pytest`` session and the cases executed in it."""

    directory: Path
    started_at: str
    cases: list[CaseResult] = field(default_factory=list)
    outcomes: dict[str, dict[str, str]] = field(default_factory=dict)

    def execute(
        self,
        name: str,
        config: dict[str, Any],
        args: Sequence[str] = (),
        *,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> CaseResult:
        """Write ``config`` in a new case folder and run ``python -m autoclip <args> <config>``."""
        case_dir = self.directory / _slug(name)
        if case_dir.name in RESERVED_CASE_NAMES:
            raise ValueError(f"e2e case {name!r} would collide with the run-level {case_dir.name}/ folder")
        if case_dir.exists():
            raise ValueError(f"e2e case {name!r} already ran in {case_dir}")
        case_dir.mkdir(parents=True)
        config_path = case_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        command = [sys.executable, "-m", "autoclip", *args, str(config_path)]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        env["PYTHONIOENCODING"] = "utf-8"  # the child writes UTF-8 even on a Windows runner
        env.setdefault("PYTHONUNBUFFERED", "1")
        started = _now()
        begin = time.perf_counter()
        timed_out = False
        try:
            done = subprocess.run(
                command,
                cwd=case_dir,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
            returncode, stdout, stderr = done.returncode, done.stdout, done.stderr
        except subprocess.TimeoutExpired as e:
            timed_out = True
            returncode = None
            stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", "replace")
            stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", "replace")
        duration = time.perf_counter() - begin
        stdout_path, stderr_path = case_dir / "stdout.txt", case_dir / "stderr.txt"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        log_name = config.get("logging", {}).get("file") if isinstance(config.get("logging"), dict) else None
        log_path = case_dir / str(log_name) if log_name else None
        result = CaseResult(
            name=name,
            directory=case_dir,
            config_path=config_path,
            command=command,
            started_at=started,
            duration=duration,
            returncode=returncode,
            timed_out=timed_out,
            stdout=stdout,
            stderr=stderr,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            log_path=log_path if log_path and log_path.is_file() else None,
        )
        self.cases.append(result)
        (case_dir / "case.json").write_text(json.dumps(result.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    def record(self, nodeid: str, when: str, outcome: str) -> None:
        """Remember the pytest outcome of one phase (the summary is written when the session ends)."""
        self.outcomes.setdefault(nodeid, {})[when] = outcome

    def write_summary(self) -> None:
        """Collect the rendered clips and write ``run.json``: cases, exit status, outcomes and clips."""
        clips = collect_clips(self)
        data = {
            "started_at": self.started_at,
            "finished_at": _now(),
            "duration_seconds": round(time.time() - self._begin, 3),
            "directory": str(self.directory),
            "cases": [case.to_json() for case in self.cases],
            "outcomes": self.outcomes,
            "clips": clips,
        }
        (self.directory / "run.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    _begin: float = 0.0


def _git_commit() -> str | None:
    """The commit under test, when the results are produced from a checkout."""
    try:
        done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


def _environment(directory: Path) -> dict[str, Any]:
    """What produced this run (versions, platform, flags), written once per results folder."""
    return {
        "started_at": _now(),
        "autoclip_version": __version__,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": _git_commit(),
        "pytest_arguments": sys.argv[1:],
        "flags": {
            "AUTOCLIP_E2E_TESTS": os.environ.get("AUTOCLIP_E2E_TESTS"),
            "AUTOCLIP_NETWORK_TESTS": os.environ.get("AUTOCLIP_NETWORK_TESTS"),
            "AUTOCLIP_E2E_RESULTS": os.environ.get("AUTOCLIP_E2E_RESULTS"),
            "AUTOCLIP_E2E_CACHE": os.environ.get("AUTOCLIP_E2E_CACHE"),
            "AUTOCLIP_E2E_LEVEL": os.environ.get("AUTOCLIP_E2E_LEVEL"),
            "AUTOCLIP_E2E_URL": os.environ.get("AUTOCLIP_E2E_URL"),
            "AUTOCLIP_E2E_COOKIES": os.environ.get("AUTOCLIP_E2E_COOKIES"),
        },
        "directory": str(directory),
    }


@dataclass(frozen=True)
class DownloadedMedia:
    """A real download shared by the pipeline cases."""

    result: CaseResult
    path: Path
    subtitle: Path
    name: str


def download_config(output_dir: Path, url: str, *, time_range: str | None = None, original_dir: Path | None = None) -> dict[str, Any]:
    """A config that downloads ``url`` at the best quality with English subtitles and the metadata sidecar.

    ``original_dir`` is the persistent cache of the ladder: it lives outside the results folder so a later
    run (a deeper level, the weekly network job) reuses the file instead of downloading it again.
    """
    download: dict[str, Any] = {
        "quality": "best",
        "subtitles": {"enabled": True, "languages": ["en"], "use_as_transcript": "always"},
        "metadata": {"info_json": True},
    }
    if time_range:
        download["time_range"] = time_range
    if original_dir is not None:
        download["original_dir"] = str(original_dir)
    cookies = os.environ.get("AUTOCLIP_E2E_COOKIES")
    if cookies:
        download["cookies"] = {"file": cookies}
    return {
        "input": [url],
        "output_dir": str(output_dir),
        "download": download,
        "logging": {"enabled": True, "file": "logs/autoclip.log", "level": "INFO", "console_level": "INFO"},
        "resources": {"max_workers": 1},
    }


def pipeline_config(output_dir: Path, source: Path, transcript: Path, clips: dict[str, Any] | None = None) -> dict[str, Any]:
    """A small, fast pipeline config over an already downloaded local file and its SRT."""
    data: dict[str, Any] = {
        "input": [str(source)],
        "output_dir": str(output_dir),
        "clips": {
            "kinds": ["short"],
            "scenes": ["best"],
            "short": {"count": 1, "min_seconds": 15, "target_seconds": 25, "max_seconds": 45},
        },
        "transcript": {"mode": "file", "file": str(transcript)},
        "video": {"short_side": 144, "preset": "ultrafast", "gpu": "off", "thumbnail": True},
        "audio": {"denoise": "none"},
        "compute": {"device": "cpu"},
        "logging": {"enabled": True, "file": "logs/autoclip.log", "level": "INFO", "console_level": "INFO"},
        "resources": {"max_workers": 1},
    }
    data["clips"] = merge(data["clips"], clips or {})
    return data


def downloaded_files(case: CaseResult) -> list[Path]:
    """Every media file in the case's originals cache (``download.original_dir`` when the config sets one)."""
    from autoclip.core.download import MEDIA_EXTS

    folder = case.directory / "out" / ".work" / "downloads"
    if case.config_path is not None and case.config_path.is_file():
        config = json.loads(case.config_path.read_text(encoding="utf-8"))
        download = config.get("download") if isinstance(config, dict) else None
        if isinstance(download, dict) and download.get("original_dir"):
            folder = Path(str(download["original_dir"]))
            if not folder.is_absolute():
                folder = case.directory / folder
    if not folder.is_dir():
        return []
    return [
        item for item in folder.iterdir() if item.is_file() and item.suffix.lower().lstrip(".") in MEDIA_EXTS and ".media." not in item.name
    ]


def english_subtitle(media: Path) -> Path:
    """The English subtitle downloaded next to ``media`` (a human-made one wins over automatic captions)."""
    folder = media.parent
    manual = sorted(folder.glob("*.manual.en.srt"))
    automatic = sorted(folder.glob("*.auto.en.srt"))
    subtitles = manual or automatic
    assert subtitles, f"no English subtitle next to {media.name}: {[p.name for p in folder.iterdir()]}"
    return subtitles[0]


def video_id(url: str) -> str:
    """The YouTube id in ``url`` (the test id and case folder name)."""
    match = re.search(r"[?&]v=([A-Za-z0-9_-]+)", url)
    return match.group(1) if match else re.sub(r"\W+", "_", url)[-40:]


def read_clips(folder: Path) -> list[dict[str, Any]]:
    """The ``clips.json`` of a rendered kind."""
    data = json.loads((folder / "clips.json").read_text(encoding="utf-8"))
    assert isinstance(data, list), f"{folder / 'clips.json'} is not a JSON list"
    assert all(isinstance(item, dict) for item in data), f"{folder / 'clips.json'} holds a non-object"
    return data


def clip_paths(folder: Path) -> list[Path]:
    """The rendered ``.mp4`` files of a clip folder."""
    return sorted(folder.glob("*.mp4"))


def rendered_clips(case: CaseResult) -> list[Path]:
    """Every rendered ``.mp4`` of a case (downloaded originals under ``.work`` are not clips)."""
    out = case.directory / "out"
    if not out.is_dir():
        return []
    return sorted(path for path in out.rglob("*.mp4") if path.is_file() and ".work" not in path.relative_to(out).parts)


def clip_metadata(clip: Path) -> dict[str, Any]:
    """The ``clips.json`` row that describes ``clip`` (an empty object when there is none)."""
    try:
        rows = read_clips(clip.parent)
    except (AssertionError, OSError, ValueError):
        return {}
    return next((row for row in rows if Path(str(row.get("file", ""))).name == clip.name), {})


def collect_clips(run: E2ERun) -> list[dict[str, Any]]:
    """Copy every rendered clip and its sidecars into ``<run>/clips`` and write the gallery.

    The per-case ``out/`` folders stay the source of truth; this folder is the flat, browsable
    view of what the run produced, next to ``run.json`` and ``environment.json``. Rebuilt from
    scratch on every call so a repeated summary never leaves stale files behind.
    """
    target = run.directory / "clips"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    entries: list[dict[str, Any]] = []
    for case in run.cases:
        out = case.directory / "out"
        for clip in rendered_clips(case):
            relative = clip.relative_to(out)
            destination = target / _slug(case.name) / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(clip, destination)
            sidecars: list[str] = []
            for suffix in (".srt", ".jpg"):
                sidecar = clip.with_suffix(suffix)
                if sidecar.is_file():
                    shutil.copy2(sidecar, destination.with_suffix(suffix))
                    sidecars.append(destination.with_suffix(suffix).relative_to(target).as_posix())
            entries.append(
                {
                    "case": case.name,
                    "kind": relative.parts[0] if len(relative.parts) > 2 else "",
                    "video": relative.parent.name,
                    "file": destination.relative_to(target).as_posix(),
                    "source": str(clip),
                    "size_bytes": destination.stat().st_size,
                    "sidecars": sidecars,
                    "metadata": clip_metadata(clip),
                }
            )
    (run.directory / "clips.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    (target / "index.html").write_text(_gallery(run, entries), encoding="utf-8")
    return entries


def _clip_seconds(metadata: dict[str, Any]) -> str:
    """The final clip length (``trimmed`` is the duration after the pause cuts)."""
    try:
        return f"{float(metadata['trimmed']):.1f}s"
    except (KeyError, TypeError, ValueError):
        return ""


def _short(text: str, limit: int = 220) -> str:
    """One line of transcript, short enough for a card."""
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 3].rstrip() + "..."


def _gallery_card(entry: dict[str, Any]) -> str:
    """One ``<article>``: the video element, its poster and what the clips.json row says."""
    metadata = entry["metadata"]
    name = Path(entry["file"]).name
    url = quote(entry["file"])
    poster = next((sidecar for sidecar in entry["sidecars"] if sidecar.lower().endswith(".jpg")), None)
    poster_attr = f' poster="{quote(poster)}"' if poster else ""
    details = " &middot; ".join(
        html.escape(part)
        for part in (entry["case"], entry["kind"], entry["video"], _clip_seconds(metadata), str(metadata.get("scene", "")))
        if part
    )
    title = html.escape(str(metadata.get("title") or name))
    text = html.escape(_short(str(metadata.get("text") or "")))
    return f"""<article>
<h2>{title}</h2>
<p class="details">{details}</p>
<video controls preload="metadata"{poster_attr}>
<source src="{url}" type="video/mp4">
<a href="{url}">{html.escape(name)}</a>
</video>
<p class="text">{text}</p>
</article>"""


def _gallery(run: E2ERun, entries: Sequence[dict[str, Any]]) -> str:
    """A self-contained HTML page that plays every collected clip of the run."""
    cards = "\n".join(_gallery_card(entry) for entry in entries) or '<p class="empty">No clip was rendered in this run.</p>'
    cases = len({entry["case"] for entry in entries})
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>autoclip e2e clips</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 72rem; padding: 0 1rem; background: #111; color: #eee; }}
h1 {{ margin-bottom: 0.25rem; }}
.summary, .details, .text {{ color: #aaa; }}
main {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(18rem, 1fr)); gap: 1.5rem; }}
article {{ background: #1c1c1c; border-radius: 0.5rem; padding: 0.75rem; }}
article h2 {{ font-size: 1rem; margin: 0 0 0.25rem; overflow-wrap: anywhere; }}
video {{ width: 100%; height: auto; border-radius: 0.25rem; background: #000; }}
.text {{ font-size: 0.85rem; margin-bottom: 0; }}
a {{ color: #6cf; }}
</style>
</head>
<body>
<h1>autoclip e2e clips</h1>
<p class="summary">{len(entries)} clip(s) from {cases} case(s) &middot; {html.escape(run.directory.name)}
&middot; <a href="../run.json">run.json</a></p>
<main>
{cards}
</main>
</body>
</html>
"""


def assert_database_done(output_dir: Path, stem: str) -> None:
    """Every row of the run's SQLite database finished (a pending row would mean a lost worker)."""
    database = output_dir / ".work" / stem / "autoclip.db"
    assert database.is_file(), f"missing {database}"
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT status FROM clips").fetchall()
    assert rows and all(status == "done" for (status,) in rows), rows
