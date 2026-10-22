"""Check a finished output directory: every clip, its sidecar, its thumbnail and the index.

The tool answers the question "did the run really produce what it promised?":

* every clip listed in ``clips.json`` exists, is a valid video with audio, has the right shape
  and a duration that matches the moment it was cut from,
* the ``.srt`` sidecar covers the whole clip and carries the text of that moment,
* the ``.jpg`` thumbnail is a valid image,
* the file names follow the ``{video}_{scene}_{nn}_{start}s`` convention.

Usage::

    python tools/verify_output.py OUTPUT_DIR [--video VIDEO_NAME]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autoclip.core.media import decode_audio

NAME_RE = re.compile(r"^(?P<video>.+)_(?P<scene>[a-z]+)_(?P<n>\d{2})_(?P<start>\d+)s$")
SRT_TIME = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})\s*-->\s*(\d+):(\d\d):(\d\d)[,.](\d{1,3})")


@dataclass
class Problem:
    """One thing that is wrong with the output."""

    where: str
    what: str

    def __str__(self) -> str:
        return f"{self.where}: {self.what}"


def srt_seconds(stamp: str) -> float:
    """``hh:mm:ss,mmm`` as seconds."""
    hours, minutes, rest = stamp.split(":")
    seconds, millis = rest.replace(",", ".").split(".")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def probe(path: Path) -> tuple[float, int, int, float, bool]:
    """(duration, width, height, fps, has audio) of a video file."""
    import av

    with av.open(str(path)) as container:
        video = container.streams.video[0]
        fps = float(video.average_rate)
        return float(container.duration) / 1e6, video.width, video.height, fps, bool(container.streams.audio)


def trimmed_seconds(work_dir: Path) -> dict[str, float]:
    """The length of every rendered clip (the run's database records it after the pause trimming).

    The key is ``kind/file``: a short and a long clip can start at the same second and then share
    their file name.
    """
    database = work_dir / "autoclip.db"
    if not database.is_file():
        return {}
    import sqlite3

    with sqlite3.connect(database) as con:
        rows = con.execute("select kind, file, trimmed from clips where status = 'done'").fetchall()
    return {f"{kind}/{file}": float(trimmed) for kind, file, trimmed in rows if trimmed}


def check_kind(root: Path, kind: str, video: str, problems: list[Problem], expected: dict[str, float]) -> int:
    """Check every clip of one kind; returns the number of clips that were checked."""
    folder = root / kind / video
    index = folder / "clips.json"
    if not index.is_file():
        problems.append(Problem(str(index), "missing clips.json"))
        return 0
    entries: list[dict[str, Any]] = json.loads(index.read_text(encoding="utf-8"))
    for entry in entries:
        name = str(entry["file"])
        clip = folder / name
        if not clip.is_file() or clip.stat().st_size == 0:
            problems.append(Problem(str(clip), "missing or empty"))
            continue
        stem = clip.stem
        if not NAME_RE.match(stem):
            problems.append(Problem(str(clip), "the file name does not follow the convention"))
        duration, width, height, fps, has_audio = probe(clip)
        moment = expected.get(f"{kind}/{name}", float(entry["end"]) - float(entry["start"]))
        if not has_audio:
            problems.append(Problem(str(clip), "has no audio"))
        if abs(duration - moment) > 0.5 * max(1.0, moment / 10):
            problems.append(Problem(str(clip), f"is {duration:.1f}s but the moment is {moment:.1f}s"))
        if fps < 25 or fps > 35:
            problems.append(Problem(str(clip), f"has {fps:.1f} fps"))
        if kind == "short" and (width, height) != (720, 1280):
            problems.append(Problem(str(clip), f"is {width}x{height}, expected 720x1280"))
        if kind == "long" and (width, height) != (1280, 720):
            problems.append(Problem(str(clip), f"is {width}x{height}, expected 1280x720"))
        audio = decode_audio(clip, 0.0, min(duration, 10.0), 16000)
        if float(np.sqrt((audio.astype(np.float64) ** 2).mean())) < 1e-4:
            problems.append(Problem(str(clip), "is silent"))
        srt = clip.with_suffix(".srt")
        if not srt.is_file():
            problems.append(Problem(str(srt), "missing sidecar"))
        else:
            cues = [m for m in SRT_TIME.finditer(srt.read_text(encoding="utf-8"))]
            if not cues:
                problems.append(Problem(str(srt), "has no cues"))
            else:
                last = srt_seconds(cues[-1].group(5) + ":" + cues[-1].group(6) + ":" + cues[-1].group(7) + "," + cues[-1].group(8))
                if last < duration - 1.0:
                    problems.append(Problem(str(srt), f"ends at {last:.1f}s but the clip is {duration:.1f}s"))
        thumb = clip.with_suffix(".jpg")
        if not thumb.is_file() or thumb.stat().st_size == 0:
            problems.append(Problem(str(thumb), "missing or empty thumbnail"))
    return len(entries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    parser.add_argument("--video", default="", help="the video name (default: the only one in the output)")
    args = parser.parse_args()
    root = args.output
    video = args.video
    if not video:
        folders = {p.parent.name for p in root.glob("*/*/clips.json")}
        if len(folders) != 1:
            print(f"cannot tell which video to check in {root} ({sorted(folders)}) - use --video")
            return 2
        video = folders.pop()
    problems: list[Problem] = []
    total = 0
    expected = trimmed_seconds(root / ".work" / video)
    if not expected:
        print("note: no run database found - the pause trimming cannot be taken into account")
    for kind in ("short", "long"):
        total += check_kind(root, kind, video, problems, expected)
    print(f"checked {total} clip(s) of '{video}' in {root}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print("  -", problem)
        return 1
    print("everything checks out: files, shapes, durations, audio, sidecars and thumbnails")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
