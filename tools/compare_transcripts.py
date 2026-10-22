"""Compare two transcripts of the same video, aligned by time."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, "/home/rayen/Documents/content-tools/autoclip-project/tools")

from compare_stt import char_error_rate, script_ratio


def load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = data["segments"]
    return data["meta"], segments, " ".join(s["text"] for s in segments)


def aligned_cer(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> tuple[float, int, int]:
    """Mean CER over the pairs of segments that overlap in time, and the drift counter."""
    scores: list[float] = []
    drift = 0
    compared = 0
    j = 0
    for sa in a:
        while j < len(b) and b[j]["end"] <= sa["start"]:
            j += 1
        if j >= len(b):
            break
        sb = b[j]
        if sb["start"] > sa["end"]:
            continue
        compared += 1
        scores.append(char_error_rate(sa["text"], sb["text"]))
        a_ar, a_la = script_ratio(sa["text"])
        b_ar, b_la = script_ratio(sb["text"])
        if (a_ar > a_la) != (b_ar > b_la) and min(a_ar + a_la, b_ar + b_la) > 3:
            drift += 1
    return (sum(scores) / len(scores) if scores else 0.0), compared, drift


def main() -> int:
    v1 = Path("runs/jutilities-package/.work/jutilities-package/jutilities-package.transcript.json")
    v2 = Path("runs/jutilities-package-v2/.work/jutilities-package/jutilities-package.transcript.json")
    meta1, seg1, text1 = load(v1)
    meta2, seg2, text2 = load(v2)
    print("v1:", {k: v for k, v in meta1.items() if k != "source"})
    print("v2:", {k: v for k, v in meta2.items() if k != "source"})
    print()
    for name, segs, text in (("v1 (small)   ", seg1, text1), ("v2 (large-v3)", seg2, text2)):
        ar, la = script_ratio(text)
        share = 100 * ar / max(1, ar + la)
        print(f"{name}: {len(segs):4d} segments, {len(text):6d} chars, arabic {share:5.1f}%, latin {la:5d}")
    mean, compared, drift = aligned_cer(seg1, seg2)
    print(f"\ntime-aligned segments compared: {compared} | mean CER {mean:.3f} | script drift in {drift} of them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
