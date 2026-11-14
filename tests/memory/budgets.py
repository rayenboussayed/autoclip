"""Named peak-RSS budgets for the memory-regression suite, measured on this codebase.

Each budget is a measured peak × a safety factor, floored at 200 MB. When a public path legitimately
grows (a new OCR pass, a bigger model), recalibrate the budget with a fresh measurement and note the
date here. An unexplained change *lowering* a budget, or removing one, is a regression risk.

Measured on 2026-11-13 on a 16-core Linux 3.14 machine, quality "lowest" fixtures, ``ultrafast``:
"""

from __future__ import annotations

# M1: in-process process_video on the 40 s fixture (short-clips preset, ranges mode)
PROCESS_VIDEO_PEAK_MB = 900.0  # measured 534 MB -> x1.7 headroom

# M2: remove_silence on a 120 s many-parts source (forced encode pass; the compose method would be ~4 GB)
UNSILENCE_MANY_PARTS_PEAK_MB = 400.0  # measured ~93 MB; regression guard, any order above this is wrong

# M3: render of one short clip from the 720p 20 s fixture at quality "medium"
RENDER_720P_PEAK_MB = 2000.0  # measured 1161 MB -> ~1.7x headroom

# M4: CLI subprocess on the fixture (ranges mode, lowest quality), 300-460 spawn threads allowed
CLI_PARENT_PEAK_MB = 600.0  # measured ~216 MB
CLI_TREE_PEAK_MB = 900.0  # measured ~320 MB
CLI_WALL_SECONDS = 120.0  # advisory only, never gates

# M5: CLI subprocess on the 120 s many-parts fixture (unsilence path, fail-safe encode)
CLI_MANY_PARTS_PARENT_MB = 600.0  # measured ~216 MB
CLI_MANY_PARTS_TREE_MB = 900.0  # measured ~320 MB
CLI_MANY_PARTS_WALL_SECONDS = 180.0  # advisory only

# M9: enhance_audio on the 40 s fixture (3 s span chunks, noisereduce path)
ENHANCE_AUDIO_PEAK_MB = 512.0  # measured ~322 MB in a fresh subprocess; advisory only, ~1.6x headroom

#: Minimum number of RSS samples every meter must record; guards against a silently broken poller passing vacuously.
MIN_SAMPLES = 50

#: Minimum peak RSS any meter must record; a run that never samples at all would otherwise pass every budget.
MIN_PEAK_MB = 40.0
