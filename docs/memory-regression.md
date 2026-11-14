# Memory regression suite

`tests/memory/` is the opt-in memory-regression suite: it exists to catch the class of bugs where a
fine-looking refactor silently balloons RSS (e.g. `method="compose"` in the unsilence re-encode, which
grew linearly with the number of kept speech segments and OOM-killed a several-hour recording).

It is **opt-in** and **Linux-only**, like the e2e suite. Nothing in it runs during the default
offline `pytest`; `AUTOCLIP_MEMORY_TESTS=1` enables it and the collection hook skips it everywhere else.

## What it measures

Every case runs the real engine in a **child process** (except the advisory ones), polls the parent's
parent-plus-children RSS tree every 10 ms, and compares the peak against a named budget in
`tests/memory/budgets.py`. Subprocess isolation is deliberate: the parent pytest process already holds
numpy, opencv and the transcript fixtures, and budgets measured inside it were meaningless (21-22
samples in a fast in-process run, ±10 % noise between runs).

| Case | Budget | Measured (2026-11-13, 16-core Linux, 3.14) |
| --- | --- | --- |
| M1 `process_video` on the 40 s fixture | 900 MB | ~534 MB |
| M2 unsilence many-parts re-encode (~30 kept spans) | 400 MB | ~93 MB (vs ~4 GB for the `compose` regression) |
| M3 one short clip from a 720p source, `quality=medium` | 2000 MB | ~1161 MB |
| M4 CLI subprocess on the fixture (parent + tree) | 600/900 MB | ~216/320 MB |
| M5 CLI subprocess on the many-parts fixture (parent + tree) | 600/900 MB | ~216/320 MB |
| M9 `enhance_audio` advisory | 512 MB | ~322 MB |

`MIN_SAMPLES` (50) and `MIN_PEAK_MB` (40 MB) are **sanity floors**: a poller that recorded almost no
samples, or a process that never reached 40 MB, fails the case instead of passing every budget.

## Running it

```bash
AUTOCLIP_MEMORY_TESTS=1 pytest tests/memory --no-cov   # Linux only
make test-memory                                       # the same
```

Each run writes `tests/results/memory/<timestamp>/metrics.json` (wall time, peak parent/tree RSS
against the budget, per case). The fixtures are generated on the fly (`lavfi`), so the suite needs no
network and no downloaded model; subtitles are provided through sidecar `.json` transcripts.

## Advisory cases

`test_memory_advisory.py` holds wall-clock expectations (the fixture CLI run must stay under the
advisory wall budget) and a schema check for `metrics.json`. The enhancement check in it is advisory:
the budget is generous and exists to catch order-of-magnitude regressions, not noise.

## Recalibrating a budget

When a change legitimately raises memory demand, measure the new peak, update the budget **and** the
dated measurement note in `budgets.py` in the same commit, and say why in the commit message. Never
recalibrate a budget to green without a measurement. Budgets should never be tightened below a
measured value on this machine; a budget that starts flapping is a bug in the meter, not in the budget.

## Known limits

- RSS is sampled at 10 ms; a peak that lives for less is invisible (the engine does nothing with
  micro-bursts).
- The tree RSS counts children reparented to init as zero if they are reaped before the poll; ffprobe
  and ffmpeg children are short-lived but the poll is tight enough in practice.
- It is calibrated on Linux only; macOS/Windows runners skip the suite.
