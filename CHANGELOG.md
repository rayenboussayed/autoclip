# Changelog

All notable changes to autoclip, oldest to newest. Release timestamps come from
Git tags (and commit metadata for untagged but version-stamped commits). Each
release mentions the source tag/commit for traceability.

## [4.0.0] — 2026-11-06 14:15:01 +0100 (commit `92d47c7`)

- feat: reduce the config file to nine keys (seven then two: `llm.system_prompt` and `image.prompt`); everything else is derived at run time or passed as a runtime override
- feat: invalid config *values* now warn and fall back to the default instead of aborting the run
- feat: remove the LLM api_key field (Ollama is local and needs no API key)
- feat: the file sets the LLM system prompt (`llm.system_prompt`, default empty = the built-in editor messages) and the Stable Diffusion thumbnail prompt (`image.prompt`)
- feat: Whisper model downloads run automatically on first use; the consent prompt and `consent` option are gone
- feat: every Ollama call has a 180 s timeout and an 8k context window (a server-side overflow silently truncated the prompt before)
- feat: the LLM re-rank constrains the reply with a JSON schema and validates it (scores clamped to 0-10, titles cut to 80 characters, ids outside the candidate list dropped)
- feat: the run asks Ollama once for its model list (`/api/tags`); a missing model disables the AI helpers instead of failing inside every clip
- fix: a worker killed by the OS (out of memory) is no longer counted as finished before the in-process retry runs — progress could over-count otherwise
- fix: `record_job` now folds the current RSS of the process into the peak history; the adaptive budget no longer sees only the synthetic utilization of in-process runs
- feat: a resource-admission gate in the render pool: before each submission the engine re-checks memory pressure and, when a job is in flight, holds back new submissions until a clip finishes (never starves an idle batch); an OOM-killed worker defers the next submission too
- feat: the engine now reports real peak RSS into its resource history instead of a synthetic 0, so the adaptive CPU budget is fed by measurement, not by threads x workers / cores
- feat: the AI helpers share a per-run circuit breaker: one failed Ollama call (timeout, missing model, off-schema reply) disables them for the rest of the run with one log line, so a hung server costs one 180 s stall instead of N — the heuristic result is kept
- chore: the dead EMA helper was removed; the `silence_threshold` runtime decision is labelled accurately; persisted utilization samples are restored on configure()
- refactor: the desktop app lost the unused control kinds and the advanced/visible_when machinery; the option catalogue stays exactly the nine file keys
- feat: a sidecar `.json`/`.srt` next to the video (`<stem>.json` preferred) is the transcript; `--transcript`, `AUTOCLIP_TRANSCRIPT` and `transcript_file` are gone
- feat: downloaded subtitles are fetched for the video's original language; the `subtitles` option is gone
- feat: language for downloads always follows the video's original track
- feat: explicit device selection is removed; GPU use is probed and automatic (no `--device` flag)
- feat: `--offline` is removed; AI is controlled by `--skip llm` and Whisper downloads are automatic
- feat: the proxy is gone entirely (`--proxy`, `AUTOCLIP_PROXY`, `download.proxy`); downloads use the network directly
- feat: `--log-level`, `AUTOCLIP_LOG_LEVEL` and the `logging` key are gone; logging is internal: DEBUG in debug mode (`AUTOCLIP_DEBUG=1`), INFO otherwise
- feat: logging of file and console is always on (no `OFF` level; the internal level suffices)
- feat: `TitleCfg` and `LoggingCfg` are removed; the title overlay is controlled by `--skip title` alone
- feat: `download.platforms.allow`/`deny` are now `download.allow`/`download.deny` (the `FileDownloadCfg` wrapper is gone); the frozen layout is exactly nine keys
- feat: optional Ollama re-ranking, titles and social captions through one serialized AI queue
- feat: optional Stable Diffusion thumbnails (serialized in the same queue; frame JPEG kept otherwise)
- feat: social captions as `<clip>.txt` sidecars and `clips.json` entries
- feat: URL-shape playlist handling (single for watch/shorts/video URLs, all otherwise, capped at 50)
- feat: automatic download format policy (quality tier, direct HTTPS over HLS/DASH, original audio track); at equal quality the least-burdensome bitrate is chosen
- feat: a stalled or failing download no longer costs the video: after the preferred format fails, the downloader tries the next-best candidate(s) (up to two alternates) before giving up; a different candidate reuses its cache when it is already present; the chosen format is logged with its resolution, codec, bitrate and protocol
- feat: automatic silence removal, speech enhancement, captions, titles and thumbnails
- fix: a video without an audio track no longer crashes the transcription stage (it is transcribed-with-skip instead)
- refactor: runtime `Config` keeps every field; `FileConfig` owns the file schema; `RuntimeOverrides` for CLI/env/app/server
- docs: rewrite README/AGENTS/docs for the nine-key schema and the 4.0 upgrade notes
- chore: bump version to 4.0.0

## [2.1.0] — 2026-10-24 14:01:53 +0100 (tag `v2.1.0`, commit `780850e`)

- feat: speech enhancement before transcription and a global device switch
- feat: clip lengths as minimum, recommended and maximum
- fix: probe the encoder with the ffmpeg MoviePy writes with
- feat: validate the config file with Pydantic instead of hand-written checks
- refactor: split the package into `core`, `cli` and `app`, including the desktop app
- fix: the output check follows the run, not a fixed config
- test: reach 100% branch coverage again
- feat: build standalone binaries for Linux, macOS and Windows (x64/arm64)
- chore: run ruff, mypy and the test suite on every commit
- feat: ship the Linux build as an AppImage and a snap
- fix: type-check against the stubs of the oldest supported dependencies
- fix: make the snapcraft recipe valid for core24
- test: pin multilingual in the CPU transcription test
- fix: make desktop app tests independent of host operating system
- Report CI failures as annotations the runner can parse
- Serialise log writes on Windows, where appends are not atomic
- Capture pytest output at the Python level, not the fd level
- Build the check video with the bundled ffmpeg; drop Windows on ARM
- Release on a version bump instead of a tag push
- Resolve the AppImage tool and AppDir before changing directory
- Make the AppImage path fix portable; surface snap failures
- Run snapcraft as root so destructive mode can install its snaps

## [2.1.1] — 2026-10-24 15:09:50 +0100 (tag `v2.1.1`, commit `e827ac8`)

- Detect the language per segment by default

## [2.2.0] — 2026-10-27 12:10:58 +0100 (tag `v2.2.0`, commit `947dadd`)

- Release 2.2.0

## [2.3.0] — 2026-10-29 10:18:05 +0100 (tag `v2.3.0`, commit `fb6c192`)

- Assert the PyAV time base before converting it
- Start from in-memory defaults when no config file exists
- Add unit tests for transcript parsing and version consistency
- Add unit tests for E2E harness and support functions
- Refactor E2E tests and enhance manifest handling
- Implement job management and web interface for video processing
- Fix server test hang and make background work loop-safe
- Refine the HTTP server, download format choice and E2E ladder
- Make server and E2E harness tests platform-independent (Windows CI)
- Remove unused pathlib import in server websocket test

## [3.0.0] — 2026-10-29 21:36:45 +0100 (tag `v3.0.0`, commit `e3b28d2`)

- autoclip 3.0: refactor pipeline, subcommand CLI, direct-call app, packaging/installer

## [3.0.1] — 2026-10-30 07:55:34 +0100 (tag `v3.0.1`, commit `c2beed2`)

- Fix ruff SIM105 / lint in new tests (CI)
- Fix a frozen build without arguments opens the desktop app

## [3.0.2] — 2026-10-30 15:24:44 +0100 (`fix(3.0.2): section 4`, commit `5fe48b6`, merged)

- fix(3.0.2): render, cancel, bundles, transcript, process coverage
- fix(3.0.2): scoped reporter, render progress/cancel, supervision, stdin, doctor, sherpa
- fix(3.0.2): version 3.0.2, CLI help and init errors
- feat: GUI consent dialog and transcript consent config option
- feat(app): bounded resizable panes (20-80%) and taller footer buttons
- feat: vulture and deptry quality gates; finish Whisper fallback packaging
- Add pydub/MoviePy unsilence preprocessing and config option
- Harden worst-case paths: ffmpeg wrapper retries, robust pool shutdown,
  server archive/eviction, whisper download retries
- Reach full coverage; fix lint/mypy/vulture/deptry checks

## [3.0.5] — 2026-11-01 17:53:31 +0100 (commit `c02cdb8`)

- Gracefully fall back when noisereduce is unavailable
- Fix impossible source assertion error type
- Fix HEVC rendering of composited captioned clips
- Decode unsilence audio without ffprobe; fix installer job
- Fix the curl readiness flag in the installer job
- Make the snap build non-blocking and robust

## [3.0.8] — 2026-11-03 12:00:13 +0100 (commit `924d1bf`)

- Add clip processing mode, unsilence-by-default and OS videos output default

## [3.0.9] — 2026-11-03 13:44:29 +0100 (commit `74d2c42`)

- 3.0.9: default half of CPU cores, sequential, lighter OOM-safe renders
- Tests: adapt defaults to half CPUs and coverage

## [3.0.10] — 2026-11-03 18:31:40 +0100 (commit `23cfb77`)

- 3.0.10: logger_decorator by spec + stdout tee to `.log`
- 3.0.10: selected wrapping, v3.0.10 release

## [3.0.11] — 2026-11-05 07:05:09 +0100 (version bump + merged operation decorator)

### Added

- System health logging (`autoclip/core/log.py`) exposing `redact_url`, `safe_path_text`,
  `system_profile` and `health_check`; “system health before X” lines emitted before each
  pipeline stage (input, preprocess/unsilence, transcript, selection, render, package,
  download, application).
- Input metadata, downloader, and output-file metadata logging with paths and sidecars.
- New tests in `tests/unit/test_infra.py::test_log_redaction_helpers` plus updated
  `test_download` assertions for the new output lines.

### Changed

- `batch.py`, `pipeline.py`, `runner.py`, `render.py`, `download.py`, `unsilence.py`,
  `app/session.py` and `cli/__init__.py` now emit safe-path and health lines.
- `batch.py`: per-input `system health before input X/Y` plus verbose per-input logging.
- `log.py`: safe import of `MB` and `cgroup_memory_available` inside `health_check`;
  all error handling uses `LOG`, avoiding undefined names inside `system_profile`.

### Fixed

- `download.py`: redundant cache block removed; correct handling of `cached` early-return.
- `log.py`: exception logging inside `system_profile` now uses the module logger.
- CLI/app `input` logging expanded to show all supplied items with metadata.

### Tests / quality

- Full `pytest` suite passes locally (~940+ tests); `ruff`, format, and `mypy` are all
  clean after the change.
- Coverage remains ≥ 95% (current repository target is 100%).
- Files modified: ~25, 700+ insertions / 100+ deletions vs `main`.
- Unified `logger_decorator` and `traced` into one universal `traced` decorator: direct use or with a label,
  with `[START]` / `[END]` prints and structured `START` / `END` / `FAIL` logging on one path. Removed the
  duplicated decorator stack and updated every call site.
- Every production function (including private helpers) is now wrapped with `traced`; nested calls are
  intentionally not double-logged so the inner tree does not swamp the outer operation's lines.
- `.log` records are flushed and `os.fsync()` is called on the file descriptor before the caller regains
  control, so completed log lines are durable even after a crash.
- New `docs/logging.md`: logging architecture, redaction/privacy rules, durability guarantees and verification
  recipes.
- Release script now expects `CHANGELOG.md`, `README.md`, `AGENTS.md` and `docs/` to be updated together.
- Tests: `test_logging_file_syncs_to_disk` proves the file handler syncs to disk.
