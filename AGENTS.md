# AGENTS.md - autoclip

Guidance for AI coding agents (and humans) working on this repository. Read it fully before changing code; the
"Invariants" and "Gotchas" sections record decisions and failures that are not obvious from the code.

## 1. What this project is

`autoclip` turns long videos into the best **short** (default 9:16, 20-60 s) and **long** (default 16:9, 3-10 min)
clips, per requested **scene type** (`best`, `funny`, `sad`, ... or `"all"`). One config file drives everything.
Pipeline: transcribe (faster-whisper or .srt/.json) -> score sentence-aligned windows per scene -> pick the best
non-overlapping ones -> render (enhance audio, cut pauses/fillers, layout, burn captions, encode on GPU if one really works).

* Free, local, no service dependencies (optional local LLM re-rank over HTTP).
* Runs on Linux, macOS and Windows, Python >= 3.10 (developed and verified on 3.12). No system ffmpeg required.
* Three parts: `core` (the engine), `cli` (the command line) and `app` (the desktop app, Tk).
* Version 2.1.0. ~5,200 lines of package code in 28 modules, ~3,700 lines of tests (284 tests, 100% statement + branch coverage).

## 2. Commands

```bash
pip install -r requirements-dev.txt            # runtime + dev tools (pytest, pytest-cov, mypy, ruff)
pip install -U "Pillow>=12.3"                  # security fixes; MoviePy's "<12" pin is conservative (CI runs both)
pip install -e .                               # optional: gives the `autoclip` console command

python -m autoclip --init config.json          # tiny starter config      (--init-full: every option + default)
python -m autoclip config.json                 # run (exit 0 = all ok, 1 = something failed, 2 = bad config)
python -m autoclip --gui                       # the desktop app (same as: python -m autoclip.app)

python -m PyInstaller autoclip.spec --noconfirm  # standalone build (needs pyinstaller; see the build workflow)
python tools/make_appimage.py                   # the AppImage (Linux) from that build
sudo snapcraft --destructive-mode               # the snap (Linux) from that build (needs root)
pre-commit install                              # ruff, mypy and the test suite on every commit
ruff check . && ruff format --check .          # lint + format (line length 140)
mypy autoclip tests tools                       # strict typing, tests and tools included
pytest                                         # whole suite; FAILS if branch coverage < 95%
pytest tests/test_moments.py -k scene --no-cov # a focused run (always add --no-cov for partial runs)
```

**Definition of done** (all four must pass before you finish): `ruff check .`, `ruff format --check .`,
`mypy autoclip tests tools`, `pytest`. Coverage is currently 100%; the gate is 95% and must never be lowered.
A full `pytest` takes about one minute (real ffmpeg encodes, real worker processes).

**Release:** bump `version` in `pyproject.toml`, `autoclip/__init__.py` and `snapcraft.yaml` (they must match -
`tests/test_version.py` checks it) and push to `main`. The build workflow checks that `v<version>` is not tagged
yet, builds the five targets, then creates the tag and publishes the GitHub release. A push that does not change
the version skips the build; a pull request builds without releasing.

## 3. Repository layout

```
autoclip/
  __main__.py     python -m autoclip entry (freeze_support for frozen Windows builds)
  core/           THE ENGINE - no user interface, heavy imports stay inside functions
    config.py       typed config dataclasses (Pydantic validates them), strict loader, input globbing
    pipeline.py     process_video (probe -> transcript -> moments -> render), split_video, ranges_moments
    styles.py       DATA ONLY: STYLES (scene weights), STYLE_HELP, LEXICONS
    models.py       small slotted dataclasses: Span/Word/Segment/Moment, MediaInfo, Encoder, RenderJob
    errors.py       AutoclipError, ConfigError
    resources.py    ResourceMonitor + RESOURCES singleton: CPU/RAM -> worker plan, cache budget
    log.py          LOG, setup_logging, traced decorator, speed_text
    store.py        SQLite Store (WAL), MemoryLRU, Cache (RAM LRU in front of SQLite kv), is_boundary
    media.py        ffmpeg discovery, probe (PyAV), energy track, decode_audio, scene cuts, snapping
    audio.py        enhance_audio (denoise/loudness/limiter), compute_keep (pauses/fillers), Timeline, build_audio
    transcript.py   parse srt/json, faster-whisper (chunked for long videos), disk cache, get_transcript
    denoise.py      speech enhancement before transcription: SpeechEnhancer (sherpa-onnx ONNX | noisereduce)
    moments.py      candidates, vectorised features, per-scene scoring, pick(), LLM re-rank, select_moments
    hardware.py     GPU/CPU encoder choice: probe by really encoding; Encoder builders; HW_MAX_WORKERS
    fonts.py        cross-platform font discovery + glyph coverage (fontTools)
    captions.py     text sprites (balanced wrapping, word highlight), CaptionTrack, build_captions
    render.py       layout, OpenCV resize/blur, encode, render_clip
    runner.py       RenderJob execution: render_job (never raises), render_all (adaptive process pool)
  cli/            THE COMMAND LINE - python -m autoclip config.json
    __init__.py     argument parsing, main(), --init / --init-full, --gui
  app/            THE DESKTOP APP - python -m autoclip.app (Tk, native look)
    fields.py       the catalogue: every config option with what / why / when (the app is built from it)
    form.py         the form: one row per option, the control that fits its kind, the scroll area
    session.py      the behaviour without widgets: read/write/validate the config, start/stop the pipeline
    window.py       the window: tabs, details panel, log, Run / Stop / Save / Load / Reset
tests/            mirrors the modules; conftest.py generates the fixture video
                  test_version.py guards the three version files
autoclip.spec     the PyInstaller spec: a standalone build (engine + command line + desktop app)
snapcraft.yaml    the snap: the same build as a snap (linux, amd64 + arm64)
assets/           the icon and the desktop file of the AppImage and the snap
tools/            frozen_entry.py - the entry point of the frozen build (a frozen __main__ has no package around it)
                  make_appimage.py - builds the AppDir and the AppImage from the PyInstaller build
                  make_icon.py     - draws the icon (assets/autoclip.png)
                  compare_stt.py   - development aid: STT models x enhancement backends on one scene
                  verify_output.py - development aid: checks a finished output directory
docs/             stt-comparison.md - the measured comparison of transcription models and denoisers
AGENTS.md        this file (guidance for coding agents)
pyproject.toml    deps, ruff, mypy(strict), pytest(+coverage gate), coverage(multiprocessing)
.github/workflows/ci.yml   ubuntu/macos/windows x py3.10/3.12, then re-run with Pillow>=12.3
```

### Import layers (no cycles - keep it this way)

```
errors, styles, fonts              leaves
config            <- errors, styles, pydantic
models, resources <- config
log               <- resources
media             <- errors, log, models
store             <- models, resources
hardware, audio   <- config, log, media, models (+store for transcript/moments)
denoise           <- config, errors, media
transcript, moments, captions, render <- the layers above
runner            <- hardware, log, models, render, resources, store
pipeline          <- everything of the engine (the per-video run)
cli               <- core.pipeline, core.config, core.errors, core.hardware, core.log, core.resources
app               <- core.config, core.errors (+ tkinter)
```
`RenderJob` lives in `models.py` (not `runner.py`) precisely to avoid a `render <-> runner` cycle; `Encoder` is in
`models.py` and its logic in `hardware.py` to avoid `models <-> media`. If you need a new shared type, put it in `models.py`.

Heavy libraries are imported **inside functions** (`moviepy`, `faster_whisper`, `cv2`, `noisereduce`, `scenedetect`,
`imageio_ffmpeg`, `arabic_reshaper`, `bidi`) so `--init`, config errors and unit tests stay fast and optional extras
degrade gracefully. Keep that pattern (ruff rule `PLC0415` is intentionally ignored).

## 4. Data flow

```
main()
  load_config -> setup_logging -> RESOURCES.configure -> pick_encoder (once per run; probes GPUs)
  for each video in cfg.input (isolated: one failure never stops the batch):
    process_video(src, cfg, log_file, encoder)
      probe(src)                                   PyAV: duration, audio?, size, fps
      mode == "split": ffmpeg stream-copy parts -> {out}/long/{video}/   (done)
      Store(.work/{video}/autoclip.db, fresh=True) + Cache
      energy_track(src)                            0.5 s loudness track (cached in kv)
      get_transcript(...)                          file | disk cache | faster-whisper (in 5 min chunks) -> SQLite
                                             (each chunk is enhanced by `SpeechEnhancer` before it is transcribed)
      for kind in cfg.clips.kinds ("short", "long"):
        moments = ranges_moments(...)  if mode == "ranges"   (scene "custom")
                  select_moments(...)  otherwise              (one pass per scene type)
        no moments -> warning, kind skipped (all kinds empty -> AutoclipError)
        Store.save_moment(...) -> clip_id ; build RenderJob (dst = {out}/{kind}/{video}/{video}_{scene}_{nn}_{start}s.mp4)
      render_all(jobs)                             adaptive pool; results land in the `clips` table
      write {out}/{kind}/{video}/clips.json from the DB rows with status == "done"
```

### select_moments (per kind)
1. `text_candidates`: every run of consecutive transcript segments that starts and ends on a sentence boundary and
   fits the kind's `[min_seconds, max_seconds]` (ignores segments ending after the media duration - a subtitle file may
   be longer than its video). Falls back to `audio_candidates` (windows every 2 s) with a warning when none exist.
   `target_seconds` (the recommended length) is a preference, not a limit: `length_bonus` gives candidates near it a
   small score bonus, so the clips tend to have that length while the scene type still decides the content.
2. Features are computed **once per kind** as NumPy arrays using prefix sums (O(1) per candidate regardless of window
   length): `audio_features` (energy, variety, pause) + `text_features` (rate, punct, digits, hook, one per lexicon, `kw`).
3. For each scene: `scene_score` = sum of `weight * zscore(feature)` from `STYLES[scene]` (+ `kw` weight 1.5 if
   `clips.keywords`), then `pick()` greedily takes the best candidates that overlap neither each other nor `taken`
   windows of earlier scenes (1 s gap). **Scene order therefore means priority.**
4. Optional LLM re-rank per scene (`_rerank`: `0.4*z(heuristic) + 0.6*(llm-5)/2`; any failure keeps heuristic order).
5. Padding (`pad_start/pad_end`) for text picks; audio-only picks snap to quiet frames; optional snap to shot cuts.

Candidates are **not stored in SQLite** (an earlier design did; it exploded to millions of rows for long clips x scenes).

### render_clip
`enhance_audio` (highpass, denoise in 60 s pieces, loudness over speech frames, soft limiter) -> `compute_keep`
(shorten long pauses, cut fillers; skipped without audio) -> clamp sub-clips to the real stream length -> `layout`
(fit on blurred/colour/image background, or fill) -> `build_captions` (sprites cached, one band track) -> optional title ->
fades -> `encode` with the chosen `Encoder` -> JPEG cover. `render_job` writes the `.srt` sidecar and the DB outcome.

`compute_keep` gets the kind's `min_seconds`: when the trimming would cut the clip below it, the smallest cuts are
given back (the fillers first), so a clip never ends up shorter than the user asked for.

### Parallelism
`render_all` plans workers from `RESOURCES.plan_workers(per_job_bytes, jobs, limit)` **before every submission**.
`limit = HW_MAX_WORKERS (3)` for GPU encoders. One usable core (or one job) runs in-process; otherwise a
`ProcessPoolExecutor` with the `spawn` context (same on every OS). A worker killed by the OS (`BrokenProcessPool`) is
retried alone, in-process. Workers re-run `setup_logging(log_file, level, None)` and `RESOURCES.configure(cfg)`.

## 5. Invariants - do not break these

1. **Fresh database per video, every run.** `Store(..., fresh=True)` deletes the DB and its `-wal/-shm/-journal`.
   The only thing persisted between runs is `{video}.transcript.json` (keyed by source name/size/mtime, model, language,
   multilingual, word_timestamps and the enhancement settings; a mismatch re-transcribes).
2. **Results flow through SQLite, not return values.** `render_job` returns a bool but the truth is the `clips` row
   (`pending` -> `done`/`failed`). `clips.json` is built from rows. A row still `pending` after `render_all` means the
   worker never finished and is reported as failed.
3. **`render_job` never raises** (except `KeyboardInterrupt`): a failed clip must not stop the batch. `process_video`
   exceptions are caught per video in `main` (`AutoclipError` is logged without a traceback, anything else with one).
4. **Global selection, segmented transcription only.** The transcript is produced in chunks of
   `processing.chunk_minutes` (5) - never more audio in memory than that, and the same chunks are enhanced and
   transcribed. Selection stays global so clips never cut at a chunk edge and a long clip may exceed a chunk. Do not
   "simplify" this into physically splitting the file.
5. **`RESOURCES` is a process-wide singleton** that is *reconfigured*, never rebound (`RESOURCES.configure(cfg)`);
   modules import the object once. Memory/CPU balancing happens in `traced` operations (`after_operation`), before each
   render submission, and inside `Cache.trim`. Trim hooks are best-effort and swallow exceptions (logged at DEBUG).
6. **GPU encoders are used only if proven.** `hardware.works()` really encodes a 0.4 s test clip with the exact arguments
   MoviePy will use (preset first, then `Encoder.params`) *and the binary MoviePy writes with* (`encoding_ffmpeg()`, not
   the `PATH` ffmpeg used for decoding). Listing an encoder in `ffmpeg -encoders` proves nothing.
   `compute.device: "cpu"` keeps the GPU out of the run entirely (the encoder and the transcriber).
7. **Config is strict.** Unknown keys, wrong JSON types and invalid values raise `ConfigError` at load; `"_comment"`
   keys are ignored. All defaults live in the dataclass fields (single source of truth for `--init-full`).
8. **Outputs:** `{output_dir}/{short|long}/{video}/` holds only clips (+ `.jpg`, `.srt`, `clips.json`);
   `{output_dir}/.work/{video}/` holds the DB and transcript cache; `{output_dir}/logs/` holds run logs.
   File names: `{video}_{scene}_{nn:02d}_{int(start)}s.mp4` (`nn` counts per kind+scene). Two inputs with the same file
   stem are rejected at load time (they would share outputs).
9. **Words/segments must be shorter than `store.MAX_SPAN` (120 s)**: window queries only look that far back.
10. **No work on the GPU except encoding (and Whisper via CUDA).** Do not claim end-to-end GPU acceleration.
11. **The desktop app shows every config option.** `app/fields.py` is the catalogue: one entry per option with
    *what* it does, *why* you would change it and *when*. `tests/test_app.py` compares it with the config
    dataclasses, so a new option that is not in the catalogue (or an entry that is not in the config) fails
    the suite. The app writes the same config file the command line reads - never a second format.

## 6. Configuration (`config.py`)

`Config` = nested dataclasses: `input` (list; files or globs), `output_dir`, `clips` (`mode`, `kinds`, `scenes`,
`short: KindCfg`, `long: KindCfg`, `keywords`, `lexicon_file`, `skip_intro_seconds`, `snap_to_cuts`, `pad_*`,
`split_seconds`), `transcript`, `llm`, `video` (`short_side`, `fps`, `layout`, `background`, `fade_seconds`, `gpu`,
`crf`, `preset`, `maxrate_mbps`, `thumbnail`), `captions`, `title`, `audio`, `silence`, `processing`, `resources`,
`storage`, `logging`. `KindCfg` = `count` (per scene), `min_seconds`, `target_seconds` (recommended, optional),
`max_seconds`, `aspect`, `ranges` - the length of a clip is `clip(minimum, recommended, maximum)`.

* **Pydantic validates the file** (`pydantic.dataclasses.dataclass` + `ConfigDict(extra="forbid")`): an unknown key, a
  wrong type or an out-of-range value becomes a `ConfigError` (`parse()` wraps the `ValidationError`). Numeric fields use
  `Annotated[int|float|bool, Strict()]` so JSON keeps its type (a bool is not an int, a string is not a number); formats
  use `StringConstraints(pattern=...)`; cross-field rules (min <= max, scene names) are `model_validator(mode="after")`.
  Keys starting with `_` are ignored, so a config file can carry `"_comment"` entries. **Do not hand-roll type checks.**
* `load_config` resolves every path relative to the config file, expands `input` globs (sorted, de-duplicated, absolute),
  and requires `{stem}` in `transcript.file` when several videos are given.
* `scenes`: `"all"` | one name | list; `ClipsCfg.scene_list()` expands it. `"scenes"` means the scene *types* (the keys of
  `STYLES`); shot-change snapping is `snap_to_cuts` (renamed from `snap_scenes` to avoid the clash).

### Adding a config option (checklist)
1. Add a typed field with a default (and a comment) to the right dataclass. Put the constraint in the type
   (`Field(ge=..., le=...)`, `StringConstraints(pattern=...)`, `Annotated[int, Strict()]`) or, for a rule that spans
   fields, in a `model_validator(mode="after")` that raises `ConfigError` - never in a hand-written type check.
2. Use it from code via `cfg.<section>.<field>` (never read raw JSON elsewhere).
3. Tests: valid + invalid case in `tests/test_config.py` (the `--init-full` round-trip test already guards defaults), plus a behavioural test.
4. If it changes behaviour users see, update `README.md` (and the upgrade notes if it is breaking).

## 7. Conventions

* **Typing:** `mypy --strict` on package *and* tests *and* tools. Use `X | None`, `list[...]`, `from __future__ import annotations`.
  Models are `@dataclass(slots=True)` (no `TypedDict`, no raw `dict[str, Any]` for domain data; `Any` only for MoviePy
  objects). NumPy arrays are `NDArray[np.float32]` etc. (`F32`, `F64`).
* **Docstrings:** every public function/class has one (what, args that are not obvious, returns). Comments explain *why*.
* **Logging:** use the `autoclip` logger (`log = logging.getLogger("autoclip")`). Wrap meaningful operations in
  `@traced("name")` (START/END with wall time, RSS, free RAM, CPU, pressure; FAIL with the exception). Use `speed_text`
  for throughput. Never log the LLM API key (`LlmCfg.api_key` has `repr=False`).
* **Errors:** raise `AutoclipError` (or `ConfigError`) for expected, user-fixable problems; let real bugs raise.
  Never `except:`; catching `Exception` is allowed only at isolation boundaries (per video, per clip, best-effort hooks) and must log.
* **The desktop app is native Tk:** `ttk` widgets with the platform theme (`system_theme()`), no custom colours
  beyond the grey of the help lines, no third-party GUI toolkit. The behaviour (config, process) stays in
  `app/session.py` so it can be tested without a display; `app/window.py` only draws and wires.
* **Minimal and DRY:** no dead code, no unused parameters/types, no speculative options. Prefer deleting to adding.
  Modules stay small and single-purpose (largest is ~360 lines). Prefer vectorised NumPy over Python loops for per-candidate work.
* **Style:** ruff (rules E,F,I,B,UP,S,SIM,RUF,PL; bandit `S` enforced), line length 140, `ruff format`.
* **Cross-platform rules:** `pathlib` everywhere; open text files with `encoding="utf-8"`; no `fork`, no POSIX-only
  APIs without a guard (`hasattr(os, "sched_getaffinity")`, cgroup reads tolerate missing files); no shell; fonts come from the
  per-OS folders in `fonts.font_dirs`; subprocess arguments are lists; ffmpeg comes from PATH or `imageio-ffmpeg`
  (`media.ffmpeg_exe`), probing uses PyAV (no `ffprobe` dependency).
* **Security:** no `shell=True`, `eval`, `exec`, pickle or `yaml.load`; `np.load(..., allow_pickle=False)`; SQL is
  parameterised (the one f-string query in `Store.clips` interpolates a constant fragment - keep it that way, `# noqa: S608`);
  `llm.base_url` must be http(s); image decoding only for a background the user names.
* **Dependencies:** adding one needs a strong reason (affects installs on three OSes). Pin a lower bound in
  `pyproject.toml` *and* `requirements.txt`; optional features must degrade gracefully when the package is absent.
  `sherpa-onnx` (the speech enhancement backend) is **optional**: without it the audio is transcribed unenhanced.
  `pydantic` validates the config file (schema, types, ranges) - do not replace it with hand-written checks.
  Rejected on purpose: `clearvoice` (pins numpy<2, scenedetect 0.6.6, opencv 4.10 - it would downgrade the environment),
  `deepfilternet` (numpy<2 + a Rust toolchain), `resemble-enhance` (numpy==1.26.2, torch==2.1.1).

## 8. Testing guide

Philosophy: **real ffmpeg, real files, real processes** wherever feasible; inject only what cannot exist in CI
(a Whisper model, a GPU, other operating systems).

* pytest runs with `--capture=sys` (see gotcha 21): pytest's default fd-level capture intermittently breaks the
  Tcl file channels on Windows and fails the desktop-app tests at random.

* `tests/conftest.py`
  * `media` (session): generates a 40 s 320x180 video (speech-like tone bursts, 3 s on / 1 s off, 10 known sentences)
    with the bundled ffmpeg, plus `talk.json` (word timings), `talk.srt` and a `silent.mp4` without audio.
  * `make_cfg(media, out, source=None, **sections)`: a small fast `Config` (144 px short side, ultrafast, GPU off,
    logging off, one worker); sections are **deep-merged** with `merge()`. The parameter is called `source` because
    `video` is a config section name.
  * `_restore_logger` (autouse): `setup_logging` sets `propagate=False`; this resets it so `caplog` works.
* Patterns already in use - reuse them:
  * Whisper: `install_stub_whisper` in `tests/test_transcript.py` (real call signatures; returns the fixture transcript,
    or one chunk-relative segment per call when given an ndarray). Also imported by `test_runner_cli.py`.
  * Process pool: `ThreadPool(ThreadPoolExecutor)` monkeypatched over `runner.ProcessPoolExecutor` so scheduling logic runs
    (and is measured) in-process; one test uses real `spawn` workers.
  * LLM: the `llm_server` fixture is a real local `http.server` speaking Ollama and OpenAI-compatible formats.
  * OS-specific logic (font folders, GPU priority per OS, cgroups) is tested by **passing the platform/paths in**
    (`font_dirs(system, env, home)`, `pick_encoder(cfg, system)`, `cgroup_*(root)`), never by running on that OS.
  * Hardware: `hardware.works` is `functools.cache`d - the test module clears it (`_fresh_probe_cache`). A software
    libx264 `Encoder(..., hardware=True, ...)` stands in for a GPU to test argument plumbing and the worker cap.
  * Caches to clear when patching: `media.ffmpeg_exe.cache_clear()`, `fonts.font_index.cache_clear()`,
    `captions._warn_no_raqm.cache_clear()`.
* **CLI tests read the run's log file**, not `caplog`: `main()` reconfigures the logger.
* Coverage uses `branch = true`, `concurrency = ["multiprocessing", "thread"]`, `parallel = true`: code that runs inside
  spawned workers is measured. `if __name__ == "__main__":` is excluded; `__main__.py` is exercised with `runpy`.
* The desktop app: the behaviour lives in `app/session.py` and is tested without a display; the widgets need one, so
  those tests skip when there is none (`has_display()`). CI runs them under `xvfb-run` on Linux.
* Every new behaviour needs a test that would fail without it; every bug fix needs a regression test. Assertions should
  state the contract (e.g. "no moment is reused under another label"), not incidental values.
* Keep tests fast: prefer the 40 s fixture, `short_side` 96-144, `ultrafast`, `audio.denoise: "none"` unless denoise is the subject.
* Platform-specific code is tested by *injecting* the platform: `sys.platform` for the open-folder branch of
  `app.session`, `os.name` for its process group. Never let a test depend on the OS it happens to run on.
* Under GitHub Actions `conftest.py` prints every failure as an `::error file=...,line=...` annotation, so the failing test
  and its traceback appear on the commit page (a job log needs an account to read). Locally the hook does nothing.

## 9. Gotchas (learned the hard way)

1. **MoviePy 2.2.1 writes `-preset <value>` unconditionally** (it crashes on `preset=None`) and appends `ffmpeg_params`
   after it. ffmpeg's last repeated option wins, and an encoder without a preset option only warns. Hence
   `Encoder.preset` is always a string (`PLACEHOLDER_PRESET = "medium"` for GPU encoders) and encoders with real presets
   (NVENC `p4`, QSV `medium`) set theirs in `params`. `hardware.works` builds the command in the same order.
2. **`ruff format` reflows code, so literal-string patch scripts silently fail to match.** Anchor edits on stable tokens
   (`def name(`), assert the match (`assert old in s`), and re-run tests after scripted edits.
3. **Pillow's Windows/macOS wheels often lack libraqm.** `captions.Face` uses Raqm when present, else pure-Python
   Arabic/Hebrew shaping (`arabic-reshaper` + `python-bidi`). Hindi/Thai/Khmer need Raqm (warning is logged once).
   Pillow's *built-in* font (`ImageFont.load_default`) is always `Layout.BASIC` and raises
   `KeyError: setting text direction, language or font features is not supported without libraqm` when a
   `direction` is passed - so `Face` may only ask for a direction for a real font file. One font must cover
   95% of the letters *and* digits of a caption, so a mixed Arabic+Latin line finds no font at all
   (per-word fallback does not exist yet).
4. **MoviePy pins `pillow<12`**, but the suite passes on Pillow 12.3 (security fixes) - CI runs both; README tells users to upgrade.
5. **PySceneDetect >= 0.7:** use `scene[0].seconds` (`get_seconds()` is deprecated).
6. **The transcription audio is always decoded by us** (`media.decode_audio`, 16 kHz), never handed to Whisper as a
   file path: the chunks have to be enhanced first. The whisper stub in `tests/test_transcript.py` tells a chunk from the
   whole file by its length.
7. **The sherpa-onnx wheel is CPU-only** (compiled without `SHERPA_ONNX_ENABLE_GPU`): the "cuda" provider silently falls
   back to the CPU. The denoiser is small, so *more* threads make it slower (RTF 0.21 at 4 threads, 0.30 at 16).
   `gtcrn` is ~4x faster than `dpdfnet8` at the same quality class - measure before changing the default.
8. **Denoising clean audio makes the ASR model drift into English.** Measured on a screen recording with an 82 dB
   signal-to-noise ratio: `gtcrn` and `noisereduce` turned the second half of an Arabic transcript into English, and
   every DPDFNet setting changed the text (CER 0.3-0.5 against the un-enhanced transcript, while the run-to-run noise
   of the same audio is ~0.1). That is why `enhance.snr_threshold_db` measures every chunk first and leaves clean
   audio alone. Only the DPDFNet models support `attenuation_limit_db`; a *larger* value allows *more* suppression.
9. **The padding and the cut snapping can push a clip over its maximum.** `pad_start`/`pad_end` and `snap_to_cut`
   add up to ~3 s to a window that was within the range, so the window is clamped to `max_seconds` again after the
   snapping. The minimum is the harder half: the pause removal happens at render time, so `compute_keep` gets the
   kind's `min_seconds` and gives the smallest cuts back when the clip would end up too short.
10. **Drawing text straight onto an RGBA canvas leaves dark fringes.** `render_text_image` draws text/outline/highlight on
   separate pre-coloured transparent layers and alpha-composites them.
11. **MoviePy's default resize is Pillow-LANCZOS (slow).** `render._resize_effect` swaps in OpenCV (~3x faster overall).
12. **Containers report a slightly longer duration than the video stream;** sub-clips are clamped to `vclip.duration`.
13. **Cache hooks and SQLite connections are per thread/process.** A hook from another thread's `Cache` raises; that is
   why `after_operation` swallows hook exceptions. Never share a `Store` across threads.
14. **Scenes compete for a finite set of moments.** In short videos later scenes may get fewer clips than `count`; this is
    intended (and documented), not a bug.
15. **Do not use `caplog` for CLI tests** (see section 8) and do not call `setup_logging` in library code.
16. **Config key names matter:** `scenes` = scene types; `snap_to_cuts` = shot-change snapping. Do not reintroduce `style` or `snap_scenes`.
17. **A frozen build has no Python and no package.** `tools/frozen_entry.py` is its entry point (a frozen
    `__main__` cannot use a relative import), and the app runs *itself* for a pipeline
    (`session.pipeline_command` checks `sys.frozen`) instead of `python -m autoclip`. The bundle is a
    folder, not a single file: numpy, OpenCV, the Whisper runtime and ffmpeg are big and a folder
    starts much faster. The optional extras are not bundled - the app degrades gracefully without them.
18. **The Linux packages wrap the same build.** The AppImage and the snap do not compile anything:
    they stage the PyInstaller folder. So the entry point is a *script* (without arguments the
    desktop app opens, with arguments the command line runs) and the payload keeps its place - a
    symlink would break PyInstaller, which looks for its files next to the binary.
19. **Two ffmpegs are in play.** `media.ffmpeg_exe()` prefers the one on `PATH` (decoding, `split` mode), but MoviePy
    encodes with `FFMPEG_BINARY` or its bundled `imageio-ffmpeg` binary. The probe therefore uses
    `hardware.encoding_ffmpeg()`: probing with the `PATH` ffmpeg once "proved" NVENC on a machine whose bundled build
    rejects `-rc`, and every clip then died at render time.
20. **Windows appends are not atomic, POSIX ones are.** The C runtime implements `O_APPEND` as seek-to-end *then* write,
    so two workers logging at the same moment overwrite each other's line (worker START lines went missing on Windows
    CI). `log._SharedFileHandler` therefore writes every record under a one-byte `msvcrt.locking` lock of the log file
    itself (past EOF on a new file is allowed); `log._appends_need_locking` turns this off on POSIX, where a `write`
    with `O_APPEND` is atomic. A lock that cannot be taken falls back to an unlocked write - a lost record is worse
    than an interleaved one. Do not replace the handler with a plain `logging.FileHandler`.
21. **pytest's fd capture breaks Tcl on Windows.** With the default `--capture=fd`, the desktop-app tests fail at
    random on the Windows runners (`Can't find a usable init.tcl ... couldn't read file ...: No error`, or
    `ttk/utils.tcl: no such file or directory`) while other Tk tests in the same process pass: the fd capture dup2s
    over the standard handles the Tcl channels depend on. `pyproject.toml` therefore sets `--capture=sys` (no test
    asserts on output written straight to a file descriptor). Do not "restore" the default.

## 10. Recipes

* **New scene type:** add weights to `STYLES` and a one-line description to `STYLE_HELP` (and lexicon words to `LEXICONS`
  if a new feature name is used). `ClipsCfg` validation and `"all"` pick it up automatically. Add it to the expectations in
  `tests/test_moments.py::test_every_scene_type_prefers_its_own_window`.
* **New GPU/CPU encoder:** add a builder to `hardware.HARDWARE` (and the OS to `PRIORITY`), add the literal to `VideoCfg.gpu`,
  test arguments in `tests/test_hardware.py`. If it needs its own `-preset` names, put them in `params`.
* **New clip kind:** add a `KindCfg` field on `ClipsCfg` with defaults, extend the `kinds` `Literal`, `ClipsCfg.kind()`, and
  the output-folder logic in `cli.process_video`; update tests and README.
* **DB schema change:** edit `store.SCHEMA` (the DB is recreated every run - no migrations), update `Store` methods and
  `tests/test_infra.py`. Remember workers open their own connection.
* **Caption styling:** `captions.render_text_image` takes a `CaptionsCfg`; sprite cache keys include the whole style
  (`sprite_key`), so new style fields are picked up automatically if they are in `CaptionsCfg`.
* **Performance work:** profile one clip with `python -m cProfile -o p.out -m autoclip cfg.json` (use `mode: "ranges"`).
  Typical hot spots are MoviePy's compositor and per-frame resize/blur; most remaining time is inside MoviePy.
* **Debugging a run:** set `logging.level: "DEBUG"`, read `{output_dir}/logs/autoclip_*.log` (START/END per operation,
  `submit clip ... allowed=N threads=M`, `pid=` shows which worker did what), inspect `.work/{video}/autoclip.db`
  (`SELECT * FROM clips`).

## 11. Environment notes for sandboxed agents

* The Hugging Face host (Whisper models) may be unreachable: use the stub (`install_stub_whisper`) or `transcript.mode: "file"`.
* No GPU may be present: GPU code paths are verified by probe/fallback behaviour and software stand-ins only.
* One CPU core: force the parallel code path with `resources.cpus: 2` and `max_workers: 2` (real spawn) or the thread fake.
* Long-running commands (e.g. 30+ minute media) must be detached (`setsid nohup ... &`) and polled.
* A 35-minute synthetic video with a chunk-aware Whisper stub is the reference for chunking behaviour (7 chunks, ~19 MB
  of audio each, ~30 s end to end on one core, peak RSS ~435 MB).

### What is verified vs not
* Verified here: Linux; real ffmpeg encode/decode; spawned workers; GPU *detection and CPU fallback*; chunked
  transcription with a stub; per-OS logic through injected platforms; 100% branch coverage.
* **Not verified on real systems:** Windows and macOS execution (CI matrix exists), real GPU encoders (NVENC/QSV/AMF/
  VideoToolbox arguments are unit-tested and probe-guarded but never ran on hardware), real Whisper inference.
  Never state these as verified.

## 12. Boundaries

**Always**
* Run the four gates (section 2) and keep coverage >= 95% (aim for 100%).
* Add/adjust tests with every change; update `README.md` (and this file when architecture, invariants or commands change).
* Keep config backward-compatibility notes in the README "Upgrading" section when keys change.
* Preserve cross-platform behaviour (paths, encodings, spawn, optional dependencies).

**Ask first**
* Adding or upgrading dependencies; raising the minimum Python version.
* Breaking changes to config keys, output layout or exit codes.
* Changing DB persistence semantics (fresh-per-run) or the transcript-cache key.
* Anything that touches how GPU encoders are selected or how workers are capped.

**Never**
* `shell=True`, `eval`/`exec`, pickle/`allow_pickle=True`, string-built SQL with user data, network calls other than the configured LLM URL.
* Log or print secrets; weaken `llm.base_url` validation.
* Lower the coverage gate, add `# pragma: no cover` to dodge a test, or silence mypy/ruff globally to get green.
* Rebind `RESOURCES`, share a `Store` across threads, use `fork`, or hard-code POSIX paths/separators.
* Reintroduce candidate storage in SQLite, blended multi-style ranking, or file-level video splitting for chunking.
* Claim GPU, Windows, macOS or real-Whisper behaviour is tested when it is not.

## 13. Glossary

* **kind** - `short` or `long`: a family of output clips with its own length range, count and aspect ratio.
* **scene (type)** - the style a moment is chosen for (`best`, `funny`, ...); defined by feature weights in `STYLES`.
* **moment** - a selected window (`Moment`) before rendering; **clip** - the rendered file.
* **candidate** - a possible window (run of sentences) considered during selection.
* **chunk** - a ~5-minute piece of audio given to Whisper and to the denoiser (not a file on disk).
* **energy track** - per-0.5 s loudness (dB) of the whole file, used for scoring, snapping and chunk boundaries.
* **sprite** - a pre-rendered RGBA caption image cached compressed and blitted by `CaptionTrack`.
* **work dir** - `{output_dir}/.work/{video}/`: database and transcript cache for one video.
