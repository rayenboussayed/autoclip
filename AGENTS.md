# AGENTS.md - autoclip

Guidance for AI coding agents (and humans) working on this repository. Read it fully before changing code; the
"Invariants" and "Gotchas" sections record decisions and failures that are not obvious from the code.

## 1. What this project is

`autoclip` turns long videos into the best **short** (default 9:16, 20-60 s) and **long** (default 16:9, 3-10 min)
clips, per requested **scene type** (`best`, `funny`, `sad`, ... or `"all"`). One config file drives everything.
Pipeline: download the http(s) URLs of `input` with yt-dlp (only when there are any) -> transcribe (faster-whisper,
a downloaded subtitle or .srt/.json) -> score sentence-aligned windows per scene -> pick the best non-overlapping ones ->
render (enhance audio, cut pauses/fillers, layout, burn captions, encode on GPU if one really works).

* Free, local, no service dependencies (downloading is off unless `input` holds an http(s) URL; the optional local
  LLM re-rank is the only other network use).
* Runs on Linux, macOS and Windows, Python >= 3.11 (developed and verified on 3.12). No system ffmpeg required.
* Four parts: `core` (the engine), `cli` (the command line), `app` (the desktop app, Tk) and `server` (the optional
  FastAPI/WebSocket service).
* Version 4.0.0 (see `pyproject.toml`, `autoclip/__init__.py` and `snapcraft.yaml`). ~12,000 lines of
  package code across `core`/`cli`/`app`/`server`, ~14,400 lines of tests (100% statement + branch coverage).

## 2. Commands

```bash
pip install -r requirements-dev.txt            # runtime + dev tools (pytest, pytest-cov, mypy, ruff)
pip install -U "Pillow>=12.3"                  # security fixes; MoviePy's "<12" pin is conservative (CI runs both)
pip install -e .                               # optional: gives the `autoclip` console command
pip install -e ".[server]"                     # optional: the FastAPI server (tests get it through requirements-dev.txt)

python -m autoclip --init config.json          # tiny starter config      (--init-full: every option + default)
python -m autoclip [config.json]               # run (no config: ./config.json if present, else ./input.mp4 + defaults; exit 0/1/2)
python -m autoclip --gui                       # the desktop app (same as: python -m autoclip.app)
python -m autoclip --formats URL               # every format of a URL and the one autoclip would pick (no config needed)
python -m autoclip --platforms [URL] config.json        # the allow/deny policy (and where that URL would go)
python -m autoclip --download-only config.json # download the URLs of `input` and stop (originals in .work/downloads)
python -m autoclip --allow youtube,vimeo --deny tiktok config.json   # override the lists for one run
python -m autoclip.server                      # the HTTP/WebSocket service (/view/, /docs; PORT, AUTH_*, MAX_FILE_SIZE)

python -m PyInstaller autoclip.spec --noconfirm  # standalone build (needs pyinstaller; see the build workflow)
python tools/make_appimage.py                   # the AppImage (Linux) from that build
sudo snapcraft --destructive-mode               # the snap (Linux) from that build (needs root)
pre-commit install                              # ruff, mypy and the test suite on every commit
ruff check . && ruff format --check .          # lint + format (line length 140)
mypy autoclip tests tools                       # strict typing, tests and tools included
vulture                                        # dead code (80%+ confidence; uses [tool.vulture])
deptry .                                      # dependency hygiene (uses [tool.deptry])
pytest                                         # offline unit suite; FAILS if branch coverage < 100%
pytest tests/unit/test_moments.py -k scene --no-cov   # a focused run (always add --no-cov for partial runs)
AUTOCLIP_NETWORK_TESTS=1 pytest -m "network and not e2e" --no-cov   # the real-YouTube downloader checks
AUTOCLIP_E2E_TESTS=1 pytest tests/e2e -m e2e --no-cov               # CLI/server end-to-end runs (results in tests/results)
AUTOCLIP_E2E_TESTS=1 AUTOCLIP_E2E_LEVEL=3 pytest tests/e2e -m e2e --no-cov   # 18-minute ladder level (cache is reused)
```

The `Makefile` is now the command catalogue: `make check`, `make test`, `make test-focused`,
`make test-config`, `make test-app`, `make test-network`, `make test-e2e`, `make test-ladder`,
`make test-all`, `make lint`, `make format`, `make format-check`, `make mypy`, `make vulture`,
and `make deptry`. Use `PYTHON=...` and `CONFIG=...` overrides.

**Definition of done** (these must pass before you finish; `make check` runs them as one
script): `ruff check .`, `ruff format --check .`, `mypy autoclip tests tools`, `vulture`, `deptry .`,
`pytest` (100% statement + branch coverage gate). The repository now ships a
`Makefile`; `make check` is the required final verification, and `make test-all` runs `test` plus
the network and e2e tiers. Before every commit, rerun `make check`, fix every warning and error
it reports, and only then commit. A full `pytest` takes about one minute (real ffmpeg encodes,
real worker processes).

**Release:** bump `version` in `pyproject.toml`, `autoclip/__init__.py` and `snapcraft.yaml` (they must match -
`tests/unit/test_version.py` checks it), **add a new section to `CHANGELOG.md` with the release number and its git
timestamp** (run `git log -1 --format=\"%ci|%H|%s\" HEAD` for the timestamp, and include the changes that went in), and
**update `README.md`, `AGENTS.md` and `docs/` when behaviour or operating instructions changed**, and push to `main`.
The build workflow checks that `v<version>` is not tagged yet, builds the five targets, then creates the tag and publishes
the GitHub release. A push that does not change the version skips the build; a pull request builds without releasing.
A release that forgets to touch `CHANGELOG.md`, `README.md`, `AGENTS.md` or `docs/` is incomplete - all four are part
of the release commit, next to the version bump.

## 3. Repository layout

```
autoclip/
  __main__.py     python -m autoclip entry (freeze_support for frozen Windows builds)
  frozen.py       the one-file build: no argument -> desktop app, any argument -> command line
  core/           THE ENGINE - no user interface, heavy imports stay inside functions
    config.py       typed config dataclasses (Pydantic validates them), strict loader, input globbing + URLs
    platforms.py    download policy: extractor families, built-in/user groups, allow/deny, per-platform settings
    formats.py      chooses one explicit yt-dlp format id (short-side quality, fps, codec, container, audio)
    download.py     Downloader: yt-dlp parameters, originals cache, time ranges, subtitles, sidecars, naming
    pipeline.py     process_video (probe -> transcript -> moments -> render; name/subtitle overrides), split_video, ranges_moments
    styles.py       DATA ONLY: STYLES (scene weights), STYLE_HELP, LEXICONS
    models.py       small slotted dataclasses: Span/Word/Segment/Moment, MediaInfo, Encoder, RenderJob, Downloaded
    errors.py       AutoclipError, ConfigError, DownloadError
    resources.py    ResourceMonitor + RESOURCES singleton: CPU/RAM -> worker plan, cache budget, adaptive tuning
    internal/       derived runtime decisions and constants (mode profiles, adaptive helpers, decision logs,
                    admission policy for the render pool);
                    not user-facing configuration, logged as `automation ...`
    log.py          LOG, setup_logging, traced decorator, speed_text
    store.py        SQLite Store (WAL), MemoryLRU, Cache (RAM LRU in front of SQLite kv), is_boundary
    media.py        ffmpeg discovery, probe (PyAV), energy track, decode_audio, scene cuts, snapping, trim_video
    audio.py        enhance_audio (denoise/loudness/limiter), compute_keep (pauses/fillers), Timeline, build_audio
    unsilence.py    remove_silence: pydub silence detection + MoviePy re-encode (audio decoded by media.decode_audio)
    transcript.py   parse srt/json, downloaded subtitles, faster-whisper (chunked), disk cache, get_transcript
    denoise.py      speech enhancement before transcription: SpeechEnhancer (sherpa-onnx ONNX | noisereduce)
    moments.py      candidates, vectorised features, per-scene scoring, pick(), LLM re-rank, select_moments
    hardware.py     GPU/CPU encoder choice: probe by really encoding; Encoder builders; HW_MAX_WORKERS
    fonts.py        cross-platform font discovery + glyph coverage (fontTools)
    captions.py     text sprites (balanced wrapping, word highlight), CaptionTrack, build_captions
    render.py       layout, OpenCV resize/blur, encode, render_clip
    runner.py       RenderJob execution: render_job (never raises), render_all (adaptive process pool)
  cli/            THE COMMAND LINE - python -m autoclip config.json
    __init__.py     argument parsing, main(), --init / --init-full, --gui, --formats / --platforms / --download-only
  app/            THE DESKTOP APP - python -m autoclip.app (Tk, native look)
    fields.py       the catalogue: every config option with what / why / when (the app is built from it)
    form.py         the form: one row per option, the control that fits its kind (incl. JSON), the scroll area
    session.py      the behaviour without widgets: read/write/validate the config, default_config_path, start/stop
    window.py       the window: tabs, details panel, log, Run / Stop / Save / Load / Reset, debug mode
  server/         THE OPTIONAL HTTP SERVICE - python -m autoclip.server (FastAPI + Jinja2/HTMX/DaisyUI)
    config.py       Settings from the environment (PORT, MAX_FILE_SIZE, AUTH_*, OUTPUT_DIR, CACHE_DIR, TTL)
    auth.py         HTTP Basic, enforced only when both AUTH_USERNAME and AUTH_PASSWORD are set
    state.py        Job + JobRegistry (in memory, lock-protected, TTL expiry)
    progress.py     ProgressManager: per-job queues -> WebSocket frames (loop-safe publish)
    service.py      the pipeline stages one by one, config preparation, ZIP packaging
    schemas.py      Pydantic API models (JobCreated, JobStatus, ProgressEvent)
    routers/api.py  /api/jobs, status, ZIP download, delete, WebSocket (the JSON API),
                    /api/preview (one frame with the configured layout/captions, mock text)
    routers/view.py /view/ upload page and /view/jobs/{id} (the HTML pages)
    main.py         create_app, lifespan evictor, OpenTelemetry (OTLP only when configured)
    templates/      base/index/job/404 (Jinja2 + HTMX + DaisyUI from a CDN)
tests/            support.py - pure helpers shared by both suites; conftest.py - fixtures and the e2e/network gates
  unit/           mirrors the modules (the default, offline `pytest` run); test_version.py guards the versions
                  test_support.py / test_e2e_harness.py / test_conftest.py - the shared helpers, the e2e harness and
                  the suite's own gating/annotations, all exercised offline
  e2e/            opt-in real-network checks: test_network.py, the CLI cases and the localhost server case
    harness.py      pure e2e logic: results folders, subprocess runs, config builders, clip/DB readers, ladder/cache
    conftest.py     the e2e fixtures and the outcome-recording hook around that harness
  assets/         youtube.json - the videos the e2e cases use
  results/        per-run configs, logs and outputs (ignored by git)
autoclip.spec     the PyInstaller spec: a standalone build (engine + command line + desktop app + yt-dlp/EJS)
snapcraft.yaml    the snap: the same build as a snap (linux, amd64 + arm64)
Dockerfile        the server image (python:3.12-slim + ffmpeg + the `server` extra)
docker-compose.yml  the same image with volumes and the environment variables
assets/           the icon and the desktop file of the AppImage and the snap
tools/            frozen_entry.py - the frozen build's entry point: freeze_support, then autoclip.frozen.main
                  (a frozen __main__ has no package around it)
                  make_appimage.py - builds the AppDir and the AppImage from the PyInstaller build
                  make_icon.py     - draws the icon (assets/autoclip.png)
                  compare_stt.py   - development aid: STT models x enhancement backends on one scene
                  verify_output.py - development aid: checks a finished output directory
docs/             stt-comparison.md - the measured comparison of transcription models and denoisers
AGENTS.md        this file (guidance for coding agents)
pyproject.toml    deps, ruff, mypy(strict), pytest(+coverage gate, `network` marker), coverage(multiprocessing)
.github/workflows/ci.yml       ubuntu/macos/windows x py3.11/3.12/3.13, then re-run with Pillow>=12.3
.github/workflows/pr-gate.yml  pull-request gate: ci checks, all OS builds, and Docker smoke build
.github/workflows/network.yml  weekly (and manual) real-platform downloader tests
.github/workflows/e2e.yml      manual command-line end-to-end runs (results uploaded as an artifact)
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
platforms         <- errors (yt-dlp imported inside the functions that need its extractor list)
formats           <- config, errors
download          <- config, errors, formats, log, media, models, platforms
transcript, moments, captions, render <- the layers above
runner            <- hardware, log, models, render, resources, store
pipeline          <- everything of the engine (the per-video run)
frozen            <- app, cli (both imported inside main: the entry point of the one-file build)
cli               <- core.pipeline, core.download, core.config, core.errors, core.hardware, core.log, core.resources
app               <- core.config, core.errors (+ tkinter)
server            <- core.config, core.hardware, core.media, core.moments, core.pipeline, core.render, core.runner,
                     core.store, core.transcript (+ fastapi/pydantic-settings, imported only by the server)
```
`RenderJob` lives in `models.py` (not `runner.py`) precisely to avoid a `render <-> runner` cycle; `Encoder` is in
`models.py` and its logic in `hardware.py` to avoid `models <-> media`. `Downloaded` is in `models.py` for the same
reason: the CLI and the app need its type without importing the downloader (and yt-dlp). If you need a new shared
type, put it in `models.py`.

Heavy libraries are imported **inside functions** (`moviepy`, `faster_whisper`, `cv2`, `noisereduce`, `scenedetect`,
`imageio_ffmpeg`, `arabic_reshaper`, `bidi`, `yt_dlp`) so `--init`, config errors, local-only runs and unit tests stay
fast and optional extras degrade gracefully. Keep that pattern (ruff rule `PLC0415` is intentionally ignored).

## 4. Data flow

```
main()
  load_config -> setup_logging -> RESOURCES.configure -> pick_encoder (once per run; probes GPUs)
  Downloader(cfg)                                only when input holds an http(s) URL; yt_dlp is imported here
    download(url) -> [Downloaded(path, name, platform, subtitle, subtitle_manual)]
                                                 original in {out}/.work/downloads,
                                                 reused when the prefix (title [id] [format] + range) exists
  for each input item (isolated: one failure never stops the batch):
    local file -> one source; URL -> the downloaded entries (names deduplicated across the batch)
    process_video(src, cfg, log_file, encoder, name=name, subtitle=subtitle, subtitle_manual=...)
      probe(src)                                   PyAV: duration, audio?, size, fps
      mode == "split": ffmpeg stream-copy parts -> {out}/long/{video}/   (done)
      Store(.work/{video}/autoclip.db, fresh=True) + Cache      ({video} = name or src.stem)
      energy_track(src)                            0.5 s loudness track (cached in kv)
      get_transcript(..., needs_text(cfg))        sidecar file | downloaded subtitle | disk cache | faster-whisper
                                             (5 min chunks) -> SQLite (each chunk enhanced by `SpeechEnhancer` first);
                                             subtitles follow the video's original language; consent is always auto;
                                             `skip.stt` stops before Whisper, `skip.denoise` skips the enhancer
      for kind in cfg.clips.kinds ("short", "long"):
        moments = ranges_moments(...)  if mode == "ranges"   (scene "custom")
                  select_moments(...)  otherwise              (one pass per scene type)
        no moments -> warning, kind skipped (all kinds empty -> AutoclipError)
        Store.save_moment(...) -> clip_id ; build RenderJob (dst = {out}/{kind}/{video}/{video}_{scene}_{nn}_{start}s.mp4)
      render_all(jobs)                             adaptive pool; results land in the `clips` table
      write {out}/{kind}/{video}/clips.json from the DB rows with status == "done"
```

### Downloading (only for URLs in `input`)
`Downloader` holds the whole flow so the CLI and the app behave identically: platform policy (`platforms.py`), the
effective settings for the platform/group (`effective_download`), format choice (`formats.py`), the cache of originals
and the yt-dlp parameters. A URL is identified by its host and the extractor that matches it (`identify`), checked
against allow/deny **before** any request, and then extracted. `allowed_extractors` is passed to yt-dlp as a second
line of defence. Each entry is downloaded with an explicit `<video>+<audio>` format id, moved out of yt-dlp's temporary
name into `<safe title> [<id>] [<format id>]` (+ range suffix), and the selected subtitle is copied next to it as a real
`.srt`. `metadata` sidecars (`.info.json`, thumbnail, description) land next to the original. A stalled or failing
media download does not cost the video: `formats.choose_candidates` returns the ranked candidates (at most three; the
best plus two alternates) and `_media` tries them in order, reusing the cache of any candidate that is already
present; the chosen candidate is logged with its resolution, codec, bitrate and protocol. A range is downloaded by
yt-dlp only when the real `ffmpeg` *and* `ffprobe` are on `PATH`; otherwise (or when the result is short/long by more
than `max(3 s, 10%)`) the whole video is downloaded and `media.trim_video` cuts it locally (the SRT is shifted and
clipped too). `--download-only` stops after this stage. The sort prefers direct HTTPS at equal quality and then the
least-burdensome bitrate (an unknown bitrate or frame rate always loses), so the smallest adequate format is downloaded first. Video-only candidates pair with different audio ids (the next-best audio for each), because the SABR audio stall hurts every candidate equally when they all share one audio id.

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
4. Optional LLM re-rank per scene (`_rerank`: `0.4*z(heuristic) + 0.6*(llm-5)/2`; any failure keeps heuristic order);
   `skip.llm` turns it off. Padding for text picks uses the internal constants `CLIP_PAD_START` and
   `CLIP_PAD_END`; audio-only picks snap to quiet frames; optional snap to shot cuts is controlled by
   `skip.snap`.

Candidates are **not stored in SQLite** (an earlier design did; it exploded to millions of rows for long clips x scenes).

### render_clip
`enhance_audio` (highpass, denoise in 60 s pieces, loudness over speech frames, soft limiter) -> `compute_keep`
(shorten long pauses, cut fillers; skipped without audio) -> clamp sub-clips to the real stream length -> `layout`
(fit on blurred/colour/image background, or fill) -> `build_captions` (sprites cached, one band track) -> optional title ->
fades -> `encode` with the chosen `Encoder` -> JPEG cover. `render_job` writes the `.srt` sidecar and the DB outcome.
The `skip.*` switches gate the matching stage here: `audio` mutes (and stops decoding), `audio_enhance`/`denoise`
pick the audio halves, `pauses`, `caption`, `title`, `fade` and `thumbnail` skip their step. `audio.enhance: false`
remains the master switch of all clip-audio processing.

`compute_keep` gets the kind's `min_seconds`: when the trimming would cut the clip below it, the smallest cuts are
given back (the fillers first), so a clip never ends up shorter than the user asked for.

### Parallelism
`render_all` plans workers from `RESOURCES.plan_workers(per_job_bytes, jobs, limit)` **before every submission**.
`limit = HW_MAX_WORKERS (3)` for GPU encoders. One usable core (or one job) runs in-process; otherwise a
`ProcessPoolExecutor` with the `spawn` context (same on every OS). A worker killed by the OS (`BrokenProcessPool`) is
retried alone, in-process. Workers re-run `setup_logging(log_file, level, None)` and `RESOURCES.configure(cfg)`.

### The HTTP server (`server/`)
`create_app` builds a FastAPI app with the JSON API under `/api`, the HTML pages under `/view` and the WebSocket
`/api/jobs/{id}/ws`. A job lives in the in-memory `JobRegistry` (id, uploaded file, job folder, prepared config, state,
step, percent, outputs); nothing else is persisted except the job folder and the ZIP cache.

1. `POST /api/jobs` streams the upload to `OUTPUT_DIR/<job_id>/input<ext>` while enforcing `MAX_FILE_SIZE` (413 above
   it), parses `config_json`/`config` and replaces `input`/`output_dir` with the uploaded file and job folder.
2. The job is queued behind an `asyncio.Semaphore(MAX_CONCURRENT_JOBS)`; a FastAPI `BackgroundTask` runs `run_job` in
   the thread pool, so the event loop stays free.
3. `run_job` repeats `process_video` stage by stage so each one can publish progress: `validating` 2 % -> `probing` 5 %
   -> `transcribing` 12/35 % -> `selecting` 45 % -> `rendering` 60 % -> `packaging` 90 % -> `completed` 100 %
   (`failed` keeps the error text). Split mode short-circuits at `rendering` 50 %. It creates a fresh `Store`, a
   `Cache`, picks the encoder once and calls `render_all` with the same `RenderJob`s the CLI builds.
4. `ProgressManager` keeps the set of WebSockets watching each job; `publish` is safe from the worker thread
   (`call_soon_threadsafe`) and keeps task references so the frames are not garbage-collected. The WebSocket sends the
   current snapshot, then every update, and a keepalive frame every `KEEPALIVE_SECONDS` (`api.py`).
5. `GET /api/jobs/{id}` reports state/step/percent/error/outputs; `GET .../download` builds the ZIP in `CACHE_DIR`
   (clips, `.srt`/`.jpg` and every `clips.json`, under their relative output paths so two `clips.json` files do not
   collide; `.work` is left out); `DELETE` removes the job, its input and its output folder.
6. The lifespan task evicts expired jobs every `EVICT_SECONDS` (`main.py`); the TTL is `JOB_TTL_SECONDS`.
7. Auth is HTTP Basic and is skipped unless **both** `AUTH_USERNAME` and `AUTH_PASSWORD` are set; the WebSocket accepts
   `?token=<user>:<password>` because browsers cannot set an `Authorization` header on it. OpenTelemetry instruments
   FastAPI and exports OTLP only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
8. `/api/preview` renders one representative frame with the configured layout/captions and **mock** text and returns
   it as a PNG. It never creates a job or a registry entry, never transcribes, and its throwaway upload under
   `CACHE_DIR/preview/<uuid>` is deleted before the response returns.

## 5. Invariants - do not break these

1. **Fresh database per video, every run.** `Store(..., fresh=True)` deletes the DB and its `-wal/-shm/-journal`.
   The only things persisted between runs are `{video}.transcript.json` (keyed by source name/size/mtime, model,
   language, multilingual, word_timestamps and the enhancement settings; a mismatch re-transcribes) and the downloaded
   originals under `.work/downloads` (reused when the name prefix - title, id, format id, range - matches).
2. **Results flow through SQLite, not return values.** `render_job` returns a bool but the truth is the `clips` row
   (`pending` -> `done`/`failed`). `clips.json` is built from rows. A row still `pending` after `render_all` means the
   worker never finished and is reported as failed.
3. **`render_job` never raises** (except `KeyboardInterrupt`): a failed clip must not stop the batch. `process_video`
   exceptions are caught per video in `main` (`AutoclipError` is logged without a traceback, anything else with one).
4. **Global selection, segmented transcription only.** The transcript is produced in chunks of
   fixed 5-minute chunks (internal constant) - never more audio in memory than that, and the same chunks are enhanced and
   transcribed. Selection stays global so clips never cut at a chunk edge and a long clip may exceed a chunk. Do not
   "simplify" this into physically splitting the file.
5. **`RESOURCES` is a process-wide singleton** that is *reconfigured*, never rebound (`RESOURCES.configure(cfg)`);
   modules import the object once. Memory/CPU balancing happens in `traced` operations (`after_operation`), during the render
   batch, and inside `Cache.trim`. `mode` (`performance`/`resources`/`balance`) selects the policy; the adaptive tuner
   logs `automation ...` and adjusts the CPU budget over time. Trim hooks are best-effort and swallow exceptions (logged at DEBUG).
6. **GPU encoders are used only if proven.** `hardware.works()` really encodes a 0.4 s test clip with the exact arguments
   MoviePy will use (preset first, then `Encoder.params`) *and the binary MoviePy writes with* (`encoding_ffmpeg()`, not
   the `PATH` ffmpeg used for decoding). Listing an encoder in `ffmpeg -encoders` proves nothing.
   GPU use is probed automatically; never enabled blindly (no `--device` flag exists).
7. **Config is strict, with a soft fallback for values.** Unknown keys and wrong JSON types raise `ConfigError` at
   load; invalid *values* warn and fall back to the field default (`_coerce_defaults`); `"_comment"` keys are
   ignored. All defaults live in the dataclass fields (single source of truth for `--init`/`--init-full`).
8. **Outputs:** `{output_dir}/{short|long}/{video}/` holds only clips (+ `.jpg`, `.srt`, `clips.json`);
   `{output_dir}/.work/{video}/` holds the DB and transcript cache; `{output_dir}/.work/downloads/` holds
   downloaded originals and their sidecars; `{output_dir}/logs/` holds run logs.
   File names: `{video}_{scene}_{nn:02d}_{int(start)}s.mp4` (`nn` counts per kind+scene). Two local inputs with the same
   file stem are rejected at load time (they would share outputs); downloaded names are deduplicated per batch
   (`Name (2)`).
9. **Words/segments must be shorter than `store.MAX_SPAN` (120 s)**: window queries only look that far back.
10. **No work on the GPU except encoding (and Whisper via CUDA).** Do not claim end-to-end GPU acceleration.
11. **The desktop app shows every config option.** `app/fields.py` is the catalogue: one entry per option with
    *what* it does, *why* you would change it and *when*. `tests/unit/test_app.py` compares it with the config
    dataclasses, so a new option that is not in the catalogue (or an entry that is not in the config) fails
    the suite. The app writes the same config file the command line reads - never a second format.
    The catalogue must stay complete: every file option appears there (nine today), and nothing else does.
12. **A local-only run never imports yt-dlp and never makes a network request.** `is_url` decides what is downloaded;
    `yt_dlp` is imported inside `Downloader.__init__`/`platforms` helpers, and `Downloader` is only constructed when
    `input` holds a URL (inspection modes included). `tests/unit/test_cli_download.py` proves it with a poisoned
    `sys.modules["yt_dlp"] = None`.
13. **The platform policy runs before the request.** `identify` (host + extractor match) -> allow/deny (deny wins,
    groups expand, aliases canonicalise) -> `check_families` raises `DownloadError`, and only then is yt-dlp asked
    anything. `generic` (any web page) is never in a group and is refused unless named. `allowed_extractors` is also
    handed to yt-dlp, but it is the second line of defence, not the policy.
14. **The downloader never executes anything.** No `exec`, no `netrc_cmd`, no shell: `cookies` are a browser store or a
    Netscape file, the JavaScript runtime is probed from the `PATH`, `ffmpeg` paths come from `media.ffmpeg_exe` /
    `hardware.encoding_ffmpeg`. Do not expose yt-dlp options that run commands.
15. **Downloaded subtitles are a transcript source, never a forced one.** A sidecar `.json`/`.srt` next to the video
    wins (`<stem>.json` before `<stem>.srt`); otherwise a downloaded subtitle is used (a manual track is preferred;
    automatic captions are used when no manual one exists), then the disk cache, then Whisper. `--skip stt` stops
    before Whisper. Only a real subtitle/transcript file is handed to `get_transcript` (other formats are converted
    or dropped); an empty one is unusable and falls back.
16. **Nothing is skipped by default; `skip.*` only removes steps.** The eleven switches are `stt`, `denoise`,
    `audio_enhance`, `audio`, `pauses`, `caption`, `title`, `fade`, `thumbnail`, `llm` and `snap`, all `false` in
    `SkipCfg`. `skip.stt` still uses a sidecar, a downloaded subtitle or the disk cache and never calls Whisper;
    `skip.audio` mutes the clip and stops audio decoding, so pause removal cannot run; `skip.denoise` transcribes raw
    audio and disables noise reduction of the clip audio; `skip.thumbnail` keeps the frame JPEG; `skip.llm` forbids
    Ollama calls. `captions.enabled` and `--skip title` are the remaining switches; titles are shown only when the
    model generated one. `needs_text(cfg)` decides whether a transcript is loaded at all (highlights, or captions/titles
    that are not skipped).
17. **The server owns the input and the output folder.** `prepare_config` always drops `input` and `output_dir` from the
    uploaded config and sets them to the uploaded file and the job folder; a server job can never write outside its
    own folder. The registry is in memory (a restart loses the jobs, not their files), and `OUTPUT_DIR`/`CACHE_DIR`
    are the only locations it writes to.
18. **Server background work must be loop-safe.** `run_job` runs in a worker thread, so `ProgressManager.publish` goes
    through `loop.call_soon_threadsafe`; the concurrency semaphore is recreated when the running loop changes (each
    `TestClient` starts a new one). Never patch `asyncio.sleep` globally in a test - shrink the `KEEPALIVE_SECONDS` /
    `EVICT_SECONDS` module constants instead, or every await in the client starves.
19. **Server auth is opt-in and compared in constant time.** Basic auth is enforced only when **both** `AUTH_USERNAME`
    and `AUTH_PASSWORD` are non-empty (`secrets.compare_digest`); an empty pair means a public API. Never log the
    credentials, and keep the WebSocket `?token=` fallback in sync with the header check.
20. **The preview is not a job.** `/api/preview` renders one frame with mock text and returns a PNG; it never creates a
    registry entry, never enqueues work, and the throwaway folder under `CACHE_DIR/preview/<uuid>` is deleted before
    the response returns. It shares the jobs' upload discipline (`_read_custom_config`, filename sanitisation,
    `MAX_FILE_SIZE`) but not their output folder.

## 6. Configuration (`config.py`)

The **file** schema is exactly nine keys (`FileConfig`): `input`, `clips.preset`, `clips.scenes`, `clips.keywords`,
`clips.ranges`, `download.allow`, `download.deny`, `llm.system_prompt`, `image.prompt`. Everything else is a runtime override
(`RuntimeOverrides`) or derived internally. `FileConfig.to_config(overrides)` builds the full runtime `Config`
(quality tier, built-in short/long profiles, derived download settings). Removed keys are dropped by
`_strip_removed` with a warning via `REMOVED_KEYS` (see `docs/UPGRADING-4.0.md`); unknown keys still raise
`ConfigError`.

`Config` (runtime) = `input`, `output_dir`, `mode`, `quality`, `download`, `clips`,
`llm`, `image`, `video`, `captions`, `skip`. `KindCfg` = `count`, `min_seconds`, `max_seconds`, `aspect`,
`ranges` (no `target_seconds` in 4.0). `SkipCfg` = `stt`, `denoise`, `audio_enhance`, `audio`, `pauses`, `caption`,
`title`, `fade`, `thumbnail`, `llm`, `snap` (all `false`; see invariant 16). `DownloadCfg` = `cookies`, `time_range`,
`platforms` (runtime overrides or derived from the allow/deny lists; never file keys). `PlatformsCfg` = `allow`, `deny` only in the file; `groups`,
`settings`, `group_by_platform` are gone (groups are named directly in allow/deny). `LlmCfg` = `base_url`, `model` and
`system_prompt` (the last from the file; empty keeps the built-in editor messages). `ImageCfg` = `prompt` (from the
file; the Stable Diffusion prompt). Runtime values are derived internally and logged, never accepted from the config file.

The transcript comes from: a sidecar `.json`/`.srt` next to the video (`<stem>.json` preferred over `<stem>.srt`),
then a downloaded subtitle (manual track preferred, automatic captions otherwise), then the disk cache, then
Whisper. The language is always auto-detected and consent is always `auto`; there is no `TranscriptCfg` and no
`--transcript` flag. Subtitles are downloaded for the video's original language when available; there is no
`SubtitlesCfg`. `--skip stt` stops before Whisper.

Invalid *values* fall back: `_coerce_defaults` replaces a value that fails validation with the field default and logs
`config: invalid value for <path>; falling back to default`. Wrong *types* and unknown keys remain fatal (`ConfigError`).

Four previously user-facing knobs are still internally derived (see `autoclip/core/internal/automation.py`):
`transcript.enhance.num_threads`, `transcript.beam_size`, `captions.max_chars`, and
`captions.safe_top` / `captions.safe_bottom`. Do not reintroduce them in user-facing config. Download quality derives
from output quality via `DOWNLOAD_QUALITY_BY_OUTPUT` (1080/720/480/360/240 for highest/high/medium/low/lowest).
`formats.choose` prefers a direct HTTPS format over HLS/DASH at equal quality and fps (the segmented variants report
inflated bitrates and throttle); `audio_language: null` follows `info["language"]` (the original track).

* **Pydantic validates the file** (`pydantic.dataclasses.dataclass` + `ConfigDict(extra="forbid")`): an unknown key, a
  wrong type or an out-of-range value becomes a `ConfigError` (`parse()` wraps the `ValidationError`). Numeric fields use
  `Annotated[int|float|bool, Strict()]` so JSON keeps its type (a bool is not an int, a string is not a number); formats
  use `StringConstraints(pattern=...)`; cross-field rules (min <= max, scene names) are `model_validator(mode="after")`.
  Keys starting with `_` are ignored, so a config file can carry `"_comment"` entries. **Do not hand-roll type checks.**
* `load_config` resolves every path relative to the config file, expands `input` globs (sorted, de-duplicated,
  absolute) and leaves http(s) URLs **verbatim**. **Exception:** a relative `output_dir` is resolved against the current working directory instead, so the output
  folder is independent of where the config file lives.
* `scenes`: `"all"` | one name | list; `ClipsCfg.scene_list()` expands it. `"scenes"` means the scene *types* (the keys of
  `STYLES`); shot-change snapping is automatic (`snap` skip switch).
* `download.allow`/`download.deny`: names are extractor **families** (`youtube`, `tiktok`, ...), aliases (`x`, `vm.tiktok`, `html5`)
  or groups. User groups must match `@[a-z0-9][a-z0-9_-]*`, may not shadow a built-in group, are checked for unknown
  members and cycles at load time (a cycle is a `ConfigError`); family names are checked against yt-dlp at run time
  (a missing yt-dlp is a `DownloadError`). The `generic` family (any web page) is only used when named.

### Adding a config option (checklist)

The catalogue must stay complete (every file option, nothing else).

1. Add a typed field with a default (and a comment) to the right dataclass. Put the constraint in the type
   (`Field(ge=..., le=...)`, `StringConstraints(pattern=...)`, `Annotated[int, Strict()]`) or, for a rule that spans
   fields, in a `model_validator(mode="after")` that raises `ConfigError` - never in a hand-written type check.
2. Use it from code via `cfg.<section>.<field>` (never read raw JSON elsewhere).
3. Add the option to `app/fields.py` (the suite fails without it) and regenerate `docs/options-reference.md`.
4. Tests: valid + invalid case in `tests/unit/test_config.py`, plus a behavioural test.
5. If it changes behaviour users see, update `README.md`, `docs/UPGRADING-*.md` and `CHANGELOG.md`.

## 7. Conventions

* **Typing:** `mypy --strict` on package *and* tests *and* tools. Use `X | None`, `list[...]`, `from __future__ import annotations`.
  Models are `@dataclass(slots=True)` (no `TypedDict`, no raw `dict[str, Any]` for domain data; `Any` only for MoviePy
  objects). NumPy arrays are `NDArray[np.float32]` etc. (`F32`, `F64`).
* **Docstrings:** every public function/class has one (what, args that are not obvious, returns). Comments explain *why*.
* **Logging:** use the `autoclip` logger (`log = logging.getLogger("autoclip")`). Every production function is
  wrapped with `@traced` (nested calls are intentionally not double-logged, so the inner tree does not drown
  callers). `.log` records are flushed and fsync'ed before the caller returns. The LLM is a local
  Ollama server and needs no API key.
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

* `tests/support.py` holds the pure helpers both suites import (`SENTENCES`, `Media`, `merge`, `make_cfg`,
  `write_config`, `install_stub_whisper`, the manifest loader); `tests/conftest.py` keeps only fixtures and gating.
* `tests/conftest.py`
  * `media` (session): generates a 40 s 320x180 video (speech-like tone bursts, 3 s on / 1 s off, 10 known sentences)
    with the bundled ffmpeg, plus `talk.json` (word timings), `talk.srt` and a `silent.mp4` without audio.
  * `make_cfg(media, out, source=None, **sections)`: a small fast `Config` (144 px short side, ultrafast, GPU off,
    one worker); sections are **deep-merged** with `merge()`. The parameter is called `source` because
    `video` is a config section name.
  * `bare_media(media, tmp_path, name="bare.mp4")`: a copy of the fixture video without the sidecar transcript
    (for Whisper/cache paths).
  * `_restore_logger` (autouse): `setup_logging` sets `propagate=False`; this resets it so `caplog` works.
* Patterns already in use - reuse them:
  * Whisper: `install_stub_whisper` in `tests/support.py` (real call signatures; returns the fixture transcript,
    or one chunk-relative segment per call when given an ndarray). Used by `unit/test_transcript.py` and
    `unit/test_runner_cli.py`.
  * Process pool: `ThreadPool(ThreadPoolExecutor)` monkeypatched over `runner.ProcessPoolExecutor` so scheduling logic runs
    (and is measured) in-process; one test uses real `spawn` workers.
  * LLM: the `llm_server` fixture is a real local `http.server` speaking Ollama and OpenAI-compatible formats.
  * Downloader: `tests/unit/test_download.py` replaces `yt_dlp.YoutubeDL` with `FakeYtDlp`, a scripted session that
    records every parameter, extracts synthetic info dicts and writes the **real fixture video** where yt-dlp would
    have (so probing, caching and trimming really run). The yt-dlp exceptions are the real classes (`YoutubeDLError`
    and its subclasses), so error handling is not faked. `tests/unit/test_cli_download.py` drives the flags that keep
    the downloader out of a run (local-only batches, `--allow`/`--deny`, inspection modes).
  * Network: tests that need the real internet carry `@pytest.mark.network`; `conftest.pytest_collection_modifyitems`
    skips them unless `AUTOCLIP_NETWORK_TESTS=1` or `AUTOCLIP_E2E_TESTS=1`. `tests/e2e/test_network.py` uses yt-dlp's
    own test video, and `.github/workflows/network.yml` runs it weekly with `-m "network and not e2e"` (the normal
    suite stays offline).
  * End-to-end: `tests/e2e/test_cli_e2e.py` carries both `e2e` and `network` and only runs with
    `AUTOCLIP_E2E_TESTS=1` (which implies network access). It drives `python -m autoclip` against the URLs of
    `tests/assets/youtube.json`, then asserts the real outputs (formats report, originals, subtitles, `clips.json`,
    `clips` table, probe dimensions/duration). Every case keeps its config, stdout, stderr, log and outputs under
    `tests/results/<timestamp>/` (`run.json`, `environment.json`, per-case `case.json`); the run also gets a flat
    `clips/` folder with every rendered `.mp4` and its `.srt`/`.jpg` sidecars, a `clips.json` index and
    `clips/index.html` (a gallery that plays them in a browser). The folder is ignored by git.
    `AUTOCLIP_E2E_RESULTS`, `AUTOCLIP_E2E_URL` and `AUTOCLIP_E2E_COOKIES` override the defaults. **The results
    folder is the source of truth for a run, not a tee'd stdout file**: a second session writing the same
    log path (shared `/tmp` folders are shared between OpenCode sessions) clobbers it mid-run. Never mark a
    fake-based test `e2e`.
  * E2E ladder: every manifest case downloads the first `AUTOCLIP_E2E_LEVEL * 360` seconds of its video (default
    level 1 = six minutes; the last level is the full video) into `tests/.cache/e2e/<id>/level_<n>` (or
    `AUTOCLIP_E2E_CACHE`), renders one short and one long clip, and writes `levels.json` so the run can compare the
    clips with the previous level. Deeper levels reuse the cached originals. `tests/e2e/test_server_e2e.py` starts a
    real `python -m autoclip.server` on a free port, uploads the fixture video, follows the WebSocket, downloads the
    ZIP, probes the clip and deletes the job (localhost only).
  * Server: `tests/unit/test_server.py` uses FastAPI's `TestClient` (the `fastapi` import is skipped only when the
    optional extra is genuinely absent); `tests/e2e/test_server_e2e.py` is the real-process check. The module-level
    `KEEPALIVE_SECONDS`/`EVICT_SECONDS` are patched instead of `asyncio.sleep`, `get_settings.cache_clear()` runs
    between tests, and the auth env vars are reset so one test cannot make another public or private.
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
  those tests skip when there is none (`has_display()`). CI runs them under `xvfb-run` on Linux. The missing-config
  flow is tested by injecting the platform (`default_config_path(system, env, home)`) and patching `messagebox`; an
  autouse fixture sets `AUTOCLIP_DEBUG=1` so a widget error fails the test with its traceback instead of opening a dialog.
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
   file path: the chunks have to be enhanced first. The whisper stub in `tests/support.py` tells a chunk from the
   whole file by its length.
7. **The sherpa-onnx wheel is CPU-only** (compiled without `SHERPA_ONNX_ENABLE_GPU`): the "cuda" provider silently falls
   back to the CPU. The denoiser is small, so *more* threads make it slower (RTF 0.21 at 4 threads, 0.30 at 16).
   `gtcrn` is ~4x faster than `dpdfnet8` at the same quality class - measure before changing the default.
8. **Denoising clean audio makes the ASR model drift into English.** Measured on a screen recording with an 82 dB
   signal-to-noise ratio: `gtcrn` and `noisereduce` turned the second half of an Arabic transcript into English, and
   every DPDFNet setting changed the text (CER 0.3-0.5 against the un-enhanced transcript, while the run-to-run noise
   of the same audio is ~0.1). That is why `enhance.snr_threshold_db` measures every chunk first and leaves clean
   audio alone. Only the DPDFNet models support `attenuation_limit_db`; a *larger* value allows *more* suppression.
9. **The padding and the cut snapping can push a clip over its maximum.** `CLIP_PAD_START`/`CLIP_PAD_END` and the
   cut snapping add up to ~3 s to a window that was within the range, so the window is clamped to `max_seconds` again
   after the snapping. The minimum is the harder half: the pause removal happens at render time, so `compute_keep`
   gets the kind's `min_seconds` and gives the smallest cuts back when the clip would end up too short.
10. **Drawing text straight onto an RGBA canvas leaves dark fringes.** `render_text_image` draws text/outline/highlight on
   separate pre-coloured transparent layers and alpha-composites them.
11. **MoviePy's default resize is Pillow-LANCZOS (slow).** `render._resize_effect` swaps in OpenCV (~3x faster overall).
12. **Containers report a slightly longer duration than the video stream;** sub-clips are clamped to `vclip.duration`.
13. **Cache hooks and SQLite connections are per thread/process.** A hook from another thread's `Cache` raises; that is
   why `after_operation` swallows hook exceptions. Never share a `Store` across threads.
14. **Scenes compete for a finite set of moments.** In short videos later scenes may get fewer clips than `count`; this is
    intended (and documented), not a bug.
15. **Do not use `caplog` for CLI tests** (see section 8) and do not call `setup_logging` in library code.
16. **Config key names matter:** `scenes` = scene types (a list is per-type clips, not a blend); shot-change snapping is
    `skip.snap`. Do not reintroduce `style`, `snap_scenes` or a `snap_to_cuts` file key.
17. **A frozen build has no Python and no package.** `tools/frozen_entry.py` is its entry point (a frozen
    `__main__` cannot use a relative import); it only calls `autoclip.frozen.main`, where no argument opens
    the desktop app and any argument runs the command line. The app runs *itself* for a pipeline
    (`core.process.pipeline_command` checks `sys.frozen`) instead of `python -m autoclip`. The bundle is a
    folder, not a single file: numpy, OpenCV, the Whisper runtime and ffmpeg are big and a folder
    starts much faster. The optional extras are not bundled - the app degrades gracefully without them.
18. **All AI model calls share one serialized queue.** Ollama (ranking, titles, social captions) and Stable
    Diffusion (thumbnails) never run concurrently, even in `performance` mode: every call takes the
    process-wide lock in `autoclip.core.ai`. A missing or failing backend is logged and skipped; the
    clip keeps its heuristic result. Stable Diffusion thumbnails are generated after `render_all` in the
    main process and replace the frame JPEG only when they succeed. LLM calls carry the file's
    `llm.system_prompt` (empty keeps the built-in messages); the thumbnails use the file's `image.prompt`.
    Every Ollama call is made with a 180 s timeout and an 8k context window (`LLM_TIMEOUT` / `LLM_NUM_CTX`), and
    the temperature is 0, so the response never depends on a sampling seed. The run asks the server once for its
    model list (`llm_available(base_url, model)`, cached) and disables the AI helpers when the configured model is
    missing instead of erroring per clip; the old per-call probing is gone. Re-rank replies are constrained with a
    JSON schema (`_RERANK_SCHEMA`) and validated with pydantic: scores are clamped to 0-10, titles are cut to 80
    characters, ids outside the candidate list are ignored, and an off-schema reply raises - `moments._rerank`
    catches it and keeps the heuristic order.
19. **The Linux packages wrap the same build.** The AppImage and the snap do not compile anything:
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
22. **Two different `DownloadError` classes exist.** yt-dlp has `yt_dlp.utils.DownloadError` (and it is *not* a
    subclass of autoclip's), and importing it into `download.py` would shadow `errors.DownloadError`. The downloader
    keeps yt-dlp's base class as `self._ytdl_error = YoutubeDLError` and catches that; anything else would either
    miss yt-dlp's failures or swallow autoclip's own. Never `except DownloadError` around a yt-dlp call.
23. **The bundled imageio-ffmpeg has no ffprobe.** `download.time_range` is handed to yt-dlp only when a *real*
    `ffmpeg` **and** `ffprobe` are on `PATH` (`Downloader._range_supported`); otherwise the whole video is downloaded
    and `media.trim_video` cuts it locally (the SRT is shifted and clipped too). Unsilence therefore decodes audio with
    `media.decode_audio`, never `pydub.AudioSegment.from_file` (which calls `ffprobe` and silently fell back to the
    original on every machine without a system ffmpeg, including CI and the frozen builds). yt-dlp needs
    both binaries to cut a range itself, so a "range" that looks supported is not: verify the duration
    after the download.
24. **The format id is part of the originals cache prefix** (`<safe title> [<id>] [<format id>]`). Changing `quality`,
    `fps`, `codec`, `container` or a `time_range` therefore downloads a new file instead of reusing one that no longer
    matches the request. The price is that the cache is only checked *after* extraction - the format id can only come
    from yt-dlp - so even a fully cached URL makes a small network request.
25. **YouTube needs a JavaScript runtime to offer every format.** `js_runtime.name: "auto"` probes `deno -> node ->
    bun -> quickjs` and logs a warning when none is installed; `remote_components: "ejs"` lets yt-dlp fetch its EJS
    component from GitHub. A machine without a runtime still works but may only see the SABR/muxed formats - do not
    treat "few formats" as a bug in `formats.choose`.
26. **The offline suite proves our code, not yt-dlp compatibility.** `tests/unit/test_download.py` replaces `YoutubeDL`
    with a scripted fake, so a yt-dlp release can rename an internal import (`yt_dlp.globals`,
    `yt_dlp.utils.dfxp2srt`, `download_range_func`) without a failure here. `tests/e2e/test_network.py`
    (`@pytest.mark.network`, gated by `AUTOCLIP_NETWORK_TESTS=1`, run weekly by `.github/workflows/network.yml`) and
    the CLI end-to-end cases (`@pytest.mark.e2e`, gated by `AUTOCLIP_E2E_TESTS=1`) are the checks that the real API
    still works. Keep both: never mark a fake-based test `network` or `e2e`.
27. **Never patch `asyncio.sleep` globally in a server test.** Replacing it with a no-op starves every await in
    `TestClient.__enter__` and the suite hangs forever instead of failing. `KEEPALIVE_SECONDS` (`routers/api.py`) and
    `EVICT_SECONDS` (`main.py`) are module constants precisely so a test can shrink just those loops.
28. **Starlette's test client now prefers `httpx2`.** `starlette.testclient` catches `ModuleNotFoundError` (not the
    broader `ImportError`) when falling back to `httpx`, so a broken `httpx2` install fails loudly. `httpx>=0.27` is in
    the `dev` extra/`requirements-dev.txt` for Python 3.10, where the newer package may not be available; the runtime
    `server` extra does not need it.
29. **Do not add a console span exporter to the server.** It printed every span and, once the test session closed its
    streams, raised `ValueError: I/O operation on closed file` at shutdown. OTel is configured only when
    `OTEL_EXPORTER_OTLP_ENDPOINT` is set; otherwise no span processor is added.
30. **YouTube's HLS/DASH variants can throttle a range download to ~24 KB/s while looking healthy.** The old format
    choice (`625+251`) took eight minutes for six minutes of video; the direct HTTPS VP9 variant (`313+140`) downloads
    at normal speed. `formats._protocol_rank` therefore prefers a direct HTTP(S) format at equal quality and fps.
    Related: `audio_language: null` must follow `info["language"]`, not the highest-bitrate track - otherwise a Malay
    dub wins over the original English audio. A direct-HTTPS format is no guarantee either: an `313+140-18` stall
    (read timeouts through the tv-samsung/visionos SABR path) burned a whole video on Windows until the alternates
    fallback (`choose_candidates`, at most three tries) made it resilient. Note yt-dlp emits SABR format ids with a
    `-N` instance suffix (`140-18` among nineteen `140-*` entries); the 0.001 kbps bitrate difference tips the sort,
    which is fine as long as the fallback treats a stall as retryable.
31. **MoviePy ignores its `pixel_format` argument.** `write_videofile` derives the pixel format from the clip (an RGBA
    composite stays RGBA) and only appends its own `-pix_fmt` for libx264/libvpx/h264_nvenc, so libx265 received i444
    and its `main` profile rejected it (`main profile not compatible with i444 input chroma subsampling`). `render.encode`
    therefore puts `-pix_fmt yuv420p` *before* the encoder parameters, and `render_clip` flattens the composite with
    `without_mask()`. Keep both: the forced format is what makes HEVC work, the flattening drops needless alpha work.
    `tests/unit/test_render.py::test_render_hevc_with_captions_writes_yuv420p` is the real-encode regression test.
32. **A frozen Windows launch hides its console, it does not detach it.** The binary is a console program because
    the command line needs stdout, so a double-click opens a console before the app appears. `app.console.hide_console`
    only calls `ShowWindow(SW_HIDE)`, and only when `GetConsoleProcessList` returns 1 (this process is the sole user;
    a terminal launch also lists the shell). Detaching (`FreeConsole`) would close the handles the workers write to and
    would give console-subsystem children a new visible console; it also does not reliably close a Windows Terminal tab.
    The handle from `GetConsoleWindow` is pointer-sized (`restype = c_void_p`): truncated to 32 bits it hides nothing on
    64-bit Windows. The tkinter check runs before hiding, so a missing toolkit still produces a readable error. Windows
    Terminal may ignore the request - the app still opens. Unit tests use a fake `ctypes.windll`; real Windows
    behaviour is not verified in this sandbox.
33. **The unsilence re-encode must never use MoviePy `method="compose"`.** Compose holds all concatenated parts as simultaneous layers; RSS grows linearly with the segment count (~15 MB per segment — measured 437 MB at 10 parts, 4.9 GB at 300), which OOM-killed a several-hour recording. `method="chain"` splices frames in order, is flat (~98 MB) and ~6x faster; `tests/unit/test_unsilence.py::test_remove_silence_concatenates_with_chain_not_compose` pins it. All parts come from one source, so they share size, fps and aspect — compose's resolution-mismatch handling is never needed.


## 10. Recipes

* **New scene type:** add weights to `STYLES` and a one-line description to `STYLE_HELP` (and lexicon words to `LEXICONS`
  if a new feature name is used). `ClipsCfg` validation and `"all"` pick it up automatically. Add it to the expectations in
  `tests/unit/test_moments.py::test_every_scene_type_prefers_its_own_window`.
* **New GPU/CPU encoder:** add a builder to `hardware.HARDWARE` (and the OS to `PRIORITY`), add the literal to `VideoCfg.gpu`,
  test arguments in `tests/unit/test_hardware.py`. If it needs its own `-preset` names, put them in `params`.
* **New clip kind:** add a `KindCfg` field on `ClipsCfg` with defaults, extend the `kinds` `Literal`, `ClipsCfg.kind()`, and
  the output-folder logic in `cli.process_video`; update tests and README.
* **New download option:** add a typed field to the right `download.*` dataclass (default included), use it in
  `Downloader._base_params` / `_media_params` / `formats`, add its entry to `app/fields.py` (the suite fails without it),
  document it in README's "Downloading URLs" and cover it in `tests/unit/test_config.py` (valid + invalid) and
  `tests/unit/test_download.py` (the recorded yt-dlp parameters).
* **New platform family, alias or group:** edit `platforms.py` (`FAMILIES`, `ALIASES`, `GROUPS`), test it offline in
  `tests/unit/test_platforms.py` and update the README table. Family names are checked against yt-dlp only at run time, so
  none of this needs the network.
* **DB schema change:** edit `store.SCHEMA` (the DB is recreated every run - no migrations), update `Store` methods and
  `tests/unit/test_infra.py`. Remember workers open their own connection.
* **Caption styling:** `captions.render_text_image` takes a `CaptionsCfg`; sprite cache keys include the whole style
  (`sprite_key`), so new style fields are picked up automatically if they are in `CaptionsCfg`.
* **Performance work:** profile one clip with `python -m cProfile -o p.out -m autoclip cfg.json` (use `mode: "ranges"`).
  Typical hot spots are MoviePy's compositor and per-frame resize/blur; most remaining time is inside MoviePy.
* **Debugging a run:** run with `AUTOCLIP_DEBUG=1` (the dev mode: DEBUG everywhere), read `{output_dir}/logs/autoclip_*.log` (START/END per operation,
  `submit clip ... allowed=N threads=M`, `pid=` shows which worker did what), inspect `.work/{video}/autoclip.db`
  (`SELECT * FROM clips`).

## 11. Environment notes for sandboxed agents

* The Hugging Face host (Whisper models) may be unreachable: use the stub (`install_stub_whisper`) or a sidecar `.srt` next to the video.
* No GPU may be present: GPU code paths are verified by probe/fallback behaviour and software stand-ins only.
* One CPU core: use `mode: "performance"` and the thread fake (or monkeypatch `RESOURCES.cpus` to more than one) to force the parallel code path.
* Long-running commands (e.g. 30+ minute media) must be detached (`setsid nohup ... &`) and polled.
* A 35-minute synthetic video with a chunk-aware Whisper stub is the reference for chunking behaviour (7 chunks, ~19 MB
  of audio each, ~30 s end to end on one core, peak RSS ~435 MB).

### What is verified vs not
* Verified here: Linux; real ffmpeg encode/decode; spawned workers; GPU *detection and CPU fallback*; chunked
  transcription with a stub; per-OS logic through injected platforms; 100% branch coverage.
* Downloader: the yt-dlp **Python API** was exercised here with the installed version against a small YouTube
  video (format listing/choice, extractor filtering, subtitle attachment, sidecars, imageio-ffmpeg merge, range
  trimming and the originals cache) - see the release notes for the exact checks. Every other platform, real
  cookies, proxies and JS runtimes beyond the one installed here were **not** exercised end to end.
* **Not verified on real systems:** Windows and macOS execution (CI matrix exists), real GPU encoders (NVENC/QSV/AMF/
  VideoToolbox arguments are unit-tested and probe-guarded but never ran on hardware), real Whisper inference.
  Never state these as verified.

## 12. Boundaries

**Always**
* Run the four gates (section 2) and keep coverage at or above the 100% gate.
* Add/adjust tests with every change; update `README.md`, `AGENTS.md` and `docs/` when architecture,
  commands, release flow, behaviour or operational guidance change.
* Keep config backward-compatibility notes in the README "Upgrading" section when keys change.
* Preserve cross-platform behaviour (paths, encodings, spawn, optional dependencies).
* Keep the clip folders clips-only: downloaded originals, subtitles and metadata sidecars live in
  `.work/downloads/`, never next to the clips.
* Keep the downloader policy (allow/deny, groups, aliases, format choice) offline-testable: no test in the normal
  suite may need the network.
* Keep the config catalogue complete: every file option appears in `app/fields.py` (nine today) and the docs match.

**Ask first**
* Adding or upgrading dependencies; raising the minimum Python version.
* Breaking changes to config keys, output layout or exit codes.
* Changing DB persistence semantics (fresh-per-run) or the transcript-cache key.
* Anything that touches how GPU encoders are selected or how workers are capped.
* Changing the originals cache key, the platform policy defaults (`@popular`), or the subtitle precedence rules.

**Never**
* `shell=True`, `eval`/`exec`, pickle/`allow_pickle=True`, string-built SQL with user data, network calls other than the
  configured LLM URL and the downloader of URLs that are in `input`.
* Log or print secrets; weaken `llm.base_url` validation.
* Lower the coverage gate, add `# pragma: no cover` to dodge a test, or silence mypy/ruff globally to get green.
* Rebind `RESOURCES`, share a `Store` across threads, use `fork`, or hard-code POSIX paths/separators.
* Reintroduce candidate storage in SQLite, blended multi-style ranking, or file-level video splitting for chunking.
* Bypass the platform policy, expose a yt-dlp option that runs a command (`netrc_cmd`, `exec`, ...), or claim the
  fake-based downloader tests prove compatibility with a real yt-dlp release.
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
* **originals cache** - `{output_dir}/.work/downloads/`: the downloaded files and their
  sidecars, reused between runs when the name prefix (title, id, format id, range) matches.
* **platform family** - the canonical yt-dlp extractor family of a URL (`youtube`, `tiktok`, ...); what
  `download.allow`/`download.deny` talk about. `generic` means "any other web page".
* **group** - a named set of families (`@popular`, `@social`, `@video`, `@all` or a user group) usable in
  `allow`/`deny` and for per-group settings.
