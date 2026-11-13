# autoclip

Cut long videos into the best **short** and **long** clips (TikTok / Reels / Shorts and YouTube-style highlights).
Free, local, one command. Runs on **Linux, macOS and Windows** (Python 3.11+). No system ffmpeg needed.

```bash
pip install -r requirements.txt
pip install -U "Pillow>=12.3"          # security fixes; MoviePy's "<12" pin is only conservative (CI tests both)
python -m autoclip --init config.json  # starter config (nine keys)
python -m autoclip config.json         # or: autoclip config.json  (after `pip install -e .`)
python -m autoclip                     # no config: ./input.mp4 with the built-in defaults
python -m autoclip.app                 # the desktop app (same options, explained)
```

## The desktop app

`python -m autoclip.app` opens a native window (Tk, the look of your system) with **every option of
the config file**:

- four config tabs - *Videos*, *Clips*, *Download*, *AI* - each option with a one-line summary under it,
  plus a runtime output-folder control that is never written to the config file;
- a details panel that explains the option you clicked: **what** it does, **why** you would change
  it and **when** it is the right moment;
- *All options...* opens the same explanations in a searchable list;
- *Run* writes the config and starts the pipeline (the log appears in the window), *Stop* ends it;
  *Save config*, *Load config...* and *Reset* work on the config file;
- the config file lives in the usual per-OS folder (Linux `~/.config/autoclip/config.json`,
  macOS `~/Library/Application Support/autoclip/config.json`, Windows `%APPDATA%\autoclip\config.json`);
  a missing file is fine - the built-in defaults are kept in memory (no dialog, nothing written) until
  you press *Save config*, *Save as...* or *Run*; a broken config resets the form to the defaults
  without overwriting the file;
- drag video files onto the **Videos** list to add them (when `tkinterdnd2` is installed; otherwise the
  *Add videos...* button does the same), and drag the divider above the log: every pane keeps between
  20% and 80% of the height, so no section can be squeezed away, and the action buttons at the bottom
  are a little taller than the platform default so they are easy to hit;
- *Copy error* and *Copy log* in the log header put the last error (or the whole log) on the clipboard;
- `AUTOCLIP_DEBUG=1` makes an unexpected window error print its traceback, close the window and
  re-raise it (the default keeps Tk's error dialog).

The app and the command line share the same engine and the same config file - what the window builds
is exactly what `python -m autoclip config.json` would run.

## Output layout

```
{output_dir}/
  short/{video}/   {video}_{scene}_{nn}_{start}s.mp4 + .jpg cover + .srt, clips.json     (9:16, 20-60 s by default)
  long/{video}/    same, for the long clips                                               (16:9, 3-10 min by default)
  logs/            autoclip_<timestamp>.log
  .work/{video}/   autoclip.db (recreated empty on every start) and the transcript cache
  .work/downloads/ originals downloaded from URLs (reused by later runs)
```

A kind that is not requested (or finds nothing) gets no folder. Several videos share the same `short/` and `long/` folders. A *relative* `output_dir` is resolved against the current working
directory - not against the config file - so the output folder is independent of where your config lives. An absolute `output_dir`
(or the default `~/Videos/clips`) is used as-is.

## Models and caches

Nothing is downloaded twice on purpose:

* **Whisper models** live in the normal HuggingFace cache. The model is downloaded automatically on first use
  (or prefetched with `autoclip models pull`).
* **The speech enhancement model (sherpa ONNX)** is kept in `~/.cache/autoclip/models` (on Linux; the platform's cache folder
  everywhere else) and is shared by every output folder, so changing the output location no longer re-downloads it.
* **Transcripts** are cached per video in `{output_dir}/.work/{video}/{video}.transcript.json` and reused while input, model,
  language and settings agree; a stale cache is re-transcribed.
* **Downloaded originals** plus a small `autoclip-cache.json` in the originals folder make a second identical run (same URL
  and settings) succeed **without a single network call** - the originals are reused.
* **Caption sprites** are pre-rendered once per word and cached compressed (RAM LRU over SQLite), so re-runs are instant.

## Configuration

The config file has **nine keys**. Everything else is a runtime override (command-line flag or environment
variable) or derived automatically.

```json
{
  "input": ["talks/*.mp4", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
  "clips": {
    "preset": "both",
    "scenes": "best",
    "keywords": [],
    "ranges": []
  },
  "download": {
    "allow": ["@popular"],
    "deny": []
  }
}
```

| Key | What it controls |
|---|---|
| `input` | videos, globs or http(s) URLs |
| `clips.preset` | `both`, `shorts` or `long` - which clip families to render |
| `clips.scenes` | `best`, a list, or `all` - the scene types to pick moments for |
| `clips.keywords` | words or phrases that boost a moment |
| `clips.ranges` | exact cuts (`"00:01:00-00:01:45"`), one per line |
| `download.allow` | extractor families or groups (`@popular`, …) that may be downloaded |
| `download.deny` | families that are never downloaded, even when an allow entry covers them |
| `llm.system_prompt` | the system message of the local LLM (moment ranking, titles, captions); empty keeps the built-in editor messages |
| `image.prompt` | the Stable Diffusion prompt of every AI thumbnail |

The built-in clip profiles are fixed: **short** = 20-60 s at 9:16, **long** = 3-10 min at 16:9. Use `clips.ranges`
or `--split` for other shapes. `clips.scenes` is `"all"`, one type, or a list: `best attractive exciting funny sad
emotional inspiring calm educational controversial`.

Runtime overrides (flags win over environment variables):

| Flag | Env var | What it does |
|---|---|---|
| `--output-dir DIR` | `AUTOCLIP_OUTPUT_DIR` | output folder |
| `--cookies BROWSER\|FILE` | `AUTOCLIP_COOKIES` | cookies for downloads |
| `--skip NAME` | `AUTOCLIP_SKIP` | skip pipeline steps (comma separated) |
| `--llm-model MODEL` | `AUTOCLIP_LLM_MODEL` / `AUTOCLIP_LLM_URL` | local LLM model / URL |
| `--split` | — | cut the video into equal 600 s parts |
| `--mode …` | `AUTOCLIP_MODE` | performance / resources / balance |
| `--quality …` | `AUTOCLIP_QUALITY` | output quality tier |

Env values apply first; an explicit flag always wins. `--init` and `--init-full` both write the complete nine-key file.

A wrong *type* or an unknown key is still fatal (`ConfigError`). A wrong *value* warns
(`config: invalid value for <path>; falling back to default`) and uses the field default, so a typo in one option no
longer aborts the run. Keys starting with `_` are ignored, so `"_comment"` entries are fine. Removed keys from 3.x
warn and are ignored (see `docs/UPGRADING-4.0.md`).

See `docs/options-reference.md` for the generated reference of the nine keys.

## Short and long clips, scene types

```json
{
  "input": ["talks/*.mp4", "interview.mov"],
  "clips": {"preset": "both", "scenes": "all", "keywords": [], "ranges": []}
}
```

* `scenes` is `"all"`, one type, or a list. **Every scene type gets its own clips** (`count` is per scene type and per
  kind), so `"all"` with the default counts means up to 10 short clips and 2 long clips per scene type. A moment picked
  for one scene is never reused under another label; scenes pick in the order listed, so in a short video the later ones
  may find fewer free moments than `count`.
* Each kind has its own length range, count and `aspect` (`9:16`, `1:1`, `4:5`, `16:9`, `original`).
  If a video is too short for a kind (e.g. 2-minute video, long clips from 3 minutes) that kind is skipped with a warning.
* **The length of a clip is `clip(minimum, maximum)`:** `min_seconds` is a hard floor (the pause removal gives cuts back
  rather than going below it), `max_seconds` is a hard ceiling (padding around the cut cannot push it over).
* `clips.ranges` cuts your own ranges (scene type `custom`), applies to both kinds; `--split` cuts equal 600 s parts into
  `long/{video}/` and cannot be combined with `clips.ranges`.
* `input` is a path, a glob or a list. One broken video or clip never stops the batch. Exit status: `0` all done,
  `1` something failed, `2` bad configuration. A transcript is picked up from a sidecar `.json`/`.srt` next to each video.

## Downloading URLs (YouTube, TikTok, Vimeo, ...)

`input` accepts **http(s) URLs next to files and globs**. A URL is downloaded with
[yt-dlp](https://github.com/yt-dlp/yt-dlp) (installed with autoclip) and then goes through exactly the
same pipeline as a local file:

```json
{
  "input": ["talks/*.mp4", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
  "download": {"allow": ["@popular", "vimeo"], "deny": ["tiktok"]}
}
```

* **Quality and format are automatic**: the download short side follows the output quality tier (1080/720/480/360/240 for
  highest/high/medium/low/lowest). A direct HTTP(S) format beats an HLS/DASH variant with the same quality and fps (the
  segmented one reports an inflated bitrate and throttles badly), and the video's original audio track is kept over a
  higher-bitrate dub. Cookies (`--cookies`/`AUTOCLIP_COOKIES`) are a runtime-only choice; downloads use the network directly (there is no proxy option).
* **Originals** are kept in `{output_dir}/.work/downloads/` and reused by later runs; the clip folders stay clips-only.
  The output name is `<title> [<video id>]`, and a time range adds `[<start>-<end>]`.
* **Resilience:** if the chosen format stalls or fails to download, the downloader tries the next-best candidate(s)
  (at most two alternates, then the video is reported failed). A different candidate is reused from the cache when it is
  already there; the chosen format is logged with its resolution, codec, bitrate and protocol.
* **Platform policy:** `download.allow` / `download.deny` hold extractor families (`youtube`, `tiktok`, `vimeo`, ...) or
  groups. `deny` wins, the default allow list is `@popular`, and the `generic` family (any web page with a video) is only
  used when it is named - so a random link is refused before a request is made. Built-in groups are `@popular`,
  `@social`, `@video` and `@all`; name families or groups directly in allow/deny (there is no `groups` file key).
* **Playlists:** a URL whose shape names a list (channel/playlist/album) downloads every entry up to 50; a watch URL stays
  single. Metadata sidecars (`.info.json`, thumbnail, description) are always written next to the original.
* **Subtitles:** the video's original track is downloaded (a manual track is preferred; automatic captions are used
  when there is no manual one). A downloaded subtitle is used when it exists. A sidecar transcript (`<video-stem>.json`
  preferred over `<video-stem>.srt`) next to the video always wins.
* **Command line:** without a config file the built-in defaults run `./input.mp4` (a missing file is reported with a hint,
  exit status 2, and nothing is written); `--download-only config.json` downloads the URLs and stops; `--formats URL`
  prints every available format and the one autoclip would choose (no config file needed); `--platforms [URL]` prints the
  allow/deny policy and where a URL would go; `--allow a,b` and `--deny c` override the loaded lists for one run.
  Inspection modes create no folders and configure no logging. A run with only local files never imports yt-dlp and makes
  no network request.

## AI (optional, local)

All local LLM calls (Ollama or any OpenAI-compatible server) and Stable Diffusion thumbnails go through one
process-wide lock, so even in `performance` mode they never run concurrently. A missing backend is logged and skipped,
never fatal; the clips keep their heuristic result.

* `--llm-model MODEL` (`AUTOCLIP_LLM_MODEL`/`AUTOCLIP_LLM_URL`) selects the model and server; a local Ollama at
  `http://localhost:11434` with `bonsai-2:2b` is the default. At the start of a run autoclip asks the server once for
  its model list (`/api/tags`, cached) and keeps the heuristic ranking when the configured model is not installed.
* Every Ollama call is made with a 180 s request timeout and an 8k context window (the server default can be as low
  as 4096, and an overflow silently truncates the prompt); the temperature is 0. Re-rank replies are constrained with
  a JSON schema and validated: scores are clamped to 0-10, titles are cut to 80 characters, ids outside the candidate
  list are ignored; anything invalid falls back to the heuristic ranking.
* With an LLM available, highlight moments get re-rank scores, generated titles and social captions; long clips get
  heuristic titles otherwise. Captions are written as `<clip>.txt` sidecars.
* Stable Diffusion thumbnails replace the frame JPEG only when `torch`/`diffusers` are installed; otherwise the frame
  JPEG is kept.
* `llm.system_prompt` (file) replaces the system message of every LLM call for your house style or rating rubric
  (empty keeps the built-in editor messages); `image.prompt` (file) sets the Stable Diffusion prompt of the AI thumbnails.
* `--skip llm` skips every Ollama call. Whisper model downloads are automatic; `--skip stt` avoids Whisper entirely (a sidecar, downloaded or cached transcript is used instead).

## Automatic resource use

The engine watches its own headroom and degrades gracefully instead of dying:

* The render pool re-checks memory **before every submission**; when another clip is already in flight and RAM
  is tight (or an OOM was seen), the next submission waits until a clip finishes. An idle pool always starts the
  batch - no machine configuration can deadlock the run.
* Once an Ollama call fails (timeout, missing model, off-schema reply), the AI helpers are switched off for the
  rest of the run with one log line; the heuristic result is kept.
* The resource history (peak RSS, CPU) feeds the adaptive CPU budget, so the engine gets quieter or louder on its
  own as the machine's condition changes - see the `automation ...` log lines and `AUTOCLIP_DEBUG=1` for the details.

## HTTP server (optional)

The same pipeline runs as a small FastAPI service for a web front end or another program: upload a
video, follow the steps over a WebSocket, then download a ZIP with the clips, their `.srt`/`.jpg`
sidecars and every `clips.json` (relative paths mirror the output folder, so the two indexes of a
short and a long clip do not collide). The service calls the pipeline stages one by one (probe,
transcript, selection, render) so progress is visible while the job runs; it is stateless apart from
the render cache on disk.

The upload page offers a **Preview**: it takes one representative frame of the video and renders it
with the captions and layout your config would use (a mock sentence, highlighted word, the configured
font, theme and position — never a Whisper call and never the real clip), so you can see the look
before committing to the run. The preview uploads the file once more and the PNG is discarded after
the response; nothing is written to the output folder and no job is created. A preview is an
illustration — real captions follow the transcript and the cut.

```bash
pip install -e ".[server]"
python -m autoclip.server              # http://127.0.0.1:8000/view/  (HTML) and /docs (OpenAPI)
# or
docker compose up --build
```

| endpoint | what it does |
| --- | --- |
| `GET /view/` | the HTML upload page (Jinja2 + HTMX + DaisyUI) |
| `GET /view/jobs/{job_id}` | the job page with progress and the clip links |
| `POST /api/jobs` | multipart upload: `file` plus an optional `config_json` string or `config` JSON file |
| `GET /api/jobs/{job_id}` | state (`queued`/`running`/`completed`/`failed`), step, percent, error, outputs |
| `GET /api/jobs/{job_id}/download` | a ZIP of the finished clips and their sidecars |
| `DELETE /api/jobs/{job_id}` | remove the job and its files |
| `WS /api/jobs/{job_id}/ws` | the current snapshot, then every step/percent update (plus keepalive frames) |

The uploaded config keeps only the nine keys; `input` and `output_dir` are always overridden because the server owns
the uploaded file and the job folder. The defaults are the same as the command line - notably the download short side
follows the quality tier, source fps, no upscaling and `quality: "high"` - so a server job produces high-quality clips
unless the config asks for something else.

| environment variable | default | meaning |
| --- | --- | --- |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | where uvicorn listens |
| `MAX_FILE_SIZE` | `1073741824` (1 GiB) | larger uploads get HTTP 413 |
| `MAX_CONCURRENT_JOBS` | `1` | renders running at once (the rest wait in the queue) |
| `OUTPUT_DIR` / `CACHE_DIR` | `.server/outputs` / `.server/cache` | jobs, work folders and ZIPs |
| `JOB_TTL_SECONDS` | `3600` | expired jobs are evicted in the background |
| `AUTH_USERNAME` / `AUTH_PASSWORD` | empty | HTTP Basic auth **only when both are set**; otherwise the API is public |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | empty | export OpenTelemetry spans to a collector (none are exported when unset) |

If auth is enabled, the WebSocket takes the same credentials as `?token=<user>:<password>` (put the
server behind HTTPS when it is reachable from outside). The Docker image installs ffmpeg and the
`server` extra; `docker-compose.yml` mounts two named volumes and reads the same variables.

## Where the work runs

The GPU is used automatically when there is one and it really works (probed by encoding a test clip; NVENC/QSV/AMF/VideoToolbox,
transcriber with CUDA). Otherwise everything runs on the CPU and the log says so. A `GPU` you cannot really use is never selected.

On the CPU the transcriber is much slower: `large-v3` needs roughly ten times the time of `small`, so pick a smaller
model tier (via `--quality`) for long videos on a CPU-only machine. `--quality` also selects the transcription model:
highest/high use `large-v3` (high drops to `small` with no GPU), medium `small`, low `base`, lowest `tiny`.

## Media quality (`--quality`)

`--quality` is the single global output-quality control instead of low-level encoder knobs:

| value | resolution vs 1080 short side | frame rate | audio |
| --- | --- | --- | --- |
| `highest` | 100% | source | 256 kbps AAC |
| `high` (default) | 80% | source | 192 kbps AAC |
| `medium` | 60% | capped at 30 fps | 128 kbps AAC |
| `low` | 40% | capped at 24 fps | 96 kbps AAC |
| `lowest` | 20% | capped at 15 fps | 64 kbps AAC |

Dimensions are rounded to even numbers and the source is never enlarged. The chosen level also selects CPU/GPU encoder
arguments; users no longer configure CRF, FFmpeg presets, bitrate caps or audio bitrates.

## GPU acceleration (automatic, with CPU fallback)

The encoder is used only if ffmpeg can **actually encode a test clip with it on this machine** (ffmpeg builds list
encoders such as `h264_nvenc` even on computers without that GPU); otherwise libx264 on the CPU is used and the log says so.
* GPU workers are capped at 3 (consumer GPUs limit simultaneous encode sessions).
* Transcription uses the GPU too when CUDA is available. A GPU that fails mid-file (missing cuBLAS, out of memory, ...) is
  retried on the CPU for that chunk; if the CPU also fails the step is skipped and logged ("transcription skipped (...):
  selecting on audio only; captions and titles need a transcript") instead of aborting the run. Voice-activity detection is
  retried without VAD when its model is unavailable.
* Only the **encode** step is on the GPU: decoding and compositing (captions, blur, layout) stay on the CPU, so expect a
  speed-up in encoding, not a 10x end-to-end gain.

## Very long videos

The transcript is produced in **fixed 5-minute chunks chosen automatically**; the boundaries move to the quietest
moment nearby and the language is detected once and reused. Neither the denoiser nor Whisper ever holds more than one
chunk in memory, so a 3-hour file no longer needs 3 hours of audio at once. Analysis already streams, noise reduction
works in 60 s pieces, and selection stays **global**, so a clip is never cut at a chunk edge and a long clip may be
longer than a chunk. Very short videos are transcribed in a single chunk; chunking is not user-tunable.

## Unsilencing (automatic)

Silence removal is always attempted before processing: the media is copied to `{output}/.work/{stem}/unsilence_{stem}.mp4`
using pydub silence detection and a conservative MoviePy cut. Every later stage, including transcription, selection and
rendering, then operates only on the audible stretches. pydub needs FFmpeg to load the audio, and the feature fails
softly to the original file when that dependency or the audio itself are unavailable (and platform subtitles are dropped,
since the new timeline is no longer aligned).

## Clear speech for the transcriber (on by default)

Before the audio is transcribed it is **enhanced**, so the ASR model hears clear words instead of room noise,
keyboard clicks or hum. Enhancement is automatic:

* `sherpa` (small ONNX denoiser) or `noisereduce` (spectral gating, no model) is chosen internally; `gtcrn` is the
  fastest model, `dpdfnet8` the best.
* **Denoising only helps when there is noise.** A clean recording is damaged by it and the ASR model then drifts into
  the wrong language, so every chunk is measured first and left alone when it is clean enough.
* Enhancement is part of the transcript cache key: changing it re-transcribes the video.
* `--skip denoise` transcribes the untouched audio (and changes the cache key).
* `--skip audio_enhance` leaves the clip audio untouched (no high-pass, loudness or limiter).

## Transcription

* The model follows `--quality` (highest/high `large-v3`, medium `small`, low `base`, lowest `tiny`; high drops to
  `small` with no GPU). On the CPU a model that does not fit into the free RAM is replaced by a smaller one.
* The audio is transcribed in fixed 5-minute chunks: neither the denoiser nor Whisper ever holds more than that,
  and the language is detected once and reused for the remaining chunks.
* A sidecar transcript (`<video-stem>.json` preferred, `.srt` otherwise) next to the video is used; `--skip stt` stops
  before Whisper (a sidecar, downloaded or cached transcript is still used). The language is always auto-detected.

## What it does

1. Enhances the audio and transcribes it with faster-whisper (auto language), or loads `.srt` / `.json` (or a downloaded subtitle).
2. Scores sentence-aligned windows per scene type from audio energy, speech rate, hooks and lexicons; optional local-LLM re-rank.
3. Renders each clip: denoise + loudness, cuts long pauses and "um/uh", blurred background, word-highlighted captions inside
   the safe zone, fades, H.264 + AAC (`+faststart`).

## Skipping steps

Every step runs by default. `--skip NAME` (or `AUTOCLIP_SKIP=a,b`) leaves individual steps out without changing the
rest of the config - useful for a quick render, a silent clip or a run that must not download a model:

| name | what is left out |
| --- | --- |
| `stt` | Whisper. An explicit, downloaded or cached transcript is still used; without one the run continues without text (highlight selection then falls back to audio-only windows). |
| `denoise` | Speech enhancement before Whisper **and** noise reduction of the clip audio. The transcript cache key changes. |
| `audio_enhance` | High-pass, loudness normalisation and the limiter of the clip audio; the audio stays at its original level. |
| `audio` | The audio track: clips are muted and the audio is not decoded (so pauses cannot be removed either). |
| `pauses` | Keep long pauses and filler sounds in the clips. |
| `caption` | No burned-in captions. |
| `title` | No overlay title, even when the model generated one. |
| `fade` | No video or audio fades. |
| `thumbnail` | Keep the frame JPEG instead of generating an AI thumbnail. |
| `llm` | No Ollama calls (ranking, titles, social captions). |
| `snap` | Do not snap cut points to shot changes. |

`--skip audio` mutes the clips and stops audio decoding. Setting `--skip caption`/`--skip title` lets ranges/split runs
avoid transcription entirely when no text is needed.

## Automatic performance mode (`--mode`)

`--mode` decides how the run balances throughput against resource usage:

| value | meaning | typical use |
| --- | --- | --- |
| `performance` | Use as much CPU and memory as the configured limits allow | dedicated machine / servers |
| `resources` | Stay as lean as possible; render in-process and keep memory pressure low | laptops, shared machines |
| `balance` | Start in the middle and adapt after every batch (default) | default |

The mode is **adaptive, not static**: before each render batch the worker plan is recomputed from current free
memory and CPU, and after each completed batch the monitor smooths the observed utilisation (EMA) and nudges the
CPU budget up or down. If memory becomes tight or a worker dies, parallelism drops automatically; when utilisation
is below target and memory is calm, budget is raised (within the mode's bounds). Containers/cgroups are honoured
on Linux, and macOS/Windows/Linux all use the same affinity/`psutil`-based accounting. Automatic values are logged
as `automation ...`, and the current CPU-budget choice is persisted under `<output_dir>/.work/adaptive.json` so the
next run starts close to the learned optimum.

## Resources, storage, logs

* Before and after every operation the script re-reads CPU count (affinity + container quota) and *free* RAM. The number
  of parallel renders, encoder threads, cache size and CPU budget are derived automatically from these readings and the
  active `--mode`; they shrink under memory pressure and recover when memory is available again. A smaller Whisper
  model is chosen if the requested one will not fit in free RAM.
* One SQLite database (WAL) per video, recreated empty at every start. Only the transcript JSON is kept between runs.
* Every production function is wrapped with `traced`: callers get `[START]` / `[END]` output, `START` /
  `END` / `FAIL` log records with machine state, and post-operation memory rebalancing. Nested calls are
  not double-logged, keeping the file usable instead of exploding it.
* `os.fsync()` is called after every `.log` record (after Python-level flush), so completed log lines
  are on disk before the caller regains control. See [`docs/logging.md`](docs/logging.md) for design,
  safety, and failure semantics.

## Standalone binaries

The build workflow (`.github/workflows/build.yml`) builds a standalone app for **Linux and macOS on
x64 and arm64, and Windows on x64** - no Python installation needed on the machine that runs it.
To release, bump `version` in `pyproject.toml`, `autoclip/__init__.py` and `snapcraft.yaml` (they
must match) and push to `main`: the workflow checks that `v<version>` is not tagged yet, builds all
five targets, then creates the tag and publishes the release - one zip per target, plus the AppImage
and the snap of the two Linux targets. A push to `main` that does not change the version skips the
build; a pull request that touches the build still builds, without releasing.

To build it yourself (for the machine you are on):

```bash
python -m pip install -e . "pyinstaller>=6.10"
python -m PyInstaller autoclip.spec --noconfirm
./dist/autoclip/autoclip              # the desktop app (no argument)
./dist/autoclip/autoclip --gui        # the same, explicitly
./dist/autoclip/autoclip config.json  # the command line
```

`dist/autoclip/` holds the program and everything it needs, ffmpeg included. It is a folder and
not a single file because the program is big (numpy, OpenCV, the Whisper runtime and ffmpeg are
inside it) and a folder starts much faster.

Every standalone binary follows one rule: **without an argument it opens the desktop app, with one
it runs the command line** - so `autoclip config.json`, `autoclip run`, `autoclip --gui`, `--help`
and the rest work unchanged. On Windows a double-click opens a console first; the app hides it when
it is the only program using it (a launch from an existing terminal leaves that terminal alone),
because the command line still needs its output. Windows Terminal may keep its window visible; the
app opens anyway. On macOS the zip contains the `autoclip` folder, not an `.app`: open
`autoclip/autoclip` from Finder (Terminal starts, then the window opens); Gatekeeper still applies
to an unsigned download.

`python -m autoclip` keeps its old rule: with no arguments it runs `./config.json` or
`./input.mp4`. That rule belongs to the command line; the no-argument desktop app is the convention
of the packaged binary.

The optional extras - the speech enhancement and the spectral denoising - are not in the binary:
they are optional, the app works without them, and they would make it noticeably bigger.

### Linux: AppImage and snap

```bash
# the AppImage (one file, runs everywhere, no installation)
python tools/make_appimage.py --dist dist/autoclip --out dist/autoclip-x86_64.AppImage

# the snap (needs snapcraft; destructive mode installs snaps, so it needs root)
sudo snapcraft --destructive-mode
```

The build workflow does both for every Linux target and uploads them next to the other builds.
Without an argument the AppImage and the snap open the desktop app, with one they run the command
line - so `./autoclip-x86_64.AppImage config.json` still cuts clips.

## Platform notes

* **ffmpeg**: decoding uses the one on `PATH` if present, otherwise the binary bundled with `imageio-ffmpeg`. Encoding goes
  through MoviePy, which uses `FFMPEG_BINARY` if you set it, else its bundled binary - and the encoder is probed with that
  same binary, so a GPU encoder is only chosen when the build that will really encode can use it.
* **Fonts**: each OS's font folders are searched for a bold font that covers the caption's script; install a Noto font for scripts your OS lacks.
* **Arabic / Hebrew** captions are shaped in pure Python when Pillow has no libraqm (typical for Windows / macOS wheels).
  **Hindi / Thai / Khmer** need Pillow built with libraqm; a warning is logged when it is missing.
* Worker processes use `spawn` everywhere, so behaviour is identical on all three systems.

See `docs/options-reference.md` for every option and its explanation (this is the complete reference).

## Upgrading from 3.x

See `docs/UPGRADING-4.0.md` for the nine-key schema, the removed-key → runtime replacement table, the
`--flag`/`AUTOCLIP_*` equivalents, and the behaviour that is automatic now. `UPGRADING-3.0.md` covers the 2.x → 3.0
changes and is kept as history. 4.0 is a breaking config-schema release: old config files still load (removed keys warn
and are ignored), but their settings must move to flags or environment variables.

## Upgrading from 2.1

2.2 was additive: the new `download` section had defaults, so a 2.1 config ran unchanged. Nothing was removed or renamed,
so a 2.1 config is a valid 2.2 config.

## Upgrading from 2.0

`clips.style/count/min_seconds/max_seconds/ranges` -> `clips.scenes` and `clips.short` / `clips.long`; `video.aspect` and
`video.size` -> each kind's `aspect`; `clips.snap_to_cuts` -> `clips.snap_to_cuts`; clips now go to `{output_dir}/{short|long}/{video}/`.
A list of styles used to be *blended* into one ranking; now each gets separate clips.

## Security

Config files are trusted input, but: no shell is ever invoked (argument lists only), `llm.base_url` must be http(s), SQL is
parameterised, cached arrays load with `allow_pickle=False`, ruff's bandit rules are enforced.

Downloading only happens for http(s) URLs in `input`; the platform policy is checked before any request, the
`generic` family (any web page) is refused unless it is named, and `deny` always wins. The downloader exposes
no `exec`/`netrc_cmd`-style option: cookies come from a browser store or a Netscape `cookies.txt`, never from a
command. Metadata files (`.info.json`, thumbnail, description) are written next to the original in
`.work/downloads/`, not into the clip folders.

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md) for release-by-release notes. The current code is version `4.0.0`.

## Development

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check . && mypy autoclip tests tools
pytest            # fails below 100% branch coverage (currently 100%); real ffmpeg, real worker processes
pre-commit install   # the same checks on every commit (ruff, mypy and the test suite)
```

`tests/unit` is the offline suite that `pytest` runs by default. `tests/e2e` holds the opt-in checks
that use the real internet: the downloader smoke test and the command-line end-to-end cases driven
by [`tests/assets/youtube.json`](tests/assets/youtube.json).

```bash
AUTOCLIP_NETWORK_TESTS=1 pytest tests/e2e/test_network.py --no-cov   # the real yt-dlp API
AUTOCLIP_E2E_TESTS=1 pytest tests/e2e -m e2e --no-cov                # downloads, renders, keeps results
```

Every end-to-end case writes its config, stdout, stderr, log and output files to
`tests/results/<timestamp>/`; `run.json` and `environment.json` describe the run. The run also keeps a
flat `clips/` folder with every rendered `.mp4` (and its `.srt`/`.jpg` sidecars), a `clips.json`
index and `clips/index.html` - a gallery that plays the clips in a browser, so the result can be
inspected without opening the nested case folders. The folder is ignored by git.
`AUTOCLIP_E2E_RESULTS` changes its parent, `AUTOCLIP_E2E_URL` replaces the short BBC News clip, and
`AUTOCLIP_E2E_COOKIES` points at a Netscape `cookies.txt` when YouTube asks for one.
`AUTOCLIP_E2E_TESTS=1` also enables the `network` tests; `AUTOCLIP_NETWORK_TESTS=1` alone does not
run the e2e cases.

The manifest cases walk a **download ladder**: each level keeps the first six minutes of every video,
and every extra level adds six minutes (`AUTOCLIP_E2E_LEVEL`, default `1`; the last level is the full
video). Originals live in `tests/.cache/e2e/` (or `AUTOCLIP_E2E_CACHE`), so a deeper level reuses what
an earlier one downloaded instead of starting over, and `levels.json` compares the clips of a level
with the previous one so a quality regression is visible in the results. Use `AUTOCLIP_E2E_LEVEL=3`
for eighteen-minute sources, or a high level to reach the full videos.

`tests/e2e/test_server_e2e.py` starts the real server (`python -m autoclip.server`) on a free port,
uploads the generated fixture video, follows the WebSocket, downloads the ZIP, probes the rendered
clip and deletes the job - no network needed beyond localhost. It runs with the other `e2e` cases:

```bash
AUTOCLIP_E2E_TESTS=1 pytest tests/e2e/test_server_e2e.py -m e2e --no-cov
```

The `e2e` GitHub workflow runs the end-to-end cases by hand (Actions -> e2e -> Run workflow) and keeps
`tests/results/` as a downloadable artifact; `network.yml` runs the downloader checks weekly.

The package has four parts:

```
autoclip/
  core/   the engine: config, media, audio, transcript, moments, captions, render, runner, store
  cli/    the command line: python -m autoclip config.json
  app/    the desktop app:  python -m autoclip.app   (Tk, native look)
  server/ the optional HTTP/WebSocket service: python -m autoclip.server
```

`core` knows nothing about a user interface; the CLI, the app and the server are thin layers on top of
it, and the app keeps its behaviour (reading and writing the config, starting and stopping the
pipeline) in `app/session.py`, so it can be tested without a display. The server imports FastAPI only
when it is used, because the runtime `server` extra is optional.

`tools/compare_stt.py` compares transcription models and enhancement backends on one scene of a video:

```bash
python tools/compare_stt.py VIDEO --start 4956 --duration 120 \
    --conditions raw,gtcrn,dpdfnet2,dpdfnet8,noisereduce --models tiny,base,small,medium,large-v3
```

The measured result is in [docs/stt-comparison.md](docs/stt-comparison.md).

`tools/verify_output.py` checks a finished run: every clip listed in `clips.json` exists, is a valid
video with audio, has the right shape and the length the run recorded, and comes with its `.srt`
sidecar and `.jpg` thumbnail.

```bash
python tools/verify_output.py OUTPUT_DIR
```
