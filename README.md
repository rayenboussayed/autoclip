# autoclip

Cut long videos into the best **short** and **long** clips (TikTok / Reels / Shorts and YouTube-style highlights).
Free, local, one command. Runs on **Linux, macOS and Windows** (Python 3.10+). No system ffmpeg needed.

```bash
pip install -r requirements.txt
pip install -U "Pillow>=12.3"          # security fixes; MoviePy's "<12" pin is only conservative (CI tests both)
python -m autoclip --init config.json  # tiny starter config   (--init-full lists every option and default)
python -m autoclip config.json         # or: autoclip config.json  (after `pip install -e .`)
python -m autoclip                     # no config: ./input.mp4 with the built-in defaults
python -m autoclip.app                 # the desktop app (same options, explained)
```

## The desktop app

`python -m autoclip.app` opens a native window (Tk, the look of your system) with **every option of
the config file**:

- ten tabs - *Videos, Download, Clips, Transcript, Video, Captions, Audio, Skip, Speed, LLM & log* - each
  option with a one-line summary under it;
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
  .work/downloads/ originals downloaded from URLs (reused by later runs; or download.original_dir)
```
A kind that is not requested (or finds nothing) gets no folder. Several videos share the same `short/` and `long/` folders.

## Short and long clips, scene types

```json
{
  "input": ["talks/*.mp4", "interview.mov"],
  "output_dir": "clips",
  "clips": {
    "kinds": ["short", "long"],
    "scenes": "all",
    "short": {"count": 3, "min_seconds": 20, "target_seconds": 35, "max_seconds": 60,  "aspect": "9:16"},
    "long":  {"count": 1, "min_seconds": 180, "target_seconds": 240, "max_seconds": 600, "aspect": "16:9"}
  }
}
```
* `scenes` is `"all"`, one type, or a list: `best attractive exciting funny sad emotional inspiring calm educational controversial`.
  **Every scene type gets its own clips** (`count` is per scene type and per kind), so `"all"` with `count: 3` means up to
  30 clips per kind. A moment picked for one scene is never reused under another label; scenes pick in the order listed,
  so in a short video the later ones may find fewer free moments than `count`.
* Each kind has its own length range, count and `aspect` (`9:16`, `1:1`, `4:5`, `16:9`, `original`).
  If a video is too short for a kind (e.g. 2-minute video, long clips from 3 minutes) that kind is skipped with a warning.
* **The length of a clip is `clip(minimum, target, maximum)`:**
  * `min_seconds` is a hard floor: the clip is never shorter. If the pause removal would cut it below the minimum, the
    least important cuts are given back (the fillers first, then the shortest pauses).
  * `target_seconds` is the *recommended* length - a preference, not a limit. Candidates near it get a small score bonus,
    so the clips tend to have that length while the scene type still decides the content. Leave it out (`null`) to take
    whatever length the content offers.
  * `max_seconds` is a hard ceiling: a clip is never longer, not even by the padding around it.
* `mode: "ranges"` cuts your own ranges per kind (`clips.short.ranges`, `clips.long.ranges`, scene type `custom`);
  `mode: "split"` cuts equal parts into `long/{video}/`.
* The file is validated before anything is touched: an unknown key, a wrong type or a value out of range is reported with
  the option that caused it (`error: clips.short.count: Input should be a valid integer`) and the run stops. Keys starting
  with `_` are ignored, so `"_comment"` entries are fine.
* `input` is a path, a glob or a list. One broken video or clip never stops the batch. Exit status: `0` all done, `1` something
  failed, `2` bad configuration. With several videos `transcript.file` needs `{stem}` (e.g. `subs/{stem}.srt`).
  Unknown config keys are errors.

## Downloading URLs (YouTube, TikTok, Vimeo, ...)
`input` accepts **http(s) URLs next to files and globs**. A URL is downloaded with
[yt-dlp](https://github.com/yt-dlp/yt-dlp) (installed with autoclip) and then goes through exactly the
same pipeline as a local file:

```json
{
  "input": ["talks/*.mp4", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
  "output_dir": "clips",
  "download": {
    "quality": 1080,
    "prefer": "at_most",
    "codec": "h264",
    "container": "mp4",
    "platforms": {"allow": ["@popular", "vimeo"], "deny": ["tiktok"]},
    "subtitles": {"enabled": true, "languages": ["en"], "use_as_transcript": "manual"}
  }
}
```

* **Quality and format:** `quality` is a short side in pixels (144-4320) or `"best"`/`"worst"`; `fps`
  (`null` = any), `codec` (`auto`, `h264`, `vp9`, `av1`, `h265`), `container` (`auto`, `mp4`, `mkv`,
  `webm`; a preference, not a forced remux) and `audio_quality`/`audio_language`. `prefer` is
  `closest` (default), `at_most` or `at_least`, and applies to quality and fps.
* **Originals** are kept in `{output_dir}/.work/downloads/` (or `download.original_dir`) and reused
  by later runs; the clip folders stay clips-only. The output name is `<title> [<video id>]`, and a
  time range adds `[<start>-<end>]`.
* **Platform policy:** `download.platforms.allow` / `deny` hold extractor families (`youtube`,
  `tiktok`, `vimeo`, ...) or groups. `deny` wins, the default allow list is `@popular`, and the
  `generic` family (any web page with a video) is only used when it is named - so a random link is
  refused before a request is made. Built-in groups are `@popular`, `@social`, `@video` and `@all`;
  define your own with `groups: {"@news": ["youtube", "vimeo"]}`. `settings` overrides options for
  one family or group (`{"youtube": {"quality": 720}}`), and `group_by_platform` puts the originals
  in one folder per platform.
* **Lists and extras:** `playlist` is `single` (default; a playlist URL still expands, capped at one
  entry) or `all` with `max_videos` (50). `metadata` writes `.info.json`/thumbnail/description next
  to the original. `time_range` (`"MM:SS-MM:SS"`/`"HH:MM:SS-HH:MM:SS"`) downloads only that part
  when ffmpeg *and* ffprobe are on `PATH`; otherwise the whole video is downloaded and trimmed
  locally. `cookies` reads a browser store (`from_browser`) or a Netscape `cookies.txt` (`file`) -
  never both and never a shell command. `network` sets proxy, rate limit, retries, socket timeout and
  a pause between playlist entries. `js_runtime` selects/points at Deno, Node, Bun or QuickJS for
  YouTube's signature challenges.
* **Subtitles:** `download.subtitles` saves the platform's subtitles (`.srt` when offered, `.vtt`
  converted) for `languages`. `use_as_transcript` is `never` (default; files only), `manual` (only
  human-made subtitles may replace Whisper) or `always` (automatic captions too). An explicit
  `transcript.file` always wins; `transcript.mode: "auto"` prefers a downloaded subtitle, then the
  transcript cache, then Whisper; `"whisper"` ignores them unless `use_as_transcript` is `always`.
* **Command line:** without a config file the built-in defaults run `./input.mp4` (a missing file is
  reported with a hint, exit status 2, and nothing is written); `--download-only config.json` downloads
  the URLs and stops; `--formats URL` prints every available format and the one autoclip would choose
  (no config file needed); `--platforms [URL]` prints the allow/deny policy and where a URL would go;
  `--allow a,b` and `--deny c` override the loaded lists for one run. Inspection modes create no
  folders and configure no logging. A run with only local files never imports yt-dlp and makes no
  network request.

## Where the work runs
`compute.device` decides whether the GPU is used at all:

| value | effect |
| --- | --- |
| `auto` (default) | the GPU is used when there is one and it really works |
| `cpu` | **nothing touches the GPU**: the transcriber runs on the CPU and the clips are encoded with libx264 |
| `gpu` | the GPU is preferred (the transcriber with CUDA, the video encoder with NVENC/QSV/AMF/VideoToolbox) |

`transcript.device` (`auto`/`cpu`/`cuda`) still selects the device of the transcriber alone and wins
over `compute.device: gpu`; `compute.device: cpu` always wins. On the CPU the transcriber is much
slower: `large-v3` needs roughly ten times the time of `small`, so pick a smaller
`transcript.model` for long videos on a CPU-only machine.

## GPU acceleration (automatic, with CPU fallback)
`video.gpu` is `auto` (default), `off`, or one of `nvenc` (NVIDIA), `videotoolbox` (macOS), `qsv` (Intel), `amf` (AMD).
An encoder is used only if ffmpeg can **actually encode a test clip with it on this machine** (ffmpeg builds list
encoders such as `h264_nvenc` even on computers without that GPU); otherwise libx264 on the CPU is used and the log says so.
* GPU workers are capped at 3 (consumer GPUs limit simultaneous encode sessions).
* Transcription uses the GPU too when CUDA is available (`transcript.device: auto`).
* Only the **encode** step is on the GPU: decoding and compositing (captions, blur, layout) stay on the CPU, so expect a
  speed-up in encoding, not a 10x end-to-end gain.

## Very long videos
The transcript is produced in **chunks of `processing.chunk_minutes` (5 minutes)**; the boundaries move to the quietest
moment nearby and the language is detected once and reused. Neither the denoiser nor Whisper ever holds more than one
chunk in memory, so a 3-hour file no longer needs 3 hours of audio at once. Analysis already streams, noise reduction
works in 60 s pieces, and selection stays **global**, so a clip is never cut at a chunk edge and a long clip may be
longer than a chunk. Set `processing.chunk_threshold_minutes` to a positive number to transcribe short videos in one go.

## Clear speech for the transcriber (on by default)
Before the audio is transcribed it is **enhanced**, so the ASR model hears clear words instead of room noise,
keyboard clicks or hum. `transcript.enhance` is enabled by default:

| option | default | meaning |
| --- | --- | --- |
| `enabled` | `true` | denoise the audio before transcription |
| `backend` | `sherpa` | `sherpa` (small ONNX denoiser) or `noisereduce` (spectral gating, no model) |
| `model` | `gtcrn` | `gtcrn` (0.5 MB, fastest) or `dpdfnet_baseline` / `dpdfnet2` / `dpdfnet4` / `dpdfnet8` (16 kHz, better) |
| `attenuation_limit_db` | `12.0` | how much noise may be removed (0 = off, 100 = no suppression) |
| `denoise_strength` | `0.75` | only for the `noisereduce` backend |
| `snr_threshold_db` | `20.0` | only enhance a chunk whose speech-to-noise ratio is below this (`null` = always) |
| `num_threads` | `4` | CPU threads of the denoiser |
| `model_dir` | `<output_dir>/.work/models` | where the ONNX model is kept |

* The `sherpa` backend needs the optional package `sherpa-onnx` (`pip install sherpa-onnx`); its model is downloaded
  once from the sherpa-onnx release. Without the package the run continues with the original audio and says so.
* **Denoising only helps when there is noise.** A clean recording is damaged by it and the ASR model then drifts into
  the wrong language, so every chunk is measured first and left alone when it is clean enough (`snr_threshold_db`).
* Enhancement is part of the transcript cache key: changing it re-transcribes the video.
* Set `transcript.enhance.enabled: false` to transcribe the untouched audio.

## Transcription
* `transcript.model` is `large-v3` (the most accurate multilingual model); `tiny`, `base`, `small` and `medium` are
  faster and smaller. On the CPU a model that does not fit into the free RAM is replaced by a smaller one.
* The audio is transcribed in chunks of `processing.chunk_minutes` (5 minutes): neither the denoiser nor Whisper ever
  holds more than that, and the language is detected once and reused for the remaining chunks.

## What it does
1. Enhances the audio and transcribes it with faster-whisper (auto language) or loads `.srt` / `.json`.
2. Scores sentence-aligned windows per scene type from audio energy, speech rate, hooks and lexicons; optional local-LLM re-rank.
3. Renders each clip: denoise + loudness, cuts long pauses and "um/uh", blurred background, word-highlighted captions inside
   the app-UI safe zone, fades, H.264 High + AAC (`+faststart`, capped bitrate).

## Skipping steps
Every step runs by default. `skip.*` leaves individual steps out without changing the rest of the
config - useful for a quick render, a silent clip or a run that must not download a model:

```json
{
  "skip": {
    "stt": false,
    "denoise": false,
    "audio_enhance": false,
    "audio": false,
    "pauses": false,
    "caption": false,
    "title": false,
    "fade": false,
    "thumbnail": false,
    "llm": false,
    "snap": false
  }
}
```

| option | what is left out |
| --- | --- |
| `stt` | Whisper. An explicit `transcript.file`, a downloaded subtitle or a cached transcript is still used; without one the run continues without text (highlight selection then falls back to audio-only windows). Cannot be combined with `transcript.mode: "whisper"`. |
| `denoise` | Speech enhancement before Whisper **and** noise reduction of the clip audio. The transcript cache key changes, so an enhanced cache is not reused for raw audio. |
| `audio_enhance` | High-pass, loudness normalisation and the limiter of the clip audio; the audio stays at its original level. Denoising is controlled separately. |
| `audio` | The audio track: clips are muted and the audio is not decoded (so pauses cannot be removed either). |
| `pauses` | Pause shortening and filler-word removal; the clip keeps its whole selected window. |
| `caption` | Burned-in captions (even when `captions.enabled` is true). |
| `title` | The hook title (even when `title.enabled` is true). |
| `fade` | Video and audio fades at the clip edges. |
| `thumbnail` | The `.jpg` cover next to each clip. |
| `llm` | The optional local-LLM re-rank; the heuristic ranking is used and no request is made. |
| `snap` | Snapping a text-selected window to shot cuts; audio-only quiet snapping still applies. |

`audio.enhance: false` remains the master switch for all clip-audio processing and turns both
`audio_enhance` and `denoise` off. Skipping `stt`, `caption` and `title` also lets ranges/split runs
avoid transcription entirely when no text is needed.

## Resources, storage, logs
* Before and after every operation the script re-reads CPU count (affinity + container quota) and *free* RAM, runs as many
  parallel renders as both allow (spare cores become encoder threads), shrinks the cache and worker count under memory
  pressure, and retries a worker the OS killed. A smaller Whisper model is chosen if the requested one will not fit in free RAM.
* One SQLite database (WAL) per video, recreated empty at every start. Only the transcript JSON is kept between runs.
* The log has START/END of every operation with wall time, RSS, free RAM, CPU and throughput; workers log into the same file.

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
./dist/autoclip/autoclip --gui          # the desktop app
./dist/autoclip/autoclip config.json    # the command line
```

`dist/autoclip/` holds the program and everything it needs, ffmpeg included. It is a folder and
not a single file because the program is big (numpy, OpenCV, the Whisper runtime and ffmpeg are
inside it) and a folder starts much faster.

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

## Upgrading from 2.1
2.2 is additive: the new `download` section has defaults, so a 2.1 config runs unchanged, and a run whose
`input` holds only local files never imports yt-dlp and makes no network request. What is new:

* `input` also accepts http(s) URLs; files and globs keep their old meaning. A URL is downloaded first and
  then goes through the usual pipeline.
* `download.*` controls quality and format, platform allow/deny, playlists, metadata, time ranges, cookies,
  network settings, the JavaScript runtime and subtitles (see [Downloading URLs](#downloading-urls-youtube-tiktok-vimeo-)).
* The command line gained `--download-only`, `--formats URL`, `--platforms [URL]`, `--allow` and `--deny`.
* The desktop app gained the *Download* tab, *Copy error* / *Copy log* and `AUTOCLIP_DEBUG=1`; a
  missing config file now starts from the built-in defaults in memory instead of asking where to save.
* `skip.*` leaves individual pipeline steps out (all eleven switches are `false` by default, so
  nothing changes unless you ask for it).

Nothing was removed or renamed, so a 2.1 config is a valid 2.2 config.

## Upgrading from 2.0
`clips.style/count/min_seconds/max_seconds/ranges` -> `clips.scenes` and `clips.short` / `clips.long`; `video.aspect` and
`video.size` -> each kind's `aspect`; `clips.snap_scenes` -> `clips.snap_to_cuts`; clips now go to `{output_dir}/{short|long}/{video}/`.
A list of styles used to be *blended* into one ranking; now each gets separate clips.

## Security
Config files are trusted input, but: no shell is ever invoked (argument lists only), `llm.base_url` must be http(s), the API key is
never logged, SQL is parameterised, cached arrays load with `allow_pickle=False`, ruff's bandit rules are enforced.

Downloading only happens for http(s) URLs in `input`; the platform policy is checked before any request, the
`generic` family (any web page) is refused unless it is named, and `deny` always wins. The downloader exposes
no `exec`/`netrc_cmd`-style option: cookies come from a browser store or a Netscape `cookies.txt`, never from a
command. Metadata files (`.info.json`, thumbnail, description) are written next to the original in
`.work/downloads/`, not into the clip folders.

## Development
```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check . && mypy autoclip tests tools
pytest            # fails below 95% branch coverage (currently 100%); real ffmpeg, real worker processes
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

The `e2e` GitHub workflow runs the end-to-end cases by hand (Actions -> e2e -> Run workflow) and keeps
`tests/results/` as a downloadable artifact; `network.yml` runs the downloader checks weekly.

The package has three parts:

```
autoclip/
  core/   the engine: config, media, audio, transcript, moments, captions, render, runner, store
  cli/    the command line: python -m autoclip config.json
  app/    the desktop app:  python -m autoclip.app   (Tk, native look)
```

`core` knows nothing about a user interface; the CLI and the app are thin layers on top of it, and
the app keeps its behaviour (reading and writing the config, starting and stopping the pipeline) in
`app/session.py`, so it can be tested without a display.

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
