# Upgrading from 3.x to 4.0

> Current documentation target: 4.0.0. See `../CHANGELOG.md` for the full release history.
> `UPGRADING-3.0.md` covers the 2.x → 3.0 changes and is kept as history.

4.0 is a **breaking configuration-schema release**. The config file shrinks from dozens of nested keys to **nine**:
everything else is derived at run time or passed on the command line. Old config files still *load* (removed keys warn
and are ignored), but the behaviour you used to configure is now either automatic or a runtime override.

## The new file schema

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
  },
  "llm": {
    "system_prompt": ""
  },
  "image": {
    "prompt": "cinematic thumbnail, high quality, sharp focus"
  }
}
```

| Key | What it controls |
|---|---|
| `input` | videos, globs or http(s) URLs |
| `clips.preset` | `both`, `shorts` or `long` — which clip families to render |
| `clips.scenes` | `best`, a list, or `all` — the scene types to pick moments for |
| `clips.keywords` | words or phrases that boost a moment |
| `clips.ranges` | exact cuts (`"00:01:00-00:01:45"`), one per line |
| `download.allow` | extractor families or groups (`@popular`, …) that may be downloaded |
| `download.deny` | families that are never downloaded, even when an allow entry covers them |
| `llm.system_prompt` | the system message of every local LLM call (moment ranking, titles, social captions); empty keeps the built-in editor messages |
| `image.prompt` | the Stable Diffusion prompt of every AI thumbnail (used only when torch/diffusers are installed) |

All other options are runtime overrides or automatic.

## Removed keys and their replacements

A removed key no longer exists in the file. When you set it, autoclip prints
`config: '<key>' is no longer an option; <hint>` and ignores it, so an old config file still runs.

| Removed | Replacement |
|---|---|
| `output_dir` | `--output-dir DIR`, `AUTOCLIP_OUTPUT_DIR`, or the folder control in the app |
| `mode` | `--mode performance\|resources\|balance` (`AUTOCLIP_MODE`) |
| `quality` | `--quality highest\|high\|medium\|low\|lowest` (`AUTOCLIP_QUALITY`) |
| `logging` | logging is internal: DEBUG in debug mode (`AUTOCLIP_DEBUG=1`), INFO otherwise |
| `clips.mode` | `clips.preset` for shorts/long/both; `clips.ranges` for exact cuts; `--split` for equal parts |
| `clips.kinds`, `clips.short`, `clips.long` | `clips.preset` (the short and long profiles are built in) |
| `clips.lexicon_file` | drop a `lexicon.json` next to the config file |
| `clips.skip_intro_seconds`, `clips.pad_start`, `clips.pad_end` | chosen automatically |
| `clips.snap_to_cuts` | shot-change snapping is automatic |
| `clips.split_seconds` | `--split` |
| `transcript` (file/consent/language) | a sidecar `.json`/`.srt` next to the video is used automatically; then a downloaded subtitle, the disk cache, then Whisper |
| `llm` (everything but `llm.system_prompt`) | `--llm-model MODEL` (`AUTOCLIP_LLM_MODEL`/`AUTOCLIP_LLM_URL`); a local Ollama is detected automatically |
| `compute`, `device` | the GPU is probed automatically and used when it works (there is no `--device` flag) |
| `offline` | `--skip llm` controls the local LLM; Whisper model downloads are automatic |
| `video` | the output stack (layout, background, codec, fps) is chosen automatically |
| `captions` | the automatic social style; `--skip caption` disables |
| `title` | titles are automatic; `--skip title` disables |
| `audio` | audio processing is automatic; `--skip audio_enhance` disables |
| `unsilence`, `silence` | silence removal is automatic |
| `skip` | `--skip NAME` (see `autoclip run --help`) |
| `metadata`, `cookies`, `js_runtime`, `network` | always written / `--cookies` (`AUTOCLIP_COOKIES`) / probed automatically; proxies are not supported |
| `download.quality`, `download.fps`, `download.prefer`, `download.codec`, `download.container` | the best matching format is chosen automatically |
| `download.audio_quality`, `download.audio_language` | the best audio track / the video's original audio track |
| `download.original_dir` | originals live in `<output>/.work/downloads` |
| `download.metadata`, `download.cache` | metadata sidecars are always written; originals are always cached |
| `download.subtitles` | a downloaded subtitle is used when it exists (`--skip stt` forbids Whisper) |
| `download.playlist`, `download.max_videos` | the URL shape decides (one video, or the whole playlist/channel, max 50) |
| `download.time_range` | cut the source beforehand, or use `clips.ranges` |
| `download.cookies`, `download.network`, `download.js_runtime` | `--cookies`, proxies are not supported, probed automatically |
| `download.platforms.groups`, `download.platforms.settings` | name families directly in `download.allow`/`download.deny` |
| `download.platforms`, `download.platforms.allow`, `download.platforms.deny` | use `download.allow` / `download.deny` directly |

## Runtime overrides

Every option that used to live in the file is now a runtime override:

| Flag | Env var | What it does |
|---|---|---|
| `--output-dir DIR` | `AUTOCLIP_OUTPUT_DIR` | output folder |
| `--cookies BROWSER\|FILE` | `AUTOCLIP_COOKIES` | cookies for downloads |
| `--skip NAME` | `AUTOCLIP_SKIP` | skip pipeline steps (comma separated) |
| `--llm-model MODEL` | `AUTOCLIP_LLM_MODEL` / `AUTOCLIP_LLM_URL` | local LLM model / URL |
| `--split` | — | cut the video into equal 600 s parts |
| `--mode …` | `AUTOCLIP_MODE` | performance/resources/balance |
| `--quality …` | `AUTOCLIP_QUALITY` | output quality tier |

Env values apply first; an explicit flag always wins. The desktop app keeps the same overrides in its controls.

## What is automatic now

- **Output stack**: layout `fit`, `blur` background, `h264`, automatic fps (source-capped by quality).
- **Download format policy**: the best matching format is chosen from the output quality tier (1080/720/480/360/240 short side); a direct HTTP(S) format beats an HLS/DASH variant; the video's original audio track is kept.
- **Playlists**: a URL whose shape names a list (channel/playlist/album) downloads every entry up to 50; a watch URL stays single.
- **Subtitles**: a downloaded subtitle is used when it exists; a sidecar transcript (`<video-stem>.json` preferred over `.srt`) next to the video always wins.
- **Captions**: the social theme, lower position; `.txt` sidecar written when a caption exists.
- **Titles**: heuristic for long clips; LLM-generated when available.
- **Silence removal**: always attempted to `.work/{stem}/unsilence_{stem}.mp4`, fails soft.
- **Speech enhancement**: chosen per quality tier; clean audio is left untouched.
- **Thumbnails**: a Stable Diffusion thumbnail replaces the frame JPEG only when `torch`/`diffusers` are installed, otherwise the frame JPEG is kept.

## Invalid values fall back instead of failing

A wrong *type* or an unknown key is still fatal (`ConfigError`). A wrong *value* now warns
(`config: invalid value for <path>; falling back to default`) and uses the field default, so a typo in one option no
longer aborts the run. Removed keys warn and are ignored.

## AI queue

All local LLM (Ollama) calls and Stable Diffusion thumbnails go through one process-wide lock, so even in
`performance` mode they never run concurrently. `llm.system_prompt` (file) is sent as the system message of every call
(moment ranking, titles and social captions); empty keeps the built-in editor messages. `--skip llm` skips every
Ollama call; Whisper models download on first use automatically. The clips keep their heuristic result. A missing
backend is logged and skipped, never fatal.

## Notes for 3.x users

- The `clips.short` / `clips.long` length/count/aspect blocks are gone: the built-in short profile is 20–60 s at 9:16
  and the long profile is 3–10 min at 16:9. Use `--split` or `clips.ranges` for other shapes.
- `transcript.consent` is now always `auto` (no prompt); Whisper models download on first use.
- GPU use is probed automatically; there is no `--device` flag or `compute.device`.
- `--init` and `--init-full` both write the complete nine-key file now (they are identical).
- The desktop app has four config tabs now: *Videos*, *Clips*, *Download*, *AI* (plus a runtime output-folder control that is
  never written to the config file).
- Version string: 4.0.0 (`pyproject.toml`, `autoclip/__init__.py`, `snapcraft.yaml`).

See also: `docs/options-reference.md` for the generated reference of the nine keys.
