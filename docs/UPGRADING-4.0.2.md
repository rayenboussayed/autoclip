# Upgrading from 4.0 / 4.0.1 to 4.0.2

This document applies from the 4.0.2 release. The **nine config-file keys are unchanged** — no config-file edit is
needed. What changes is the last set of *machine-policy* knobs: they are removed from the command line and the
environment, and the resource-aware engine chooses them per run.

## Removed flags and environment variables

| Removed | Migration |
|---|---|
| `--quality …` | Nothing to pass — the engine picks the encode tier from the measured machine (see below). |
| `AUTOCLIP_QUALITY` | Nothing to set — same; the variable is ignored. |
| `--mode …` | Nothing to pass — the engine derives `performance` / `resources` / `balance` from the machine. |
| `AUTOCLIP_MODE` | Nothing to set — the variable is ignored. |
| `--llm-model MODEL` | Nothing to pass — a local Ollama at `http://localhost:11434` is used automatically; the engine chooses the model (below). |
| `AUTOCLIP_LLM_MODEL`, `AUTOCLIP_LLM_URL` | Nothing to set — a local Ollama at `http://localhost:11434` is used automatically. There is no way to point autoclip elsewhere from the file, the environment or the command line. (The Python `Config` still carries `llm.base_url` / `llm.model`, defaulting to that local server with the engine's pick, for tests and embedders.) |
| `MAX_CONCURRENT_JOBS` (server) | Nothing to set — the HTTP server renders up to `min(4, cpus // 2, free_RAM // 1.5 GiB)` jobs at once (at least one, even on a tight box), derived once per server process at the first job and logged as `automation max_concurrent_jobs=N`. `MAX_PENDING_JOBS` is unchanged. |
| `RuntimeOverrides.quality` / `.mode` / `.llm_base_url` / `.llm_model` | These fields are gone from the Python API; `RuntimeOverrides(input=…, output_dir=…, cookies=…, skip=[…], split=True)` and `.from_env()` are all that remain. |

Scripts that still pass these flags fail at startup with `unrecognized arguments` (exit code 2, before any work is
done) — nothing is silently ignored.

## What the engine derives now

- **Encode quality**: `high` when the machine sustains it (≥ 4 cores and ≥ 3 GiB free RAM), `medium` (≥ 2 cores and
  ≥ 2 GiB), `low` otherwise. A GPU only affects speed, never the tier or the picture. Only three encoder profiles
  exist now (`high`, `medium`, `low`); the internal `highest` and `lowest` profiles are gone. Download short side
  follows the picked tier (720/480/360 for high/medium/low, respectively — previously up to 1080).
- **Performance mode**: `resources` for ≤ 2 cores or < 1.5 GiB free RAM, `performance` for ≥ 8 cores and
  ≥ 8 GiB free RAM, `balance` otherwise. The adaptive CPU tuner refines the budget inside the chosen profile during
  the run, exactly as before.
- **Transcription model**: `large-v3` on CUDA; on the CPU `medium` on performance-class machines (8+ cores and 8+ GiB
  free) and `small` otherwise (no longer tied to the quality tier, and `base` is never an engine pick). A model that does
  not fit the free RAM is replaced by a smaller one — but the shrink is CPU-only: a CUDA pick is used as-is, since GPU
  memory is separate. On a CUDA load failure the pick is capped to the CPU default before the RAM fit, so a GPU-sized
  model is never loaded on the CPU for hours. Beam size: 5 on CUDA, 3 on the CPU.
- **Ollama model**: once per process, at the first AI call, autoclip asks the local server for its model list
  (`/api/tags`, cached for the process). It
  uses the largest installed model whose on-disk size (`size`) fits inside half of the free RAM (binary units), falling back to the
  smallest known one (or the alphabetically-first name when no sizes are reported at all).
  `llm.system_prompt` is unchanged. When the server is unreachable or lists no
  models, the AI helpers are off for the run and clips keep their heuristic result. The probe response is capped at
  4 MB and never follows a redirect off the local server.
- **Caption and fade timing**: the beam, caption and fade helpers lost their old quality parameterization and follow
  the picked profile / the device instead; render fades are scaled only for very short clips.
- **Diagnostics**: `runtime_decisions()` logs the derived values (`quality`, `mode`, the profile-derived threads, CPU
  percent, reserve and memory fraction). The Whisper-model row names the *requested* model (before the free-RAM fit can
  shrink it), and `max_cpu_percent` reports the profile default (the adaptive value is logged separately by the
  monitor).

## Notes

- `config.json` files that mention `quality`/`mode`/`llm.base_url`/`llm.model` still load (removed keys warn and are
  ignored; the warning hint now points at the engine instead of the removed flags).
- The transcript disk cache format was bumped (`format: 2`) when the streaming rework landed — entries written by
  older builds never match and re-transcribe once, on every such machine (there is no in-place migration, by design).
  Because CPU runs now also use `small` instead of `tiny`/`base`, a machine that previously ran at `lowest`/`low`
  re-transcribes for that reason too. The 4.0.3 engine changes recorded under `CHANGELOG.md` → `[4.0.3]`
  (detect-once decoding with a hallucination guard, a new cache key, bundled fonts and the CPU `medium` pick) cause one
  further re-transcribe on affected machines.
- Whisper model downloads remain automatic; `--skip stt` still avoids Whisper entirely.
- With effectively no usable free RAM the engine names the smallest known model, and Whisper itself refuses to load
  anything below its minimum; the video is then skipped with a clear error instead of swapping into oblivion. The rest
  of the batch is unaffected.

## What did not change

The nine file keys; the split/ranges semantics; the downloader and platform policy; the app and its option catalogue;
the server's other environment variables; `--skip`.
