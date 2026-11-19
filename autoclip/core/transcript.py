"""Transcripts: parse .srt/.json, run faster-whisper, cache on disk and load into SQLite."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .cutmap import CutMap
from .denoise import SpeechEnhancer, default_models_dir
from .errors import AutoclipError
from .events import CancelToken
from .internal import CHUNK_MINUTES as _CHUNK_MINUTES
from .internal.automation import ENHANCE_ATTENUATION_DB, ENHANCE_SNR_THRESHOLD_DB, transcription_beam_size
from .internal.runtime import whisper_model_for
from .log import speed_text, traced
from .media import atomic_write_text, decode_audio
from .model_engine import release_allocations
from .models import MediaInfo, Segment, Word
from .resources import MB, RESOURCES
from .store import Store

log = logging.getLogger("autoclip")

WHISPER_SR = 16000  # Whisper models expect 16 kHz mono audio
CJK = "\u3040-\u30ff\u3400-\u9fff\u0e00-\u0e7f"  # scripts written without spaces: Japanese, Chinese, Thai
SRT_TIME = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})\s*-->\s*(\d+):(\d\d):(\d\d)[,.](\d{1,3})")
#: Approximate RAM (MB, int8 on CPU) of faster-whisper models, smallest first.
WHISPER_RAM_MB = {"tiny": 400, "base": 500, "small": 1100, "medium": 2600, "large-v3": 4200}


def _strip_rolling(text: str, previous: str) -> str:
    """Remove the previous screen's text when a rolling caption repeats it at the start of this cue."""
    prev, cur = previous.split(), text.split()
    if len(cur) > len(prev) and [_bare(w) for w in cur[: len(prev)]] == [_bare(w) for w in prev]:
        return " ".join(cur[len(prev) :])
    return text


def _bare(word: str) -> str:
    """A word without its surrounding punctuation, for rolling-caption comparisons."""
    return word.strip(".,;:!?\"'()[]{}<>-").casefold()


def normalize_cues(raw: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Clean subtitle cues: drop empties and duplicates, unroll repeated text and clamp overlaps.

    Automatic captions (``youtube``'s in particular) often emit, for every new line, a cue with the previous
    line still on screen and a second cue carrying only that previous line. Burned in as-is, the caption text
    jumps back and forth and every screen lives for a few milliseconds. This collapses that pattern into one
    cue per spoken line, sorted, with no empty, zero-length or overlapping cues.
    """
    cues = sorted((dict(item) for item in raw), key=lambda c: float(c["start"]))
    out: list[dict[str, Any]] = []
    for cue in cues:
        start, end = float(cue["start"]), float(cue["end"])
        text = " ".join(str(cue["text"]).split())
        if not text or end <= start:
            continue
        previous = out[-1] if out else None
        if previous is not None and start - float(previous["end"]) <= 0.35:
            if text == str(previous["text"]):
                previous["end"] = max(float(previous["end"]), end)
                continue
            text = _strip_rolling(text, str(previous["text"]))
        if previous is not None and start < float(previous["end"]):
            start = float(previous["end"])
            if end <= start:
                previous["end"] = max(float(previous["end"]), end)
                continue
        out.append({"start": start, "end": end, "text": text})
    return out


def parse_srt(text: str) -> list[dict[str, Any]]:
    """SubRip text -> ``[{"start", "end", "text"}]`` (normalized: no rolling duplicates or overlaps)."""
    out: list[dict[str, Any]] = []
    for block in re.split(r"\n\s*\n", text.replace("\r", "").strip()):
        lines = block.split("\n")
        for idx, line in enumerate(lines):
            if m := SRT_TIME.match(line.strip()):
                g = [int(x.ljust(3, "0")) if k in (3, 7) else int(x) for k, x in enumerate(m.groups())]
                body = " ".join(s.strip() for s in lines[idx + 1 :] if s.strip())
                if body:
                    out.append(
                        {
                            "start": g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
                            "end": g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000,
                            "text": body,
                        }
                    )
                break
    return normalize_cues(out)


def tokenize(text: str) -> list[str]:
    """Display tokens: words for spaced scripts, single characters for CJK/Thai (their punctuation sticks)."""
    tokens: list[str] = []
    for raw in text.split():
        for piece in re.findall(rf"[{CJK}]|[^{CJK}\s]+", raw):
            if tokens and re.fullmatch(r"[\u3000-\u303f\uff00-\uffef]+", piece):
                tokens[-1] += piece
            else:
                tokens.append(piece)
    return tokens


def _map_segments(segments: list[Segment], cutmap: CutMap | None) -> list[Segment]:
    """Translate source-time segments and words to a playback timeline, retaining both times."""
    if cutmap is None:
        return segments
    mapped: list[Segment] = []
    for segment in segments:
        source_start = segment.source_start if segment.source_start is not None else segment.start
        source_end = segment.source_end if segment.source_end is not None else segment.end
        optimized_start, optimized_end = cutmap.source_span_to_optimized(source_start, source_end)
        words = []
        for word in segment.words:
            word_source_start = word.source_start if word.source_start is not None else word.start
            word_source_end = word.source_end if word.source_end is not None else word.end
            start, end = cutmap.source_span_to_optimized(word_source_start, word_source_end)
            if end > start:  # words entirely inside a removed span collapse to zero width and are dropped
                words.append(Word(start, end, word.text, word.pseudo, word_source_start, word_source_end))
        if optimized_end <= optimized_start or not segment.text.strip():
            continue
        mapped.append(Segment(optimized_start, optimized_end, segment.text, words, segment.pseudo, source_start, source_end))
    return mapped


def to_segments(raw: Iterable[Mapping[str, Any]]) -> list[Segment]:
    """Raw dicts -> time-ordered segments; segments without word times get proportional pseudo-words."""
    out: list[Segment] = []
    for item in raw:
        text = str(item["text"]).strip()
        if not text:
            continue
        start, end = float(item["start"]), max(float(item["end"]), float(item["start"]) + 0.05)
        words = [
            Word(
                float(w["start"]),
                float(w["end"]),
                str(w.get("word") or w["text"]).strip(),
                False,
                w.get("source_start"),
                w.get("source_end"),
            )
            for w in item.get("words") or []
        ]
        pseudo = not words
        if pseudo:
            tokens = tokenize(text)
            source_start = float(item.get("source_start", start) or start)
            source_end = float(item.get("source_end", end) or end)
            source_width = source_end - source_start
            playback_width = end - start
            chars = sum(map(len, tokens))
            total, t, source_pos = chars, start, source_start
            for tok in tokens:
                share = len(tok) / total
                width = playback_width * share
                source_width_token = source_width * share
                words.append(Word(t, t + width, tok, True, source_pos, source_pos + source_width_token))
                t += width
                source_pos += source_width_token
        out.append(Segment(start, end, text, words, pseudo, item.get("source_start"), item.get("source_end")))
    return sorted(out, key=lambda s: s.start)


def load_transcript_file(path: Path) -> list[Segment]:
    """An ``.srt`` or ``.json`` transcript (plain list, ``{"segments": [...]}`` or autoclip's cache format)."""
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data["segments"] if isinstance(data, dict) else data
        return to_segments(normalize_cues(raw))
    return to_segments(parse_srt(path.read_text(encoding="utf-8-sig")))


def fit_whisper_model(name: str, device: str) -> str:
    """A smaller Whisper model when free RAM cannot hold ``name`` (CPU only; GPU memory is separate)."""
    if device == "cuda" or name not in WHISPER_RAM_MB:
        return name
    usable = RESOURCES.usable_memory() // MB
    fitting = [m for m, mb in WHISPER_RAM_MB.items() if mb <= min(usable, WHISPER_RAM_MB[name])]
    chosen = fitting[-1] if fitting else next(iter(WHISPER_RAM_MB))
    if not fitting:
        raise AutoclipError(f"less than {WHISPER_RAM_MB['tiny']}MB of RAM is usable; refusing to load a Whisper model")
    if chosen != name:
        log.warning("whisper '%s' needs ~%dMB but ~%dMB is usable: using '%s'", name, WHISPER_RAM_MB[name], usable, chosen)
    return chosen


def transcript_model() -> str:
    """The Whisper model chosen for this run, after the free-memory fit is applied."""
    device = asr_device()
    return fit_whisper_model(whisper_model_for(device), device)


def asr_device() -> str:
    """The device the ASR model runs on (the GPU when CUDA is present, otherwise the CPU)."""
    import ctranslate2

    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"


def _model_is_cached(model_name: str) -> bool:
    """Whether faster-whisper already has ``model_name`` in its default cache (never touches the network)."""
    from faster_whisper.utils import download_model

    try:
        download_model(model_name, local_files_only=True)
    except Exception:
        return False
    return True


def _download_model(model_name: str) -> None:
    """Download ``model_name`` into the default cache (called automatically on first use)."""
    from faster_whisper.utils import download_model

    download_model(model_name, local_files_only=False)


def _warmup_model(model: Any) -> None:
    """Run a second of silence through ``model`` so a broken CUDA runtime fails now, not mid-file.

    ``ctranslate2`` links a specific CUDA/cuBLAS/cuDNN combination: a newer toolkit or a
    half-installed driver can let the model *load* and only fail on the first real inference.
    Probing a library name was Linux-only and missed cuDNN; an actual inference is portable and
    fails exactly where the real work would.
    """
    stream, _ = model.transcribe(np.zeros(WHISPER_SR, dtype=np.float32), vad_filter=False, word_timestamps=False)
    for _ in stream:  # consume the generator: the compute happens here
        pass


def _vad_failure(exc: Exception) -> bool:
    """Whether ``exc`` is a failure of faster-whisper's voice-activity detection (VAD) step."""
    text = str(exc).lower()
    return "vad" in text or "silero" in text or "onnxruntime" in text or "onnx" in text


def _device_label(dev: str) -> str:
    return "GPU" if dev == "cuda" else dev.upper() if dev == "cpu" else dev


def _device_failure(attempts: list[tuple[str, Exception]]) -> str:
    """A compact description of every device that was tried and why it failed."""
    last: dict[str, Exception] = {}
    for dev, exc in attempts:
        last[dev] = exc
    parts: list[str] = []
    for dev in ("cuda", "cpu"):
        if dev in last:
            exc = last[dev]
            parts.append(f"{_device_label(dev)}: {type(exc).__name__}: {exc}")
    return "; ".join(parts)


def _partial_append(path: Path, header: dict[str, Any], row: dict[str, Any]) -> None:
    """Append one resumable chunk record; a stale partial is replaced wholesale first."""
    header_line = json.dumps(header, ensure_ascii=False, separators=(",", ":")) + "\n"
    stale = False
    if path.exists():
        try:
            stale = path.read_text(encoding="utf-8").splitlines()[0] != header_line.rstrip("\n")
        except (OSError, IndexError):
            stale = True
    if not path.exists() or stale:
        path.write_text(header_line, encoding="utf-8")
    with path.open("a", encoding="utf-8") as part:
        part.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        part.flush()
        os.fsync(part.fileno())


def _partial_load(path: Path, key: Mapping[str, Any], chunk_count: int) -> tuple[list[dict[str, Any]], str | None, int]:
    """Load a resumable transcript cache, discarding it wholesale if it cannot be trusted."""
    if not path.exists():
        return [], None, 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        header = json.loads(lines[0])
        if (
            not isinstance(header, dict)
            or header.get("type") != "meta"
            or header.get("meta") != dict(key)
            or header.get("format") != TRANSCRIPT_CACHE_FORMAT
        ):
            return [], None, 0
        raw: list[dict[str, Any]] = []
        language: str | None = None
        done = 0
        for line in lines[1:]:
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("type") != "chunk" or not isinstance(row.get("rows"), list):
                return [], None, 0
            rows = row["rows"]
            raw.extend(rows)
            language = row.get("language") or language
            done = max(done, int(row.get("done", 0)))
        if done > chunk_count:
            return [], None, 0
        return raw, language, done
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return [], None, 0


def _release_whisper(model: Any | None) -> None:
    """Release the native CTranslate2 model as soon as transcription is finished."""
    if model is None:
        return
    try:
        native = getattr(model, "model", None)
        if native is not None and hasattr(native, "unload_model"):
            try:
                native.unload_model(to_cpu=False)
            except Exception as exc:
                log.warning("could not unload Whisper's native model (%s: %s); releasing references anyway", type(exc).__name__, exc)
    finally:
        del model
        release_allocations()


def _map_local_time(t: float, pieces: Sequence[tuple[float, float, float]]) -> tuple[float, float]:
    """Map a time inside a concatenated chunk to absolute playback and source times."""
    offset = 0.0
    for index, (source_start, source_end, optimized_start) in enumerate(pieces):
        width = source_end - source_start
        limit = offset + width
        if index != len(pieces) - 1 and t > limit:
            offset = limit
            continue
        local = min(max(t - offset, 0.0), width)
        return optimized_start + local, source_start + local
    return 0.0, 0.0


@traced("transcribe")
def transcribe(
    src: Path,
    cfg: Config,
    info: MediaInfo,
    enhancer: SpeechEnhancer | None = None,
    *,
    chunks: Sequence[Sequence[tuple[float, float, float]]] | None = None,
    cutmap: CutMap | None = None,
    model_name_override: str | None = None,
    partial_path: Path | None = None,
    partial_meta: Mapping[str, Any] | None = None,
    cancel: CancelToken | None = None,
) -> list[Segment]:
    from faster_whisper import WhisperModel

    cutmap = cutmap or CutMap.identity(info.duration)
    chunks = list(chunks) if chunks is not None else cutmap.chunked_pieces(_CHUNK_MINUTES * 60.0)
    device = asr_device()
    model_name = model_name_override or transcript_model()
    model_name = fit_whisper_model(model_name, device)
    threads = RESOURCES.cpu_budget()
    # Whisper models download automatically on first use (consent is not a configuration option)
    if not _model_is_cached(model_name):
        _download_model(model_name)

    attempts: list[tuple[str, Exception]] = []

    def load(dev: str) -> Any:
        return WhisperModel(
            model_name,
            device=dev,
            compute_type="float16" if dev == "cuda" else "int8",
            cpu_threads=threads,
            local_files_only=True,
        )

    model: Any = None
    if device == "cuda":
        try:
            model = load(device)
            _warmup_model(model)
        except Exception as e:
            log.warning("the CUDA runtime cannot run Whisper (%s): using the CPU", e)
            attempts.append(("cuda", e))
            _release_whisper(model)
            device = "cpu"
            model_name = fit_whisper_model(model_name, device)  # a GPU-sized pick is re-fit for the CPU
            try:
                model = load(device)
            except Exception as ce:
                attempts.append(("cpu", ce))
                raise AutoclipError(f"Whisper failed on every available device ({_device_failure(attempts)})") from ce
    else:
        try:
            model = load(device)
        except Exception as e:
            attempts.append(("cpu", e))
            raise AutoclipError(f"Whisper failed on every available device ({_device_failure(attempts)})") from e

    log.info("whisper model=%s device=%s%s cpu_threads=%d", model_name, device, " (GPU)" if device == "cuda" else "", threads)
    language, t0 = None, time.perf_counter()
    raw, language, start_chunk = (
        _partial_load(partial_path, partial_meta, len(chunks)) if partial_path is not None and partial_meta is not None else ([], None, 0)
    )
    if start_chunk:
        log.info("resuming transcript from partial cache after chunk %d", start_chunk)
    vad = True
    try:
        for k, pieces in enumerate(chunks, 1):
            if cancel is not None:
                cancel.check()
            if k <= start_chunk:
                continue
            if not pieces:
                continue
            audio = np.concatenate([decode_audio(src, start, end - start, WHISPER_SR) for start, end, _ in pieces])
            if enhancer is not None:
                audio = enhancer(audio)
            while True:
                options: dict[str, Any] = {"multilingual": True}
                try:
                    stream, found = model.transcribe(
                        audio,
                        vad_filter=vad,
                        word_timestamps=True,
                        language=language,
                        condition_on_previous_text=False,
                        beam_size=transcription_beam_size(device),
                        **options,
                    )
                    chunk_rows: list[dict[str, Any]] = []
                    for seg in stream:
                        word_rows = []
                        for word in seg.words or []:
                            playback_start, source_start = _map_local_time(float(word.start), pieces)
                            playback_end, source_end = _map_local_time(float(word.end), pieces)
                            word_rows.append(
                                {
                                    "start": playback_start,
                                    "end": playback_end,
                                    "word": word.word,
                                    "source_start": source_start,
                                    "source_end": source_end,
                                }
                            )
                        playback_start, source_start = _map_local_time(float(seg.start), pieces)
                        playback_end, source_end = _map_local_time(float(seg.end), pieces)
                        chunk_rows.append(
                            {
                                "start": playback_start,
                                "end": playback_end,
                                "text": seg.text,
                                "source_start": source_start,
                                "source_end": source_end,
                                "words": word_rows,
                            }
                        )
                    old_count = len(raw)
                    if cancel is not None:
                        cancel.check()
                    raw.extend(chunk_rows)
                    language = language or found.language
                    if partial_path is not None and partial_meta is not None:
                        new_rows = raw[old_count:]
                        _partial_append(
                            partial_path,
                            {"type": "meta", "meta": dict(partial_meta), "format": TRANSCRIPT_CACHE_FORMAT},
                            {"type": "chunk", "done": k, "language": language, "rows": new_rows},
                        )
                    del audio

                    log.info(
                        "  transcribed chunk %d/%d (%.0f-%.0fs): %d segments so far, %.0fs elapsed",
                        k,
                        len(chunks),
                        pieces[0][0],
                        sum(end - start for start, end, _ in pieces),
                        len(raw),
                        time.perf_counter() - t0,
                    )
                    break
                except Exception as e:
                    if vad and _vad_failure(e):
                        log.warning("VAD failed on %s (%s); retrying this chunk without VAD", _device_label(device), e)
                        vad = False
                        continue
                    if device == "cuda":
                        log.warning("Whisper failed on the GPU for this chunk (%s); retrying on the CPU", e)
                        attempts.append(("cuda", e))
                        device = "cpu"
                        _release_whisper(model)
                        try:
                            model = load(device)
                        except Exception as ce:
                            attempts.append(("cpu", ce))
                            raise AutoclipError(f"Whisper failed on every available device ({_device_failure(attempts)})") from ce
                        continue
                    attempts.append(("cpu", e))
                    raise AutoclipError(f"Whisper failed on every available device ({_device_failure(attempts)})") from e
    finally:
        _release_whisper(model)
        if enhancer is not None:
            enhancer.close()
    log.info(
        "  language %s, %d segments in %.1fs (%s)",
        language,
        len(raw),
        time.perf_counter() - t0,
        speed_text(info.duration, time.perf_counter() - t0),
    )
    return to_segments(raw)


TRANSCRIPT_CACHE_FORMAT = 2


def _cache_key(src: Path, model: str, cutmap: CutMap, *, enhance: bool = True) -> dict[str, Any]:
    """Identity of a transcription run: the cached transcript is reused only while this is unchanged."""
    st = src.stat()
    return {
        "source": f"{src.name}:{st.st_size}:{st.st_mtime_ns}",
        "model": model,
        "language": None,
        "multilingual": True,
        "word_timestamps": True,
        "format": TRANSCRIPT_CACHE_FORMAT,
        "cut_map": cutmap.fingerprint,
        "enhance": (
            {"backend": "gtcrn", "snr_threshold_db": ENHANCE_SNR_THRESHOLD_DB, "attenuation_db": ENHANCE_ATTENUATION_DB}
            if enhance
            else None
        ),
        "chunk_minutes": _CHUNK_MINUTES,
    }


def _sidecar_transcript(src: Path) -> Path | None:
    """A transcript that ships next to the source: ``<name>.json`` first, then ``<name>.srt``."""
    for suffix in (".json", ".srt"):
        candidate = src.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def _sidecar_segments(path: Path, cutmap: CutMap | None = None) -> list[Segment]:
    """The segments of a sidecar transcript, or ``[]`` when it cannot be used (falls back to the next source)."""
    try:
        try:
            peeked = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            peeked = None
        own_cache = isinstance(peeked, dict) and peeked.get("format") == TRANSCRIPT_CACHE_FORMAT
        segments = load_transcript_file(path)
        if not segments:
            raise ValueError("it has no cues")
        if own_cache:
            # an autoclip transcript cache stores playback times already; re-mapping would double-shift them
            log.info("using the sidecar transcript '%s' as autoclip-format segments (%d segments)", path.name, len(segments))
            return segments
    except Exception as e:  # a broken sidecar must not cost the video when Whisper can replace it
        log.warning("the sidecar transcript '%s' cannot be used (%s: %s); using another source", path.name, type(e).__name__, e)
        return []
    log.info("using the sidecar transcript '%s' (%d segments)", path.name, len(segments))
    return _map_segments(segments, cutmap)


@traced("get_transcript")
def get_transcript(
    src: Path,
    cfg: Config,
    work_dir: Path,
    store: Store,
    info: MediaInfo,
    need_text: bool = True,
    subtitle: Path | None = None,
    subtitle_manual: bool = False,
    name: str | None = None,
    cutmap: CutMap | None = None,
    cache_source: Path | None = None,
    cancel: CancelToken | None = None,
) -> bool:
    """Put the transcript of ``src`` into ``store`` (sidecar file, downloaded subtitles, disk cache or Whisper).

    All mapping is monotone and piecewise: ``start/end`` are playback times, while ``source_start/source_end`` retain
    the original source timeline. On a re-encoded source, every caller path maps through the one cut map.
    """
    if not need_text:
        return False
    cutmap = cutmap or CutMap.identity(info.duration)
    cache_src = cache_source or src
    stem = name or cache_src.stem
    sidecar = _sidecar_transcript(cache_src)
    if sidecar is not None:
        segments = _sidecar_segments(sidecar, cutmap)
        if segments:
            store.put_segments(segments)
            return True
    if subtitle is not None:
        downloaded = _downloaded_segments(subtitle, subtitle_manual, cutmap)
        if downloaded:
            store.put_segments(downloaded)
            return True
    selected_model = transcript_model()
    cache_file, key = work_dir / f"{stem}.transcript.json", _cache_key(cache_src, selected_model, cutmap, enhance=not cfg.skip.denoise)
    partial = work_dir / f"{stem}.transcript.partial.jsonl"
    cached_segments: list[Segment] = []
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
                raise ValueError("the cached transcript does not contain a segment list")
            if data.get("meta") == key and data.get("format") == TRANSCRIPT_CACHE_FORMAT:
                cached_segments = to_segments(data["segments"])
                log.info("reusing cached transcript %s (%d segments)", cache_file.name, len(cached_segments))
            else:
                log.warning("cached transcript is stale (input, model, language or optimized timeline changed): transcribing again")
        except (OSError, json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
            log.warning("cached transcript '%s' is unreadable (%s): transcribing again", cache_file.name, exc)
    segments = cached_segments
    if not segments:
        if cfg.skip.stt:
            log.info("skip.stt is set: no cached transcript or subtitle is available")
            partial.unlink(missing_ok=True)
            return False
        if not info.has_audio:
            log.info("the video has no audio: there is no transcript to produce")
            partial.unlink(missing_ok=True)
            return False
        enhancer = SpeechEnhancer(True, default_models_dir()) if not cfg.skip.denoise else None
        try:
            segments = transcribe(
                src,
                cfg,
                info,
                enhancer,
                cutmap=cutmap,
                model_name_override=selected_model,
                partial_path=partial,
                partial_meta=key,
                cancel=cancel,
            )
        except Exception as e:
            detail = str(e) if isinstance(e, AutoclipError) else f"{type(e).__name__}: {e}"
            log.warning("transcription skipped (%s): selecting on audio only; captions and titles need a transcript", detail)
            return False
        cache_payload = {
            "meta": key,
            "format": TRANSCRIPT_CACHE_FORMAT,
            "segments": [_plain(s) for s in segments],
        }
        atomic_write_text(cache_file, json.dumps(cache_payload, ensure_ascii=False))
    partial.unlink(missing_ok=True)  # the cache now carries the full transcript; the partial is redundant
    store.put_segments(segments)
    return bool(segments)


def _downloaded_segments(path: Path, manual: bool, cutmap: CutMap | None = None) -> list[Segment]:
    """The segments of a downloaded subtitle file, or ``[]`` when it cannot be used."""
    try:
        segments = load_transcript_file(path)
        if not segments:
            raise ValueError("it has no cues")
    except Exception as e:  # a broken subtitle must not cost the video when Whisper can replace it
        log.warning("the downloaded subtitles '%s' cannot be used (%s: %s); using another source", path.name, type(e).__name__, e)
        return []
    log.info("using the downloaded %s subtitles as the transcript (%d segments)", "manual" if manual else "automatic", len(segments))
    return _map_segments(segments, cutmap)


def _plain(seg: Segment) -> dict[str, Any]:
    """A segment as the JSON dict the cache stores (real word times only)."""
    words = (
        []
        if seg.pseudo
        else [
            {"start": w.start, "end": w.end, "word": w.text, "source_start": w.source_start, "source_end": w.source_end} for w in seg.words
        ]
    )
    return {
        "start": seg.start,
        "end": seg.end,
        "text": seg.text,
        "source_start": seg.source_start,
        "source_end": seg.source_end,
        "words": words,
    }
