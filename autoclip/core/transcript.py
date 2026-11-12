"""Transcripts: parse .srt/.json, run faster-whisper, cache on disk and load into SQLite."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import re
import time
from collections.abc import Iterable, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .denoise import SpeechEnhancer, default_models_dir, snr_db
from .errors import AutoclipError
from .internal import CHUNK_MINUTES as _CHUNK_MINUTES
from .internal.automation import ENHANCE_ATTENUATION_DB, ENHANCE_SNR_THRESHOLD_DB, transcription_beam_size
from .internal.runtime import transcript_model_for
from .log import speed_text, traced
from .media import F32, decode_audio, snap_to_quiet
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


def to_segments(raw: Iterable[Mapping[str, Any]]) -> list[Segment]:
    """Raw dicts -> time-ordered segments; segments without word times get proportional pseudo-words."""
    out: list[Segment] = []
    for item in raw:
        text = str(item["text"]).strip()
        if not text:
            continue
        start, end = float(item["start"]), max(float(item["end"]), float(item["start"]) + 0.05)
        words = [Word(float(w["start"]), float(w["end"]), str(w.get("word") or w["text"]).strip()) for w in item.get("words") or []]
        pseudo = not words
        if pseudo:
            tokens = tokenize(text)
            total, t = sum(map(len, tokens)), start
            for tok in tokens:
                words.append(Word(t, t + (end - start) * len(tok) / total, tok, pseudo=True))
                t += (end - start) * len(tok) / total
        out.append(Segment(start, end, text, words, pseudo))
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
    if chosen != name:
        log.warning("whisper '%s' needs ~%dMB but ~%dMB is usable: using '%s'", name, WHISPER_RAM_MB[name], usable, chosen)
    return chosen


def transcript_model(cfg: Config) -> str:
    """The Whisper model chosen for this run, after the free-memory fit is applied."""
    device = asr_device()
    return fit_whisper_model(transcript_model_for(cfg.quality, device), device)


def chunk_bounds(duration: float, db: F32, chunk_seconds: float) -> list[tuple[float, float]]:
    """Split ``[0, duration]`` into chunks of about ``chunk_seconds``, each boundary moved to the quietest nearby frame."""
    """Split ``[0, duration]`` into chunks of about ``chunk_seconds``, each boundary moved to the quietest nearby frame."""
    count = math.ceil(duration / chunk_seconds)
    radius = min(15.0, chunk_seconds / 4)
    cuts = [0.0, *(snap_to_quiet(db, k * chunk_seconds, radius, duration) for k in range(1, count)), duration]
    return list(pairwise(cuts))


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


@traced("transcribe")
def transcribe(
    src: Path,
    cfg: Config,
    info: MediaInfo,
    db: F32,
    enhancer: SpeechEnhancer | None = None,
    *,
    model_name_override: str | None = None,
) -> list[Segment]:
    from faster_whisper import WhisperModel

    device = asr_device()
    model_name = model_name_override or transcript_model(cfg)
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

    model: Any
    if device == "cuda":
        try:
            model = load(device)
            _warmup_model(model)
        except Exception as e:  # a broken CUDA runtime must degrade to the CPU, not fail the run
            log.warning("the CUDA runtime cannot run Whisper (%s): using the CPU", e)
            attempts.append(("cuda", e))
            device = "cpu"
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
    # Chunking is now automatic once the audio is long enough for memory safety.
    bounds = chunk_bounds(info.duration, db, _CHUNK_MINUTES * 60.0)
    language, raw, t0 = None, [], time.perf_counter()
    # `vad_filter` is applied eagerly by faster-whisper; a missing/broken VAD model (or a provider that
    # rejects it) must not abort the run - retry the same chunk with VAD disabled.
    vad = True
    for k, (start, end) in enumerate(bounds, 1):
        while True:
            audio: F32 = decode_audio(src, start, end - start, WHISPER_SR)
            if enhancer is not None and enhancer.active:
                audio = enhancer(audio)
            options: dict[str, Any] = {"multilingual": True}
            try:
                stream, found = model.transcribe(
                    audio,
                    vad_filter=vad,
                    word_timestamps=True,
                    language=language,
                    condition_on_previous_text=False,
                    beam_size=transcription_beam_size(cfg.quality),
                    **options,
                )
                chunk_rows: list[dict[str, Any]] = []
                for seg in stream:
                    words = [{"start": start + w.start, "end": start + w.end, "word": w.word} for w in seg.words or []]
                    chunk_rows.append({"start": start + seg.start, "end": start + seg.end, "text": seg.text, "words": words})
                raw.extend(chunk_rows)
                language = language or found.language  # detect once, then reuse for the remaining chunks
                del audio
                log.info(
                    "  transcribed chunk %d/%d (%.0f-%.0fs): %d segments so far, %.0fs elapsed",
                    k,
                    len(bounds),
                    start,
                    end,
                    len(raw),
                    time.perf_counter() - t0,
                )
                break
            except Exception as e:
                with contextlib.suppress(Exception):  # pragma: no cover - best effort
                    del audio
                if vad and _vad_failure(e):
                    log.warning("VAD failed on %s (%s); retrying this chunk without VAD", _device_label(device), e)
                    vad = False
                    continue
                if device == "cuda":
                    log.warning("Whisper failed on the GPU for this chunk (%s); retrying on the CPU", e)
                    attempts.append(("cuda", e))
                    device = "cpu"
                    try:
                        model = load(device)
                    except Exception as ce:
                        attempts.append(("cpu", ce))
                        raise AutoclipError(f"Whisper failed on every available device ({_device_failure(attempts)})") from ce
                    continue
                attempts.append(("cpu", e))
                raise AutoclipError(f"Whisper failed on every available device ({_device_failure(attempts)})") from e
    log.info(
        "  language %s, %d segments in %.1fs (%s)",
        language,
        len(raw),
        time.perf_counter() - t0,
        speed_text(info.duration, time.perf_counter() - t0),
    )
    return to_segments(raw)


def _cache_key(src: Path, model: str, *, enhance: bool = True) -> dict[str, Any]:
    """Identity of a transcription run: the cached transcript is reused only while this is unchanged."""
    st = src.stat()
    return {
        "source": f"{src.name}:{st.st_size}:{st.st_mtime_ns}",
        "model": model,
        "language": None,
        "multilingual": True,
        "word_timestamps": True,
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


def _sidecar_segments(path: Path) -> list[Segment]:
    """The segments of a sidecar transcript, or ``[]`` when it cannot be used (falls back to the next source)."""
    try:
        segments = load_transcript_file(path)
        if not segments:
            raise ValueError("it has no cues")
    except Exception as e:  # a broken sidecar must not cost the video when Whisper can replace it
        log.warning("the sidecar transcript '%s' cannot be used (%s: %s); using another source", path.name, type(e).__name__, e)
        return []
    log.info("using the sidecar transcript '%s' (%d segments)", path.name, len(segments))
    return segments


@traced("get_transcript")
def get_transcript(
    src: Path,
    cfg: Config,
    work_dir: Path,
    store: Store,
    info: MediaInfo,
    db: F32,
    need_text: bool,
    subtitle: Path | None = None,
    subtitle_manual: bool = False,
    name: str | None = None,
) -> bool:
    """Put the transcript of ``src`` into ``store`` (sidecar file, downloaded subtitles, disk cache or Whisper).

    A ``.srt`` or ``.json`` transcript next to the source wins (it is the natural place for one). A
    ``subtitle`` the downloader picked for this video is used next (its policy already decided whether
    automatic captions are acceptable), then the disk cache, then Whisper. Returns whether a transcript exists.

    ``skip.stt`` stops before Whisper: the sidecar, downloaded subtitle and disk cache are still used,
    and a run without any of them simply continues without text. ``skip.denoise`` builds no speech
    enhancer (and changes the cache key, so an enhanced cache is not reused for raw audio).
    """
    if not need_text:
        return False
    stem = name or src.stem
    sidecar = _sidecar_transcript(src)
    if sidecar is not None:
        segments = _sidecar_segments(sidecar)
        if segments:
            store.put_segments(segments)
            return True
    if subtitle is not None:
        downloaded = _downloaded_segments(subtitle, subtitle_manual)
        if downloaded:
            store.put_segments(downloaded)
            return True
    selected_model = transcript_model(cfg)
    cache_file, key = work_dir / f"{stem}.transcript.json", _cache_key(src, selected_model, enhance=True)
    segments = []
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        if data.get("meta") == key:
            segments = to_segments(data["segments"])
            log.info("reusing cached transcript %s (%d segments)", cache_file.name, len(segments))
        else:
            log.warning("cached transcript is stale (input, model or language changed): transcribing again")
    if not segments:
        if cfg.skip.stt:
            log.info("skip.stt is set: no cached transcript or subtitle is available")
            return False
        if not info.has_audio:
            log.info("the video has no audio: there is no transcript to produce")
            return False
        noisy = decode_audio(src, 0.0, info.duration, 16000)
        make = snr_db(noisy) < ENHANCE_SNR_THRESHOLD_DB
        enhancer = SpeechEnhancer(True, default_models_dir()) if make else None
        try:
            segments = transcribe(src, cfg, info, db, enhancer, model_name_override=selected_model)
        except Exception as e:
            detail = str(e) if isinstance(e, AutoclipError) else f"{type(e).__name__}: {e}"
            log.warning("transcription skipped (%s): selecting on audio only; captions and titles need a transcript", detail)
            return False
        cache_file.write_text(json.dumps({"meta": key, "segments": [_plain(s) for s in segments]}, ensure_ascii=False), encoding="utf-8")
    store.put_segments(segments)
    return bool(segments)


def _downloaded_segments(path: Path, manual: bool) -> list[Segment]:
    """The segments of a downloaded subtitle file, or ``[]`` when it cannot be used."""
    try:
        segments = load_transcript_file(path)
        if not segments:
            raise ValueError("it has no cues")
    except Exception as e:  # a broken subtitle must not cost the video when Whisper can replace it
        log.warning("the downloaded subtitles '%s' cannot be used (%s: %s); using another source", path.name, type(e).__name__, e)
        return []
    log.info("using the downloaded %s subtitles as the transcript (%d segments)", "manual" if manual else "automatic", len(segments))
    return segments


def _plain(seg: Segment) -> dict[str, Any]:
    """A segment as the JSON dict the cache stores (real word times only)."""
    words = [] if seg.pseudo else [{"start": w.start, "end": w.end, "word": w.text} for w in seg.words]
    return {"start": seg.start, "end": seg.end, "text": seg.text, "words": words}
