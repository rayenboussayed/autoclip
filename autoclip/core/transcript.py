"""Transcripts: parse .srt/.json, run faster-whisper, cache on disk and load into SQLite."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Iterable, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

from .config import Config, ProcessingCfg, TranscriptCfg
from .denoise import SpeechEnhancer
from .errors import AutoclipError
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


def parse_srt(text: str) -> list[dict[str, Any]]:
    """SubRip text -> ``[{"start", "end", "text"}]``."""
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
    return out


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
        return to_segments(data["segments"] if isinstance(data, dict) else data)
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


def chunk_bounds(duration: float, db: F32, chunk_seconds: float) -> list[tuple[float, float]]:
    """Split ``[0, duration]`` into chunks of about ``chunk_seconds``, each boundary moved to the quietest nearby frame."""
    count = math.ceil(duration / chunk_seconds)
    radius = min(15.0, chunk_seconds / 4)
    cuts = [0.0, *(snap_to_quiet(db, k * chunk_seconds, radius, duration) for k in range(1, count)), duration]
    return list(pairwise(cuts))


def asr_device(cfg: Config) -> str:
    """The device the ASR model runs on: the global switch first, then ``transcript.device``."""
    if cfg.compute.device == "cpu" or cfg.transcript.device == "cpu":
        return "cpu"
    import ctranslate2

    has_cuda = ctranslate2.get_cuda_device_count() > 0
    if cfg.compute.device == "gpu" or cfg.transcript.device == "cuda":
        if has_cuda:
            return "cuda"
        log.warning("a GPU was requested but no CUDA device was found: using the CPU")
        return "cpu"
    return "cuda" if has_cuda else "cpu"


@traced("transcribe")
def transcribe(src: Path, cfg: Config, info: MediaInfo, db: F32, enhancer: SpeechEnhancer | None = None) -> list[Segment]:
    """Transcribe with faster-whisper using every allowed core (and the GPU when there is one).

    The audio is processed in ``processing.chunk_minutes`` chunks (5 by default) so neither the
    denoiser nor Whisper ever holds more than that in memory; the language is detected once
    and reused for the remaining chunks.  A video shorter than one chunk is one chunk.

    ``enhancer`` denoises every chunk before it is transcribed, so the model hears clear words.
    """
    from faster_whisper import WhisperModel

    tc, proc = cfg.transcript, cfg.processing
    device = asr_device(cfg)
    model_name = fit_whisper_model(tc.model, device) if tc.auto_downgrade else tc.model
    threads = RESOURCES.cpu_budget()
    log.info("whisper model=%s device=%s%s cpu_threads=%d", model_name, device, " (GPU)" if device == "cuda" else "", threads)
    model = WhisperModel(model_name, device=device, compute_type="float16" if device == "cuda" else "int8", cpu_threads=threads)
    segmented = info.duration >= proc.chunk_threshold_minutes * 60
    bounds = chunk_bounds(info.duration, db, proc.chunk_minutes * 60) if segmented else [(0.0, info.duration)]
    language, raw, t0 = tc.language, [], time.perf_counter()
    for k, (start, end) in enumerate(bounds, 1):
        audio: F32 = decode_audio(src, start, end - start, WHISPER_SR)
        if enhancer is not None and enhancer.active:
            audio = enhancer(audio)
        options: dict[str, Any] = {"multilingual": True} if tc.multilingual else {}
        stream, found = model.transcribe(
            audio,
            vad_filter=True,
            word_timestamps=tc.word_timestamps,
            language=language,
            condition_on_previous_text=False,
            beam_size=tc.beam_size,
            **options,
        )
        for seg in stream:
            words = [{"start": start + w.start, "end": start + w.end, "word": w.word} for w in seg.words or []]
            raw.append({"start": start + seg.start, "end": start + seg.end, "text": seg.text, "words": words})
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
    log.info(
        "  language %s, %d segments in %.1fs (%s)",
        language,
        len(raw),
        time.perf_counter() - t0,
        speed_text(info.duration, time.perf_counter() - t0),
    )
    return to_segments(raw)


def _cache_key(src: Path, cfg: TranscriptCfg, proc: ProcessingCfg) -> dict[str, Any]:
    """Identity of a transcription run: the cached transcript is reused only while this is unchanged."""
    st = src.stat()
    return {
        "source": f"{src.name}:{st.st_size}:{st.st_mtime_ns}",
        "model": cfg.model,
        "language": cfg.language,
        "multilingual": cfg.multilingual,
        "word_timestamps": cfg.word_timestamps,
        "enhance": cfg.enhance.fingerprint(),
        "chunk_minutes": proc.chunk_minutes,
    }


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
    """Put the transcript of ``src`` into ``store`` (file, downloaded subtitles, disk cache or Whisper).

    ``transcript.file`` wins over everything. A ``subtitle`` the downloader picked for this video is
    used next (its policy already decided whether automatic captions are acceptable), then the disk
    cache, then Whisper. Returns whether a transcript exists.
    """
    tc = cfg.transcript
    if tc.mode == "off" or not need_text:
        return False
    if tc.mode == "file" and not tc.file and subtitle is None:
        raise AutoclipError("transcript.mode is 'file' but neither transcript.file nor a downloaded subtitle is available")
    stem = name or src.stem
    if tc.file:
        segments = load_transcript_file(Path(tc.file.replace("{stem}", stem)))
        log.info("loaded transcript: %d segments", len(segments))
    else:
        segments = _downloaded_segments(subtitle, tc.mode, subtitle_manual) if subtitle is not None else []
        if segments:
            store.put_segments(segments)
            return True
        cache_file, key = work_dir / f"{stem}.transcript.json", _cache_key(src, tc, cfg.processing)
        segments = []
        if tc.cache and cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if data.get("meta") == key:
                segments = to_segments(data["segments"])
                log.info("reusing cached transcript %s (%d segments)", cache_file.name, len(segments))
            else:
                log.warning("cached transcript is stale (input, model or language changed): transcribing again")
        if not segments:
            enhancer = SpeechEnhancer(tc.enhance, _model_dir(tc, work_dir))
            try:
                segments = transcribe(src, cfg, info, db, enhancer)
            except Exception as e:
                if tc.mode == "whisper":
                    raise AutoclipError(f"transcription failed: {type(e).__name__}: {e}") from e
                log.warning("transcription unavailable (%s: %s); continuing without text", type(e).__name__, e)
                return False
            cache_file.write_text(
                json.dumps({"meta": key, "segments": [_plain(s) for s in segments]}, ensure_ascii=False), encoding="utf-8"
            )
    store.put_segments(segments)
    return bool(segments)


def _downloaded_segments(path: Path, mode: str, manual: bool) -> list[Segment]:
    """The segments of a downloaded subtitle file, or ``[]`` when it cannot be used (an error in "file" mode)."""
    try:
        segments = load_transcript_file(path)
        if not segments:
            raise ValueError("it has no cues")
    except Exception as e:  # a broken subtitle must not cost the video when Whisper can replace it
        if mode == "file":
            raise AutoclipError(f"cannot use the downloaded subtitles '{path.name}': {type(e).__name__}: {e}") from e
        log.warning("the downloaded subtitles '%s' cannot be used (%s: %s); using another source", path.name, type(e).__name__, e)
        return []
    log.info("using the downloaded %s subtitles as the transcript (%d segments)", "manual" if manual else "automatic", len(segments))
    return segments


def _model_dir(cfg: TranscriptCfg, work_dir: Path) -> Path:
    """Where the speech enhancement model is kept: the configured folder or ``<output>/.work/models``."""
    if cfg.enhance.model_dir:
        return Path(cfg.enhance.model_dir)
    return work_dir.parent / "models"


def _plain(seg: Segment) -> dict[str, Any]:
    """A segment as the JSON dict the cache stores (real word times only)."""
    words = [] if seg.pseudo else [{"start": w.start, "end": w.end, "word": w.text} for w in seg.words]
    return {"start": seg.start, "end": seg.end, "text": seg.text, "words": words}
