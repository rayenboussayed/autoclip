"""Transcripts: parse .srt/.json, run faster-whisper, cache on disk and load into SQLite."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .cutmap import CutMap
from .denoise import SpeechEnhancer, default_models_dir
from .errors import AutoclipError
from .events import Cancelled, CancelToken
from .internal import CHUNK_MINUTES as _CHUNK_MINUTES
from .internal.automation import (
    ENHANCE_ATTENUATION_DB,
    ENHANCE_SNR_THRESHOLD_DB,
    HALLUCINATION_SILENCE_THRESHOLD,
    transcription_beam_size,
)
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
#: ``base`` is documented for its RAM cost but is never an engine pick: measured 0% Arabic and CER 0.97.
WHISPER_NEVER_FIT = frozenset({"base"})


def _cap_cpu_model(name: str) -> str:
    """A GPU failure never loads a CUDA-sized model on the CPU: cap at ``small`` (``medium`` is already ~5x slower)."""
    order = tuple(WHISPER_RAM_MB)
    if name not in order:  # custom model names/paths are never downgraded
        return name
    return name if order.index(name) <= order.index("small") else "small"


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
    fitting = [m for m, mb in WHISPER_RAM_MB.items() if m not in WHISPER_NEVER_FIT and mb <= min(usable, WHISPER_RAM_MB[name])]
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
    """Append one resumable chunk record; a stale or torn partial is repaired first."""
    header_line = json.dumps(header, ensure_ascii=False, separators=(",", ":")) + "\n"
    stale = False
    if path.exists():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines, stale = [], True
        stale = True if not lines else lines[0] != header_line.rstrip("\n")
        if not stale:
            good = [lines[0]]
            for line in lines[1:]:
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    break  # a crash mid-write left a torn line: keep the valid prefix
                if not isinstance(parsed, dict) or parsed.get("type") != "chunk" or not isinstance(parsed.get("rows"), list):
                    break
                good.append(line)
            if len(good) != len(lines):
                path.write_text("\n".join(good) + "\n", encoding="utf-8")
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


class WhisperSession:
    """One resident Whisper model for a whole transcription pass (standalone or fused into the assembly).

    The model is loaded lazily on the first chunk: when the partial cache already holds every
    chunk, the transcript is replayed without touching ``faster_whisper`` at all.
    """

    def __init__(
        self,
        src: Path,
        *,
        enhancer: SpeechEnhancer | None = None,
        model_name_override: str | None = None,
        partial_path: Path | None = None,
        partial_meta: Mapping[str, Any] | None = None,
        chunk_count: int = 0,
    ) -> None:
        self.src = src
        self.enhancer = enhancer
        self.model_name_override = model_name_override
        self.partial_path, self.partial_meta = partial_path, partial_meta
        self.chunk_count = chunk_count
        self.raw, self.language, self.start_chunk = (
            _partial_load(partial_path, partial_meta, chunk_count)
            if partial_path is not None and partial_meta is not None
            else ([], None, 0)
        )
        self._model: Any = None
        self._device = ""
        self._model_name = ""
        self._attempts: list[tuple[str, Exception]] = []
        self._vad = True
        self.failed = False  # set by the assembly if a fused chunk fails; the run then degrades to audio-only
        if self.start_chunk:
            log.info("resuming transcript from partial cache after chunk %d", self.start_chunk)

    @property
    def complete(self) -> bool:
        """Whether the partial cache already holds every chunk (no model needed)."""
        return self.chunk_count > 0 and self.start_chunk >= self.chunk_count

    def _load(self, dev: str) -> Any:
        from faster_whisper import WhisperModel

        return WhisperModel(
            self._model_name,
            device=dev,
            compute_type="float16" if dev == "cuda" else "int8",
            cpu_threads=RESOURCES.cpu_budget(),
            local_files_only=True,
        )

    def _load_cpu(self) -> None:
        try:
            self._model = self._load("cpu")
        except Exception as ce:
            self._attempts.append(("cpu", ce))
            raise AutoclipError(f"Whisper failed on every available device ({_device_failure(self._attempts)})") from ce

    def ensure_model(self) -> None:
        """Load the resident model once, falling back from CUDA to a CPU-capped model."""
        if self._model is not None:
            return
        device = asr_device()
        # The override is already the model chosen for this run; re-fitting it here would let a
        # transient RAM reading produce rows under a different key than the one we record.
        self._model_name = self.model_name_override or transcript_model()
        # Whisper models download automatically on first use (consent is not a configuration option)
        if not _model_is_cached(self._model_name):
            _download_model(self._model_name)
        if device == "cuda":
            try:
                self._model = self._load(device)
                _warmup_model(self._model)
            except Exception as e:
                log.warning("the CUDA runtime cannot run Whisper (%s): using the CPU", e)
                self._attempts.append(("cuda", e))
                _release_whisper(self._model)
                device = "cpu"
                self._model_name = fit_whisper_model(_cap_cpu_model(self._model_name), device)  # no multi-hour CPU fallback
                self._load_cpu()
        else:
            try:
                self._model = self._load(device)
            except Exception as e:
                self._attempts.append(("cpu", e))
                raise AutoclipError(f"Whisper failed on every available device ({_device_failure(self._attempts)})") from e
        self._device = device
        self._warn_slow_cpu()
        log.info(
            "whisper model=%s device=%s%s cpu_threads=%d",
            self._model_name,
            device,
            " (GPU)" if device == "cuda" else "",
            RESOURCES.cpu_budget(),
        )

    def transcribe_chunk(
        self,
        k: int,
        pieces: Sequence[tuple[float, float, float]],
        cancel: CancelToken | None = None,
    ) -> None:
        """Transcribe one already-decoded chunk and append its rows to the partial cache."""
        if k <= self.start_chunk or not pieces:
            return
        self.ensure_model()
        if cancel is not None:
            cancel.check()
        audio = np.concatenate([decode_audio(self.src, start, end - start, WHISPER_SR) for start, end, _ in pieces])
        if self.enhancer is not None:
            audio = self.enhancer(audio)
        while True:
            try:
                stream, found = self._model.transcribe(
                    audio,
                    vad_filter=self._vad,
                    word_timestamps=True,
                    language=self.language,
                    condition_on_previous_text=False,
                    beam_size=transcription_beam_size(self._device),
                    multilingual=False,  # detect once, then reuse: per-window detection drifts into other languages
                    hallucination_silence_threshold=HALLUCINATION_SILENCE_THRESHOLD,
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
                if cancel is not None:
                    cancel.check()
                self.raw.extend(chunk_rows)
                self.language = self.language or found.language
                if self.partial_path is not None and self.partial_meta is not None:
                    _partial_append(
                        self.partial_path,
                        {"type": "meta", "meta": dict(self.partial_meta), "format": TRANSCRIPT_CACHE_FORMAT},
                        {"type": "chunk", "done": k, "language": self.language, "rows": chunk_rows},
                    )
                del audio
                log.info(
                    "  transcribed chunk %d/%d (%.0f-%.0fs): %d segments so far",
                    k,
                    self.chunk_count,
                    pieces[0][0],
                    sum(end - start for start, end, _ in pieces),
                    len(self.raw),
                )
                break
            except Cancelled:
                raise
            except Exception as e:
                if self._vad and _vad_failure(e):
                    log.warning("VAD failed on %s (%s); retrying this chunk without VAD", _device_label(self._device), e)
                    self._vad = False
                    continue
                if self._device == "cuda":
                    log.warning("Whisper failed on the GPU for this chunk (%s); retrying on the CPU", e)
                    self._attempts.append(("cuda", e))
                    self._device = "cpu"
                    _release_whisper(self._model)
                    self._model = None
                    # A mid-file GPU failure must not silently load a CUDA-sized model on the CPU either.
                    self._model_name = fit_whisper_model(_cap_cpu_model(self._model_name), "cpu")
                    self._warn_slow_cpu()
                    self._load_cpu()
                    continue
                self._attempts.append(("cpu", e))
                raise AutoclipError(f"Whisper failed on every available device ({_device_failure(self._attempts)})") from e

    def _warn_slow_cpu(self) -> None:
        if self._device == "cpu" and self._model_name in ("medium", "large-v3"):
            log.warning(
                "whisper '%s' on the CPU is much slower than 'small' (about 5x); a long video can take hours",
                self._model_name,
            )

    def close(self) -> None:
        """Release the native model and enhancer (idempotent)."""
        _release_whisper(self._model)
        self._model = None
        if self.enhancer is not None:
            self.enhancer.close()
            self.enhancer = None


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
    """Transcribe one source, chunk by chunk, reusing a single resident Whisper model."""
    cutmap = cutmap or CutMap.identity(info.duration)
    chunks = list(chunks) if chunks is not None else cutmap.chunked_pieces(_CHUNK_MINUTES * 60.0)
    t0 = time.perf_counter()
    session = WhisperSession(
        src,
        enhancer=enhancer,
        model_name_override=model_name_override,
        partial_path=partial_path,
        partial_meta=partial_meta,
        chunk_count=len(chunks),
    )
    language = session.language
    try:
        if not session.complete:
            session.ensure_model()  # standalone transcription always loads once, even for empty chunks
            for k, pieces in enumerate(chunks, 1):
                if cancel is not None:
                    cancel.check()
                session.transcribe_chunk(k, pieces, cancel=cancel)
            language = session.language
    finally:
        session.close()
    log.info(
        "  language %s, %d segments in %.1fs (%s)",
        language,
        len(session.raw),
        time.perf_counter() - t0,
        speed_text(info.duration, time.perf_counter() - t0),
    )
    return to_segments(session.raw)


@dataclass(frozen=True, slots=True)
class FusedStt:
    """Arm the in-assembly Whisper pass only when the standalone pass would have run anyway.

    The cut map is only available after the speech scan, so the decision is a factory called by
    ``prepare_source`` once the kept intervals are known.
    """

    src: Path
    cfg: Config
    work: Path
    stem: str
    want: bool
    model_name: str
    subtitle: Path | None = None

    def plan(self, cut: CutMap) -> WhisperSession | None:
        """Return a ready session, or ``None`` when a cheaper transcript source wins."""
        if not self.want or self.subtitle is not None or _sidecar_transcript(self.src) is not None:
            return None
        key = _cache_key(self.src, self.model_name, cut, enhance=not self.cfg.skip.denoise)
        if _load_cached_transcript(self.work / f"{self.stem}.transcript.json", key):
            return None
        chunks = cut.chunked_pieces(_CHUNK_MINUTES * 60.0)
        session = WhisperSession(
            self.src,
            model_name_override=self.model_name,
            partial_path=self.work / f"{self.stem}.transcript.partial.jsonl",
            partial_meta=key,
            chunk_count=len(chunks),
        )
        if session.complete:
            session.close()  # nothing to do: get_transcript will replay the completed partial
            return None
        if not self.cfg.skip.denoise:
            session.enhancer = SpeechEnhancer(True, default_models_dir())
        return session


TRANSCRIPT_CACHE_FORMAT = 2


def _cache_key(src: Path, model: str, cutmap: CutMap, *, enhance: bool = True) -> dict[str, Any]:
    """Identity of a transcription run: the cached transcript is reused only while this is unchanged."""
    st = src.stat()
    return {
        "source": f"{src.name}:{st.st_size}:{st.st_mtime_ns}",
        "model": model,
        "language": None,
        "multilingual": False,
        "hallucination_silence_threshold": HALLUCINATION_SILENCE_THRESHOLD,
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


def _load_cached_transcript(cache_file: Path, key: Mapping[str, Any]) -> list[Segment]:
    """The cached segments when the cache matches ``key``; an empty list otherwise."""
    if not cache_file.exists():
        return []
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
            raise ValueError("the cached transcript does not contain a segment list")
        if data.get("meta") == key and data.get("format") == TRANSCRIPT_CACHE_FORMAT:
            segments = to_segments(data["segments"])
            log.info("reusing cached transcript %s (%d segments)", cache_file.name, len(segments))
            return segments
        log.warning("cached transcript is stale (input, model, language or optimized timeline changed): transcribing again")
    except (OSError, json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
        log.warning("cached transcript '%s' is unreadable (%s): transcribing again", cache_file.name, exc)
    return []


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
    model_name_override: str | None = None,
    stt_failed: bool = False,
    cancel: CancelToken | None = None,
) -> bool:
    """Put the transcript of ``src`` into ``store`` (sidecar file, downloaded subtitles, disk cache or Whisper).

    All mapping is monotone and piecewise: ``start/end`` are playback times, while ``source_start/source_end`` retain
    the original source timeline. On a re-encoded source, every caller path maps through the one cut map.

    ``model_name_override`` pins the cache key to the model already chosen for this run. ``stt_failed`` suppresses a
    fresh Whisper pass after a fused failure; a complete partial is still replayed and kept for a later resume.
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
    selected_model = model_name_override or transcript_model()
    cache_file, key = work_dir / f"{stem}.transcript.json", _cache_key(cache_src, selected_model, cutmap, enhance=not cfg.skip.denoise)
    partial = work_dir / f"{stem}.transcript.partial.jsonl"
    segments = _load_cached_transcript(cache_file, key)
    if not segments:
        if stt_failed:
            log.warning("transcription was disabled after a failure while assembling; selecting on audio only")
            return False  # the partial is kept so a later run can resume it
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
        except Cancelled:
            raise
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
