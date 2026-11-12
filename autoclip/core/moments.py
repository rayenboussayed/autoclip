"""Choosing the clips: candidate windows, vectorised style scoring, non-overlapping picks, optional LLM re-rank."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import Config, KindCfg
from .internal.automation import CLIP_PAD_END, CLIP_PAD_START, DEFAULT_FILLERS, LLM_TOP_K
from .internal.content import load_lexicon
from .log import traced
from .media import F32, FRAME, frame_energy_db, snap_to_quiet
from .models import MediaInfo, Moment
from .store import Cache, Store
from .styles import LEXICONS, STYLES

log = logging.getLogger("autoclip")
F64 = NDArray[np.float64]
Counter = Callable[[str], int]
Segs = list[tuple[float, float, str, bool]]
CJK_CHARS = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\u0e00-\u0e7f\u3000-\u303f\uff00-\uffef]")


def count_units(text: str) -> int:
    """Speech-rate unit that works for every script: words for spaced scripts plus CJK/Thai characters."""
    return len(text.split()) + len(CJK_CHARS.findall(text))


def compile_lexicon(words: list[str]) -> Counter:
    """A fast phrase counter: spaced words match whole words, CJK/Thai entries match as substrings."""
    spaced = [re.escape(w) for w in sorted(words, key=len, reverse=True) if not CJK_CHARS.search(w)]
    unspaced = [re.escape(w) for w in words if CJK_CHARS.search(w)]
    parts = ([rf"(?<!\w)(?:{'|'.join(spaced)})(?!\w)"] if spaced else []) + unspaced
    if not parts:
        return lambda text: 0
    pattern = re.compile("|".join(parts), re.IGNORECASE)
    return lambda text: len(pattern.findall(text))


def build_lexicons(cfg: Config, lexicon: Mapping[str, Any] | None = None) -> tuple[dict[str, Counter], Counter | None]:
    """Per-feature counters (built-in + ``<config-dir>/lexicon.json``) and the optional ``clips.keywords`` counter."""
    words = {name: list(entries) for name, entries in LEXICONS.items()}
    if lexicon:
        for name, entries in lexicon.items():
            if name == "fillers":
                continue
            if isinstance(entries, list) and all(isinstance(w, str) for w in entries):
                words.setdefault(name, []).extend(entries)
    keywords = [k for k in cfg.clips.keywords if k]
    return {name: compile_lexicon(entries) for name, entries in words.items()}, compile_lexicon(keywords) if keywords else None


@dataclass
class Candidates:
    """Candidate windows as parallel arrays: segment index range plus start/end seconds (text stays in SQLite)."""

    i: NDArray[np.int64]
    j: NDArray[np.int64]
    start: F64
    end: F64

    def __len__(self) -> int:
        return len(self.start)


def text_candidates(segs: Segs, kind: KindCfg, duration: float) -> Candidates:
    """Every run of consecutive segments that starts and ends on a sentence boundary and fits the length range.

    Segments ending after ``duration`` are ignored (a subtitle file may be longer than its video).
    """
    pairs = []
    for i, (start, *_rest) in enumerate(segs):
        if (i and not segs[i - 1][3]) or start < 0:
            continue
        for j in range(i, len(segs)):
            length = segs[j][1] - start
            if length > kind.max_seconds or segs[j][1] > duration:
                break
            if length >= kind.min_seconds and segs[j][3]:
                pairs.append((i, j, start, segs[j][1]))
    if not pairs:
        return Candidates(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), np.zeros(0))
    firsts, lasts, starts, ends = zip(*pairs, strict=True)
    return Candidates(np.array(firsts, np.int64), np.array(lasts, np.int64), np.array(starts), np.array(ends))


def audio_candidates(total: float, kind: KindCfg) -> Candidates:
    """Fixed-length windows every 2 s (the shortest and the longest length) for videos without a usable transcript."""
    spans = []
    for length in sorted({kind.min_seconds, kind.max_seconds}):
        t = 0.0
        while t + length <= total + 0.01:
            spans.append((t, min(t + length, total)))
            t += 2.0
    start, end = (np.array(c, dtype=np.float64) for c in zip(*spans, strict=True)) if spans else (np.zeros(0),) * 2
    none = np.full(len(spans), -1, dtype=np.int64)
    return Candidates(none, none, start, end)


def zscore(v: F64) -> F64:
    """Standardise to zero mean and unit variance (zeros for a constant feature)."""
    sd = v.std()
    return (v - v.mean()) / sd if sd > 1e-9 else np.zeros_like(v)


def text_features(cands: Candidates, segs: Segs, lex: dict[str, Counter], kwfn: Counter | None) -> dict[str, F64]:
    """Per-candidate text features from prefix sums over per-segment counts (O(1) per candidate)."""
    names = sorted(lex)
    counts = np.zeros((len(segs), 3 + len(names) + (kwfn is not None)))
    question = np.zeros(len(segs), dtype=bool)
    digit = np.zeros(len(segs), dtype=bool)
    emphasis = np.zeros(len(segs), dtype=np.int64)
    for row, (_s, _e, text, _b) in enumerate(segs):
        counts[row, :3] = (count_units(text), sum(text.count(c) for c in "?!\uff1f\uff01\u061f\u00bf\u00a1"), len(re.findall(r"\d+", text)))
        counts[row, 3 : 3 + len(names)] = [lex[n](text) for n in names]
        if kwfn:
            counts[row, -1] = kwfn(text)
        question[row], digit[row], emphasis[row] = (
            bool(re.search("[?\uff1f\u061f]", text)),
            bool(re.search(r"\d", text)),
            lex["emphasis"](text),
        )
    prefix = np.vstack([np.zeros((1, counts.shape[1])), np.cumsum(counts, axis=0)])
    sums = prefix[cands.j + 1] - prefix[cands.i]
    dur = np.maximum(cands.end - cands.start, 1.0)
    feats = {"rate": sums[:, 0] / dur, "punct": sums[:, 1] / dur * 10, "digits": sums[:, 2] / dur * 30}
    feats.update({name: sums[:, 3 + k] / dur * 30 for k, name in enumerate(names)})
    if kwfn:
        feats["kw"] = sums[:, -1] / dur * 30
    starts = np.array([s[0] for s in segs])
    hook = np.zeros(len(cands))
    for n, (i, j, s0) in enumerate(zip(cands.i, cands.j, cands.start, strict=True)):
        first = slice(i, min(j + 1, np.searchsorted(starts, s0 + 5)))  # segments starting in the first 5 s
        hook[n] = float(question[first].any()) + 0.5 * min(2, int(emphasis[first].sum())) + 0.5 * float(digit[first].any())
    return {**feats, "hook": hook}


def audio_features(cands: Candidates, db: F32) -> dict[str, F64]:
    """Per-candidate audio features from prefix sums of the 0.5 s energy track: loudness, variety, share of pauses."""
    d = db.astype(np.float64)
    c1, c2, c3 = (np.concatenate([[0.0], np.cumsum(v)]) for v in (d, d**2, d < np.percentile(d, 10) + 6))
    a = np.minimum((cands.start / FRAME).astype(np.int64), len(d) - 1)
    b = np.minimum(np.maximum((cands.end / FRAME).astype(np.int64), a + 1), len(d))
    n = (b - a).astype(np.float64)
    mean = (c1[b] - c1[a]) / n
    return {"energy": mean, "variety": np.sqrt(np.maximum((c2[b] - c2[a]) / n - mean**2, 0.0)), "pause": (c3[b] - c3[a]) / n}


def scene_score(feats: dict[str, F64], scene: str, keywords: bool) -> F64:
    """Weighted sum of the z-scored features that define ``scene`` (plus the keyword boost when keywords are set)."""
    weights = {**STYLES[scene], **({"kw": 1.5} if keywords else {})}
    return sum((w * zscore(feats[f]) for f, w in weights.items() if f in feats), np.zeros(len(feats["energy"])))


def pick(cands: Candidates, scores: F64, count: int, taken: list[tuple[float, float]], gap: float = 1.0) -> list[int]:
    """Indices of the best candidates that overlap neither each other nor ``taken`` windows (``gap`` seconds apart)."""
    alive = np.ones(len(cands), dtype=bool)

    def exclude(start: float, end: float) -> None:
        alive[(cands.end + gap > start) & (cands.start < end + gap)] = False

    for start, end in taken:
        exclude(start, end)
    chosen: list[int] = []
    while len(chosen) < count and alive.any():
        best = int(np.argmax(np.where(alive, scores, -np.inf)))
        chosen.append(best)
        exclude(float(cands.start[best]), float(cands.end[best]))
    return chosen


@traced("llm_rerank")
def llm_rerank(top: list[Moment], cfg: Config, scene: str) -> dict[int, tuple[float, str]]:
    """Ask the local Ollama server for a 0-10 fit score and a title for each candidate."""
    from . import ai

    return ai.llm_rerank(top, cfg, scene)


def make_title(text: str, fillers: frozenset[str] | set[str] | None = None) -> str:
    """First sentence of ``text`` with filler sounds removed, at most 70 characters."""
    if fillers:
        lowered = sorted({f.lower() for f in fillers})
        pattern = r"(?<!\w)(?:" + "|".join(map(re.escape, lowered)) + r")(?!\w)[,.]?\s*"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    sentences = re.split(r"(?<=[.?!])\s+|(?<=[\u3002\uff01\uff1f\u061f])", " ".join(text.split()))
    # a model that speaks in short sentences ("Omit... T... Break...") would give a useless title
    first = next((s for s in sentences if len(s.split()) >= 3), sentences[0])
    return first[:70].rstrip() + ("\u2026" if len(first) > 70 else "")


def _rerank(top: list[Moment], count: int, cfg: Config, scene: str) -> list[Moment]:
    """Blend the heuristic score with the LLM score; on any failure keep the heuristic order."""
    try:
        ratings = llm_rerank(top, cfg, scene)
    except Exception as e:
        log.warning("LLM re-rank failed (%s: %s); using the heuristic ranking", type(e).__name__, e)
        return top[:count]
    z = zscore(np.array([m.score for m in top], dtype=np.float64))
    for k, m in enumerate(top):
        m.llm_score, m.title = ratings.get(k, (5.0, ""))
        m.score = float(0.4 * z[k] + 0.6 * (m.llm_score - 5.0) / 2.0)
    log.info("  LLM re-ranked %d '%s' candidates with '%s'", len(top), scene, cfg.llm.model)
    return sorted(top, key=lambda m: m.score or 0.0, reverse=True)[:count]


def energy_track(src: Path, info: MediaInfo, cache: Cache) -> F32:
    """The 0.5 s loudness track of ``src`` (cached; flat zeros for a video without audio)."""
    st = src.stat()

    def compute() -> F32:
        if not info.has_audio:
            return np.zeros(int(info.duration / FRAME) + 1, dtype=np.float32)
        return frame_energy_db(src)

    track: F32 = cache.memoize_array(f"energy:{src.name}:{st.st_size}:{st.st_mtime_ns}", compute).astype(np.float32, copy=False)
    return track


@traced("select_moments")
def select_moments(
    src: Path,
    info: MediaInfo,
    store: Store,
    cache: Cache,
    cfg: Config,
    kind: str,
    db: F32,
    have_text: bool,
    config_dir: Path | None = None,
) -> list[Moment]:
    """The best moments of ``src`` for every requested scene type, as ``kind`` ("short" or "long") clips.

    Candidates and their features are computed once; each scene type then scores them with its own weights and
    picks its best non-overlapping windows. A moment picked for one scene type is never reused for another.
    """
    cc, kcfg = cfg.clips, cfg.clips.kind(kind)
    segs = store.light_segments() if have_text else []
    cands = text_candidates(segs, kcfg, info.duration)
    text_mode = len(cands) > 0
    if not text_mode:
        if segs:
            log.warning("no sentence-aligned %s candidates in the transcript; falling back to audio-only selection", kind)
        cands = audio_candidates(info.duration, kcfg)
    if not len(cands):
        return []
    log.info("%s: %d candidate windows, scenes: %s", kind, len(cands), ", ".join(cc.scene_list()))
    lex, kwfn = build_lexicons(cfg, load_lexicon(config_dir))
    feats = audio_features(cands, db) | (text_features(cands, segs, lex, kwfn) if text_mode else {})
    from . import ai

    use_llm = not cfg.offline and ai.llm_available(cfg.llm.base_url)
    moments: list[Moment] = []
    taken: list[tuple[float, float]] = []
    for scene in cc.scene_list():
        scores = scene_score(feats, scene, kwfn is not None)
        picks = pick(cands, scores, max(kcfg.count, LLM_TOP_K) if use_llm and text_mode else kcfg.count, taken)
        found = [
            Moment(float(cands.start[k]), float(cands.end[k]), store.segment_text(int(cands.i[k]), int(cands.j[k])) if text_mode else "",
                   score=float(scores[k]), scene=scene, i=int(cands.i[k]), j=int(cands.j[k]))
            for k in picks
        ]  # fmt: skip
        if use_llm and text_mode:
            found = _rerank(found, kcfg.count, cfg, scene)
            for m in found:
                m.caption = m.caption or ai.generate_caption(cfg, m.text)
        elif not use_llm and text_mode and kind == "long":
            for m in found:
                m.title = make_title(m.text, set(DEFAULT_FILLERS))
        taken += [(m.start, m.end) for m in found]
        for m in sorted(found, key=lambda m: m.score or 0.0, reverse=True):
            if text_mode:
                s, e = max(0.0, m.start - CLIP_PAD_START), min(info.duration, m.end + CLIP_PAD_END)
            else:
                s, e = snap_to_quiet(db, m.start, 2.0, info.duration), snap_to_quiet(db, m.end, 2.0, info.duration)
                # Snapping may move both edges: keep it only when the window still honours the length the user asked for.
                if not kcfg.min_seconds <= e - s <= kcfg.max_seconds:
                    s, e = m.start, m.end
            m.start, m.end = s, e
            if m.end - m.start > kcfg.max_seconds:
                m.end = m.start + kcfg.max_seconds  # the padding must not push the clip over its maximum
            m.title = m.title or make_title(m.text, frozenset(DEFAULT_FILLERS))
            moments.append(m)
    return moments
