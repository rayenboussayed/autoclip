"""Choosing the clips: candidate windows, vectorised style scoring, non-overlapping picks, optional LLM re-rank."""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import ClipsCfg, Config, KindCfg
from .log import traced
from .media import F32, FRAME, frame_energy_db, scene_cuts, snap_to_cut, snap_to_quiet
from .models import MediaInfo, Moment
from .store import Cache, Store
from .styles import LEXICONS, STYLE_HELP, STYLES

log = logging.getLogger("autoclip")
F64 = NDArray[np.float64]
Counter = Callable[[str], int]
Segs = list[tuple[float, float, str, bool]]
CJK_CHARS = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\u0e00-\u0e7f\u3000-\u303f\uff00-\uffef]")
#: How much a candidate near the recommended length may beat an equally good one (scene scores are z-scores).
LENGTH_BONUS = 0.75


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


def build_lexicons(cfg: Config) -> tuple[dict[str, Counter], Counter | None]:
    """Per-feature counters (built-in + ``clips.lexicon_file``) and the optional ``clips.keywords`` counter."""
    words = {name: list(entries) for name, entries in LEXICONS.items()}
    if cfg.clips.lexicon_file:
        for name, entries in json.loads(Path(cfg.clips.lexicon_file).read_text(encoding="utf-8")).items():
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


def text_candidates(segs: Segs, cc: ClipsCfg, kind: KindCfg, duration: float) -> Candidates:
    """Every run of consecutive segments that starts and ends on a sentence boundary and fits the length range.

    Segments ending after ``duration`` are ignored (a subtitle file may be longer than its video).
    """
    pairs = []
    for i, (start, *_rest) in enumerate(segs):
        if (i and not segs[i - 1][3]) or start < cc.skip_intro_seconds:
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


def audio_candidates(total: float, cc: ClipsCfg, kind: KindCfg) -> Candidates:
    """Fixed-length windows every 2 s (the shortest, the recommended and the longest length) for videos without a usable transcript."""
    middle = kind.target_seconds if kind.target_seconds is not None else (kind.min_seconds + kind.max_seconds) / 2
    spans = []
    for length in sorted({kind.min_seconds, middle, kind.max_seconds}):
        t = float(cc.skip_intro_seconds)
        while t + length <= total + 0.01:
            spans.append((t, min(t + length, total)))
            t += 2.0
    start, end = (np.array(c, dtype=np.float64) for c in zip(*spans, strict=True)) if spans else (np.zeros(0),) * 2
    none = np.full(len(spans), -1, dtype=np.int64)
    return Candidates(none, none, start, end)


def length_bonus(cands: Candidates, kind: KindCfg) -> F64:
    """A small bonus for candidates whose length is close to the recommended one.

    The recommendation is a preference, not a limit: the scene type still decides the content, the
    length only decides how close to ``target_seconds`` the winner should be.  The candidate at
    the target gets the full bonus, the one at the bound opposite the target gets nothing.
    """
    if kind.target_seconds is None:
        return np.zeros(len(cands))
    target = kind.target_seconds
    span = max(abs(target - kind.min_seconds), abs(kind.max_seconds - target), 1e-6)
    length = cands.end - cands.start
    return LENGTH_BONUS * (1.0 - np.minimum(np.abs(length - target) / span, 1.0))


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
    """Ask a local LLM (Ollama or OpenAI-compatible) for a 0-10 fit score and a title for each candidate."""
    lc = cfg.llm
    items = "\n".join(f"[{k}] ({m.end - m.start:.0f}s) {m.text[:700]}" for k, m in enumerate(top))
    prompt = (
        f"You pick moments from a long video for short-form clips. Target scene type -> {scene}: {STYLE_HELP[scene]}.\n"
        "Rate each candidate 0-10 for how well it fits AND stands alone (clear hook, complete thought). "
        "Also write a punchy title (max 8 words) in the SAME language as the candidate text.\n"
        'Return ONLY JSON: {"scores":[{"id":0,"score":7,"title":"..."}]}\n\n'
        f"Candidates:\n{items}"
    )
    messages = [
        {"role": "system", "content": "You are a precise short-form video editor. Reply with JSON only."},
        {"role": "user", "content": prompt},
    ]
    base = lc.base_url.rstrip("/")
    body: dict[str, Any]
    if base.endswith("/v1"):
        url, body = f"{base}/chat/completions", {"model": lc.model, "messages": messages, "response_format": {"type": "json_object"}}
    else:
        url, body = f"{base}/api/chat", {"model": lc.model, "messages": messages, "stream": False, "format": "json"}
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {lc.api_key}"} if lc.api_key else {})}
    request = urllib.request.Request(url, json.dumps(body).encode(), headers)  # noqa: S310 - scheme validated in LlmCfg
    with urllib.request.urlopen(request, timeout=lc.timeout) as response:  # noqa: S310
        reply = json.loads(response.read())
    content = (reply.get("message") or reply["choices"][0]["message"])["content"]
    return {int(x["id"]): (float(x["score"]), str(x.get("title", ""))[:80]) for x in json.loads(content)["scores"]}


def make_title(text: str, cfg: Config) -> str:
    """First sentence of ``text`` with a few words (fillers removed), at most 70 characters."""
    if cfg.silence.remove_fillers and cfg.silence.fillers:
        text = re.sub(rf"(?<!\w)(?:{'|'.join(map(re.escape, cfg.silence.fillers))})(?!\w)[,.]?\s*", "", text, flags=re.IGNORECASE)
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
        db = frame_energy_db(src)
        return db if db is not None else np.zeros(int(info.duration / FRAME) + 1, dtype=np.float32)

    track: F32 = cache.memoize_array(f"energy:{src.name}:{st.st_size}:{st.st_mtime_ns}", compute).astype(np.float32, copy=False)
    return track


@traced("select_moments")
def select_moments(
    src: Path, info: MediaInfo, store: Store, cache: Cache, cfg: Config, kind: str, db: F32, have_text: bool
) -> list[Moment]:
    """The best moments of ``src`` for every requested scene type, as ``kind`` ("short" or "long") clips.

    Candidates and their features are computed once; each scene type then scores them with its own weights and
    picks its best non-overlapping windows. A moment picked for one scene type is never reused for another.
    """
    cc, kcfg = cfg.clips, cfg.clips.kind(kind)
    segs = store.light_segments() if have_text else []
    cands = text_candidates(segs, cc, kcfg, info.duration)
    text_mode = len(cands) > 0
    if not text_mode:
        if segs:
            log.warning("no sentence-aligned %s candidates in the transcript; falling back to audio-only selection", kind)
        cands = audio_candidates(info.duration, cc, kcfg)
    if not len(cands):
        return []
    log.info("%s: %d candidate windows, scenes: %s", kind, len(cands), ", ".join(cc.scene_list()))
    lex, kwfn = build_lexicons(cfg)
    feats = audio_features(cands, db) | (text_features(cands, segs, lex, kwfn) if text_mode else {})
    st = src.stat()
    cuts = (
        cache.memoize_array(f"cuts:{src.name}:{st.st_size}:{st.st_mtime_ns}", lambda: np.array(scene_cuts(src))).tolist()
        if cc.snap_to_cuts
        else []
    )
    moments: list[Moment] = []
    taken: list[tuple[float, float]] = []
    for scene in cc.scene_list():
        scores = scene_score(feats, scene, kwfn is not None) + length_bonus(cands, kcfg)
        picks = pick(cands, scores, max(kcfg.count, cfg.llm.top_k) if cfg.llm.enabled and text_mode else kcfg.count, taken)
        found = [
            Moment(float(cands.start[k]), float(cands.end[k]), store.segment_text(int(cands.i[k]), int(cands.j[k])) if text_mode else "",
                   score=float(scores[k]), scene=scene, i=int(cands.i[k]), j=int(cands.j[k]))
            for k in picks
        ]  # fmt: skip
        if cfg.llm.enabled and text_mode:
            found = _rerank(found, kcfg.count, cfg, scene)
        taken += [(m.start, m.end) for m in found]
        for m in sorted(found, key=lambda m: m.score or 0.0, reverse=True):
            if text_mode:
                s, e = max(0.0, m.start - cc.pad_start), min(info.duration, m.end + cc.pad_end)
            else:
                s, e = snap_to_quiet(db, m.start, 2.0, info.duration), snap_to_quiet(db, m.end, 2.0, info.duration)
                if not kcfg.min_seconds * 0.6 <= e - s <= kcfg.max_seconds * 1.4:
                    s, e = m.start, m.end
            m.start, m.end = snap_to_cut(s, cuts, 2.5), snap_to_cut(e, cuts, 2.5)
            if m.end - m.start > kcfg.max_seconds:
                m.end = m.start + kcfg.max_seconds  # the padding and the cut snapping must not push the clip over
            m.title = m.title or make_title(m.text, cfg)
            moments.append(m)
    return moments
