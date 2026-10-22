"""Candidate generation, per-scene scoring, picking, LLM re-ranking and end-to-end moment selection."""

from __future__ import annotations

import http.server
import json
import logging
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autoclip import moments
from autoclip.config import Config, parse
from autoclip.media import probe
from autoclip.models import MediaInfo, Moment
from autoclip.moments import (
    audio_candidates,
    audio_features,
    build_lexicons,
    compile_lexicon,
    count_units,
    energy_track,
    llm_rerank,
    make_title,
    pick,
    scene_score,
    select_moments,
    text_candidates,
    text_features,
    zscore,
)
from autoclip.resources import RESOURCES
from autoclip.store import Cache, Store
from autoclip.styles import STYLES

from .conftest import SENTENCES, Media, make_cfg, merge

SEGS = [(k * 4.0, k * 4.0 + 3.0, t, True) for k, t in enumerate(SENTENCES)]


def cfg(**sections: dict[str, Any]) -> Config:
    base = {
        "clips": {"short": {"count": 2, "min_seconds": 6, "max_seconds": 12}, "long": {"count": 1, "min_seconds": 12, "max_seconds": 30}}
    }
    return parse(Config, merge(base, sections))


def covers(c: Any, k: int, sentence: int) -> bool:
    return int(c.i[k]) <= sentence <= int(c.j[k])


def test_count_units_and_lexicons() -> None:
    assert count_units("one two three") == 3 and count_units("\u4eca\u5929\u597d") == 3 + 1
    count = compile_lexicon(["lol", "can't believe", "\u7b11"])
    assert count("LOL I can't believe it, lolly \u7b11\u7b11") == 4  # whole words only, CJK as substrings
    assert compile_lexicon([])("anything") == 0
    assert compile_lexicon(["only-cjk\u7b11"])("only-cjk\u7b11") == 1


def test_build_lexicons_merges_file_and_keywords(tmp_path: Path) -> None:
    (tmp_path / "lex.json").write_text(json.dumps({"funny": ["chistoso"], "newkind": ["zzz"]}), encoding="utf-8")
    lex, kw = build_lexicons(cfg(clips={"lexicon_file": str(tmp_path / "lex.json"), "keywords": ["mother", ""]}))
    assert lex["funny"]("muy chistoso, lol") == 2 and lex["newkind"]("zzz") == 1
    assert kw is not None and kw("My mother") == 1
    assert build_lexicons(cfg())[1] is None


def test_text_candidates_use_the_length_range_of_their_kind() -> None:
    c = cfg()
    short = text_candidates(SEGS, c.clips, c.clips.short, duration=40)
    long = text_candidates(SEGS, c.clips, c.clips.long, duration=40)
    assert np.all((short.end - short.start >= 6) & (short.end - short.start <= 12))
    assert np.all((long.end - long.start >= 12) & (long.end - long.start <= 30)) and np.max(long.end) - np.min(long.start) > 12
    assert np.all(short.i <= short.j) and np.all(short.start == np.array([SEGS[i][0] for i in short.i]))
    late = text_candidates(SEGS, cfg(clips={"skip_intro_seconds": 20}).clips, c.clips.short, 40)
    assert np.min(late.start) >= 20
    # a subtitle file longer than its video
    assert np.max(text_candidates(SEGS, c.clips, c.clips.short, duration=15).end) <= 15
    assert len(text_candidates(SEGS, c.clips, c.clips.long.__class__(min_seconds=50, max_seconds=60), 40)) == 0
    open_ended = [(0.0, 5.0, "no end", False), (5.0, 10.0, "still going", False), (10.0, 15.0, "Done.", True)]
    kind = c.clips.long.__class__(min_seconds=5, max_seconds=20)
    found = text_candidates(open_ended, c.clips, kind, 20)
    assert list(zip(found.i.tolist(), found.j.tolist(), strict=True)) == [(0, 2)]  # only the run that ends on a sentence


def test_audio_candidates() -> None:
    c = cfg(clips={"skip_intro_seconds": 4})
    cands = audio_candidates(30, c.clips, c.clips.short)
    assert np.all(cands.i == -1) and np.min(cands.start) == 4 and np.max(cands.end) <= 30
    assert len(audio_candidates(3, c.clips, c.clips.short)) == 0


def test_zscore() -> None:
    z = zscore(np.array([1.0, 2.0, 3.0]))
    assert z.mean() == pytest.approx(0) and z.std() == pytest.approx(1)
    assert not zscore(np.ones(4)).any()


def feature_set(config: Config) -> tuple[Any, dict[str, Any]]:
    cands = text_candidates(SEGS, config.clips, config.clips.short, 40)
    lex, kw = build_lexicons(config)
    db = np.full(80, -30.0, np.float32)
    return cands, audio_features(cands, db) | text_features(cands, SEGS, lex, kw)


def test_every_scene_type_prefers_its_own_window() -> None:
    cands, feats = feature_set(cfg())
    assert {"energy", "variety", "pause", "rate", "punct", "digits", "hook", "funny", "sad"} <= feats.keys()
    best = {s: int(np.argmax(scene_score(feats, s, False))) for s in STYLES}
    assert covers(cands, best["funny"], 3) and covers(cands, best["sad"], 4)  # "haha ... lol" / "mother passed away"
    assert covers(cands, best["exciting"], 6) and covers(cands, best["inspiring"], 7) and covers(cands, best["controversial"], 8)
    assert covers(cands, best["educational"], 5)  # "how to learn step by step"
    assert feats["hook"].max() >= 1.0  # "Why does this matter ...?" opens with a question


def test_keywords_boost_every_scene() -> None:
    _, feats = feature_set(cfg(clips={"keywords": ["incredible"]}))
    assert "kw" in feats
    plain, boosted = scene_score(feats, "calm", False), scene_score(feats, "calm", True)
    assert not np.allclose(plain, boosted) and boosted[int(np.argmax(feats["kw"]))] > plain[int(np.argmax(feats["kw"]))]


def test_audio_only_scoring_prefers_energetic_windows() -> None:
    c = cfg()
    cands = audio_candidates(40, c.clips, c.clips.short)
    db = np.concatenate([np.full(40, -50.0), np.full(40, -10.0)]).astype(np.float32)  # quiet first half, loud second
    feats = audio_features(cands, db)
    assert "hook" not in feats
    assert cands.start[int(np.argmax(scene_score(feats, "calm", False)))] < 20  # "calm" favours the quiet part
    assert cands.start[int(np.argmax(scene_score(feats, "exciting", False)))] >= 18  # "exciting" the loud one


def test_pick_is_greedy_non_overlapping_and_respects_taken() -> None:
    c = cfg()
    cands = audio_candidates(40, c.clips, c.clips.short)
    scores = -np.abs(cands.start - 10.0)  # best window starts at 10 s
    chosen = pick(cands, scores, 3, [])
    spans = sorted((float(cands.start[k]), float(cands.end[k])) for k in chosen)
    assert len(chosen) == 3 and cands.start[chosen[0]] == pytest.approx(10.0)
    assert all(b[0] >= a[1] + 1.0 for a, b in zip(spans, spans[1:], strict=False))  # at least the 1 s gap
    avoiding = pick(cands, scores, 1, [(5.0, 20.0)])
    assert cands.end[avoiding[0]] + 1.0 <= 5.0 or cands.start[avoiding[0]] >= 21.0  # stays clear of already-taken windows
    assert len(pick(cands, scores, 100, [])) < 100  # runs out of room instead of overlapping
    assert pick(cands, scores, 3, [(0.0, 100.0)]) == []


def test_the_recommended_length_is_preferred_but_never_forced() -> None:
    """``target_seconds`` is a preference: a candidate near it wins, the range stays the limit."""
    c = cfg()
    kind = c.clips.short.__class__(min_seconds=5, max_seconds=20, target_seconds=12.5)
    cands = audio_candidates(40, c.clips, kind)
    lengths = cands.end - cands.start
    assert 12.5 in set(lengths.round(2))  # the recommended length is among the windows
    bonus = moments.length_bonus(cands, kind)
    assert bonus[np.isclose(lengths, 12.5)][0] == pytest.approx(moments.LENGTH_BONUS)
    assert bonus[np.isclose(lengths, 5.0)][0] == pytest.approx(0.0)  # the minimum gets nothing
    assert bonus[np.isclose(lengths, 20.0)][0] == pytest.approx(0.0)  # the maximum gets nothing
    assert moments.length_bonus(cands, c.clips.short.__class__(min_seconds=5, max_seconds=20)).tolist() == [0.0] * len(cands)
    # the same scene score, a different length: the one near the target wins
    same = np.zeros(len(cands))
    near = int(np.argmax(same + bonus))
    assert lengths[near] == pytest.approx(12.5)


@pytest.fixture
def llm_server() -> Iterator[dict[str, Any]]:
    """A local HTTP server answering like Ollama (/api/chat) or an OpenAI-compatible API (/v1/chat/completions)."""
    state: dict[str, Any] = {"requests": [], "reply": None}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, self.headers.get("Authorization"), body))
            payload = state["reply"](self.path) if callable(state["reply"]) else state["reply"]
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    yield state
    server.shutdown()
    server.server_close()


def ratings(n: int) -> str:
    return json.dumps({"scores": [{"id": k, "score": 9 - k, "title": f"LLM title {k}" + "x" * 100} for k in range(n)]})


def test_llm_rerank_ollama_and_openai_styles(llm_server: dict[str, Any]) -> None:
    top = [Moment(0, 10, "some text " * 100, score=1.0) for _ in range(3)]
    llm_server["reply"] = lambda path: (
        {"message": {"content": ratings(3)}} if path == "/api/chat" else {"choices": [{"message": {"content": ratings(3)}}]}
    )
    out = llm_rerank(top, cfg(llm={"enabled": True, "base_url": llm_server["url"] + "/"}), "funny")
    assert out[0][0] == 9 and out[0][1].startswith("LLM title 0") and len(out[0][1]) <= 80
    path, auth, body = llm_server["requests"][-1]
    assert path == "/api/chat" and auth is None and body["format"] == "json" and "funny" in body["messages"][1]["content"]
    assert "humorous" in body["messages"][1]["content"]  # the scene's description is part of the prompt
    out2 = llm_rerank(top, cfg(llm={"enabled": True, "base_url": llm_server["url"] + "/v1", "api_key": "sk-test"}), "sad")
    path, auth, body = llm_server["requests"][-1]
    assert path == "/v1/chat/completions" and auth == "Bearer sk-test" and "response_format" in body and len(out2) == 3


def test_make_title() -> None:
    assert make_title("Um, I think so. And more.", cfg()) == "I think so."
    assert make_title("x" * 100, cfg()).endswith("\u2026") and len(make_title("x" * 100, cfg())) == 71
    assert make_title("Um, hello.", cfg(silence={"remove_fillers": False})) == "Um, hello."
    assert make_title("\u4eca\u5929\u597d\u3002\u660e\u5929\u4e5f\u597d\u3002", cfg()) == "\u4eca\u5929\u597d\u3002"
    # a model that speaks in short sentences must not produce a one-word title
    assert make_title("Omit... T... Break... Key and value.", cfg()) == "Key and value."
    assert make_title("Omit.", cfg()) == "Omit."  # nothing longer is available


# ---- select_moments (real fixture video) ---------------------------------------------------------
@pytest.fixture
def pipeline(media: Media, tmp_path: Path):  # type: ignore[no-untyped-def]
    store = Store(tmp_path / "p.db")
    cache = Cache(store, RESOURCES)
    info = probe(media.video)
    yield store, cache, info, energy_track(media.video, info, cache)
    cache.close()
    store.close()


def load_segments(store: Store, media: Media) -> None:
    from autoclip.transcript import load_transcript_file

    store.put_segments(load_transcript_file(media.transcript))


def select(
    media: Media, pipeline: Any, tmp_path: Path, kind: str = "short", have_text: bool = True, **sections: dict[str, Any]
) -> list[Moment]:
    store, cache, info, db = pipeline
    if have_text:
        load_segments(store, media)
    return select_moments(media.video, info, store, cache, make_cfg(media, tmp_path, None, **sections), kind, db, have_text)


def test_energy_track_is_cached_and_flat_without_audio(media: Media, pipeline: Any) -> None:
    store, cache, info, db = pipeline
    assert db.dtype == np.float32 and len(db) == pytest.approx(80, abs=2) and db.max() > db.min() + 20
    assert store.conn.execute("SELECT COUNT(*) FROM kv WHERE k LIKE 'energy:%'").fetchone()[0] == 1
    assert energy_track(media.video, info, cache).shape == db.shape  # second call: from the cache
    silent = energy_track(media.silent, probe(media.silent), cache)
    assert len(silent) == pytest.approx(80, abs=2) and not silent.any()


def test_each_scene_type_gets_its_own_distinct_moments(media: Media, pipeline: Any, tmp_path: Path) -> None:
    picks = select(media, pipeline, tmp_path, clips={"scenes": ["funny", "sad"], "short": {"count": 2}})
    scenes = [m.scene for m in picks]
    assert scenes.count("funny") == 2 and 1 <= scenes.count("sad") <= 2  # a short video has room for fewer clips than asked:
    assert scenes == sorted(scenes, key=["funny", "sad"].index)  # ... the scene listed first picks first
    spans = sorted((m.start, m.end) for m in picks)
    assert all(b[0] >= a[1] for a, b in zip(spans, spans[1:], strict=False))  # no moment is reused under another label
    best_funny = max((m for m in picks if m.scene == "funny"), key=lambda m: m.score or 0)
    assert "haha" in best_funny.text and all(m.title for m in picks) and all(0 <= m.start < m.end <= 40 for m in picks)


def test_all_scene_types(media: Media, pipeline: Any, tmp_path: Path) -> None:
    picks = select(media, pipeline, tmp_path, clips={"scenes": "all", "short": {"count": 1, "min_seconds": 3, "max_seconds": 4}})
    assert {m.scene for m in picks} <= set(STYLES) and len({m.scene for m in picks}) >= 8  # ten scenes compete for ten sentences
    assert len({(m.start, m.end) for m in picks}) == len(picks)  # all different


def test_long_kind_uses_longer_windows(media: Media, pipeline: Any, tmp_path: Path) -> None:
    short = select(media, pipeline, tmp_path, "short")
    long = select(media, pipeline, tmp_path, "long", clips={"long": {"count": 1, "min_seconds": 20, "max_seconds": 30}})
    assert all(m.end - m.start <= 13 for m in short) and all(m.end - m.start >= 19 for m in long)  # padding adds a fraction of a second
    assert len(long) == 1 and long[0].scene == "best"


def test_select_moments_audio_only_and_silent_video(media: Media, pipeline: Any, tmp_path: Path) -> None:
    sections: dict[str, Any] = {"clips": {"short": {"count": 2, "min_seconds": 8, "max_seconds": 10}}}
    picks = select(media, pipeline, tmp_path, have_text=False, **sections)
    assert len(picks) == 2 and all(m.text == "" and m.i == -1 for m in picks)
    assert all(8 * 0.6 <= m.end - m.start <= 10 * 1.4 for m in picks)
    store, cache, _, _ = pipeline
    silent = probe(media.silent)
    quiet = select_moments(
        media.silent,
        silent,
        store,
        cache,
        make_cfg(media, tmp_path, None, **sections),
        "short",
        energy_track(media.silent, silent, cache),
        False,
    )
    assert len(quiet) == 2  # no audio: flat energy, windows are still produced


def test_audio_snap_falls_back_when_the_window_gets_too_short(
    media: Media, pipeline: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(moments, "snap_to_quiet", lambda _db, t, _r, _total: 0.0 if t < 20 else 1.0)
    (pick_,) = select(media, pipeline, tmp_path, have_text=False, clips={"short": {"count": 1, "min_seconds": 8, "max_seconds": 10}})
    assert pick_.end - pick_.start >= 8  # the bogus snap result was rejected and the original window kept


def test_no_candidates_means_no_moments(media: Media, pipeline: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    too_long: dict[str, Any] = {"clips": {"short": {"min_seconds": 39.5, "max_seconds": 40}}}
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        picks = select(media, pipeline, tmp_path, **too_long)
    assert "falling back to audio-only" in caplog.text and len(picks) >= 1  # a 39.5-40 s audio window still fits the 40 s video
    store, cache, _, db = pipeline
    tiny = MediaInfo(5.0, True, 320, 180, 15.0)
    assert select_moments(media.video, tiny, store, cache, make_cfg(media, tmp_path, None, **too_long), "short", db, False) == []


def test_snap_to_shot_changes(media: Media, pipeline: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(moments, "scene_cuts", lambda _src: [8.0, 21.0])
    picks = select(media, pipeline, tmp_path, clips={"snap_to_cuts": True, "pad_start": 0.0, "pad_end": 0.0, "short": {"count": 3}})
    assert any(m.start in (8.0, 21.0) or m.end in (8.0, 21.0) for m in picks)


def test_padding_never_pushes_a_moment_over_the_maximum(media: Media, pipeline: Any, tmp_path: Path) -> None:
    """``pad_start``/``pad_end`` add time around a window: the result still respects ``max_seconds``."""
    kind: dict[str, Any] = {"min_seconds": 6, "max_seconds": 12}
    picks = select(media, pipeline, tmp_path, clips={"pad_start": 4, "pad_end": 4, "short": {"count": 2, **kind}})
    assert len(picks) == 2 and all(m.end - m.start <= 12 for m in picks)
    assert any(m.end - m.start == pytest.approx(12.0) for m in picks)  # padded to 14+ s, then clamped


def test_llm_reranks_every_scene(
    media: Media, pipeline: Any, tmp_path: Path, llm_server: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    llm_server["reply"] = {
        "message": {
            "content": json.dumps({"scores": [{"id": 0, "score": 1, "title": "Bad"}, {"id": 1, "score": 10, "title": "Great one"}]})
        }
    }
    llm = {"enabled": True, "base_url": llm_server["url"], "top_k": 4}
    picks = select(media, pipeline, tmp_path, clips={"scenes": ["funny", "sad"], "short": {"count": 1}}, llm=llm)
    assert len(llm_server["requests"]) == 2 and {m.scene for m in picks} == {"funny", "sad"}  # one request per scene type
    assert all(m.title == "Great one" and m.llm_score == 10 for m in picks)
    broken = {"enabled": True, "base_url": "http://127.0.0.1:9", "timeout": 1}
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        fallback = select(media, pipeline, tmp_path, llm=broken)
    assert "LLM re-rank failed" in caplog.text and len(fallback) == 2 and fallback[0].llm_score is None
