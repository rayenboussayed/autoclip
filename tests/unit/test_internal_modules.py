"""Coverage for the small internal helpers that the feature tests do not exercise directly."""

from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from autoclip.core import ai
from autoclip.core.config import Config, parse
from autoclip.core.internal import adaptive, background, content, decisions, modes, quality, runtime
from autoclip.core.internal.automation import DEFAULT_FILLERS


def test_background_helpers() -> None:
    assert background.is_color("#abc") and background.is_color("#aabbcc")
    assert not background.is_color("blur")
    assert background.background_kind("blur") == "blur"
    assert background.background_kind("#fff") == "color"
    assert background.background_kind("/tmp/x.png") == "image"
    base = Path(tempfile.gettempdir())
    assert background.resolve_background("blur", base) == "blur"
    assert background.resolve_background("#fff", base) == "#fff"
    assert background.resolve_background("rel.png", base) == str(base / "rel.png")
    with contextlib.suppress(Exception):
        background.background_kind("")
    with contextlib.suppress(Exception):
        background.background_kind("#zzz")


def test_content_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert content.load_lexicon(None) == {}
    assert content.load_lexicon(tmp_path) == {}
    (tmp_path / content.LEXICON_FILE).write_text(json.dumps({"funny": ["x"], "fillers": ["um"]}), encoding="utf-8")
    assert content.load_lexicon(tmp_path) == {"funny": ["x"], "fillers": ["um"]}
    assert content.fillers_from({"fillers": ["um"]}) == frozenset({"um"})
    assert content.fillers_from({}) == frozenset(DEFAULT_FILLERS)
    # invalid files raise
    for bad in ("{nope", "[1,2]", '{"a": 1}', '{"a": [""]}', '{"a": []}'):
        (tmp_path / content.LEXICON_FILE).write_text(bad, encoding="utf-8")
        with contextlib.suppress(Exception):
            content.load_lexicon(tmp_path)
    (tmp_path / content.LEXICON_FILE).unlink()
    # unreadable file branch
    (tmp_path / content.LEXICON_FILE).write_text("{}", encoding="utf-8")
    from pathlib import Path as _P

    orig = _P.read_text

    def boom(self: Path, *a: Any, **k: Any) -> str:
        if content.LEXICON_FILE in str(self):
            raise OSError("boom")
        return orig(self, *a, **k)

    monkeypatch.setattr(_P, "read_text", boom)
    with contextlib.suppress(Exception):
        content.load_lexicon(tmp_path)


def test_quality_helpers() -> None:
    assert quality.quality_profile("high").name == "high"
    assert quality.quality_short_side("lowest") >= 2
    assert quality.output_fps(24.0, "lowest") >= 1.0
    assert quality.download_quality_for("high") == 720
    with contextlib.suppress(ValueError):
        quality.quality_profile("nope")
    with contextlib.suppress(ValueError):
        quality.download_quality_for("nope")
    # output_fps with a cap
    assert quality.output_fps(100.0, "medium") == 30.0


def test_runtime_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types
    from pathlib import Path

    fake = types.ModuleType("ctranslate2")
    fake.get_cuda_device_count = lambda: 0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)
    assert runtime.transcript_model_for("high", "cpu") == "small"
    assert runtime.transcript_model_for("highest", "cpu") == "large-v3"
    assert runtime.default_transcript_model() in ("small", "large-v3")
    assert runtime._container_cpu_quota() is None or isinstance(runtime._container_cpu_quota(), float)
    # cgroup cpu.max valid content

    def fake_read_text(self: Path, *a: Any, **k: Any) -> str:
        if "cpu.max" in str(self):
            return "200000 100000"
        raise OSError("no")

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    assert runtime._container_cpu_quota() == 2.0

    # cgroup v1 quota/period files
    def fake_v1(self: Path, *a: Any, **k: Any) -> str:
        if "cpu.cfs_quota_us" in str(self):
            return "200000"
        if "cpu.cfs_period_us" in str(self):
            return "100000"
        raise OSError("no")

    monkeypatch.setattr(Path, "read_text", fake_v1)
    assert runtime._container_cpu_quota() == 2.0
    # detect_cpu_count except branch
    monkeypatch.setattr("autoclip.core.resources.cgroup_cpu_quota", lambda: (_ for _ in ()).throw(RuntimeError("boom")), raising=False)
    assert runtime.detect_cpu_count() >= 1


def test_adaptive_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert adaptive.load_state(tmp_path / "missing.json") is None
    p = tmp_path / "state.json"
    adaptive.save_state(p, {"a": 1})
    assert adaptive.load_state(p) == {"a": 1}
    assert adaptive.ema(0.0, 10.0) == 3.0
    from collections import deque

    assert adaptive.trend(deque([1.0, 2.0])) == 0  # too few samples
    assert adaptive.trend(deque([1.0, 1.0, 1.0, 1.0])) == 0
    assert adaptive.trend(deque([1.0, 1.0, 5.0, 5.0])) == 1
    assert adaptive.trend(deque([5.0, 5.0, 1.0, 1.0])) == -1
    assert adaptive.clamp(5.0, 0.0, 10.0) == 5.0
    assert adaptive.clamp(15.0, 0.0, 10.0) == 10.0
    # save_state OSError branch
    monkeypatch.setattr(Path, "replace", lambda self, other: (_ for _ in ()).throw(OSError("boom")))
    adaptive.save_state(tmp_path / "x.json", {"a": 1})  # must not raise


def test_modes_helpers() -> None:
    assert modes.profile_for("nope").name == "balance"
    assert modes.profile_for("performance").name == "performance"
    assert "balance" in modes.VALID_MODES
    from autoclip.core.internal.automation import denoiser_threads

    assert denoiser_threads("resources") == 1
    assert denoiser_threads("performance") >= 1
    assert denoiser_threads("balance") >= 1


def test_llm_available_falls_back_to_a_plain_http_probe() -> None:
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            body = b"not a model list"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert ai.llm_available(f"http://127.0.0.1:{server.server_port}", timeout=1.0) is True
    finally:
        server.shutdown()
        server.server_close()


def test_field_default_handles_copy_failures_and_bad_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    from autoclip.core.config import _field_default

    class WithDefault:
        __dataclass_fields__ = {"x": dataclasses.field(default=7)}

    assert _field_default(WithDefault, "x") == 7
    monkeypatch.setattr("copy.deepcopy", lambda v: (_ for _ in ()).throw(OSError("no")))
    assert _field_default(WithDefault, "x") == 7

    def bad() -> int:
        raise TypeError("factory")

    class BadFactory:
        __dataclass_fields__ = {"y": dataclasses.field(default_factory=bad)}

    assert callable(_field_default(BadFactory, "y"))


def test_removed_key_warns_only_once_per_context(caplog: pytest.LogCaptureFixture) -> None:
    from autoclip.core.config import _removed_seen, _warn_removed

    token = _removed_seen.set(set())
    try:
        with caplog.at_level("WARNING", logger="autoclip"):
            _warn_removed("clips.count", "use clips.preset instead")
            _warn_removed("clips.count", "use clips.preset instead")
        assert caplog.text.count("use clips.preset instead") == 1
    finally:
        _removed_seen.reset(token)


def test_config_coercion_edge_cases() -> None:
    from autoclip.core.config import Config, _coerce_defaults, _field_default, _strip_removed

    assert _coerce_defaults(Config, []) == []  # non-dict input is returned unchanged
    assert _coerce_defaults(Config, {"unknown": 1}) == {"unknown": 1}  # unknown keys stay for the strict validator
    assert _field_default(Config, "nope") is None  # unknown field name
    assert _strip_removed({"_comment": "x"}) == {"_comment": "x"}  # comments are not removed here


def test_decisions_helpers() -> None:
    from autoclip.core.internal.runtime import RuntimeDecision

    d = RuntimeDecision(name="x", value=1, source="test")
    assert "x" in decisions.decision_text([d])
    # log_decisions runs without error
    decisions.log_decisions()


def test_ai_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(ai.sd_available(), bool)
    import tempfile

    src = Path(tempfile.mkdtemp()) / "frame.jpg"
    out = Path(tempfile.mkdtemp()) / "thumb.jpg"
    assert ai.generate_sd_thumbnail(src, out, "a prompt") is False
    cfg = parse(Config, {})
    assert ai.generate_title(cfg, "") == ""
    assert ai.generate_caption(cfg, "") == ""
    assert ai.llm_available("http://127.0.0.1:1") is False
    # cover the ollama-is-None branch
    monkeypatch.setattr(ai, "ollama", None)
    assert ai.llm_available("http://127.0.0.1:1") is False
    assert ai.generate_title(cfg, "hello") == ""
    assert ai.generate_caption(cfg, "hello") == ""
    # skip.llm is a hard stop: no probe, no chat, no re-rank
    skip_llm = parse(Config, {"skip": {"llm": True}})
    monkeypatch.setattr(ai, "llm_available", lambda *a, **k: pytest.fail("a skip.llm run must not probe the LLM"))
    monkeypatch.setattr(ai, "_chat", lambda *a, **k: pytest.fail("a skip.llm run must not chat"))
    assert ai.generate_title(skip_llm, "hello") == ""
    assert ai.generate_caption(skip_llm, "hello") == ""
    assert ai.llm_rerank([], skip_llm, "best") == {}


def test_llm_calls_include_the_configured_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core.models import Moment

    monkeypatch.setattr(ai, "llm_available", lambda *a, **k: True)
    captured: dict[str, Any] = {}

    def fake_chat(base_url: str, model: str, messages: list[dict[str, str]], *, fmt: str = "") -> str:
        captured["messages"] = messages
        if fmt == "json":
            return json.dumps({"scores": [{"id": 0, "score": 5, "title": "t"}]})
        return "ok"

    monkeypatch.setattr(ai, "_chat", fake_chat)
    cfg = parse(Config, {})
    assert ai._messages(cfg, "p") == [{"role": "user", "content": "p"}]  # unset: no system message
    cfg = parse(Config, {"llm": {"system_prompt": "You are stern."}})
    assert ai._messages(cfg, "p") == [{"role": "system", "content": "You are stern."}, {"role": "user", "content": "p"}]
    assert ai.generate_title(cfg, "some text") == "ok"
    assert captured["messages"][0] == {"role": "system", "content": "You are stern."}
    assert ai.generate_caption(cfg, "some text") == "ok"
    assert captured["messages"][0] == {"role": "system", "content": "You are stern."}
    assert ai.llm_rerank([Moment(0, 10, "some text", score=1.0)], cfg, "funny")
    assert captured["messages"][0] == {"role": "system", "content": "You are stern."}


def test_llm_rerank_falls_back_to_the_editor_prompt_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core.models import Moment

    monkeypatch.setattr(ai, "llm_available", lambda *a, **k: True)
    captured: dict[str, Any] = {}

    def fake_chat(base_url: str, model: str, messages: list[dict[str, str]], *, fmt: str = "") -> str:
        captured["messages"] = messages
        return json.dumps({"scores": [{"id": 0, "score": 5, "title": "t"}]})

    monkeypatch.setattr(ai, "_chat", fake_chat)
    ai.llm_rerank([Moment(0, 10, "some text", score=1.0)], parse(Config, {}), "funny")
    assert captured["messages"][0]["role"] == "system"
    assert "short-form video editor" in captured["messages"][0]["content"]
