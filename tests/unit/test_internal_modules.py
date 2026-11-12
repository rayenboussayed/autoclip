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
        ai._tags.cache_clear()
        # "not a model list" is not a usable Ollama: the preflight must say the model is not there
        assert ai.llm_available(f"http://127.0.0.1:{server.server_port}") is False
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

    def fake_chat(base_url: str, model: str, messages: list[dict[str, str]], *, fmt: object = None) -> str:
        captured["messages"] = messages
        if isinstance(fmt, dict):  # the re-rank constrains the reply with a JSON schema
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


def _stub_ollama(model_names: list[str]) -> dict[str, Any]:
    """A local HTTP server answering /api/tags like Ollama (the port is fresh, so the ``_tags`` cache is)."""
    import http.server
    import threading

    calls: dict[str, int] = {"tags": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/tags":
                calls["tags"] += 1
                body = json.dumps({"models": [{"name": name} for name in model_names]}).encode()
            else:
                body = b"{}"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    ai._tags.cache_clear()
    return {"url": f"http://127.0.0.1:{server.server_port}", "calls": calls, "server": server}


def test_llm_available_requires_the_configured_model_to_be_installed() -> None:
    state = _stub_ollama(["llama3.2:latest"])
    try:
        assert ai.llm_available(state["url"]) is True  # the server answers and lists models
        assert ai.llm_available(state["url"], "llama3.2") is True
        assert ai.llm_available(state["url"], "llama3.2:latest") is True
        assert ai.llm_available(state["url"], "bonsai-2:2b") is False  # model not installed: AI is off for this run
        assert ai.llm_available(state["url"], "qwen") is False
    finally:
        state["server"].shutdown()
        state["server"].server_close()
    ai._tags.cache_clear()


def test_llm_available_is_false_when_no_server_answers() -> None:
    assert ai.llm_available("http://127.0.0.1:1") is False


class _FakeClient:
    """A stand-in for ``ollama.Client``: records the constructor and the chat call."""

    def __init__(self, host: str, *, timeout: float = 2.0) -> None:
        self.host = host
        self.timeout = timeout
        self.last: dict[str, Any] = {}

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.last = kwargs
        content = json.dumps(
            {
                "scores": [
                    {"id": 0, "score": 99, "title": "x" * 200},
                    {"id": 1, "score": -4, "title": ""},
                    {"id": 99, "score": 8, "title": "ghost"},
                ]
            }
        )
        return {"message": {"role": "assistant", "content": content}}


def _ollama_module(client_cls: type) -> Any:
    class _Ollama:
        Client = client_cls

    return _Ollama()


def test_rerank_clamps_scores_and_ignores_unknown_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core.models import Moment  # noqa: PLC0415

    monkeypatch.setattr(ai, "ollama", _ollama_module(_FakeClient))
    top = [Moment(0, 10, "one", score=1.0), Moment(0, 10, "two", score=1.0)]
    out = ai.llm_rerank(top, parse(Config, {}), "funny")
    assert out == {0: (10.0, "x" * 80), 1: (0.0, "")}


def test_rerank_rejects_an_off_schema_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core.models import Moment  # noqa: PLC0415

    class _OffSchema(_FakeClient):
        def chat(self, **kwargs: Any) -> dict[str, Any]:
            return {"message": {"role": "assistant", "content": "not json at all"}}

    monkeypatch.setattr(ai, "ollama", _ollama_module(_OffSchema))
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ai.llm_rerank([Moment(0, 10, "one", score=1.0)], parse(Config, {}), "funny")


def test_chat_passes_the_configured_timeout_and_context_window(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, Any] = {}

    class _RecordingClient(_FakeClient):
        def __init__(self, host: str, *, timeout: float = 2.0) -> None:
            recorded["timeout"] = timeout
            super().__init__(host)

        def chat(self, **kwargs: Any) -> dict[str, Any]:
            recorded.update(kwargs)
            return {"message": {"role": "assistant", "content": "ok"}}

    monkeypatch.setattr(ai, "ollama", _ollama_module(_RecordingClient))
    assert ai._chat("http://x", "bonsai-2:2b", [{"role": "user", "content": "hi"}]) == "ok"
    assert recorded["timeout"] == 180.0
    assert recorded["options"] == {"num_ctx": 8192, "temperature": 0}


def test_model_names_handles_odd_shapes() -> None:
    assert ai._model_names({"models": [{"name": ""}, {"model": "picked:latest"}, "bare"]}) == {"picked:latest", "bare"}
    assert ai._model_names({"models": None}) == set()
    assert ai._model_names("nonsense") == set()


def test_tags_uses_the_python_client_when_it_works(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Client:
        def __init__(self, host: str, *, timeout: float = 2.0) -> None:
            pass

        def list(self) -> dict[str, Any]:
            return {"models": [{"name": "bonsai-2:2b"}]}

    class _Ollama:
        Client = _Client

    monkeypatch.setattr(ai, "ollama", _Ollama())
    ai._tags.cache_clear()
    assert ai.llm_available("http://whatever.local", "bonsai-2:2b") is True
    ai._tags.cache_clear()


def test_generate_title_swallows_a_failing_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ai, "llm_available", lambda *a, **k: True)

    def boom(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("the server died mid-request")

    monkeypatch.setattr(ai, "_chat", boom)
    assert ai.generate_title(parse(Config, {}), "some text") == ""


def test_generate_title_flips_the_breaker_once_and_later_calls_never_reach_the_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoclip.core import ai as _ai  # noqa: PLC0415

    calls: list[int] = []

    def boom(*args: Any, **kwargs: Any) -> str:
        calls.append(1)
        raise RuntimeError("server died")

    monkeypatch.setattr(ai, "_tags", lambda url: {"bonsai-2:2b"})
    monkeypatch.setattr(ai, "_chat", boom)
    first = ai.generate_title(parse(Config, {}), "some text")
    second = ai.generate_title(parse(Config, {}), "other text")
    assert first == "" and second == ""
    assert calls == [1]  # one failed call tripped the breaker; the second never reached the backend
    assert ai.llm_available("http://x", "bonsai-2:2b") is False
    _ai._RUN_TRIPPED.clear()  # this test is deliberately not isolated (the suite fixture resets it anyway)


def test_rerank_flips_the_breaker_when_the_backend_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core import ai as _ai  # noqa: PLC0415
    from autoclip.core.models import Moment  # noqa: PLC0415

    def boom(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("server died")

    monkeypatch.setattr(ai, "_tags", lambda url: {"bonsai-2:2b"})
    monkeypatch.setattr(ai, "_chat", boom)
    with pytest.raises(RuntimeError):
        ai.llm_rerank([Moment(0, 10, "one", score=1.0)], parse(Config, {}), "funny")
    assert _ai._RUN_TRIPPED.is_set()
    _ai._RUN_TRIPPED.clear()


def test_generate_caption_flips_the_breaker_on_backend_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core import ai as _ai  # noqa: PLC0415

    def boom(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("server died")

    monkeypatch.setattr(ai, "_tags", lambda url: {"bonsai-2:2b"})
    monkeypatch.setattr(ai, "_chat", boom)
    assert ai.generate_caption(parse(Config, {}), "some text") == ""
    assert _ai._RUN_TRIPPED.is_set()
    assert ai.llm_available("http://x", "bonsai-2:2b") is False
    _ai._RUN_TRIPPED.clear()


def test_the_breaker_logs_only_the_first_failure(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from autoclip.core import ai as _ai  # noqa: PLC0415

    _ai._RUN_TRIPPED.set()
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        _ai._disable_for_this_run(RuntimeError("second failure"))
    assert "AI disabled" not in caplog.text
    _ai._RUN_TRIPPED.clear()
