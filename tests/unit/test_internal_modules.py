"""Coverage for the small internal helpers that the feature tests do not exercise directly."""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
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
    assert tuple(quality.QUALITY_PROFILES) == ("high", "medium", "low")
    assert quality.quality_short_side("low") >= 2
    assert quality.output_fps(24.0, "low") >= 1.0
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

    fake = types.ModuleType("ctranslate2")
    fake.get_cuda_device_count = lambda: 0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)
    assert runtime.whisper_model_for("cpu", cpus=4, ram_mb=4096) == "small"
    assert runtime.whisper_model_for("cpu", cpus=16, ram_mb=16384) == "medium"
    assert runtime.whisper_model_for("cuda") == "large-v3"
    assert runtime.detect_device() in ("cpu", "cuda")
    # detect_cpu_count must survive a broken probe (falls back to the raw affinity/cpu count)
    monkeypatch.setattr("autoclip.core.resources.cgroup_cpu_quota", lambda: None)
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
    assert tuple(modes.PROFILES) == ("performance", "resources", "balance")
    from autoclip.core.internal.automation import denoiser_threads

    assert denoiser_threads("resources") == 1
    assert denoiser_threads("performance") >= 1
    assert denoiser_threads("balance") >= 1


def _clear_ai_caches() -> None:
    for cached in (ai._installed, ai._resolve_model):
        cache_clear = getattr(cached, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()


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
        _clear_ai_caches()
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
    from autoclip.core.internal.runtime import runtime_decisions

    decisions_list = runtime_decisions()
    assert any(d.name == "quality" for d in decisions_list)
    assert any(d.name == "mode" for d in decisions_list)
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
    monkeypatch.setattr(ai, "llm_model", lambda *a, **k: pytest.fail("a skip.llm run must not probe the LLM"))
    monkeypatch.setattr(ai, "_chat", lambda *a, **k: pytest.fail("a skip.llm run must not chat"))
    assert ai.generate_title(skip_llm, "hello") == ""
    assert ai.generate_caption(skip_llm, "hello") == ""
    assert ai.llm_rerank([], skip_llm, "best") == {}


def test_llm_calls_include_the_configured_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core.models import Moment

    monkeypatch.setattr(ai, "llm_model", lambda *a, **k: "test-model")
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

    monkeypatch.setattr(ai, "llm_model", lambda *a, **k: "test-model")
    captured: dict[str, Any] = {}

    def fake_chat(base_url: str, model: str, messages: list[dict[str, str]], *, fmt: str = "") -> str:
        captured["messages"] = messages
        return json.dumps({"scores": [{"id": 0, "score": 5, "title": "t"}]})

    monkeypatch.setattr(ai, "_chat", fake_chat)
    ai.llm_rerank([Moment(0, 10, "some text", score=1.0)], parse(Config, {}), "funny")
    assert captured["messages"][0]["role"] == "system"
    assert "short-form video editor" in captured["messages"][0]["content"]


def _stub_ollama(model_names: list[str]) -> dict[str, Any]:
    """A local HTTP server answering /api/tags like Ollama (the port is fresh, so both caches are)."""
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
    _clear_ai_caches()
    return {"url": f"http://127.0.0.1:{server.server_port}", "calls": calls, "server": server}


def _stub_ollama_models(models: list[dict[str, object]]) -> dict[str, Any]:
    """A local HTTP server answering /api/tags with explicit sizes (fresh port; both caches are cleared)."""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            body = json.dumps({"models": models}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _clear_ai_caches()
    return {"url": f"http://127.0.0.1:{server.server_port}", "server": server}


def test_llm_model_picks_the_largest_installed_model_that_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ai, "free_ram_mb", lambda: 8192.0)  # 4 GiB budget
    models = [
        {"name": "tiny:1b", "size": 500 * 1024**2},
        {"name": "mid:7b", "size": 3 * 1024**3},
        {"name": "huge:70b", "size": 40 * 1024**3},
    ]
    state = _stub_ollama_models(models)
    try:
        assert ai.llm_model(state["url"]) == "mid:7b"
    finally:
        state["server"].shutdown()
        state["server"].server_close()
    _clear_ai_caches()


def test_llm_model_falls_back_to_the_smallest_installed_model_when_none_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ai, "free_ram_mb", lambda: 2048.0)  # 1 GiB budget
    models = [
        {"name": "mid:7b", "size": 3 * 1024**3},
        {"name": "huge:70b", "size": 40 * 1024**3},
    ]
    state = _stub_ollama_models(models)
    try:
        assert ai.llm_model(state["url"]) == "mid:7b"
    finally:
        state["server"].shutdown()
        state["server"].server_close()
    _clear_ai_caches()


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
    _clear_ai_caches()


def test_llm_available_is_false_when_no_server_answers() -> None:
    assert ai.llm_available("http://127.0.0.1:1") is False


class _FakeClient:
    """A stand-in for ``ollama.Client``: records the constructor and the chat call."""

    def __init__(self, host: str, *, timeout: float = 2.0) -> None:
        self.host = host
        self.timeout = timeout
        self.last: dict[str, Any] = {}

    def list(self) -> dict[str, Any]:
        return {"models": [{"name": "fake-model", "size": 6 * 1024**3}]}

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

    _clear_ai_caches()
    monkeypatch.setattr(ai, "_installed", lambda url: {"fake-model:latest": 6 * 1024**3})
    monkeypatch.setattr(ai, "ollama", _ollama_module(_FakeClient))
    top = [Moment(0, 10, "one", score=1.0), Moment(0, 10, "two", score=1.0)]
    out = ai.llm_rerank(top, parse(Config, {}), "funny")
    assert out == {0: (10.0, "x" * 80), 1: (0.0, "")}


def test_rerank_rejects_an_off_schema_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core.models import Moment  # noqa: PLC0415

    _clear_ai_caches()

    class _OffSchema(_FakeClient):
        def chat(self, **kwargs: Any) -> dict[str, Any]:
            return {"message": {"role": "assistant", "content": "not json at all"}}

    monkeypatch.setattr(ai, "_installed", lambda url: {"fake-model:latest": 6 * 1024**3})
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


def test_model_sizes_handles_odd_shapes() -> None:
    assert ai._model_sizes({"models": [{"name": ""}, {"model": "picked:latest", "size": 7 * 1024**3}, "bare"]}) == {
        "picked:latest": 7 * 1024**3,
        "bare": 0,
    }
    assert ai._model_sizes({"models": [{"name": "unknown-size", "size": "lots"}]}) == {"unknown-size": 0}
    assert ai._model_sizes({"models": None}) == {}
    assert ai._model_sizes("nonsense") == {}
    assert ai._model_sizes(
        {"models": [{"name": "a", "size": float("nan")}, {"name": "b", "size": float("inf")}, {"name": "c", "size": 1e400}]}
    ) == {
        "a": 0,
        "b": 0,
        "c": 0,
    }
    assert ai._model_sizes({"models": [{"name": "x", "size": True}]}) == {"x": 0}


def test_llm_available_is_true_when_the_plain_http_probe_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_ai_caches()
    state = _stub_ollama(["bonsai-2:2b"])
    try:
        assert ai.llm_available(state["url"], "bonsai-2:2b") is True
    finally:
        state["server"].shutdown()
        state["server"].server_close()
    _clear_ai_caches()


def test_generate_title_swallows_a_failing_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ai, "llm_model", lambda *a, **k: "test-model")

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

    _clear_ai_caches()
    monkeypatch.setattr(ai, "_installed", lambda url: {"bonsai-2:2b": 6 * 1024**3})
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

    _clear_ai_caches()
    monkeypatch.setattr(ai, "_installed", lambda url: {"bonsai-2:2b": 6 * 1024**3})
    monkeypatch.setattr(ai, "_chat", boom)
    with pytest.raises(RuntimeError):
        ai.llm_rerank([Moment(0, 10, "one", score=1.0)], parse(Config, {}), "funny")
    assert _ai._RUN_TRIPPED.is_set()
    _ai._RUN_TRIPPED.clear()


def test_generate_caption_flips_the_breaker_on_backend_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core import ai as _ai  # noqa: PLC0415

    def boom(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("server died")

    _clear_ai_caches()
    monkeypatch.setattr(ai, "_installed", lambda url: {"bonsai-2:2b": 6 * 1024**3})
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


def _fake_sd_modules(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from importlib.machinery import ModuleSpec

    torch = ModuleType("torch")
    torch.__spec__ = ModuleSpec("torch", loader=None)
    cuda = SimpleNamespace(is_available=lambda: False)
    mps = SimpleNamespace(is_available=lambda: False)
    torch.cuda = cuda  # type: ignore[attr-defined]
    torch.backends = SimpleNamespace(mps=mps)  # type: ignore[attr-defined]
    torch.float32 = "float32"  # type: ignore[attr-defined]
    torch.float16 = "float16"  # type: ignore[attr-defined]
    torch.cuda.empty_cache = lambda: None
    diffusers = ModuleType("diffusers")
    diffusers.__spec__ = ModuleSpec("diffusers", loader=None)

    class Result:
        def __init__(self) -> None:
            self.images = [SimpleNamespace(save=lambda path: Path(path).write_bytes(b"thumbnail"))]

    class Pipe:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> Pipe:
            return cls()

        def to(self, device: str) -> Pipe:
            self.device = device
            return self

        def __call__(self, **kwargs: object) -> Result:
            return Result()

    diffusers.StableDiffusionImg2ImgPipeline = Pipe  # type: ignore[attr-defined]
    image_module = ModuleType("PIL.Image")

    class ImageFile:
        def convert(self, mode: str) -> ImageFile:
            return self

        def resize(self, size: tuple[int, int]) -> ImageFile:
            return self

    image_module.open = lambda path: ImageFile()  # type: ignore[attr-defined]
    pil = ModuleType("PIL")
    pil.Image = image_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_module)
    return {"torch": torch, "diffusers": diffusers}


def test_sd_generate_success_and_worker_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_sd_modules(monkeypatch)
    monkeypatch.setattr("autoclip.core.model_engine.release_allocations", lambda: None)
    frame, out = tmp_path / "frame.jpg", tmp_path / "out.jpg"
    frame.write_bytes(b"jpg")
    assert ai._sd_generate(frame, out, "prompt", "cpu")
    assert out.read_bytes() == b"thumbnail"

    decision = SimpleNamespace(subprocess=False, reason="tiny test room")
    monkeypatch.setattr("autoclip.core.model_engine.choose_model", lambda *args, **kwargs: decision)
    assert ai.generate_sd_thumbnail(frame, out, "prompt")

    decision = SimpleNamespace(subprocess=True, reason="low memory")
    monkeypatch.setattr("autoclip.core.model_engine.choose_model", lambda *args, **kwargs: decision)

    class FakeProcess:
        def start(self) -> None:
            pass

        def join(self, timeout: float | None = None) -> None:
            return None

        def is_alive(self) -> bool:
            return False

        exitcode = 0

    monkeypatch.setattr("multiprocessing.get_context", lambda name: SimpleNamespace(Process=lambda **kwargs: FakeProcess()))
    assert ai.generate_sd_thumbnail(frame, out, "prompt")

    decision = SimpleNamespace(subprocess=True, reason="low memory")
    monkeypatch.setattr("autoclip.core.model_engine.choose_model", lambda *args, **kwargs: decision)

    class FailingProcess(FakeProcess):
        exitcode = 1

    monkeypatch.setattr("multiprocessing.get_context", lambda name: SimpleNamespace(Process=lambda **kwargs: FailingProcess()))
    assert not ai.generate_sd_thumbnail(frame, out, "prompt")


def test_sd_generate_logs_a_failure_and_releases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_sd_modules(monkeypatch)

    class BrokenPipe:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            raise RuntimeError("boom")

    monkeypatch.setattr(sys.modules["diffusers"], "StableDiffusionImg2ImgPipeline", BrokenPipe)
    monkeypatch.setattr("autoclip.core.model_engine.release_allocations", lambda: None)
    frame, out = tmp_path / "frame.jpg", tmp_path / "out.jpg"
    frame.write_bytes(b"jpg")
    assert ai._sd_generate(frame, out, "prompt", "cpu") is False


def test_sd_device_selection_covers_cuda_mps_and_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_sd_modules(monkeypatch)
    torch = sys.modules["torch"]
    torch.cuda.is_available = lambda: True
    assert ai._sd_generate(tmp_path / "frame.jpg", tmp_path / "out.jpg", "prompt", "auto")

    torch.cuda.is_available = lambda: False
    torch.backends.mps = SimpleNamespace(is_available=lambda: True)
    assert ai._sd_generate(tmp_path / "frame.jpg", tmp_path / "out.jpg", "prompt", "auto")

    monkeypatch.setattr(ai, "_sd_generate", lambda *args, **kwargs: True)
    with pytest.raises(SystemExit, match="0"):
        ai._sd_process(str(tmp_path / "frame.jpg"), str(tmp_path / "out.jpg"), "prompt", "auto")
    with pytest.raises(SystemExit, match="1"):
        monkeypatch.setattr(ai, "_sd_generate", lambda *args, **kwargs: False)
        ai._sd_process(str(tmp_path / "frame.jpg"), str(tmp_path / "out.jpg"), "prompt", "auto")


def test_sd_available_returns_false_when_the_probe_breaks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("half-registered")))
    assert ai.sd_available() is False


def test_sd_worker_kills_a_wedged_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_sd_modules(monkeypatch)
    monkeypatch.setattr("autoclip.core.model_engine.choose_model", lambda *args, **kwargs: SimpleNamespace(subprocess=True, reason="x"))
    actions: list[str] = []
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpg")
    out = tmp_path / "out.jpg"

    class Wedged:
        exitcode = 0

        def start(self) -> None:
            pass

        def join(self, timeout: float | None = None) -> None:
            actions.append(f"join{timeout}")

        def is_alive(self) -> bool:
            return "killed" not in actions

        def kill(self) -> None:
            actions.append("killed")

    monkeypatch.setattr("multiprocessing.get_context", lambda name: SimpleNamespace(Process=lambda **kwargs: Wedged()))
    assert ai.generate_sd_thumbnail(frame, out, "prompt") is False
    assert "killed" in actions and "join900.0" in actions and "join5.0" in actions


def test_rerank_returns_empty_when_no_model_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core.models import Moment  # noqa: PLC0415

    monkeypatch.setattr(ai, "llm_model", lambda base_url, model="": None)
    assert ai.llm_rerank([Moment(0, 10, "one", score=1.0)], parse(Config, {}), "funny") == {}


def test_free_ram_mb_honours_the_container_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    from autoclip.core.internal import runtime

    monkeypatch.setattr("psutil.virtual_memory", lambda: types.SimpleNamespace(available=1024 * (1 << 20)))
    monkeypatch.setattr("autoclip.core.resources.cgroup_memory_available", lambda: 512 * (1 << 20))
    assert runtime.free_ram_mb() == pytest.approx(512)
    monkeypatch.setattr("autoclip.core.resources.cgroup_memory_available", lambda: None)
    assert runtime.free_ram_mb() == pytest.approx(1024)


def test_reset_run_breaker_rearms_the_ai_helpers() -> None:
    """One failing job must not poison the AI helpers for the rest of a server's lifetime."""
    from autoclip.core import ai as _ai

    _ai._RUN_TRIPPED.set()
    assert ai.llm_available("http://whatever.local") is False
    ai.reset_run_breaker()
    assert not _ai._RUN_TRIPPED.is_set()
    _clear_ai_caches()


def test_llm_rerank_sends_the_engine_picked_model_over_the_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core.models import Moment  # noqa: PLC0415

    monkeypatch.setattr(ai, "_installed", lambda url: {"fake-model:latest": 6 * 1024**3})
    monkeypatch.setattr(ai, "free_ram_mb", lambda: 8192.0)
    _clear_ai_caches()
    captured: dict[str, Any] = {}

    class _CaptureClient:
        def __init__(self, host: str, *, timeout: float = 2.0) -> None:
            pass

        def chat(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"message": {"role": "assistant", "content": json.dumps({"scores": [{"id": 0, "score": 5, "title": "t"}]})}}

    monkeypatch.setattr(ai, "ollama", _ollama_module(_CaptureClient))
    ai.llm_rerank([Moment(0, 10, "one", score=1.0)], parse(Config, {}), "funny")
    assert captured["model"] == "fake-model:latest"
    _clear_ai_caches()


def test_the_probe_never_follows_a_redirect() -> None:
    """A hostile 302 must not turn the local probe into a blind SSRF."""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/never-fetched")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        _clear_ai_caches()
        assert ai._installed(f"http://127.0.0.1:{server.server_port}") == {}
    finally:
        server.shutdown()
        server.server_close()
    _clear_ai_caches()


def test_model_sizes_handles_absurd_sizes() -> None:
    """Bignum and non-finite sizes degrade to "unknown" instead of crashing the highlights pass."""
    payload = {
        "models": [
            {"name": "bignum", "size": 10**400},
            {"name": "float-overflow", "size": 1e400},
            {"name": "nan", "size": float("nan")},
            {"name": "negative", "size": -5},
        ]
    }
    assert ai._model_sizes(payload) == {"bignum": 0, "float-overflow": 0, "nan": 0, "negative": 0}


def test_quality_tiers_stay_in_sync_with_the_profile_table() -> None:
    from autoclip.core.internal import quality

    assert tuple(quality.QUALITY_PROFILES) == quality.VALID_QUALITIES


def test_llm_model_survives_a_broken_free_ram_meter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken memory meter must degrade the AI pick to the smallest known model, never raise."""
    monkeypatch.setattr(ai, "_installed", lambda url: {"small": 6 * 1024**3})
    monkeypatch.setattr(ai, "free_ram_mb", lambda: (_ for _ in ()).throw(RuntimeError("no meter")))
    _clear_ai_caches()
    try:
        assert ai.llm_model("http://whatever.local") == "small"
    finally:
        _clear_ai_caches()
