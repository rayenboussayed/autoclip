"""Optional AI helpers: a local Ollama server and (optionally) Stable Diffusion.

Every call goes through a single process-wide lock so that, even in ``performance`` mode, Ollama
and Stable Diffusion never run concurrently.  When a backend is missing or errors, the caller
keeps its heuristic result - AI is an enhancement, never a requirement.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import urllib.error
import urllib.request
from functools import cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaValue

from .config import Config
from .derive.constants import LLM_NUM_CTX, LLM_TIMEOUT
from .derive.policy import auto_ollama_model
from .derive.probes import free_ram_mb
from .models import Moment

log = logging.getLogger("autoclip")

ollama: Any
try:
    import ollama as _ollama

    ollama = _ollama
except Exception:  # pragma: no cover - optional dependency
    ollama = None

#: One queue for every AI model call in this process (Ollama and Stable Diffusion share it).
_AI_LOCK = threading.Lock()

#: Set when an AI call fails mid-run: every later AI call in this process is skipped and the
#: heuristic result is kept. A hung server would otherwise cost LLM_TIMEOUT per clip.
_RUN_TRIPPED = threading.Event()


#: The local Ollama server is a trusted (but not necessarily well-formed) peer: never follow a redirect off it.
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, _req: Any, _fp: Any, _code: int, _msg: str, _headers: Any, _newurl: str) -> None:
        return None


_probe_opener = urllib.request.build_opener(_NoRedirect)

#: A sane ceiling for the /api/tags body (the real answer is a few dozen model entries).
_MAX_TAGS_BYTES = 4 * 1024 * 1024


def _bytes(raw: object) -> int:
    """Size in bytes, or 0 ("unknown") for anything that is not a finite number >= 0."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0
    try:
        value = float(raw)
    except OverflowError:
        return 0
    if not math.isfinite(value) or value <= 0:
        return 0
    return int(value)


def _model_sizes(payload: object) -> dict[str, int]:
    """The installed models as name -> size in bytes on disk (0 when the server reports no usable size)."""
    models: object = payload
    if isinstance(payload, dict):
        models = payload.get("models") or []
    sizes: dict[str, int] = {}
    if isinstance(models, list):
        for item in models:
            if isinstance(item, dict):
                name = item.get("name") or item.get("model")
                size = _bytes(item.get("size", 0))
            else:
                name, size = str(item), 0
            if isinstance(name, str) and name and len(name) <= 200:
                sizes[name] = size
    return sizes


@cache
def _installed(base_url: str) -> dict[str, int]:
    """The models the server at ``base_url`` has, name -> bytes on disk (empty when it cannot be asked)."""
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with _probe_opener.open(url, timeout=2.0) as response:
            data = json.loads(response.read(_MAX_TAGS_BYTES + 1).decode("utf-8"))
            return _model_sizes(data)
    except Exception:
        return {}


@cache
def _resolve_model(base_url: str, model: str) -> str | None:
    """The resolved model name, or ``None`` when there is no usable one."""
    sizes = _installed(base_url)
    wanted = model.strip().lower()
    if wanted:
        for name in sizes:
            lowered = name.lower()
            if lowered == wanted or lowered.startswith(wanted + ":"):
                return name
        return None
    try:
        free = free_ram_mb()
    except Exception:  # a broken meter must not kill the video; treat it as "no RAM to spare"
        free = 0.0
    return auto_ollama_model(sizes, free)


def llm_model(base_url: str, model: str = "") -> str | None:
    """The model this run will use, or ``None`` when the server or the model is unavailable.

    With an empty ``model`` the engine picks the largest installed model that fits inside half of the
    free RAM, falling back to the smallest known one; an explicit ``model`` must match an installed
    one (exact, or an ``name:tag`` prefix such as ``llama3.2`` for ``llama3.2:latest``). The answer is
    cached per (``base_url``, ``model``), so a whole run asks the server once; once any AI call has
    failed in this run, the answer is ``None`` for the rest of the run (the backend is not retried per clip).
    """
    if _RUN_TRIPPED.is_set():
        return None
    return _resolve_model(base_url, model)


def llm_available(base_url: str, model: str = "") -> bool:
    """True when an Ollama server answers and a usable model exists (see :func:`llm_model`)."""
    return llm_model(base_url, model) is not None


def reset_run_breaker() -> None:
    """Clear the per-run AI state: the failure breaker and the cached model list.

    A CLI run is a fresh process, so neither ever needs clearing there. The HTTP server is one
    long-lived process for many jobs; without a reset, one failed AI call (or one failed probe)
    would disable the LLM helpers for every later job until restart. Called once per job by the
    server; the next AI call re-probes the local server.
    """
    _RUN_TRIPPED.clear()
    for cached in (_installed, _resolve_model):  # monkeypatched stand-ins may lack cache_clear
        cache_clear = getattr(cached, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()


def _chat(base_url: str, model: str, messages: list[dict[str, str]], *, fmt: JsonSchemaValue | None = None) -> str:
    if ollama is None:  # the AI helpers need the `ai` extra; the pipeline keeps its heuristic result without it
        raise RuntimeError("the 'ollama' package is not installed; install the AI extra (pip install autoclip[ai])")
    with _AI_LOCK:
        client = ollama.Client(host=base_url, timeout=LLM_TIMEOUT)
        response = client.chat(model=model, messages=messages, format=fmt, options={"num_ctx": LLM_NUM_CTX, "temperature": 0})
    return str((response.get("message") or {}).get("content", "")).strip()


def _disable_for_this_run(error: BaseException) -> None:
    """Switch AI off for the rest of this run (one log line, no spam)."""
    if not _RUN_TRIPPED.is_set():
        _RUN_TRIPPED.set()
        log.warning("AI disabled for the rest of this run (%s: %s)", type(error).__name__, error)


def llm_rerank(top: list[Moment], cfg: Config, scene: str) -> dict[int, tuple[float, str]]:
    """Ask Ollama for a 0-10 fit score and a title for each candidate."""
    from .styles import STYLE_HELP

    if cfg.skip.llm:
        return {}
    lc = cfg.llm
    model = llm_model(lc.base_url, lc.model)
    if model is None:
        return {}
    items = "\n".join(f"[{k}] ({m.end - m.start:.0f}s) {m.text[:700]}" for k, m in enumerate(top))
    prompt = (
        f"You pick moments from a long video for short-form clips. Target scene type -> {scene}: {STYLE_HELP[scene]}.\n"
        "Rate each candidate 0-10 for how well it fits AND stands alone (clear hook, complete thought). "
        "Also write a punchy title (max 8 words) in the SAME language as the candidate text.\n"
        'Return ONLY JSON: {"scores":[{"id":0,"score":7,"title":"..."}]}\n\n'
        f"Candidates:\n{items}"
    )
    system = lc.system_prompt or "You are a precise short-form video editor. Reply with JSON only."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    try:
        content = _chat(lc.base_url, model, messages, fmt=_RERANK_SCHEMA)
        parsed = _RerankReply.model_validate_json(content)  # pydantic raises on any deviation; the caller falls back
    except Exception as e:
        _disable_for_this_run(e)
        raise
    out: dict[int, tuple[float, str]] = {}
    for item in parsed.scores:
        if 0 <= item.id < len(top):  # an id outside the candidate list is ignored
            out[item.id] = (max(0.0, min(10.0, item.score)), item.title[:80])
    return out


def generate_title(cfg: Config, text: str) -> str:
    """A punchy title for ``text`` (empty when Ollama is unavailable, offline or errors)."""
    if cfg.skip.llm or not text:
        return ""
    model = llm_model(cfg.llm.base_url, cfg.llm.model)
    if model is None:
        return ""
    prompt = "Write a punchy title (max 8 words) in the same language as the text. Return only the title.\n\n" + text[:500]
    messages = _messages(cfg, prompt)
    try:
        return _chat(cfg.llm.base_url, model, messages)[:80]
    except Exception as e:
        _disable_for_this_run(e)
        log.warning("AI title failed (%s: %s)", type(e).__name__, e)
        return ""


def generate_caption(cfg: Config, text: str) -> str:
    """A short social-media caption for ``text`` (empty when Ollama is unavailable, offline or errors)."""
    if cfg.skip.llm or not text:
        return ""
    model = llm_model(cfg.llm.base_url, cfg.llm.model)
    if model is None:
        return ""
    prompt = "Write a short social-media caption (max 2 sentences) for this clip transcript. Return only the caption.\n\n" + text[:500]
    messages = _messages(cfg, prompt)
    try:
        return _chat(cfg.llm.base_url, model, messages)
    except Exception as e:
        _disable_for_this_run(e)
        log.warning("AI caption failed (%s: %s)", type(e).__name__, e)
        return ""


class _RerankItem(BaseModel):
    """One scored candidate of the re-rank request."""

    id: int
    score: float
    title: str = ""


class _RerankReply(BaseModel):
    """The shape the local LLM is constrained to answer for a re-rank."""

    scores: list[_RerankItem]


_RERANK_SCHEMA: JsonSchemaValue = _RerankReply.model_json_schema()


def _messages(cfg: Config, prompt: str) -> list[dict[str, str]]:
    """The chat messages for a prompt: the file's system prompt first (when set), then the user prompt."""
    if cfg.llm.system_prompt:
        return [{"role": "system", "content": cfg.llm.system_prompt}, {"role": "user", "content": prompt}]
    return [{"role": "user", "content": prompt}]


def sd_available() -> bool:
    """Cheap availability probe: does not import torch into the parent process."""
    import importlib.util

    try:
        return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("diffusers") is not None
    except (ValueError, ModuleNotFoundError, ImportError):  # a half-registered package must not abort the run
        return False


def _sd_generate(source_frame: Path, out_path: Path, prompt: str, device: str) -> bool:
    """Run the SD pipeline and release it before returning (native allocations are best-effort)."""
    try:
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline
        from PIL import Image

        pipe = None
        try:
            with _AI_LOCK:
                if device == "auto":
                    if torch.cuda.is_available():
                        device = "cuda"
                    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = "cpu"
                dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    "CompVis/stable-diffusion-v1-4",
                    torch_dtype=dtype,
                    revision="133a221b8aa7292a167afc5127cb63fb5005638b",
                )
                pipe = pipe.to(device)
                image = Image.open(source_frame).convert("RGB").resize((512, 512))
                result = pipe(prompt=prompt, image=image, strength=0.4, guidance_scale=7.5)
                result.images[0].save(out_path)
            return True
        finally:
            del pipe
            from .model_engine import release_allocations

            release_allocations()
    except Exception as e:  # an optional thumbnail must never fail the run
        log.warning("Stable Diffusion thumbnail failed (%s: %s)", type(e).__name__, e)
        return False


def _sd_process(frame: str, out: str, prompt: str, device: str) -> None:
    """Run one SD call in a bounded child, so a tight machine gets a strong memory release."""
    import sys

    sys.exit(0 if _sd_generate(Path(frame), Path(out), prompt, device) else 1)


def generate_sd_thumbnail(source_frame: Path, out_path: Path, prompt: str, device: str = "auto") -> bool:
    """Generate an AI thumbnail; a tight machine gets a one-shot worker, otherwise this process runs it and releases it."""
    if not sd_available():
        return False
    from .model_engine import choose_model

    decision = choose_model("stable-diffusion", 5000.0)
    if not decision.subprocess:
        return _sd_generate(source_frame, out_path, prompt, device)
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=_sd_process, args=(str(source_frame), str(out_path), prompt, device), name="autoclip-sd")
    proc.start()
    proc.join(900.0)
    if proc.is_alive():
        proc.kill()
        proc.join(5.0)
    return proc.exitcode == 0 and out_path.is_file() and out_path.stat().st_size > 0
