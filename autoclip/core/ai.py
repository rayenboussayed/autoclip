"""Optional AI helpers: a local Ollama server and (optionally) Stable Diffusion.

Every call goes through a single process-wide lock so that, even in ``performance`` mode, Ollama
and Stable Diffusion never run concurrently.  When a backend is missing or errors, the caller
keeps its heuristic result - AI is an enhancement, never a requirement.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from functools import cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaValue

from .config import Config
from .internal.automation import LLM_NUM_CTX, LLM_TIMEOUT
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


def _model_names(payload: object) -> set[str]:
    """The model names from an ``/api/tags`` response (both the list and dict shapes)."""
    models: object = payload
    if isinstance(payload, dict):
        models = payload.get("models") or []
    names: set[str] = set()
    if isinstance(models, list):
        for item in models:
            name = item.get("name") or item.get("model") if isinstance(item, dict) else str(item)
            if isinstance(name, str) and name:
                names.add(name)
    return names


@cache
def _tags(base_url: str) -> set[str]:
    """The model names the server at ``base_url`` has installed (empty when it cannot be asked)."""
    url = base_url.rstrip("/") + "/api/tags"
    if ollama is not None:
        try:
            names = _model_names(ollama.Client(host=base_url).list())
            if names:
                return names
        except Exception as e:
            log.debug("the ollama python client could not list models (%s: %s); trying a plain HTTP probe", type(e).__name__, e)
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return set()
    return _model_names(data)


def llm_available(base_url: str, model: str = "") -> bool:
    """True when an Ollama server answers AND the requested model is installed.

    The answer is cached per (base_url, model), so a whole run asks once instead of per clip;
    a local Ollama without the configured model reports ``False`` and the run keeps its
    heuristic ranking instead of failing inside every clip. Once any AI call has failed in
    this run, the answer is ``False`` for the rest of the run (the offending backend is not
    retried per clip).
    """
    if _RUN_TRIPPED.is_set():
        return False
    names = _tags(base_url)
    if not model:
        return bool(names)
    wanted = model.strip().lower()
    for name in names:
        lowered = name.lower()
        if lowered == wanted or lowered.startswith(wanted + ":") or lowered.split(":", 1)[0] == wanted:
            return True
    return False


def _chat(base_url: str, model: str, messages: list[dict[str, str]], *, fmt: JsonSchemaValue | None = None) -> str:
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

    if cfg.skip.llm or _RUN_TRIPPED.is_set():
        return {}
    lc = cfg.llm
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
        content = _chat(lc.base_url, lc.model, messages, fmt=_RERANK_SCHEMA)
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
    if cfg.skip.llm or not text or _RUN_TRIPPED.is_set() or not llm_available(cfg.llm.base_url, cfg.llm.model):
        return ""
    prompt = "Write a punchy title (max 8 words) in the same language as the text. Return only the title.\n\n" + text[:500]
    messages = _messages(cfg, prompt)
    try:
        return _chat(cfg.llm.base_url, cfg.llm.model, messages)[:80]
    except Exception as e:
        _disable_for_this_run(e)
        log.warning("AI title failed (%s: %s)", type(e).__name__, e)
        return ""


def generate_caption(cfg: Config, text: str) -> str:
    """A short social-media caption for ``text`` (empty when Ollama is unavailable, offline or errors)."""
    if cfg.skip.llm or not text or _RUN_TRIPPED.is_set() or not llm_available(cfg.llm.base_url, cfg.llm.model):
        return ""
    prompt = "Write a short social-media caption (max 2 sentences) for this clip transcript. Return only the caption.\n\n" + text[:500]
    messages = _messages(cfg, prompt)
    try:
        return _chat(cfg.llm.base_url, cfg.llm.model, messages)
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
    """True when torch + diffusers are importable."""
    try:
        import torch  # noqa: F401  # pragma: no cover
        from diffusers import StableDiffusionImg2ImgPipeline  # noqa: F401  # pragma: no cover

        return True  # pragma: no cover
    except Exception:  # pragma: no cover
        return False  # pragma: no cover


def generate_sd_thumbnail(source_frame: Path, out_path: Path, prompt: str, device: str = "auto") -> bool:  # pragma: no cover
    """Generate an AI thumbnail with Stable Diffusion img2img. Returns True on success."""  # pragma: no cover
    if not sd_available():  # pragma: no cover
        return False  # pragma: no cover
    try:  # pragma: no cover
        import torch  # pragma: no cover
        from diffusers import StableDiffusionImg2ImgPipeline  # pragma: no cover
        from PIL import Image  # pragma: no cover

        # pragma: no cover
        with _AI_LOCK:  # pragma: no cover
            if device == "auto":  # pragma: no cover
                if torch.cuda.is_available():  # pragma: no cover
                    device = "cuda"  # pragma: no cover
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # pragma: no cover
                    device = "mps"  # pragma: no cover
                else:  # pragma: no cover
                    device = "cpu"  # pragma: no cover
            dtype = torch.float16 if device in ("cuda", "mps") else torch.float32  # pragma: no cover
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=dtype)  # pragma: no cover
            pipe = pipe.to(device)  # pragma: no cover
            image = Image.open(source_frame).convert("RGB")  # pragma: no cover
            image = image.resize((512, 512))  # pragma: no cover
            result = pipe(prompt=prompt, image=image, strength=0.4, guidance_scale=7.5)  # pragma: no cover
            result.images[0].save(out_path)  # pragma: no cover
        return True  # pragma: no cover
    except Exception as e:  # pragma: no cover
        log.warning("Stable Diffusion thumbnail failed (%s: %s)", type(e).__name__, e)  # pragma: no cover
        return False  # pragma: no cover
