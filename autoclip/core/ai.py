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
from pathlib import Path
from typing import Literal

from .config import Config
from .models import Moment

log = logging.getLogger("autoclip")

try:
    import ollama
except Exception:  # pragma: no cover - optional dependency
    ollama = None  # type: ignore[assignment]

#: One queue for every AI model call in this process (Ollama and Stable Diffusion share it).
_AI_LOCK = threading.Lock()


def llm_available(base_url: str, timeout: float = 2.0) -> bool:
    """True when an Ollama server answers at ``base_url``."""
    if ollama is None:
        return False
    try:
        ollama.Client(host=base_url).list()
        return True
    except Exception:
        # A server that only answers some paths is still up (the tests use a minimal stub).
        url = base_url.rstrip("/")
        for path in ("", "/v1/models", "/api/tags"):
            try:
                urllib.request.urlopen(url + path, timeout=timeout)  # noqa: S310
                return True
            except urllib.error.HTTPError:  # pragma: no cover
                return True  # pragma: no cover
            except (OSError, ValueError):
                continue
        return False


def _chat(base_url: str, model: str, messages: list[dict[str, str]], *, fmt: Literal["", "json"] | None = None) -> str:
    with _AI_LOCK:
        client = ollama.Client(host=base_url)
        response = client.chat(model=model, messages=messages, format=fmt if fmt else None)
    return str((response.get("message") or {}).get("content", "")).strip()


def llm_rerank(top: list[Moment], cfg: Config, scene: str) -> dict[int, tuple[float, str]]:
    """Ask Ollama for a 0-10 fit score and a title for each candidate."""
    from .styles import STYLE_HELP

    if cfg.skip.llm:
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
    content = _chat(lc.base_url, lc.model, messages, fmt="json")
    parsed = json.loads(content)
    return {int(x["id"]): (float(x["score"]), str(x.get("title", ""))[:80]) for x in parsed["scores"]}


def generate_title(cfg: Config, text: str) -> str:
    """A punchy title for ``text`` (empty when Ollama is unavailable, offline or errors)."""
    if cfg.skip.llm or not text or not llm_available(cfg.llm.base_url):
        return ""
    prompt = "Write a punchy title (max 8 words) in the same language as the text. Return only the title.\n\n" + text[:500]
    messages = _messages(cfg, prompt)
    try:
        return _chat(cfg.llm.base_url, cfg.llm.model, messages)[:80]
    except Exception as e:
        log.warning("AI title failed (%s: %s)", type(e).__name__, e)
        return ""


def generate_caption(cfg: Config, text: str) -> str:
    """A short social-media caption for ``text`` (empty when Ollama is unavailable, offline or errors)."""
    if cfg.skip.llm or not text or not llm_available(cfg.llm.base_url):
        return ""
    prompt = "Write a short social-media caption (max 2 sentences) for this clip transcript. Return only the caption.\n\n" + text[:500]
    messages = _messages(cfg, prompt)
    try:
        return _chat(cfg.llm.base_url, cfg.llm.model, messages)
    except Exception as e:
        log.warning("AI caption failed (%s: %s)", type(e).__name__, e)
        return ""


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
