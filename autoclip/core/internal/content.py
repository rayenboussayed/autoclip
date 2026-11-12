"""Optional ``lexicon.json`` content file: per-feature word lists and the filler sounds.

The file sits next to the config file (``<config-dir>/lexicon.json``) and is the only override:
``{ "funny": ["..."], "fillers": ["um", "uh"] }``. A missing file means the built-in lexicons and
filler list are used.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..errors import ConfigError
from .automation import DEFAULT_FILLERS

__all__ = ["LEXICON_FILE", "fillers_from", "load_lexicon"]

#: Name of the optional file, looked up next to the config file.
LEXICON_FILE = "lexicon.json"


def load_lexicon(base: Path | None) -> dict[str, list[str]]:
    """The word lists in ``<base>/lexicon.json`` (empty when the folder has no such file).

    Raises:
        ConfigError: if the file is unreadable, not a JSON object of string lists, or holds an empty word.
    """
    if base is None:
        return {}
    path = base / LEXICON_FILE
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"cannot read '{path}': {e.strerror or e}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a JSON object")
    words: dict[str, list[str]] = {}
    for name, entries in raw.items():
        if not isinstance(entries, list) or not all(isinstance(w, str) and w.strip() for w in entries):
            raise ConfigError(f"{path}: '{name}' must be a list of non-empty strings")
        words[str(name)] = list(entries)
    return words


def fillers_from(lexicon: dict[str, list[str]]) -> frozenset[str]:
    """The filler sounds: the lexicon's ``fillers`` list or the built-in defaults (an empty list disables them)."""
    return frozenset(lexicon["fillers"]) if "fillers" in lexicon else frozenset(DEFAULT_FILLERS)
