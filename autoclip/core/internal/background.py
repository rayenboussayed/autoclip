"""Interpret the one ``video.background`` value: ``blur``, a colour or an image path."""

from __future__ import annotations

import re
from pathlib import Path

from ..errors import ConfigError

__all__ = ["background_kind", "is_color", "resolve_background"]

#: ``#RGB`` or ``#RRGGBB`` (the two forms Pillow's ``ImageColor`` reads unambiguously).
_COLOR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def is_color(value: str) -> bool:
    """Whether ``value`` is a ``#RGB``/``#RRGGBB`` colour."""
    return bool(_COLOR.fullmatch(value))


def background_kind(value: str) -> str:
    """``"blur"``, ``"color"`` or ``"image"`` for a ``video.background`` value.

    Raises:
        ConfigError: for an empty value or a ``#`` value that is not a colour.
    """
    if value == "blur":
        return "blur"
    if value.startswith("#"):
        if is_color(value):
            return "color"
        raise ConfigError(f"video.background: '{value}' is not a colour (use #RGB, #RRGGBB, 'blur' or an image path)")
    if not value.strip():
        raise ConfigError("video.background must be 'blur', a #RGB/#RRGGBB colour or an image path")
    return "image"


def resolve_background(value: str, base: Path) -> str:
    """``value`` with an image path made absolute against the config file's folder."""
    if background_kind(value) != "image":
        return value
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else base / path)
