"""HTTP/WebSocket front end for autoclip.

The server is stateless: uploads are processed in ephemeral job directories and only a small
performance cache is retained between requests.
"""

from __future__ import annotations

from .config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
