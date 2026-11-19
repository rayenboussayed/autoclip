"""Visible shot boundaries as a small signal for choosing clip edges."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from .events import CancelToken
from .internal.automation import SCENE_ADAPTIVE_THRESHOLD, SCENE_MIN_CONTENT_VAL, SCENE_MIN_SCENE_LEN_SECONDS
from .log import traced
from .models import MediaInfo

log = logging.getLogger("autoclip")


@traced("detect_scenes")
def detect_scenes(src: Path, info: MediaInfo, cancel: CancelToken | None = None) -> list[float]:
    """Detect cut times in seconds with PySceneDetect, falling back to acoustic boundaries only."""
    if cancel is not None:
        cancel.check()
    if info.duration <= 0:
        return []
    try:
        from scenedetect import AdaptiveDetector, SceneManager, open_video
    except Exception as exc:
        log.warning("scene detection unavailable (%s: %s); using acoustic boundaries only", type(exc).__name__, exc)
        return []
    video = None
    try:
        video = open_video(str(src), backend="pyav")
        manager = SceneManager()
        manager.auto_downscale = True
        manager.add_detector(
            AdaptiveDetector(
                adaptive_threshold=SCENE_ADAPTIVE_THRESHOLD,
                min_scene_len=SCENE_MIN_SCENE_LEN_SECONDS,
                min_content_val=SCENE_MIN_CONTENT_VAL,
            )
        )
        manager.detect_scenes(video)
        if cancel is not None:
            cancel.check()
        cuts = [float(scene[0].seconds) for scene in manager.get_scene_list()[1:]]
        return sorted({round(c, 6) for c in cuts if 0.0 < c < info.duration})
    except Exception as exc:
        log.warning("scene detection failed (%s: %s); using acoustic boundaries only", type(exc).__name__, exc)
        return []
    finally:
        if video is not None:
            with contextlib.suppress(Exception):  # closing a decoder must not mask a detection failure
                video.close()
