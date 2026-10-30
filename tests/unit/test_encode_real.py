"""Real encodes for every output container (the temporary file must keep the container extension)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
from moviepy import ColorClip, VideoFileClip

from autoclip.core.config import VideoCfg
from autoclip.core.hardware import cpu_encoder
from autoclip.core.media import probe
from autoclip.core.models import Moment, RenderJob
from autoclip.core.render import encode, render_clip
from autoclip.core.resources import RESOURCES
from autoclip.core.store import Cache, Store
from autoclip.core.transcript import load_transcript_file

from ..support import Media, make_cfg


def _clip() -> ColorClip:
    return ColorClip((96, 54), color=(10, 20, 30)).with_duration(1.0).with_fps(10)


@pytest.mark.parametrize("container", ["mp4", "mkv"])
def test_encode_writes_a_real_file_in_the_requested_container(tmp_path: Path, container: Literal["mp4", "mkv"]) -> None:
    video = VideoCfg(preset="ultrafast", container=container, audio_bitrate_kbps=96)
    dst = tmp_path / f"clip.{container}"
    encode(_clip(), dst, 10, cpu_encoder(video), 1, video)
    assert dst.is_file()
    assert not list(tmp_path.glob("*.part*")), "the temporary file must be renamed away"
    info = probe(dst)
    assert info.duration == pytest.approx(1.0, abs=0.2)
    assert (info.width, info.height) == (96, 54)


def test_render_clip_writes_the_container_from_the_config(media: Media, tmp_path: Path) -> None:
    """The pipeline names the output ``.<container>``; rendering it must not depend on the muxer name."""
    cfg = make_cfg(media, tmp_path, video={"container": "mkv", "preset": "ultrafast"}, captions={"enabled": False})
    store = Store(tmp_path / "r.db", fresh=True)
    store.put_segments(load_transcript_file(media.transcript))
    cache = Cache(store, RESOURCES)
    dst = tmp_path / "clip.mkv"
    job = RenderJob(
        1,
        Moment(8.0, 19.0, ""),
        media.video,
        dst,
        "short",
        "9:16",
        cpu_encoder(cfg.video),
        probe(media.video).has_audio,
        cfg,
        store.path,
        None,
        0,
        threads=1,
    )
    vclip = VideoFileClip(str(media.video), audio=False)
    try:
        size, _lines = render_clip(vclip, job, store, cache)
    finally:
        vclip.close()
        cache.close()
        store.close()
    assert dst.is_file() and size == (144, 256)
    assert not list(tmp_path.glob("*.part*")), "the temporary file must be renamed away"
    assert probe(dst).duration > 3
