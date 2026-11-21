"""The transcript clock must follow the real assembled video, not the nominal cut map."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import autoclip.core.optimizer as optimizer
from autoclip.core.media import probe
from autoclip.core.optimizer import prepare_source
from autoclip.core.store import Store
from autoclip.core.transcript import get_transcript

from ..support import Media, make_cfg


def test_last_cue_follows_the_assembled_video_end(media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cue at the final kept span must stay locked to the end of the file captions burn into."""
    monkeypatch.setattr(optimizer, "UNSILENCE_MIN_TOTAL_SECONDS", 0.0)
    source = tmp_path / "cue.mp4"
    source.write_bytes(media.video.read_bytes())
    tail_start, tail_end = 35.88, 39.12  # the last kept interval of the 40 s fixture
    source.with_suffix(".json").write_text(json.dumps([{"start": tail_start, "end": tail_end, "text": "LAST CUE"}]), encoding="utf-8")
    cfg = make_cfg(media, tmp_path / "out", source=source)
    info = probe(source)
    work = tmp_path / "work"
    prepared = prepare_source(source, cfg, info, work)

    store = Store(work / "autoclip.db", fresh=True)
    try:
        assert get_transcript(
            source,
            cfg,
            work,
            store,
            info,
            need_text=True,
            cutmap=prepared.cutmap,
            timeline=prepared.timeline,
        )
        segments = store.light_segments()
        assert len(segments) == 1
        text, start, end = segments[0][2], segments[0][0], segments[0][1]
        assert text == "LAST CUE"
        # The stored cue must track the real file, whatever the encoder's padding happens to be.
        assert end == pytest.approx(probe(prepared.path).duration, abs=0.05)
        assert start > 0.0
        assert prepared.timeline.to_source(end) == pytest.approx(tail_end, abs=0.05)
    finally:
        store.close()
