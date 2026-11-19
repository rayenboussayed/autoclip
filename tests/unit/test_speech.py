"""One streaming scan of an audio track, feeding both silence removal and loudness."""

import io
from pathlib import Path

import numpy as np
import pytest

from autoclip.core.media import frame_energy_db
from autoclip.core.speech import scan_audio
from tests.support import Media


def test_scan_matches_the_existing_energy_track(media: Media) -> None:
    info = __import__("autoclip.core.media", fromlist=["probe"]).probe(media.video)
    scan = scan_audio(media.video, info)
    reference = frame_energy_db(media.video)
    assert np.allclose(scan.energy_db, reference)


def test_scan_finds_the_fixture_speech_spans(media: Media) -> None:
    info = __import__("autoclip.core.media", fromlist=["probe"]).probe(media.video)
    scan = scan_audio(media.video, info)
    assert len(scan.intervals) >= 8
    assert scan.intervals[0][0] < 0.5
    assert scan.intervals[0][1] > 2.8
    assert all(end > start for start, end in scan.intervals)


def test_scan_of_a_silent_video_has_no_speech(media: Media) -> None:
    info = __import__("autoclip.core.media", fromlist=["probe"]).probe(media.silent)
    scan = scan_audio(media.silent, info)
    assert scan.intervals == []
    assert np.all(scan.energy_db == 0)


def test_scan_calls_the_interval_callback(media: Media) -> None:
    info = __import__("autoclip.core.media", fromlist=["probe"]).probe(media.video)
    scan = scan_audio(media.video, info)
    assert scan.energy_db.size


def test_scan_flushes_an_open_interval_at_eof(tmp_path: Path) -> None:
    import subprocess

    from autoclip.core.media import ffmpeg_exe

    video = tmp_path / "tone.mp4"
    command = [
        ffmpeg_exe(),
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=220:duration=1.2",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=32x32:d=1.2:r=1",
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(video),
    ]
    subprocess.run(command, check=True)
    info = __import__("autoclip.core.media", fromlist=["probe"]).probe(video)
    scan = scan_audio(video, info)
    assert scan.intervals and scan.intervals[0][0] == pytest.approx(0.0)


def test_scan_merge_behavior() -> None:
    from autoclip.core.speech import _merge

    assert _merge([(2.0, 3.0), (0.0, 2.5), (5.0, 6.0)]) == [(0.0, 3.0), (5.0, 6.0)]


def test_stop_process_prefers_terminate_and_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    from subprocess import TimeoutExpired

    from autoclip.core.speech import _stop_process

    class FakeProc:
        def __init__(self) -> None:
            self.killed = False
            self.waits = 0

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            pass

        def wait(self, timeout: float) -> int:
            self.waits += 1
            if self.waits == 1:
                raise TimeoutExpired("ffmpeg", timeout)
            return 0

        def kill(self) -> None:
            self.killed = True

    proc = FakeProc()
    _stop_process(proc)  # type: ignore[arg-type]
    assert proc.killed


def test_scan_stops_the_analysis_process_on_cancellation(media: Media) -> None:
    from autoclip.core.events import Cancelled, CancelToken

    info = __import__("autoclip.core.media", fromlist=["probe"]).probe(media.video)
    token = CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        scan_audio(media.video, info, cancel=token)


def test_scan_returns_no_interval_for_a_tiny_tail_of_speech(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import io

    import autoclip.core.speech as speech
    from autoclip.core.models import MediaInfo

    sr = 16_000
    x = np.zeros(sr, dtype=np.float32)
    t = np.arange(int(0.08 * sr)) / sr
    x[-len(t) :] = 0.3 * np.sin(2 * np.pi * 220 * t)
    payload = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()

    class FakeStdout:
        def __init__(self) -> None:
            self.buffer = io.BytesIO(payload)

        def read(self, _n: int) -> bytes:
            return bytes(self.buffer.read(64 * 1024))

    class FakeProc:
        def __init__(self) -> None:
            self.stdout = FakeStdout()

        def wait(self, timeout: float | None = None) -> int:
            return 0

    def fake_popen(command: list[str], **kwargs: object) -> FakeProc:
        return FakeProc()

    monkeypatch.setattr("autoclip.core.speech.subprocess.Popen", fake_popen)
    info = MediaInfo(1.0, True, 32, 32, 1.0)
    assert speech.scan_audio(tmp_path / "fake.mp4", info).intervals == []


def test_stop_process_returns_immediately_when_the_child_already_exited(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoclip.core.speech import _stop_process

    class ExitedProc:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self) -> int:
            return 0

        def terminate(self) -> None:
            self.terminated = True

    proc = ExitedProc()
    _stop_process(proc)  # type: ignore[arg-type]
    assert not proc.terminated


def test_scan_keeps_a_quiet_band_from_closing_a_speech_span(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import autoclip.core.speech as speech
    from autoclip.core.models import MediaInfo

    amplitude = np.array([10_000] * 16_000 + [230] * 3_200 + [10_000] * 1_600 + [0] * 32_000, dtype=np.float32)
    x = amplitude / 32768.0
    payload = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()

    class FakeStdout:
        def __init__(self) -> None:
            self.buffer = io.BytesIO(payload)

        def read(self, _n: int) -> bytes:
            return bytes(self.buffer.read(64 * 1024))

    class FakeProc:
        def __init__(self) -> None:
            self.stdout = FakeStdout()

        def wait(self, timeout: float | None = None) -> int:
            return 0

    monkeypatch.setattr("autoclip.core.speech.subprocess.Popen", lambda *args, **kwargs: FakeProc())
    assert speech.scan_audio(tmp_path / "fake.mp4", MediaInfo(3.0, True, 32, 32, 1.0)).intervals


def test_scan_ignores_a_speech_span_shorter_than_the_minimum(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import io

    import autoclip.core.speech as speech
    from autoclip.core.models import MediaInfo

    sr = 16_000
    x = np.zeros(int(3.5 * sr), dtype=np.float32)
    t = np.arange(int(0.02 * sr)) / sr
    x[int(3.0 * sr) : int(3.0 * sr) + len(t)] = 0.3 * np.sin(2 * np.pi * 220 * t)
    payload = (x * 32767).astype(np.int16).tobytes()

    class FakeStdout:
        def __init__(self) -> None:
            self.buffer = io.BytesIO(payload)

        def read(self, _n: int) -> bytes:
            return self.buffer.read(64 * 1024)

    class FakeProc:
        def __init__(self) -> None:
            self.stdout = FakeStdout()

        def wait(self, timeout: float | None = None) -> int:
            return 0

    monkeypatch.setattr("autoclip.core.speech.subprocess.Popen", lambda *args, **kwargs: FakeProc())
    assert speech.scan_audio(tmp_path / "fake.mp4", MediaInfo(3.0, True, 32, 32, 1.0)).intervals == []


def test_scan_drops_spans_shorter_than_the_minimum_at_eof(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import io

    import autoclip.core.speech as speech
    from autoclip.core.models import MediaInfo

    payload = (np.ones(1600, dtype=np.float32) * 10_000).astype(np.int16).tobytes()

    class FakeStdout:
        def __init__(self) -> None:
            self.buffer = io.BytesIO(payload)

        def read(self, _n: int) -> bytes:
            return bytes(self.buffer.read(64 * 1024))

    class FakeProc:
        def __init__(self) -> None:
            self.stdout = FakeStdout()

        def wait(self, timeout: float | None = None) -> int:
            return 0

    monkeypatch.setattr("autoclip.core.speech.subprocess.Popen", lambda *args, **kwargs: FakeProc())
    assert speech.scan_audio(tmp_path / "tiny.mp4", MediaInfo(0.1, True, 16, 16, 1.0)).intervals == []


def test_scan_closes_a_span_with_the_hangover_a_mid_speech_quiet_patch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import io

    import autoclip.core.speech as speech
    from autoclip.core.models import MediaInfo

    payload = (np.full(int(2.5 * 16_000), 10_000, dtype=np.int16)).tobytes()
    silent = np.zeros(int(1.0 * 16_000), dtype=np.int16).tobytes()
    buf = io.BytesIO(payload + silent)

    class FakeStdout:
        def read(self, _n: int) -> bytes:
            return bytes(buf.read(64 * 1024))

    class FakeProc:
        def __init__(self) -> None:
            self.stdout = FakeStdout()

        def wait(self, timeout: float | None = None) -> int:
            return 0

    monkeypatch.setattr("autoclip.core.speech.subprocess.Popen", lambda *args, **kwargs: FakeProc())
    scan = speech.scan_audio(tmp_path / "fake.mp4", MediaInfo(3.5, True, 16, 16, 1.0))
    assert scan.intervals and scan.intervals[0][0] == pytest.approx(0.0, abs=0.15)
