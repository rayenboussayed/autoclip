"""Reading the input video: ffmpeg discovery, probing, loudness analysis and scene detection."""

from __future__ import annotations

import functools
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray

from .errors import AutoclipError
from .log import traced
from .models import MediaInfo

F32 = NDArray[np.float32]
SR = 16000  # analysis sample rate (Hz)
FRAME = 0.5  # seconds per energy frame
AUDIO_SR = 44100  # sample rate of the clip audio that is processed and exported


@functools.cache
def bundled_ffmpeg_exe() -> str:
    """The static ffmpeg bundled with ``imageio-ffmpeg`` (it runs even under a frozen library path)."""
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, RuntimeError) as e:
        raise AutoclipError("ffmpeg not found: install it (https://ffmpeg.org) or `pip install imageio-ffmpeg`") from e


@functools.cache
def ffmpeg_exe() -> str:
    """ffmpeg on ``PATH``, else the binary bundled with ``imageio-ffmpeg`` (so Windows needs no install)."""
    found = shutil.which("ffmpeg")
    return found if found else bundled_ffmpeg_exe()


def external_env() -> dict[str, str] | None:
    """A child-process environment with the PyInstaller library path removed, if any.

    A frozen one-folder build puts its shared libraries in ``sys._MEIPASS`` and puts that folder
    on ``LD_LIBRARY_PATH`` (and the macOS/Windows equivalents) so the Python process starts.  When
    we launch a *system* ffmpeg under that environment it loads the bundled ``libssl.so.3`` /
    ``libstdc++.so.6`` and fails with missing / incompatible symbols (e.g. ``OPENSSL_3.2.0`` not
    found) even though the same binary works fine without it.  This helper returns ``None`` when
    there is nothing to remove (non-frozen run, or nothing to remove), so subprocesses inherit the
    normal environment.  Video-encoding speed is the only casualty: a frozen build falls back to a
    local trim of full downloads instead of a server-side range.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return None
    meipass = str(meipass)
    keys = ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PATH")
    env = dict(os.environ)
    changed = False
    for key in keys:
        value = env.get(key)
        if not value:
            continue
        parts = value.split(os.pathsep)
        kept = [p for p in parts if p and p != meipass]
        if len(kept) != len(parts):
            changed = True
        if kept:
            env[key] = os.pathsep.join(kept)
        else:
            env.pop(key, None)
            changed = True
    return env if changed else None


@traced("probe", logging.DEBUG)
def probe(path: Path) -> MediaInfo:
    """Duration, audio presence, frame size and rate (via PyAV - no ``ffprobe`` needed)."""
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise AutoclipError(f"'{path.name}' has no video stream")
            video = container.streams.video[0]
            duration = (container.duration or 0) / av.time_base
            return MediaInfo(duration, bool(container.streams.audio), video.width, video.height, float(video.average_rate or 0))
    except av.error.FFmpegError as e:
        raise AutoclipError(f"cannot read '{path.name}': {e}") from e


@traced("analyse_audio_energy")
def frame_energy_db(path: Path) -> F32:
    """RMS level (dB) of every 0.5 s of the file, streamed so memory stays flat.

    Raises ``AutoclipError`` if ffmpeg fails (a silent file has no audio, which is reported by
    ``energy_track`` before this is called - see there, and the test suite).  Keeping this silent
    hid a broken ffmpeg build (bundled libssl under the AppImage) as a flat-zero energy track
    for the whole file instead of a loud failure."""
    last_err: AutoclipError = AutoclipError(f"cannot analyse '{path.name}': ffmpeg failed")
    for _attempt in range(2):
        try:
            return _frame_energy_db(path)
        except AutoclipError as exc:
            last_err = exc
    raise last_err


def _frame_energy_db(path: Path) -> F32:
    n = int(SR * FRAME)
    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    levels: list[F32] = []
    leftover = b""
    env = external_env()
    with tempfile.TemporaryFile() as serr:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=serr, env=env) as proc:
            assert proc.stdout is not None
            while data := proc.stdout.read(n * 2 * 240):  # ~2 minutes per chunk
                buf = leftover + data
                usable = len(buf) // (2 * n) * (2 * n)
                x = np.frombuffer(buf[:usable], dtype=np.int16).astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(x.reshape(-1, n) ** 2, axis=1) + 1e-12)
                levels.append((20 * np.log10(rms + 1e-9)).astype(np.float32))
                leftover = buf[usable:]
            rc = proc.wait()
        if rc != 0:
            serr.seek(0)
            detail = serr.read().decode("utf-8", "replace").strip().splitlines()
            raise AutoclipError(f"cannot analyse '{path.name}': {detail[-1] if detail else 'ffmpeg failed'}")
    return np.concatenate(levels) if levels else np.zeros(1, dtype=np.float32)


def decode_audio(src: Path, start: float, dur: float, sr: int, filters: Sequence[str] = ()) -> F32:
    """Mono float32 samples of ``[start, start+dur]`` at ``sr`` Hz (zero padded if the file ends early).

    FFmpeg is streamed (never ``capture_output=True``): the sample blocks are read in ~1-minute pieces,
    so a several-hour file does not sit in one giant ``bytes`` object. Fails hard only when ffmpeg cannot
    produce samples at all; transient timeouts get one retry before we raise :class:`AutoclipError`.
    """
    cmd_base = [ffmpeg_exe(), "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr)]
    if filters:
        cmd_base += ["-af", ",".join(filters)]
    log = logging.getLogger("autoclip")
    last_detail = ""
    total = round(dur * sr)
    block = 4 * sr * 60  # one minute of f32le samples per read
    deadline = time.monotonic() + max(300.0, dur * 2.0 + 60.0)
    for attempt in range(2):
        cmd = [*cmd_base, "-f", "f32le", "-"]
        rc = None
        try:
            with tempfile.TemporaryFile() as serr:
                with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=serr, env=external_env()) as proc:
                    assert proc.stdout is not None
                    chunks: list[bytes] = []
                    while (raw := proc.stdout.read(block)) != b"":
                        if time.monotonic() > deadline:
                            proc.kill()  # pragma: no cover
                            proc.wait()  # pragma: no cover
                            raise OSError("ffmpeg decode exceeded its time budget")  # pragma: no cover
                        chunks.append(raw)
                    rc = proc.wait()
                    if rc == 0:
                        payload = np.frombuffer(b"".join(chunks), dtype=np.float32)
                        return np.pad(payload, (0, total))[:total]
                serr.seek(0)
                last_detail = serr.read().decode("utf-8", "replace").strip()[-1000:]
        except OSError as e:  # pragma: no cover
            last_detail = f"{e}"  # pragma: no cover
            log.error("ffmpeg decode could not start (attempt %d/2): %s", attempt + 1, e)  # pragma: no cover
        else:
            log.error("ffmpeg decode failed (rc=%s): %s", rc, last_detail)
    raise subprocess.CalledProcessError(1 if last_detail else -9, cmd_base, output=b"", stderr=last_detail.encode())


def snap_to_quiet(db: F32, t: float, radius: float, total: float) -> float:
    """Move ``t`` to the quietest 0.5 s frame within ``radius`` seconds (avoids cutting mid-word)."""
    lo, hi = max(0, int((t - radius) / FRAME)), min(len(db) - 1, int((t + radius) / FRAME))
    if hi <= lo:
        return t
    return min(max((lo + int(np.argmin(db[lo : hi + 1]))) * FRAME + FRAME / 2, 0.0), total)


@traced("trim_video")
def trim_video(src: Path, dst: Path, start: float, end: float) -> Path:
    """Cut ``[start, end]`` out of ``src`` into ``dst`` by re-encoding the video (accurate cuts).

    yt-dlp downloads a range itself only when a real ``ffmpeg`` *and* ``ffprobe`` are on ``PATH``;
    with the bundled imageio-ffmpeg the downloader takes the whole file and cuts it here.  A
    re-encode (rather than a stream copy) keeps both cut points exact, so subtitle timings that
    were shifted by ``start`` stay right.

    Raises:
        AutoclipError: if ffmpeg cannot produce the file.
    """
    lead = min(2.0, max(start, 0.0))
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-v",
        "error",
        "-ss",
        f"{max(start - lead, 0.0):.3f}",
        "-i",
        str(src),
        # a second, output-side seek resets the timestamps (an input seek alone leaves the
        # Matroska muxer with the original offset), while the first one keeps this fast
        "-ss",
        f"{lead:.3f}",
        "-t",
        f"{max(end - start, 0.1):.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        "-c:a",
        "copy",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, env=external_env())
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        raise AutoclipError(f"cannot trim '{src.name}': {detail[-1] if detail else 'ffmpeg failed'}") from e
    return dst
