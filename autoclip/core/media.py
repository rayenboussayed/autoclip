"""Reading the input video: ffmpeg discovery, probing, loudness analysis and scene detection."""

from __future__ import annotations

import functools
import logging
import shutil
import subprocess
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
def ffmpeg_exe() -> str:
    """ffmpeg on ``PATH``, else the binary bundled with ``imageio-ffmpeg`` (so Windows needs no install)."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, RuntimeError) as e:
        raise AutoclipError("ffmpeg not found: install it (https://ffmpeg.org) or `pip install imageio-ffmpeg`") from e


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
def frame_energy_db(path: Path) -> F32 | None:
    """RMS level (dB) of every 0.5 s of the file, streamed so memory stays flat; ``None`` without audio."""
    n = int(SR * FRAME)
    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    levels: list[F32] = []
    leftover = b""
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        assert proc.stdout is not None
        while data := proc.stdout.read(n * 2 * 240):  # ~2 minutes per chunk
            buf = leftover + data
            usable = len(buf) // (2 * n) * (2 * n)
            x = np.frombuffer(buf[:usable], dtype=np.int16).astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(x.reshape(-1, n) ** 2, axis=1) + 1e-12)
            levels.append((20 * np.log10(rms + 1e-9)).astype(np.float32))
            leftover = buf[usable:]
    return np.concatenate(levels) if levels else None


def decode_audio(src: Path, start: float, dur: float, sr: int, filters: Sequence[str] = ()) -> F32:
    """Mono float32 samples of ``[start, start+dur]`` at ``sr`` Hz (zero padded if the file ends early)."""
    cmd = [ffmpeg_exe(), "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr)]
    cmd += ["-af", ",".join(filters)] if filters else []
    raw = subprocess.run([*cmd, "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.pad(np.frombuffer(raw, dtype=np.float32), (0, round(dur * sr)))[: round(dur * sr)]


def snap_to_quiet(db: F32, t: float, radius: float, total: float) -> float:
    """Move ``t`` to the quietest 0.5 s frame within ``radius`` seconds (avoids cutting mid-word)."""
    lo, hi = max(0, int((t - radius) / FRAME)), min(len(db) - 1, int((t + radius) / FRAME))
    if hi <= lo:
        return t
    return min(max((lo + int(np.argmin(db[lo : hi + 1]))) * FRAME + FRAME / 2, 0.0), total)


def snap_to_cut(t: float, cuts: Sequence[float], radius: float) -> float:
    """The nearest scene cut within ``radius`` seconds of ``t``, else ``t``."""
    near = [c for c in cuts if abs(c - t) <= radius]
    return min(near, key=lambda c: abs(c - t)) if near else t


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
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        raise AutoclipError(f"cannot trim '{src.name}': {detail[-1] if detail else 'ffmpeg failed'}") from e
    return dst


@traced("detect_scenes")
def scene_cuts(path: Path) -> list[float]:
    """Scene-change times in seconds (PySceneDetect); empty with a warning if it is not installed."""
    try:
        from scenedetect import AdaptiveDetector, detect
    except ImportError:
        logging.getLogger("autoclip").warning("clips.snap_scenes needs `pip install scenedetect[opencv]`; ignored")
        return []
    return [float(scene[0].seconds) for scene in detect(str(path), AdaptiveDetector())[1:]]
