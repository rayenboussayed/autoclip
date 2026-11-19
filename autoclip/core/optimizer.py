"""Crash-safe construction of the silence-removed source used by the whole pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .audio import enhance_audio
from .config import Config
from .cutmap import CutMap
from .errors import AutoclipError
from .events import Cancelled, CancelToken
from .internal import CHUNK_MINUTES
from .internal.automation import UNSILENCE_MIN_TOTAL_SECONDS
from .internal.quality import quality_profile
from .log import traced
from .media import AUDIO_SR, FRAME, atomic_write_json, external_env, ffmpeg_exe, frame_energy_db, probe
from .models import MediaInfo
from .resources import RESOURCES
from .speech import scan_audio

log = logging.getLogger("autoclip")

_SCHEMA = 1
_SCANNER_SCHEMA = 1
_STALE_PATCH = ".part.mp4"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _source_fingerprint(src: Path, cfg: Config) -> str:
    st = src.stat()
    relevant = {
        "source": (src.name, st.st_size, st.st_mtime_ns),
        "quality": cfg.quality,
        "audio_ready": not (cfg.skip.audio_enhance and cfg.skip.denoise),
        "audio": not cfg.skip.audio,
        "schema": _SCHEMA,
        "scanner": _SCANNER_SCHEMA,
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode("utf-8")).hexdigest()


def _wav_path(work: Path, index: int) -> Path:
    return work / f".optimizer-{os.getpid()}-{index:06d}.wav"


def _write_enhanced_wav(src: Path, start: float, end: float, threads: int, cfg: Config, target: Path) -> None:
    y = enhance_audio(src, start, end - start, threads, enhance=not cfg.skip.audio_enhance, denoise=not cfg.skip.denoise)
    pcm = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(target), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(AUDIO_SR)
        wav.writeframes(pcm.tobytes())


def _segment_command(src: Path, wav: Path | None, interval: tuple[float, float], info: MediaInfo, cfg: Config, dst: Path) -> list[str]:
    start, end = interval
    profile = quality_profile(cfg.quality)
    fps = max(1.0, min(info.fps, profile.fps_cap or info.fps)) if profile.fps_cap is not None else max(1.0, info.fps)
    command = [
        ffmpeg_exe(),
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{end - start:.3f}",
        "-i",
        str(src),
    ]
    if wav is None:
        command += ["-pix_fmt", "yuv420p", "-r", f"{fps:.6f}", "-c:v", "libx264", "-preset", profile.preset, "-crf", str(profile.crf)]
        if info.has_audio:
            command += ["-map", "0:v:0", "-map", "0:a:0", "-c:a", "aac", "-b:a", f"{profile.aac_bitrate_kbps}k"]
        else:
            command += ["-map", "0:v:0", "-an"]
    else:
        command += [
            "-i",
            str(wav),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            profile.preset,
            "-crf",
            str(profile.crf),
            "-pix_fmt",
            "yuv420p",
            "-r",
            f"{fps:.6f}",
            "-c:a",
            "aac",
            "-b:a",
            f"{profile.aac_bitrate_kbps}k",
            "-ar",
            str(AUDIO_SR),
            "-ac",
            "1",
        ]
    command += ["-avoid_negative_ts", "make_zero", str(dst)]
    return command


def _run(command: list[str], cancel: CancelToken | None = None, allow_failure: bool = False, cwd: Path | None = None) -> int:
    """Run one ffmpeg child, with in-process cancellation and bounded polling."""
    with tempfile.TemporaryFile() as stderr:
        proc = subprocess.Popen(command, env=external_env(), stdout=subprocess.DEVNULL, stderr=stderr, cwd=cwd)
        try:
            while proc.poll() is None:
                if cancel is not None:
                    try:
                        cancel.check()
                    except BaseException:
                        proc.terminate()
                        try:
                            proc.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=1.0)
                        raise
                time.sleep(0.05)
            rc = proc.wait() or 0
            if rc and not allow_failure:
                stderr.seek(0)
                tail = stderr.read().decode("utf-8", "replace").strip().splitlines()
                detail = tail[-1][:300] if tail else "ffmpeg failed"
                raise AutoclipError(f"ffmpeg failed while optimizing a video chunk ({detail})")
            return rc
        except BaseException:
            proc.terminate()
            time.sleep(0.05)
            raise


def _probe_duration(path: Path) -> float:
    return probe(path).duration


def _load_reused_assembly(src: Path, cfg: Config, work: Path, info: MediaInfo) -> tuple[Path, CutMap] | None:
    manifest = _read_json(work / "optimizer.json")
    if manifest is None or manifest.get("assembled") is not True or manifest.get("schema") != _SCHEMA:
        return None
    if manifest.get("source_fingerprint") != _source_fingerprint(src, cfg):
        return None
    try:
        cut = CutMap.from_intervals(info.duration, [(float(start), float(end)) for start, end in manifest["intervals"]])
    except (KeyError, TypeError, ValueError):
        return None
    if manifest.get("cut_fingerprint") != cut.fingerprint:
        return None
    target = work / str(manifest.get("path", ""))
    if target.name != f"optimized_{src.stem}.mp4" or not target.is_file() or target.stat().st_size <= 0:
        return None
    return target, cut


@dataclass(frozen=True, slots=True)
class PreparedSource:
    """A source ready for the rest of the per-video pipeline."""

    path: Path
    info: MediaInfo
    cutmap: CutMap
    energy_db: Any
    audio_ready: bool
    stt_failed: bool = False  # the fused Whisper pass failed; the run degrades to audio-only selection


class SttPass(Protocol):
    """The slice of a resident Whisper session the assembly drives (avoids an optimizer→transcript import)."""

    failed: bool

    def transcribe_chunk(self, k: int, pieces: Sequence[tuple[float, float, float]], cancel: CancelToken | None = None) -> None: ...
    def close(self) -> None: ...


@traced("prepare_source")
def prepare_source(
    src: Path,
    cfg: Config,
    info: MediaInfo,
    work: Path,
    cancel: CancelToken | None = None,
    stt_factory: Callable[[CutMap], SttPass | None] | None = None,
) -> PreparedSource:
    """Detect speech once, keep it when it is worthwhile, and reuse the optimized file on a clean re-run.

    ``stt_factory`` optionally arms transcription inside the assembly: it is called with the cut map
    after the speech scan, once the cheaper transcript sources (sidecar, subtitle, disk cache) have
    been ruled out. Callers that only need the optimized video pass nothing and never load Whisper.
    """
    reused = _load_reused_assembly(src, cfg, work, info)
    if reused is not None:
        target, cut = reused
        prepared_info = probe(target)
        zeros = np.zeros(int(prepared_info.duration / FRAME) + 1, dtype=np.float32)
        energy_db = frame_energy_db(target) if prepared_info.has_audio else zeros
        return PreparedSource(target, prepared_info, cut, energy_db, audio_ready=True)
    scan = scan_audio(src, info, cancel=cancel)
    removable = max(0.0, info.duration - (sum(end - start for start, end in scan.intervals) if scan.intervals else 0.0))
    threshold = max(UNSILENCE_MIN_TOTAL_SECONDS, 0.02 * info.duration)
    if not scan.intervals or removable < threshold:
        return PreparedSource(src, info, CutMap.identity(info.duration), scan.energy_db, audio_ready=False)
    cut = CutMap.from_intervals(info.duration, scan.intervals)
    stt_failed = False
    stt: SttPass | None = None
    if stt_factory is not None:
        try:
            stt = stt_factory(cut)
        except Exception as e:  # arming is an optimization: never abort the assembly over it
            log.warning("could not arm transcription (%s: %s); selecting on audio only", type(e).__name__, e)
            stt_failed = True
    try:
        target = assemble_optimized(src, info, cfg, work, cut, cancel=cancel, stt=stt)
    finally:
        if stt is not None:
            stt.close()
    prepared_info = probe(target)
    zeros = np.zeros(int(prepared_info.duration / FRAME) + 1, dtype=np.float32)
    energy_db = frame_energy_db(target) if prepared_info.has_audio else zeros
    return PreparedSource(target, prepared_info, cut, energy_db, audio_ready=True, stt_failed=stt_failed or bool(stt and stt.failed))


@traced("assemble_optimized")
def assemble_optimized(
    src: Path,
    info: MediaInfo,
    cfg: Config,
    work: Path,
    cut: CutMap,
    cancel: CancelToken | None = None,
    stt: SttPass | None = None,
) -> Path:
    """Encode the kept intervals, transcribe each finished chunk, and concatenate into ``optimized_<stem>.mp4``."""
    work.mkdir(parents=True, exist_ok=True)
    manifest_path = work / "optimizer.json"
    segment_root = work / "optimizer"
    manifest = _read_json(manifest_path)
    assembly_key = _source_fingerprint(src, cfg)
    if (
        manifest is None
        or manifest.get("source_fingerprint") != assembly_key
        or manifest.get("schema") != _SCHEMA
        or manifest.get("cut_fingerprint") != cut.fingerprint
    ):
        shutil.rmtree(segment_root, ignore_errors=True)
        manifest_path.unlink(missing_ok=True)
        manifest = {
            "schema": _SCHEMA,
            "source_fingerprint": assembly_key,
            "scanner": _SCANNER_SCHEMA,
            "cut_fingerprint": cut.fingerprint,
            "chunks": [],
        }
        atomic_write_json(manifest_path, manifest)

    segment_root.mkdir(parents=True, exist_ok=True)
    chunks = cut.chunked_pieces(CHUNK_MINUTES * 60.0)
    pieces = [piece for chunk in chunks for piece in chunk]
    expected_duration = cut.total_kept
    done: dict[int, dict[str, Any]] = {}
    for chunk in manifest.get("chunks", []):
        try:
            index = int(chunk["index"])
            done[index] = chunk
        except (KeyError, TypeError, ValueError):
            continue
    base = 0
    for chunk_number, chunk_pieces in enumerate(chunks, 1):
        for offset, (source_start, source_end, optimized_start) in enumerate(chunk_pieces):
            index = base + offset  # flat index: matches the manifest's piece numbering
            interval = (source_start, source_end)
            relative = f"optimizer/{index:06d}.mp4"
            target = work / relative
            meta = done.get(index)
            if (
                meta is not None
                and meta.get("committed")
                and target.is_file()
                and target.stat().st_size > 0
                and abs(float(meta.get("source_start", -1.0)) - interval[0]) < 1e-6
                and abs(float(meta.get("source_end", -1.0)) - interval[1]) < 1e-6
            ):
                continue
            part = target.with_suffix(_STALE_PATCH)
            part.unlink(missing_ok=True)
            wav: Path | None = None
            try:
                if cfg.skip.audio or not info.has_audio or (cfg.skip.audio_enhance and cfg.skip.denoise):
                    _run(_segment_command(src, None, interval, info, cfg, part), cancel=cancel)
                else:
                    wav = _wav_path(work, index)
                    _write_enhanced_wav(src, source_start, source_end, RESOURCES.encoding_thread_budget(), cfg, wav)
                    _run(_segment_command(src, wav, interval, info, cfg, part), cancel=cancel)
                os.replace(part, target)
                done[index] = {
                    "index": index,
                    "source_start": interval[0],
                    "source_end": interval[1],
                    "optimized_start": optimized_start,
                    "committed": True,
                }
                payload = {
                    "schema": _SCHEMA,
                    "source_fingerprint": assembly_key,
                    "scanner": _SCANNER_SCHEMA,
                    "cut_fingerprint": cut.fingerprint,
                    "chunks": [done[i] for i in sorted(done)],
                }
                atomic_write_json(manifest_path, payload)
            finally:
                if wav is not None:
                    wav.unlink(missing_ok=True)
                part.unlink(missing_ok=True)
        base += len(chunk_pieces)
        if stt is not None and not stt.failed:
            # Outside the per-piece resume check: a chunk whose pieces were already committed
            # still needs its transcript row, e.g. after a crash between encode and STT.
            # After a failure we stop transcribing so `done` stays a contiguous prefix.
            if cancel is not None:
                cancel.check()
            try:
                stt.transcribe_chunk(chunk_number, chunk_pieces, cancel=cancel)
            except Cancelled:
                raise
            except Exception as e:
                log.warning("transcription skipped (%s: %s): selecting on audio only", type(e).__name__, e)
                stt.failed = True

    list_file = segment_root / "list.txt"
    # Fixed names, relative paths, and cwd below make the manifest immune to hostile source names.
    list_file.write_text("".join(f"file '{index:06d}.mp4'\n" for index in range(len(pieces))), encoding="utf-8")
    target = work / f"optimized_{src.stem}.mp4"
    tmp = target.with_suffix(".tmp.mp4")
    rc = _run(
        [
            ffmpeg_exe(),
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-fflags",
            "+genpts",
            "-i",
            list_file.name,
            "-c",
            "copy",
            "-y",
            str(tmp),
        ],
        cwd=segment_root,
        cancel=cancel,
        allow_failure=True,
    )
    tolerance = max(0.25, 2.0 / max(info.fps, 1.0))
    if rc != 0 or not tmp.exists() or abs(_probe_duration(tmp) - expected_duration) > tolerance:
        tmp.unlink(missing_ok=True)
        rc = _run(
            [
                ffmpeg_exe(),
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_file.name,
                "-c:v",
                "libx264",
                "-preset",
                quality_profile(cfg.quality).preset,
                "-crf",
                str(quality_profile(cfg.quality).crf),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                f"{quality_profile(cfg.quality).aac_bitrate_kbps}k",
                "-y",
                str(tmp),
            ],
            cancel=cancel,
            allow_failure=True,
            cwd=segment_root,
        )
        if rc != 0 or not tmp.exists():
            tmp.unlink(missing_ok=True)
            raise AutoclipError("cannot concatenate the optimized video chunks")
    os.replace(tmp, target)
    for pattern in segment_root.glob("*.mp4"):
        pattern.unlink()
    shutil.rmtree(segment_root, ignore_errors=True)
    manifest = {
        "schema": _SCHEMA,
        "source_fingerprint": assembly_key,
        "scanner": _SCANNER_SCHEMA,
        "cut_fingerprint": cut.fingerprint,
        "intervals": [list(interval) for interval in cut.intervals],
        "path": target.name,
        "assembled": True,
    }
    atomic_write_json(manifest_path, manifest)
    return target
