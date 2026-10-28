"""GPU / CPU encoder selection."""

from __future__ import annotations

import logging
import subprocess

import pytest

from autoclip.core import hardware
from autoclip.core.config import VideoCfg
from autoclip.core.hardware import HARDWARE, PLACEHOLDER_PRESET, PRIORITY, cpu_encoder, pick_encoder, works
from autoclip.core.models import Encoder


@pytest.fixture(autouse=True)
def _fresh_probe_cache() -> None:
    works.cache_clear()


def test_cpu_encoder_arguments() -> None:
    enc = cpu_encoder(VideoCfg(crf=18, preset="slow", maxrate_mbps=8))
    assert (enc.name, enc.hardware, enc.preset) == ("libx264", False, "slow")
    assert enc.params == ("-crf", "18", "-profile:v", "high", "-maxrate", "8M", "-bufsize", "16M")
    assert "-maxrate" not in cpu_encoder(VideoCfg(maxrate_mbps=0)).params


@pytest.mark.parametrize(
    "name, codec", [("nvenc", "h264_nvenc"), ("videotoolbox", "h264_videotoolbox"), ("qsv", "h264_qsv"), ("amf", "h264_amf")]
)
def test_every_gpu_encoder_is_described(name: str, codec: str) -> None:
    enc = HARDWARE[name](VideoCfg(crf=21, maxrate_mbps=0))
    assert enc.name == codec and enc.hardware and enc.preset == PLACEHOLDER_PRESET  # real GPU presets are set via params
    assert "-crf" not in enc.params and "libx264" not in enc.name
    assert ("21" in enc.params) == (name != "videotoolbox")  # VideoToolbox has no quality scale: it is bitrate driven
    assert HARDWARE["videotoolbox"](VideoCfg(maxrate_mbps=6)).params[:2] == ("-b:v", "6M")
    assert "-maxrate" in HARDWARE["nvenc"](VideoCfg(maxrate_mbps=6)).params


def test_probe_tells_working_from_broken_encoders_with_real_ffmpeg() -> None:
    assert works(cpu_encoder(VideoCfg(preset="ultrafast")))
    assert not works(Encoder("h264_does_not_exist", True, "medium", ()))
    assert not works(Encoder("libx264", False, "no-such-preset", ()))
    # MoviePy always passes a preset first; a later "-preset" in the arguments wins, so this placeholder is harmless
    assert works(Encoder("libx264", True, "no-such-preset", ("-preset", "ultrafast")))
    assert works(Encoder("mpeg4", True, "medium", ("-b:v", "1M")))  # an encoder without presets just ignores the placeholder


def test_probe_uses_the_ffmpeg_moviepy_encodes_with(monkeypatch: pytest.MonkeyPatch) -> None:
    """Another build can accept options the real one rejects: probe the binary MoviePy writes with, not PATH's."""
    from moviepy.video.io import ffmpeg_writer

    assert hardware.encoding_ffmpeg() == ffmpeg_writer.FFMPEG_BINARY
    seen: list[str] = []

    def record(cmd: list[str], **_kw: object) -> subprocess.CompletedProcess[bytes]:
        seen.append(cmd[0])
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", record)
    assert works(cpu_encoder(VideoCfg(preset="ultrafast"))) and seen == [ffmpeg_writer.FFMPEG_BINARY]


def test_probe_survives_hangs_and_missing_binaries(monkeypatch: pytest.MonkeyPatch) -> None:
    def hang(*_a: object, **_k: object) -> None:
        raise subprocess.TimeoutExpired("ffmpeg", 60)

    def missing(*_a: object, **_k: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", hang)
    assert not works(Encoder("a", True, "medium", ()))
    monkeypatch.setattr(subprocess, "run", missing)
    assert not works(Encoder("b", True, "medium", ()))


def test_what_is_listed_is_not_what_is_chosen() -> None:
    """On any machine: the chosen encoder is a GPU one exactly when its probe encode succeeds."""
    for name in HARDWARE:
        chosen = pick_encoder(VideoCfg(gpu=name))  # type: ignore[arg-type]
        assert chosen.hardware == works(HARDWARE[name](VideoCfg())) and (chosen.hardware or chosen.name == "libx264")


def fake_probe(monkeypatch: pytest.MonkeyPatch, working: set[str]) -> list[str]:
    tried: list[str] = []

    def probe(enc: Encoder) -> bool:
        tried.append(enc.name)
        return enc.name in working

    monkeypatch.setattr(hardware, "works", probe)
    return tried


@pytest.mark.parametrize(
    "system, working, expected, tried",
    [
        ("Windows", {"h264_qsv", "h264_amf"}, "h264_qsv", ["h264_nvenc", "h264_qsv"]),  # NVIDIA first, then Intel, then AMD
        ("Windows", {"h264_nvenc"}, "h264_nvenc", ["h264_nvenc"]),
        ("Darwin", {"h264_videotoolbox", "h264_nvenc"}, "h264_videotoolbox", ["h264_videotoolbox"]),
        ("Linux", {"h264_qsv"}, "h264_qsv", ["h264_nvenc", "h264_qsv"]),
        ("Linux", set(), "libx264", ["h264_nvenc", "h264_qsv"]),
        ("Plan9", {"h264_nvenc"}, "libx264", []),  # unknown platform: nothing to try
    ],
)
def test_auto_follows_the_platform_priority(
    monkeypatch: pytest.MonkeyPatch, system: str, working: set[str], expected: str, tried: list[str]
) -> None:
    log = fake_probe(monkeypatch, working)
    assert pick_encoder(VideoCfg(gpu="auto"), system).name == expected and log == tried
    assert PRIORITY["Darwin"] == ("videotoolbox",)


def test_off_never_probes_and_explicit_choice_warns_when_unusable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    tried = fake_probe(monkeypatch, {"h264_amf"})
    assert pick_encoder(VideoCfg(gpu="off")).name == "libx264" and tried == []
    assert pick_encoder(VideoCfg(gpu="amf")).name == "h264_amf" and tried == ["h264_amf"]
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        assert pick_encoder(VideoCfg(gpu="nvenc")).name == "libx264"
    assert "not usable on this machine" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="autoclip"):
        pick_encoder(VideoCfg(gpu="auto"), "Linux")  # nothing works on "auto": silently the CPU (not a warning)
    assert caplog.text == ""
