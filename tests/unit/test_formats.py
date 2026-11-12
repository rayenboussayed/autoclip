"""Format choice: closest short-side quality, protocol, codec, bitrate and audio language."""

from __future__ import annotations

from typing import Any, cast

import pytest

from autoclip.core.errors import DownloadError
from autoclip.core.formats import Choice, DownloadPrefs, YtInfo, choose


def video(
    format_id: str,
    width: int | None,
    height: int | None,
    *,
    vcodec: str = "avc1.640028",
    acodec: str = "none",
    fps: float | None = 30,
    tbr: float | None = 1000,
    ext: str = "mp4",
    **extra: Any,
) -> dict[str, Any]:
    """One synthetic yt-dlp video format."""
    return {
        "format_id": format_id,
        "width": width,
        "height": height,
        "vcodec": vcodec,
        "acodec": acodec,
        "fps": fps,
        "tbr": tbr,
        "ext": ext,
        **extra,
    }


def audio(
    format_id: str,
    *,
    abr: float | None = 128,
    ext: str = "m4a",
    acodec: str = "mp4a.40.2",
    language: str | None = None,
) -> dict[str, Any]:
    """One synthetic yt-dlp audio-only format."""
    fmt: dict[str, Any] = {"format_id": format_id, "vcodec": "none", "acodec": acodec, "ext": ext, "tbr": abr}
    if abr is not None:
        fmt["abr"] = abr
    if language:
        fmt["language"] = language
    return fmt


def pick(formats: list[dict[str, Any]], quality: int = 1080, **prefs: Any) -> Choice:
    """Run the chooser over a format list with a derived quality target."""
    return choose(cast(YtInfo, {"formats": formats}), DownloadPrefs(quality=quality, **prefs))


def test_closest_quality_uses_the_short_side() -> None:
    formats = [video("v720", 1280, 720), video("v1080", 1920, 1080), video("v1440", 2560, 1440)]
    assert pick(formats, quality=1080).format_id == "v1080"
    assert pick([video("vertical", 1080, 1920)], quality=1080).format_id == "vertical"  # a vertical 1080 is "1080"


def test_closest_quality_prefers_the_lower_format_on_a_tie() -> None:
    assert pick([video("v900", 1600, 900), video("v1100", 1920, 1100)], quality=1000).format_id == "v900"


def test_unknown_resolution_loses_to_a_known_one() -> None:
    formats = [video("unknown", None, None), video("v720", 1280, 720)]
    assert pick(formats, quality=1440).format_id == "v720"
    assert pick(formats, quality=360).format_id == "v720"


def test_width_only_is_a_usable_quality() -> None:
    assert pick([video("wide", 1920, None)]).format_id == "wide"


def test_codec_order_is_h264_then_vp9_then_av1_then_h265() -> None:
    formats = [
        video("av1", 1920, 1080, vcodec="av01.0.08M.08"),
        video("h265", 1920, 1080, vcodec="hev1.1.6.L93.B0"),
        video("vp9", 1920, 1080, vcodec="vp09.00.10.08"),
        video("h264", 1920, 1080, vcodec="avc1.640028"),
    ]
    assert pick(formats).format_id == "h264"
    assert pick([f for f in formats if f["format_id"] != "h264"]).format_id == "vp9"
    assert pick([f for f in formats if f["format_id"] in ("av1", "h265")]).format_id == "av1"


def test_a_higher_bitrate_breaks_a_codec_tie() -> None:
    formats = [video("low", 1920, 1080, tbr=1000), video("high", 1920, 1080, tbr=2000)]
    assert pick(formats).format_id == "high"


def test_audio_picks_the_highest_bitrate() -> None:
    formats = [video("v", 1920, 1080), audio("a128", abr=128), audio("a256", abr=256, ext="webm")]
    assert pick(formats).format_id == "v+a256"


def test_audio_language_preference_and_fallback() -> None:
    formats = [
        video("v", 1920, 1080),
        audio("en", abr=128, language="en-US"),
        audio("fr", abr=256, language="fr"),
    ]
    assert pick(formats, audio_language="en").format_id == "v+en"
    fallback = pick(formats, audio_language="de")
    assert fallback.format_id == "v+fr" and "no audio track in language 'de'" in fallback.warnings[0]


def test_the_original_language_beats_a_higher_bitrate_dub() -> None:
    formats = [
        video("v", 1920, 1080),
        audio("en", abr=128, language="en"),
        audio("ms", abr=142, language="ms"),
    ]
    choice = choose(cast(YtInfo, {"formats": formats, "language": "en"}), DownloadPrefs(quality=1080))
    assert choice.format_id == "v+en"
    assert choice.warnings == ()
    # An explicit request still wins over the original language.
    assert choose(cast(YtInfo, {"formats": formats, "language": "en"}), DownloadPrefs(1080, "ms")).format_id == "v+ms"


def test_an_unknown_original_language_falls_back_without_a_warning() -> None:
    formats = [
        video("v", 1920, 1080),
        audio("en", abr=128, language="en"),
        audio("fr", abr=256, language="fr"),
    ]
    choice = choose(cast(YtInfo, {"formats": formats, "language": "ja"}), DownloadPrefs(quality=1080))
    assert choice.format_id == "v+fr" and choice.warnings == ()


def test_a_direct_format_beats_a_segmented_one_at_equal_quality() -> None:
    formats = [
        video("hls", 1920, 1080, protocol="m3u8_native", tbr=2511),
        video("direct", 1920, 1080, protocol="https", tbr=1131),
        video("dash", 1920, 1080, protocol="http_dash_segments", tbr=3000),
        video("frag", 1920, 1080, protocol="websocket_frag", tbr=3000),
    ]
    assert pick(formats, quality=1080).format_id == "direct"
    # The protocol only breaks a tie: a better resolution still wins.
    formats.append(video("8k-hls", 7680, 4320, protocol="m3u8_native"))
    assert pick(formats, quality=4320).format_id == "8k-hls"


def test_audio_bitrate_falls_back_to_tbr() -> None:
    low = {"format_id": "low", "vcodec": "none", "acodec": "mp4a", "ext": "m4a", "tbr": 64}
    high = {"format_id": "high", "vcodec": "none", "acodec": "mp4a", "ext": "m4a", "tbr": 192}
    assert pick([video("v", 1920, 1080), low, high]).format_id == "v+high"


def test_a_progressive_format_is_not_merged() -> None:
    progressive = video("18", 640, 360, acodec="mp4a.40.2")
    assert pick([progressive], quality=360).format_id == "18"


def test_video_without_any_audio_is_reported() -> None:
    choice = pick([video("v", 1920, 1080)])
    assert choice.format_id == "v" and "no audio track" in choice.warnings[0]


def test_audio_only_info_is_accepted() -> None:
    choice = pick([audio("140", language="fr")], audio_language="en")
    assert choice.format_id == "140" and "no audio track in language 'en'" in choice.warnings[0]
    assert not choice.has_video  # the pipeline needs a video stream and must reject this download


def test_a_video_choice_reports_that_it_has_video() -> None:
    assert pick([video("v", 1920, 1080)]).has_video
    assert pick([video("v", 1280, 720, acodec="mp4a.40.2")]).has_video


def test_drm_and_storyboard_formats_are_ignored() -> None:
    formats = [
        video("drm", 3840, 2160, has_drm=True),
        video("story", 160, 90, vcodec="none", ext="mhtml"),
        video("v720", 1280, 720),
    ]
    assert pick(formats, quality=2160).format_id == "v720"


def test_nothing_downloadable_is_an_error() -> None:
    with pytest.raises(DownloadError, match="no downloadable video or audio format"):
        pick([])
    with pytest.raises(DownloadError, match="no downloadable video or audio format"):
        pick([video("drm", 1920, 1080, has_drm=True)])


def test_a_direct_media_format_without_codec_or_size_is_usable() -> None:
    """The generic extractor reports a direct MP4 with ``vcodec``/``height`` set to null."""
    direct = {"format_id": "0", "ext": "mp4", "vcodec": None, "acodec": None, "height": None}
    choice = pick([direct])
    assert choice.format_id == "0" and choice.warnings == ()
    assert pick([direct, video("v720", 1280, 720)], quality=720).format_id == "v720"
