"""Configuration: strict keys, typed values, validation, multi-video input expansion."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from autoclip.core.config import ClipsCfg, Config, KindCfg, expand_inputs, is_url, load_config, parse, parse_hms
from autoclip.core.errors import ConfigError


def write(tmp: Path, data: object, name: str = "config.json") -> Path:
    p = tmp / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def touch(tmp: Path, *names: str) -> None:
    for n in names:
        (tmp / n).parent.mkdir(parents=True, exist_ok=True)
        (tmp / n).write_bytes(b"x")


def test_defaults_roundtrip_through_json() -> None:
    """``--init-full`` output must load back into the identical defaults."""
    assert parse(Config, json.loads(json.dumps(asdict(Config())))) == Config()


def test_parse_accepts_comments_and_int_for_float() -> None:
    cfg = parse(Config, {"_comment": "x", "clips": {"short": {"min_seconds": 5, "max_seconds": 9}, "_why": 1}})
    assert cfg.clips.short.min_seconds == 5


@pytest.mark.parametrize(
    "data, fragment",
    [
        ({"cout": 1}, "cout: Unexpected keyword argument"),
        ({"clips": {"cout": 1}}, "clips.cout: Unexpected keyword argument"),
        ({"clips": {"short": {"count": "5"}}}, "clips.short.count: Input should be a valid integer"),
        ({"clips": {"short": {"count": True}}}, "clips.short.count: Input should be a valid integer"),
        ({"clips": {"short": {"min_seconds": True}}}, "clips.short.min_seconds: Input should be a valid number"),
        ({"clips": 3}, "clips: Input should be a dictionary"),
        ({"clips": {"mode": "nope"}}, "clips.mode: Input should be"),
        ({"clips": {"kinds": ["medium"]}}, "clips.kinds.0: Input should be"),
        ({"clips": {"keywords": ["a", 1]}}, "clips.keywords.1: Input should be a valid string"),
        ({"clips": {"lexicon_file": 5}}, "clips.lexicon_file: Input should be a valid string"),
        ({"silence": {"threshold_db": "loud"}}, "silence.threshold_db"),
        ({"clips": {"scenes": "nope"}}, "unknown clips.scenes"),
        ({"clips": {"scenes": ["best", "nope"]}}, "unknown clips.scenes"),
        ({"clips": {"kinds": []}}, "clips.kinds"),
        ({"clips": {"kinds": ["short", "short"]}}, "clips.kinds"),
        ({"clips": {"split_seconds": 0}}, "clips.split_seconds: Input should be greater than 0"),
        ({"clips": {"short": {"min_seconds": 50, "max_seconds": 10}}}, "min_seconds must be <= max_seconds"),
        ({"clips": {"short": {"target_seconds": 5}}}, "target_seconds must be between"),
        ({"clips": {"short": {"min_seconds": 5, "max_seconds": 10, "target_seconds": 20}}}, "target_seconds must be between"),
        ({"clips": {"long": {"count": 0}}}, "clips.long.count: Input should be greater than or equal to 1"),
        ({"clips": {"short": {"ranges": ["bad"]}}}, "clips.short.ranges.0: String should match pattern"),
        ({"clips": {"long": {"aspect": "wide"}}}, "clips.long.aspect: String should match pattern"),
        ({"llm": {"base_url": "file:///etc/passwd"}}, "llm.base_url: String should match pattern"),
        ({"video": {"aspect": "9:16"}}, "video.aspect: Unexpected keyword argument"),
        ({"video": {"gpu": "cuda"}}, "video.gpu: Input should be"),
        ({"processing": {"chunk_minutes": 0}}, "processing.chunk_minutes: Input should be greater than 0"),
        ({"transcript": {"enhance": {"backend": "nope"}}}, "transcript.enhance.backend: Input should be"),
        ({"transcript": {"enhance": {"model": "nope"}}}, "transcript.enhance.model: Input should be"),
        ({"transcript": {"enhance": {"attenuation_limit_db": 200}}}, "transcript.enhance.attenuation_limit_db:"),
        ({"transcript": {"enhance": {"denoise_strength": 2}}}, "transcript.enhance.denoise_strength:"),
        ({"transcript": {"enhance": {"num_threads": 0}}}, "transcript.enhance.num_threads:"),
        ({"transcript": {"enhance": {"snr_threshold_db": 0}}}, "transcript.enhance.snr_threshold_db:"),
        ({"compute": {"device": "tpu"}}, "compute.device: Input should be"),
        ({"download": {"quality": 100}}, "download.quality.constrained-int: Input should be greater than or equal to 144"),
        ({"download": {"quality": 5000}}, "download.quality.constrained-int: Input should be less than or equal to 4320"),
        ({"download": {"quality": "hd"}}, "download.quality.literal"),
        ({"download": {"fps": 0}}, "download.fps: Input should be greater than 0"),
        ({"download": {"prefer": "nearest"}}, "download.prefer: Input should be"),
        ({"download": {"codec": "mpeg2"}}, "download.codec: Input should be"),
        ({"download": {"container": "avi"}}, "download.container: Input should be"),
        ({"download": {"audio_quality": "medium"}}, "download.audio_quality: Input should be"),
        ({"download": {"playlist": "some"}}, "download.playlist: Input should be"),
        ({"download": {"max_videos": 0}}, "download.max_videos: Input should be greater than or equal to 1"),
        ({"download": {"subtitles": {"enabled": True, "languages": []}}}, "languages must not be empty"),
        ({"download": {"subtitles": {"use_as_transcript": "always"}}}, "needs download.subtitles.enabled"),
        ({"download": {"subtitles": {"use_as_transcript": "sometimes"}}}, "download.subtitles.use_as_transcript: Input should be"),
        ({"download": {"cookies": {"from_browser": "chrome", "file": "c.txt"}}}, "set either from_browser or file"),
        ({"download": {"network": {"proxy": "ftp://host"}}}, "download.network.proxy: String should match pattern"),
        ({"download": {"network": {"rate_limit_mbps": 0}}}, "download.network.rate_limit_mbps: Input should be greater than 0"),
        ({"download": {"network": {"retries": 21}}}, "download.network.retries: Input should be less than or equal to 20"),
        ({"download": {"network": {"socket_timeout": 0}}}, "download.network.socket_timeout: Input should be greater than 0"),
        ({"download": {"network": {"sleep_between_seconds": -1}}}, "download.network.sleep_between_seconds:"),
        ({"download": {"js_runtime": {"name": "auto", "path": "deno"}}}, "needs download.js_runtime.name"),
        ({"download": {"js_runtime": {"name": "v8"}}}, "download.js_runtime.name: Input should be"),
        ({"download": {"js_runtime": {"remote_components": "all"}}}, "download.js_runtime.remote_components: Input should be"),
        ({"download": {"time_range": "a-b"}}, "download.time_range: String should match pattern"),
        ({"download": {"time_range": "00:30-00:10"}}, "must end after it starts"),
        ({"download": {"time_range": "1::2-2:00"}}, "not a valid time range"),
        ({"download": {"metadata": {"embed": True}}}, "download.metadata.embed: Unexpected keyword"),
        ({"download": {"nope": 1}}, "download.nope: Unexpected keyword"),
        ({"skip": {"stt": "yes"}}, "skip.stt:"),
        ({"skip": {"everything": True}}, "skip.everything: Unexpected keyword"),
    ],
)
def test_invalid_config_is_rejected(data: dict[str, object], fragment: str) -> None:
    with pytest.raises(ConfigError, match=fragment):
        parse(Config, data)


def test_scene_types_all_one_or_a_list() -> None:
    every = ClipsCfg(scenes="all").scene_list()
    assert len(every) == 10 and "funny" in every and "best" in every
    assert ClipsCfg(scenes="calm").scene_list() == ["calm"]
    assert ClipsCfg(scenes=["funny", "sad"]).scene_list() == ["funny", "sad"]
    assert ClipsCfg().scene_list() == ["best"]


def test_short_and_long_kinds_have_their_own_defaults() -> None:
    cc = Config().clips
    assert cc.kinds == ["short", "long"]
    assert (cc.short.aspect, cc.short.min_seconds, cc.short.max_seconds) == ("9:16", 20, 60)
    assert (cc.long.aspect, cc.long.min_seconds, cc.long.max_seconds) == ("16:9", 180, 600)
    assert cc.short.target_seconds is None and cc.long.target_seconds is None  # no recommendation by default
    assert cc.kind("short") is cc.short and cc.kind("long") is cc.long
    custom = parse(Config, {"clips": {"kinds": ["long"], "long": {"aspect": "original", "ranges": ["0:10-0:50"]}}})
    assert custom.clips.long.aspect == "original" and custom.clips.long.ranges == ["0:10-0:50"] and custom.clips.short.count == 5


def test_the_recommended_length_is_optional_and_must_fit_the_range() -> None:
    kind = parse(Config, {"clips": {"short": {"min_seconds": 10, "max_seconds": 60, "target_seconds": 30}}}).clips.short
    assert kind.target_seconds == 30
    assert parse(Config, {"clips": {"short": {"target_seconds": None}}}).clips.short.target_seconds is None


def test_valid_unions_and_lists() -> None:
    cfg = parse(Config, {"clips": {"scenes": ["best", "funny"]}, "silence": {"threshold_db": -40}, "video": {"fps": None, "gpu": "nvenc"}})
    assert cfg.clips.scene_list() == ["best", "funny"] and KindCfg().aspect == "9:16"
    assert cfg.silence.threshold_db == -40 and cfg.video.fps is None and cfg.video.gpu == "nvenc"


SKIP_KEYS = frozenset({"stt", "denoise", "audio_enhance", "audio", "pauses", "caption", "title", "fade", "thumbnail", "llm", "snap"})


def test_nothing_is_skipped_by_default() -> None:
    """Every step runs unless the user asks otherwise: all eleven switches exist and are false."""
    skip = asdict(Config().skip)
    assert skip.keys() == SKIP_KEYS
    assert not any(skip.values())


def test_skip_switches_parse_and_reject_a_contradiction() -> None:
    cfg = parse(Config, {"skip": {"caption": True, "thumbnail": True}})
    assert cfg.skip.caption and cfg.skip.thumbnail and not cfg.skip.stt
    with pytest.raises(ConfigError, match="skip.stt cannot be combined with transcript.mode: 'whisper'"):
        parse(Config, {"skip": {"stt": True}, "transcript": {"mode": "whisper"}})


def test_expand_inputs_globs_dedupes_and_sorts(tmp_path: Path) -> None:
    touch(tmp_path, "v/b.mp4", "v/a.mp4", "c.mp4")
    (tmp_path / "v" / "folder.mp4").mkdir()
    found = expand_inputs(["v/*.mp4", "v/a.mp4", "c.mp4"], tmp_path)
    assert [Path(p).name for p in found] == ["a.mp4", "b.mp4", "c.mp4"]
    assert all(Path(p).is_absolute() for p in found)


def test_expand_inputs_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="matches no file"):
        expand_inputs(["missing/*.mp4"], tmp_path)
    touch(tmp_path, "x/same.mp4", "y/same.mp4")
    with pytest.raises(ConfigError, match="share the same file name"):
        expand_inputs(["x/same.mp4", "y/same.mp4"], tmp_path)


def test_load_config_resolves_everything_relative_to_the_file(tmp_path: Path) -> None:
    touch(tmp_path, "vids/one.mp4", "vids/two.mp4", "bg.png", "lex.json")
    cfg = load_config(
        write(
            tmp_path,
            {
                "input": "vids/*.mp4",
                "output_dir": "out",
                "logging": {"file": "run.log"},
                "transcript": {"file": "subs/{stem}.srt", "enhance": {"model_dir": "models"}},
                "clips": {"lexicon_file": "lex.json"},
                "video": {"background": {"image": "bg.png"}},
            },
        )
    )
    assert [Path(p).name for p in cfg.input] == ["one.mp4", "two.mp4"]
    for value in (
        cfg.output_dir,
        cfg.logging.file,
        cfg.transcript.file,
        cfg.clips.lexicon_file,
        cfg.video.background.image,
        cfg.transcript.enhance.model_dir,
    ):
        assert value and Path(value).is_absolute() and str(tmp_path.resolve()) in str(Path(value).resolve())


def test_single_string_input_and_single_transcript_file(tmp_path: Path) -> None:
    touch(tmp_path, "one.mp4")
    cfg = load_config(write(tmp_path, {"input": "one.mp4", "transcript": {"file": "a.srt"}}))
    assert len(cfg.input) == 1 and cfg.transcript.file and cfg.transcript.file.endswith("a.srt")


def test_several_videos_need_a_stem_placeholder(tmp_path: Path) -> None:
    touch(tmp_path, "one.mp4", "two.mp4")
    with pytest.raises(ConfigError, match=r"\{stem\}"):
        load_config(write(tmp_path, {"input": ["one.mp4", "two.mp4"], "transcript": {"file": "a.srt"}}))


def test_load_config_file_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read config"):
        load_config(tmp_path / "missing.json")
    (tmp_path / "bad.json").write_text("{nope", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config(tmp_path / "bad.json")
    with pytest.raises(ConfigError, match="JSON object"):
        load_config(write(tmp_path, [1, 2]))


def test_config_without_optional_files(tmp_path: Path) -> None:
    touch(tmp_path, "one.mp4")
    cfg = load_config(write(tmp_path, {"input": "one.mp4"}))
    assert (
        cfg.transcript.file is None and cfg.logging.file is None and cfg.clips.lexicon_file is None and cfg.video.background.image is None
    )


def test_parse_hms_accepts_every_shape() -> None:
    assert parse_hms("30") == 30
    assert parse_hms("1:30") == 90
    assert parse_hms("01:02:03.5") == 3723.5
    with pytest.raises(ValueError):
        parse_hms("nope")


@pytest.mark.parametrize("value", ["https://youtu.be/dQw4w9WgXcQ", "HTTP://example.com/a?b=1", "http://localhost:8000/v"])
def test_is_url_accepts_http_and_https(value: str) -> None:
    assert is_url(value)


@pytest.mark.parametrize(
    "value", ["C:\\videos\\a.mp4", "videos/a.mp4", "https://", "ftp://host/a", "file:///tmp/a.mp4", "", "https://[::1"]
)
def test_is_url_rejects_paths_and_other_schemes(value: str) -> None:
    assert not is_url(value)


def test_expand_inputs_keeps_urls_verbatim_and_dedupes(tmp_path: Path) -> None:
    touch(tmp_path, "a.mp4")
    url = "https://example.com/watch?v=1"
    found = expand_inputs([url, "a.mp4", url], tmp_path)
    assert found[0] == url and len(found) == 2
    assert Path(found[1]).is_absolute()


def test_expand_inputs_missing_local_pattern_still_errors_with_urls(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="matches no file or URL"):
        expand_inputs(["https://example.com/v", "missing/*.mp4"], tmp_path)


def test_download_defaults() -> None:
    d = Config().download
    assert (d.quality, d.fps, d.prefer, d.codec, d.container) == (1080, None, "closest", "auto", "auto")
    assert (d.audio_quality, d.audio_language, d.original_dir) == ("best", None, None)
    assert d.metadata.info_json and not d.metadata.thumbnail and not d.metadata.description
    assert not d.subtitles.enabled and d.subtitles.languages == ["en"] and d.subtitles.use_as_transcript == "never"
    assert (d.playlist, d.max_videos, d.time_range) == ("single", 50, None)
    assert d.cookies.from_browser is None and d.cookies.file is None
    assert (d.network.retries, d.network.socket_timeout, d.network.sleep_between_seconds) == (3, 20, 0)
    assert (d.js_runtime.name, d.js_runtime.path, d.js_runtime.remote_components) == ("auto", None, "none")
    assert d.platforms.allow == ["@popular"] and d.platforms.deny == [] and not d.platforms.group_by_platform


def test_download_settings_accept_every_literal() -> None:
    cfg = parse(
        Config,
        {
            "download": {
                "quality": "best",
                "fps": 60,
                "prefer": "at_least",
                "codec": "av1",
                "container": "mkv",
                "audio_quality": "worst",
                "audio_language": "ar",
                "playlist": "all",
                "time_range": "00:01:00-00:01:45",
                "metadata": {"thumbnail": True, "description": True},
                "subtitles": {"enabled": True, "languages": ["ar", "en"], "use_as_transcript": "always"},
                "cookies": {"from_browser": "firefox"},
                "network": {"proxy": "socks5://localhost:1080", "rate_limit_mbps": 5},
                "js_runtime": {"name": "node", "remote_components": "ejs"},
            }
        },
    )
    d = cfg.download
    assert (d.quality, d.fps, d.prefer, d.codec, d.container) == ("best", 60, "at_least", "av1", "mkv")
    assert (d.audio_quality, d.audio_language, d.playlist) == ("worst", "ar", "all")
    assert d.time_range == "00:01:00-00:01:45"
    assert d.subtitles.languages == ["ar", "en"] and d.subtitles.use_as_transcript == "always"
    assert d.cookies.from_browser == "firefox" and d.network.proxy == "socks5://localhost:1080"
    assert d.js_runtime.name == "node" and d.js_runtime.remote_components == "ejs"


def test_load_config_mixed_input_and_download_paths(tmp_path: Path) -> None:
    touch(tmp_path, "one.mp4", "cookies.txt", "deno")
    url = "https://youtu.be/dQw4w9WgXcQ"
    cfg = load_config(
        write(
            tmp_path,
            {
                "input": [url, "one.mp4"],
                "download": {
                    "original_dir": "downloads",
                    "cookies": {"file": "cookies.txt"},
                    "js_runtime": {"name": "deno", "path": "deno"},
                    "platforms": {
                        "settings": {
                            "youtube": {"original_dir": "yt", "cookies": {"file": "other.txt"}},
                            "vimeo": {"original_dir": "vimeo"},
                            "twitter": {"cookies": {"file": "other.txt"}},
                            "tiktok": {"quality": 720},
                        }
                    },
                },
            },
        )
    )
    assert cfg.input[0] == url and Path(cfg.input[1]).is_absolute()
    assert Path(cfg.download.original_dir or "").is_absolute()
    assert Path(cfg.download.cookies.file or "").name == "cookies.txt"
    assert Path(cfg.download.js_runtime.path or "").name == "deno"
    settings = cfg.download.platforms.settings
    assert Path(settings["youtube"]["original_dir"]).is_absolute() and Path(settings["youtube"]["cookies"]["file"]).is_absolute()
    assert Path(settings["vimeo"]["original_dir"]).is_absolute()  # no cookies to resolve
    assert Path(settings["twitter"]["cookies"]["file"]).is_absolute()  # no original_dir to resolve
    assert settings["tiktok"] == {"quality": 720}  # nothing path-like to resolve


def test_a_url_counts_as_a_second_video_for_the_stem_rule(tmp_path: Path) -> None:
    touch(tmp_path, "one.mp4")
    with pytest.raises(ConfigError, match=r"\{stem\}"):
        load_config(
            write(
                tmp_path,
                {"input": ["one.mp4", "https://youtu.be/dQw4w9WgXcQ"], "transcript": {"file": "subs.srt"}},
            )
        )


def test_platform_settings_may_override_nested_sections() -> None:
    cfg = parse(
        Config,
        {
            "download": {
                "metadata": {"thumbnail": False},
                "platforms": {"settings": {"youtube": {"metadata": {"thumbnail": True}, "subtitles": {"enabled": True}}}},
            }
        },
    )
    assert cfg.download.platforms.settings["youtube"] == {"metadata": {"thumbnail": True}, "subtitles": {"enabled": True}}
