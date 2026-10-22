"""Configuration: strict keys, typed values, validation, multi-video input expansion."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from autoclip.config import ClipsCfg, Config, KindCfg, build, expand_inputs, load_config
from autoclip.errors import ConfigError


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
    assert build(Config, json.loads(json.dumps(asdict(Config())))) == Config()


def test_build_accepts_comments_and_int_for_float() -> None:
    cfg = build(Config, {"_comment": "x", "clips": {"short": {"min_seconds": 5, "max_seconds": 9}, "_why": 1}})
    assert cfg.clips.short.min_seconds == 5


@pytest.mark.parametrize(
    "data, fragment",
    [
        ({"cout": 1}, "unknown config key 'cout'"),
        ({"clips": {"cout": 1}}, "'clips.cout'"),
        ({"clips": {"short": {"count": "5"}}}, "invalid value"),
        ({"clips": {"short": {"count": True}}}, "invalid value"),
        ({"clips": {"short": {"min_seconds": True}}}, "invalid value"),
        ({"clips": 3}, "must be a JSON object"),
        ({"clips": {"mode": "nope"}}, "invalid value"),
        ({"clips": {"kinds": ["medium"]}}, "invalid value"),
        ({"clips": {"keywords": ["a", 1]}}, "invalid value"),
        ({"clips": {"lexicon_file": 5}}, "invalid value"),
        ({"silence": {"threshold_db": "loud"}}, "invalid value"),
        ({"clips": {"scenes": "nope"}}, "unknown clips.scenes"),
        ({"clips": {"scenes": ["best", "nope"]}}, "unknown clips.scenes"),
        ({"clips": {"kinds": []}}, "clips.kinds"),
        ({"clips": {"kinds": ["short", "short"]}}, "clips.kinds"),
        ({"clips": {"split_seconds": 0}}, "split_seconds"),
        ({"clips": {"short": {"min_seconds": 50, "max_seconds": 10}}}, "min_seconds"),
        ({"clips": {"short": {"target_seconds": 5}}}, "target_seconds"),
        ({"clips": {"short": {"min_seconds": 5, "max_seconds": 10, "target_seconds": 20}}}, "target_seconds"),
        ({"clips": {"long": {"count": 0}}}, "at least 1"),
        ({"clips": {"short": {"ranges": ["bad"]}}}, "HH:MM:SS"),
        ({"clips": {"long": {"aspect": "wide"}}}, "aspect"),
        ({"llm": {"base_url": "file:///etc/passwd"}}, r"http\(s\)"),
        ({"video": {"aspect": "9:16"}}, "unknown config key 'video.aspect'"),
        ({"video": {"gpu": "cuda"}}, "invalid value"),
        ({"processing": {"chunk_minutes": 0}}, "chunk_minutes"),
        ({"transcript": {"enhance": {"backend": "nope"}}}, "invalid value"),
        ({"transcript": {"enhance": {"model": "nope"}}}, "invalid value"),
        ({"transcript": {"enhance": {"attenuation_limit_db": 200}}}, "attenuation_limit_db"),
        ({"transcript": {"enhance": {"denoise_strength": 2}}}, "denoise_strength"),
        ({"transcript": {"enhance": {"num_threads": 0}}}, "num_threads"),
        ({"transcript": {"enhance": {"snr_threshold_db": 0}}}, "snr_threshold_db"),
        ({"compute": {"device": "tpu"}}, "invalid value"),
    ],
)
def test_invalid_config_is_rejected(data: dict[str, object], fragment: str) -> None:
    with pytest.raises(ConfigError, match=fragment):
        build(Config, data)


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
    custom = build(Config, {"clips": {"kinds": ["long"], "long": {"aspect": "original", "ranges": ["0:10-0:50"]}}})
    assert custom.clips.long.aspect == "original" and custom.clips.long.ranges == ["0:10-0:50"] and custom.clips.short.count == 5


def test_the_recommended_length_is_optional_and_must_fit_the_range() -> None:
    kind = build(Config, {"clips": {"short": {"min_seconds": 10, "max_seconds": 60, "target_seconds": 30}}}).clips.short
    assert kind.target_seconds == 30
    assert build(Config, {"clips": {"short": {"target_seconds": None}}}).clips.short.target_seconds is None


def test_valid_unions_and_lists() -> None:
    cfg = build(Config, {"clips": {"scenes": ["best", "funny"]}, "silence": {"threshold_db": -40}, "video": {"fps": None, "gpu": "nvenc"}})
    assert cfg.clips.scene_list() == ["best", "funny"] and KindCfg().aspect == "9:16"
    assert cfg.silence.threshold_db == -40 and cfg.video.fps is None and cfg.video.gpu == "nvenc"


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
