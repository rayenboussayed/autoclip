"""Configuration: the nine-key file schema, strict keys, runtime overrides and input expansion."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

import pytest

from autoclip.core.config import (
    FileConfig,
    FileLlmCfg,
    JsonValue,
    RuntimeOverrides,
    _default_output_dir,
    expand_inputs,
    is_url,
    load_config,
    parse,
    parse_hms,
)
from autoclip.core.errors import ConfigError


def write(tmp: Path, data: object, name: str = "config.json") -> Path:
    p = tmp / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def touch(tmp: Path, *names: str) -> None:
    for n in names:
        (tmp / n).parent.mkdir(parents=True, exist_ok=True)
        (tmp / n).write_bytes(b"x")


def _leaves(data: object, prefix: str = "") -> set[str]:
    """Every scalar path of a nested config dict (lists stay leaves)."""
    if not isinstance(data, dict):
        return {prefix}
    return {path for key, value in data.items() for path in _leaves(value, f"{prefix}.{key}" if prefix else key)}


def test_the_file_schema_has_exactly_nine_keys() -> None:
    data = asdict(FileConfig())
    assert set(data) == {"input", "clips", "download", "llm", "image"}
    assert _leaves(data) == {
        "input",
        "clips.preset",
        "clips.scenes",
        "clips.keywords",
        "clips.ranges",
        "download.allow",
        "download.deny",
        "llm.system_prompt",
        "image.prompt",
    }


def test_defaults_roundtrip_through_json() -> None:
    """``--init-full`` output must load back into the identical defaults."""
    assert parse(FileConfig, json.loads(json.dumps(asdict(FileConfig())))) == FileConfig()


def test_parse_accepts_comments_and_ignores_them() -> None:
    cfg = parse(FileConfig, {"_comment": "x", "clips": {"_why": 1, "preset": "shorts"}})
    assert cfg.clips.preset == "shorts"


@pytest.mark.parametrize(
    "data, fragment",
    [
        ({"cout": 1}, "cout: Unexpected keyword"),
        ({"clips": {"cout": 1}}, "clips.cout: Unexpected keyword"),
        ({"clips": {"scenes": "nope"}}, "unknown clips.scenes"),
        ({"clips": {"scenes": ["best", "nope"]}}, "unknown clips.scenes"),
        ({"download": {"nope": 1}}, "download.nope: Unexpected keyword"),
        ({"download": {"setting": {"youtube": {}}}}, "download.setting: Unexpected keyword"),
        ({"download": {"allow": []}}, "allow must name at least one platform"),
        ({"download": {"allow": [""]}}, "must not contain empty names"),
        ({"download": {"deny": ["@nope"]}}, "unknown group @nope"),
    ],
)
def test_invalid_config_is_rejected(data: dict[str, object], fragment: str) -> None:
    with pytest.raises(ConfigError, match=fragment):
        parse(FileConfig, cast(dict[str, JsonValue], data))


def test_parse_rejects_a_non_object_document() -> None:
    with pytest.raises(ConfigError):
        parse(FileConfig, cast(dict[str, JsonValue], []))


def test_unknown_values_fall_back_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """A wrong type is not fatal: the field keeps its default and the warning names the path."""
    with caplog.at_level("WARNING", logger="autoclip"):
        cfg = parse(FileConfig, {"input": "not-a-list", "clips": {"preset": "everything"}})
    assert cfg == FileConfig()
    assert "invalid value for input" in caplog.text and "invalid value for clips.preset" in caplog.text


@pytest.mark.parametrize(
    ("old", "hint"),
    [
        ({"output_dir": "out"}, "AUTOCLIP_OUTPUT_DIR"),
        ({"mode": "performance"}, "pass --mode"),
        ({"quality": "high"}, "pass --quality"),
        ({"video": {"codec": "h265"}}, "video"),
        ({"clips": {"kinds": ["short"]}}, "use clips.preset"),
        ({"clips": {"short": {"count": 2}}}, "the short profile is built in"),
        ({"transcript": {"file": "a.srt"}}, "sidecar"),
        ({"skip": {"audio": True}}, "pass --skip NAME"),
        ({"download": {"playlist": "all"}}, "the URL shape decides"),
    ],
)
def test_removed_keys_warn_and_are_ignored(old: dict[str, object], hint: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="autoclip"):
        cfg = parse(FileConfig, {"input": ["a.mp4"], **cast(dict[str, JsonValue], old)})
    assert cfg.input == ["a.mp4"]
    assert hint in caplog.text and "no longer an option" in caplog.text


def test_a_removed_key_warns_once_per_load(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="autoclip"):
        parse(FileConfig, {"output_dir": "a"})
        parse(FileConfig, {"output_dir": "a"})
    assert caplog.text.count("'output_dir'") == 2  # one per parse call, never twice inside one


def test_a_removed_key_still_warns_when_parsed_outside_parse(caplog: pytest.LogCaptureFixture) -> None:
    from pydantic import TypeAdapter

    # ``parse`` records what it has warned about; a caller validating the dataclass directly has no such
    # record, and the warning must still be emitted (not silently dropped).
    with caplog.at_level("WARNING", logger="autoclip"):
        cfg = TypeAdapter(FileConfig).validate_python({"input": ["a.mp4"], "quality": "high"})
    assert cfg.input == ["a.mp4"]
    assert "'quality'" in caplog.text and "no longer an option" in caplog.text


def test_scene_types_all_one_or_a_list() -> None:
    assert len(FileConfig(clips={"preset": "both", "scenes": "all"}).clips.scene_list()) == 10  # type: ignore[arg-type]
    assert FileConfig(clips={"preset": "both", "scenes": "calm"}).clips.scene_list() == ["calm"]  # type: ignore[arg-type]
    assert FileConfig(clips={"preset": "both", "scenes": ["funny", "sad"]}).clips.scene_list() == ["funny", "sad"]  # type: ignore[arg-type]
    assert FileConfig().clips.scene_list() == ["best"]


def test_the_preset_selects_the_kinds() -> None:
    both = FileConfig(clips={"preset": "both"}).to_config()  # type: ignore[arg-type]
    assert both.clips.kinds == ["short", "long"]
    assert (both.clips.short.min_seconds, both.clips.short.max_seconds, both.clips.short.aspect) == (20, 60, "9:16")
    assert (both.clips.long.min_seconds, both.clips.long.max_seconds, both.clips.long.aspect) == (180, 600, "16:9")
    assert FileConfig(clips={"preset": "shorts"}).to_config().clips.kinds == ["short"]  # type: ignore[arg-type]
    assert FileConfig(clips={"preset": "long"}).to_config().clips.kinds == ["long"]  # type: ignore[arg-type]


def test_ranges_switch_to_exact_cuts_for_every_selected_kind() -> None:
    cfg = FileConfig(clips={"preset": "shorts", "ranges": ["00:00:04-00:00:10"]}).to_config()  # type: ignore[arg-type]
    assert cfg.clips.mode == "ranges"
    assert cfg.clips.short.ranges == ["00:00:04-00:00:10"]
    assert cfg.clips.long.ranges == ["00:00:04-00:00:10"]  # both kinds carry them, the preset decides which render
    assert FileConfig(clips={"preset": "shorts"}).to_config().clips.mode == "highlights"  # type: ignore[arg-type]


def test_split_and_ranges_are_mutually_exclusive() -> None:
    cfg = FileConfig(clips={"ranges": ["0:04-0:10"]})  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="--split cannot be combined"):
        cfg.to_config(RuntimeOverrides(split=True))
    assert FileConfig().to_config(RuntimeOverrides(split=True)).clips.mode == "split"


def test_file_config_derives_the_runtime_defaults() -> None:
    cfg = FileConfig().to_config()
    assert cfg.mode == "balance" and cfg.quality == "high"
    assert cfg.llm.system_prompt == "" and cfg.image.prompt == "cinematic thumbnail, high quality, sharp focus"
    assert cfg.captions.enabled and not any(asdict(cfg.skip).values())
    assert cfg.download.platforms.allow == ["@popular"] and cfg.download.platforms.deny == []
    assert cfg.llm.base_url == "http://localhost:11434"


def test_overrides_replace_one_field_at_a_time() -> None:
    cfg = FileConfig().to_config(
        RuntimeOverrides(
            output_dir="/tmp/out",
            cookies="firefox",
            skip=["caption", "title"],
            llm_base_url="http://box:11434",
            llm_model="qwen2.5",
            mode="performance",
            quality="lowest",
        )
    )
    assert cfg.output_dir == "/tmp/out"
    assert cfg.download.cookies == "firefox"
    assert cfg.captions.enabled is False
    assert cfg.skip.caption and cfg.skip.title
    assert (cfg.llm.base_url, cfg.llm.model, cfg.llm.system_prompt) == ("http://box:11434", "qwen2.5", "")
    assert (cfg.mode, cfg.quality) == ("performance", "lowest")


def test_environment_overrides_are_field_by_field(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {
        "AUTOCLIP_OUTPUT_DIR": "/env/out",
        "AUTOCLIP_QUALITY": "low",
        "AUTOCLIP_SKIP": "caption, llm",
        "AUTOCLIP_LLM_MODEL": "env-model",
    }
    explicit = RuntimeOverrides(quality="highest")
    merged = RuntimeOverrides.combined(explicit, env)
    assert merged.output_dir == "/env/out"
    assert merged.quality == "highest"  # the explicit value wins
    assert merged.skip == ["caption", "llm"] and merged.llm_model == "env-model"


def test_an_invalid_environment_override_is_ignored(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="autoclip"):
        overrides = RuntimeOverrides.from_env({"AUTOCLIP_QUALITY": "nope", "AUTOCLIP_MODE": "nope", "AUTOCLIP_OUTPUT_DIR": "/env/out"})
    assert overrides.quality is None and overrides.mode is None and overrides.output_dir == "/env/out"
    assert "invalid value for quality" in caplog.text


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


def test_load_config_resolves_paths_relative_to_the_file_but_output_dir_is_cwd(tmp_path: Path) -> None:
    touch(tmp_path, "vids/one.mp4", "vids/two.mp4", "cookies.txt", "subs/one.srt", "subs/two.srt")
    overrides = RuntimeOverrides(output_dir="out", cookies="cookies.txt")
    cfg = load_config(
        write(tmp_path, {"input": "vids/*.mp4", "clips": {"preset": "shorts", "scenes": "funny"}}),
        overrides,
    )
    assert [Path(p).name for p in cfg.input] == ["one.mp4", "two.mp4"]
    assert Path(cfg.output_dir).is_absolute() and Path(cfg.output_dir).is_relative_to(Path.cwd())
    assert Path(cfg.download.cookies or "").name == "cookies.txt"
    assert cfg.download.cookies and Path(cfg.download.cookies).is_absolute()
    assert cfg.clips.scene_list() == ["funny"] and cfg.clips.kinds == ["short"]


def test_a_cookie_browser_name_stays_a_name(tmp_path: Path) -> None:
    touch(tmp_path, "one.mp4")
    cfg = load_config(write(tmp_path, {"input": "one.mp4"}), RuntimeOverrides(cookies="firefox"))
    assert cfg.download.cookies == "firefox"


def test_single_string_input(tmp_path: Path) -> None:
    touch(tmp_path, "one.mp4")
    cfg = load_config(write(tmp_path, {"input": "one.mp4"}))
    assert len(cfg.input) == 1 and cfg.input[0].endswith("one.mp4")


def test_offline_and_device_come_from_the_environment(tmp_path: Path) -> None:
    touch(tmp_path, "one.mp4")
    cfg = load_config(write(tmp_path, {"input": "one.mp4"}), RuntimeOverrides.from_env({}))
    assert cfg.mode == "balance"


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
    assert isinstance(cfg.video.background, str)


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
    d = FileConfig().to_config().download
    assert d.time_range is None
    assert d.cookies is None
    assert d.platforms.allow == ["@popular"] and d.platforms.deny == []


def test_a_removed_download_section_still_loads(tmp_path: Path) -> None:
    touch(tmp_path, "one.mp4")
    cfg = load_config(
        write(tmp_path, {"input": "one.mp4", "download": {"quality": "best", "playlist": "all", "subtitles": {"enabled": False}}})
    )
    assert cfg.download.platforms.allow == ["@popular"]


@pytest.mark.parametrize(
    ("system", "env", "suffix"),
    [
        ("Linux", {"XDG_VIDEOS_DIR": "/videos"}, "/videos/clips"),
        ("Linux", {}, "/Videos/clips"),
        ("Darwin", {}, "/Movies/clips"),
        ("Windows", {}, "/Videos/clips"),
    ],
)
def test_the_default_output_folder_follows_the_operating_system(system: str, env: dict[str, str], suffix: str, tmp_path: Path) -> None:
    assert _default_output_dir(system, env, tmp_path).replace("\\", "/").endswith(suffix)


def test_default_config_keeps_the_stable_defaults() -> None:
    import sys

    cfg = parse(FileConfig, {}).to_config()
    assert Path(cfg.output_dir) == Path.home() / ("Movies" if sys.platform == "darwin" else "Videos") / "clips"


def test_runtime_config_validation_errors() -> None:
    from autoclip.core.config import Config

    with pytest.raises(ConfigError):
        parse(Config, {"clips": {"short": {"min_seconds": 50, "max_seconds": 10}}})
    with pytest.raises(ConfigError):
        parse(Config, {"clips": {"scenes": "nope"}})
    with pytest.raises(ConfigError):
        parse(Config, {"clips": {"kinds": []}})
    with pytest.raises(ConfigError):
        parse(Config, {"clips": {"kinds": ["short", "short"]}})
    with pytest.raises(ConfigError):
        parse(Config, {"download": {"subtitles": {"languages": [""]}}})
    with pytest.raises(ConfigError):
        parse(Config, {"download": {"time_range": "00:30-00:10"}})
    with pytest.raises(ConfigError):
        parse(Config, {"download": {"time_range": "1::2-2:00"}})
    # mode normalisation
    cfg = parse(Config, {"mode": "garbage"})
    assert cfg.mode == "balance"


def test_default_transcript_model_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    fake = types.ModuleType("ctranslate2")
    fake.get_cuda_device_count = lambda: 2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)
    from autoclip.core.internal.runtime import transcript_model_for

    assert transcript_model_for("highest") == "large-v3"


def test_the_download_allow_deny_keys_are_read_directly(tmp_path: Path) -> None:
    touch(tmp_path, "one.mp4")
    cfg = load_config(write(tmp_path, {"input": "one.mp4", "download": {"allow": ["youtube"], "deny": ["tiktok"]}}))
    assert cfg.download.platforms.allow == ["youtube"] and cfg.download.platforms.deny == ["tiktok"]


def test_the_removed_platforms_wrapper_warns_and_falls_back_to_the_default(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="autoclip"):
        cfg = parse(FileConfig, {"download": {"platforms": {"allow": ["youtube"], "deny": []}}})
    assert cfg.download.allow == ["@popular"]
    assert "download.platforms" in caplog.text and "download.allow" in caplog.text


def test_the_llm_and_image_prompts_are_file_keys(tmp_path: Path) -> None:
    touch(tmp_path, "one.mp4")
    cfg = load_config(write(tmp_path, {"input": "one.mp4", "llm": {"system_prompt": "You are strict."}, "image": {"prompt": "dark room"}}))
    assert cfg.llm.system_prompt == "You are strict."
    assert cfg.image.prompt == "dark room"


def test_the_llm_and_image_prompts_default_to_empty_and_the_sd_default(tmp_path: Path) -> None:
    touch(tmp_path, "one.mp4")
    cfg = load_config(write(tmp_path, {"input": "one.mp4"}))
    assert cfg.llm.system_prompt == ""
    assert cfg.image.prompt == "cinematic thumbnail, high quality, sharp focus"


def test_the_llm_and_image_prompts_keep_their_defaults_on_bad_values(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="autoclip"):
        cfg = parse(FileConfig, {"llm": {"system_prompt": 42}, "image": {"prompt": ["dark"]}})
    assert cfg.llm.system_prompt == "" and cfg.image.prompt == "cinematic thumbnail, high quality, sharp focus"
    assert "invalid value for llm.system_prompt" in caplog.text and "invalid value for image.prompt" in caplog.text


def test_the_llm_endpoint_keys_stay_removed_but_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="autoclip"):
        cfg = parse(FileConfig, {"llm": {"base_url": "http://box:11434", "model": "qwen2.5", "api_key": "secret"}})
    assert cfg.llm == FileLlmCfg()
    for hint in ("--llm-url", "--llm-model", "no API key"):
        assert hint in caplog.text
