"""Platform policy: extractor families, groups, allow/deny and per-platform settings."""

from __future__ import annotations

import re
from typing import Any, cast

import pytest

from autoclip.core import platforms
from autoclip.core.config import Config, JsonValue, parse
from autoclip.core.errors import ConfigError


def test_aliases_are_case_insensitive_and_trimmed() -> None:
    assert platforms.canonical(" X ") == "twitter"
    assert platforms.canonical("VM.TikTok") == "tiktok"
    assert platforms.canonical("snapchat") == "snapchatspotlight"
    assert platforms.canonical("HTML5") == "generic"  # the <video> scraper is the generic policy
    assert platforms.canonical("YouTube") == "youtube"
    assert platforms.canonical("generic") == "generic"


def test_extractor_families_come_from_the_installed_yt_dlp() -> None:
    families = platforms.extractor_families()
    assert "youtube" in families and "tiktok" in families and "generic" in families
    assert "vm.tiktok" in families  # the literal IE_NAME prefix is kept
    assert len(families) > 500


def test_family_names_resolve_aliases() -> None:
    names = platforms.family_names()
    assert "twitter" in names and "snapchatspotlight" in names
    assert "vm.tiktok" not in names and "x" not in names and "html5" not in names


@pytest.mark.parametrize(
    "url, family",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "youtube"),
        ("https://youtu.be/dQw4w9WgXcQ", "youtube"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "youtube"),
        ("https://x.com/user/status/123", "twitter"),
        ("https://twitter.com/user/status/123", "twitter"),
        ("https://www.tiktok.com/@user/video/123", "tiktok"),
        ("https://vm.tiktok.com/abc/", "tiktok"),
        ("https://vimeo.com/123", "vimeo"),
        ("https://www.instagram.com/reel/abc/", "instagram"),
        ("https://www.facebook.com/watch/?v=1", "facebook"),
        ("https://www.reddit.com/r/a/comments/x/", "reddit"),
        ("https://www.snapchat.com/spotlight/abc", "snapchatspotlight"),
    ],
)
def test_identify_finds_the_specific_family(url: str, family: str) -> None:
    assert platforms.identify(url) == family


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/video.mp4",
        "https://www.threads.net/@user/post/abc",  # yt-dlp has no Threads extractor
        "https://www.linkedin.com/posts/someone_activity-1",  # posts are not videos for the extractor
        "not a url",
    ],
)
def test_identify_falls_back_to_generic(url: str) -> None:
    assert platforms.identify(url) == platforms.GENERIC


def test_builtin_groups_are_known_families() -> None:
    families = set(platforms.family_names())
    for name, members in platforms.BUILTIN_GROUPS.items():
        if name == "@all":
            assert members == ()
        else:
            assert members and set(members) <= families


def test_group_members_expand_builtin_user_and_nested_groups() -> None:
    assert {"youtube", "tiktok"} <= platforms.group_members("@popular", {})
    assert platforms.group_members("@social", {}) == {"tiktok", "twitter", "instagram", "facebook", "reddit"}
    groups = {"@news": ["youtube", "@social"], "@mixed": ["@news", "Vimeo"]}
    assert {"youtube", "twitter", "vimeo"} <= platforms.group_members("@mixed", groups)
    assert platforms.group_members("@all", {}) == set(platforms.family_names()) - {platforms.GENERIC}
    assert "generic" not in platforms.group_members("@all", {})


def test_group_errors() -> None:
    with pytest.raises(KeyError):
        platforms.group_members("@missing", {})
    with pytest.raises(ValueError, match="cycle"):
        platforms.group_members("@a", {"@a": ["@b"], "@b": ["@a"]})


def test_expand_names_and_policy() -> None:
    groups = {"@web": ["youtube", "vimeo"]}
    assert platforms.expand_names(["@web", "X"], groups) == {"youtube", "vimeo", "twitter"}
    allow = ["@popular", "generic"]
    assert platforms.allowed("youtube", allow, [], groups)
    assert platforms.allowed("generic", allow, [], groups)
    assert not platforms.allowed("archive.org", allow, [], groups)
    assert platforms.allowed("archive.org", [*allow, "archive.org"], [], groups)
    assert platforms.allowed("archive.org", [*allow, "archive.org"], ["x"], groups)
    assert not platforms.allowed("twitter", allow, ["@social"], groups)  # deny wins
    assert not platforms.allowed("twitter", allow, ["@social", "twitter"], groups)


def test_default_policy_allows_the_popular_set_and_blocks_generic() -> None:
    cfg = Config().download.platforms
    assert platforms.allowed("youtube", cfg.allow, cfg.deny, cfg.groups)
    assert not platforms.allowed(platforms.GENERIC, cfg.allow, cfg.deny, cfg.groups)


def test_check_families_accepts_known_names_aliases_and_groups() -> None:
    platforms.check_families(
        ["@popular", "html5", "X"],  # x is the twitter alias, html5 the generic one
        ["@social"],
        {"@news": ["youtube", "@video", "archive.org"]},
        {"youtube": {"cookies": "chrome"}, "@news": {"cookies": "firefox"}},
    )


@pytest.mark.parametrize(
    "allow, deny, groups, settings, fragment",
    [
        (["youtube", "youtubee"], [], {}, {}, "allow: unknown platform 'youtubee' (did you mean 'youtube'?)"),
        (["@popular"], ["tiktokk"], {}, {}, "deny: unknown platform 'tiktokk' (did you mean 'tiktok'?)"),
        (["@popular"], [], {"@news": ["youtube", "vimeoo"]}, {}, "groups.@news: unknown platform 'vimeoo'"),
        (["@popular"], [], {}, {"instgram": {"cookies": "chrome"}}, "settings: unknown platform 'instgram'"),
    ],
)
def test_check_families_rejects_an_unknown_name(
    allow: list[str],
    deny: list[str],
    groups: dict[str, list[str]],
    settings: dict[str, dict[str, Any]],
    fragment: str,
) -> None:
    with pytest.raises(ConfigError, match=re.escape(fragment)):
        platforms.check_families(allow, deny, groups, settings)


def test_check_families_does_not_suggest_when_nothing_is_close() -> None:
    with pytest.raises(ConfigError, match=r"unknown platform 'zzzzzzzz'$"):
        platforms.check_families(["zzzzzzzz"], [], {}, {})


def test_extractor_patterns_match_yt_dlp_extractor_keys() -> None:
    patterns = platforms.extractor_patterns({"tiktok", "archive.org", "nosuchfamily"})
    assert patterns[-1] == "end"
    assert "tiktok(:.*)?" in patterns and "vm\\.tiktok(:.*)?" in patterns
    compiled = [re.compile(pattern, re.I) for pattern in patterns]
    assert any(p.fullmatch("tiktok") for p in compiled)
    assert any(p.fullmatch("vm.tiktok") for p in compiled)
    assert any(p.fullmatch("archive.org") for p in compiled)
    assert any(p.fullmatch("nosuchfamily") for p in compiled)
    assert not any(p.fullmatch("youtube") for p in compiled)


def test_deep_merge_inherits_null_and_recurses() -> None:
    base: dict[str, Any] = {"audio_language": None, "subtitles": {"languages": ["en"]}, "cookies": {"file": "a"}}
    merged = platforms.deep_merge(base, {"audio_language": "de", "subtitles": {"languages": ["fr"]}, "cookies": {"file": "b"}})
    assert merged == {"audio_language": "de", "subtitles": {"languages": ["fr"]}, "cookies": {"file": "b"}}
    assert base["subtitles"]["languages"] == ["en"]  # the input is not mutated


def test_settings_entry_accepts_an_alias() -> None:
    settings: dict[str, dict[str, Any]] = {"x": {"audio_language": "de"}, "@social": {"subtitles": {"languages": ["ar"]}}}
    assert platforms.settings_entry("twitter", settings) == {"audio_language": "de"}
    assert platforms.settings_entry("vimeo", settings) is None


def test_effective_download_precedence() -> None:
    base: dict[str, Any] = {"audio_language": None, "subtitles": {"languages": ["en"]}}
    groups = {"@chan": ["youtube", "twitter"]}
    settings: dict[str, dict[str, Any]] = {
        "@chan": {"audio_language": "de", "subtitles": {"languages": ["ar"]}},
        "youtube": {"audio_language": "fr", "subtitles": {"languages": None}},
    }
    youtube = platforms.effective_download(base, "youtube", groups, settings)
    assert youtube == {"audio_language": "fr", "subtitles": {"languages": ["ar"]}}
    twitter = platforms.effective_download(base, "twitter", groups, settings)
    assert twitter["audio_language"] == "de" and twitter["subtitles"]["languages"] == ["ar"]
    vimeo = platforms.effective_download(base, "vimeo", groups, settings)
    assert vimeo == base


def test_host_of() -> None:
    assert platforms.host_of("https://WWW.YouTube.com/watch?v=1") == "www.youtube.com"
    assert platforms.host_of("not a url") == ""


def test_download_config_accepts_aliases_groups_and_settings() -> None:
    cfg = parse(
        Config,
        {
            "download": {
                "platforms": {
                    "allow": ["@popular", "generic"],
                    "groups": {"@news": ["youtube", "@social"], "@mine": ["@news", "@news"]},
                    "settings": {"x": {"cookies": "chrome"}, "@news": {"time_range": "1:00-2:00"}},
                },
            },
        },
    )
    assert cfg.download.platforms.allow == ["@popular", "generic"]
    assert cfg.download.platforms.groups["@news"] == ["youtube", "@social"]
    assert cfg.download.platforms.groups["@mine"] == ["@news", "@news"]  # a group visited twice is only checked once
    assert cfg.download.platforms.settings["x"] == {"cookies": "chrome"}


@pytest.mark.parametrize(
    "platforms_cfg, fragment",
    [
        ({"allow": []}, "must name at least one"),
        ({"allow": ["  "]}, "must not contain empty names"),
        ({"allow": ["@nope"]}, "unknown group @nope"),
        ({"deny": ["@nope"]}, "unknown group @nope"),
        ({"groups": {"news": ["youtube"]}}, "must look like '@name'"),
        ({"groups": {"@popular": ["youtube"]}}, "built-in group cannot be redefined"),
        ({"groups": {"@news": []}}, "at least one member"),
        ({"groups": {"@news": [" "]}}, "member must not be empty"),
        ({"groups": {"@news": ["@nope"]}}, "unknown group @nope"),
        ({"groups": {"@a": ["@b"], "@b": ["@a"]}}, "has a cycle"),
        ({"settings": {" ": {"audio_language": "de"}}}, "empty platform name"),
        ({"settings": {"@nope": {"audio_language": "de"}}}, "unknown group @nope"),
        ({"settings": {"x": {}, "twitter": {}}}, "name the same platform"),
        ({"settings": {"youtube": {"platforms": {}}}}, "platforms cannot be overridden"),
        ({"settings": {"youtube": {"js_runtime": {}}}}, "js_runtime: Unexpected keyword"),
        ({"settings": {"youtube": {"nope": 1}}}, r"settings.youtube: nope: Unexpected keyword"),
    ],
)
def test_platform_config_errors(platforms_cfg: dict[str, object], fragment: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(fragment) if ":" in fragment else fragment):
        parse(Config, {"download": {"platforms": cast(JsonValue, platforms_cfg)}})
