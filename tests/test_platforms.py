"""Platform policy: extractor families, groups, allow/deny and per-platform settings."""

from __future__ import annotations

import re
from typing import Any

import pytest

from autoclip.core import platforms
from autoclip.core.config import Config, parse
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
        {"youtube": {"quality": 720}, "@news": {"fps": 30}},
    )


@pytest.mark.parametrize(
    "allow, deny, groups, settings, fragment",
    [
        (["youtube", "youtubee"], [], {}, {}, "allow: unknown platform 'youtubee' (did you mean 'youtube'?)"),
        (["@popular"], ["tiktokk"], {}, {}, "deny: unknown platform 'tiktokk' (did you mean 'tiktok'?)"),
        (["@popular"], [], {"@news": ["youtube", "vimeoo"]}, {}, "groups.@news: unknown platform 'vimeoo'"),
        (["@popular"], [], {}, {"instgram": {"quality": 720}}, "settings: unknown platform 'instgram'"),
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
    base: dict[str, Any] = {"quality": 1080, "fps": 30, "metadata": {"info_json": True, "thumbnail": False}, "cookies": {"file": "a"}}
    merged = platforms.deep_merge(base, {"quality": None, "metadata": {"thumbnail": True}, "cookies": {"file": "b"}})
    assert merged == {"quality": 1080, "fps": 30, "metadata": {"info_json": True, "thumbnail": True}, "cookies": {"file": "b"}}
    assert base["metadata"]["thumbnail"] is False  # the input is not mutated


def test_settings_entry_accepts_an_alias() -> None:
    settings = {"x": {"quality": 720}, "@social": {"fps": 30}}
    assert platforms.settings_entry("twitter", settings) == {"quality": 720}
    assert platforms.settings_entry("vimeo", settings) is None


def test_effective_download_precedence() -> None:
    base: dict[str, Any] = {"quality": 1080, "fps": 30, "metadata": {"info_json": True, "thumbnail": False}}
    groups = {"@chan": ["youtube", "twitter"]}
    settings: dict[str, dict[str, Any]] = {
        "@chan": {"quality": 720, "metadata": {"thumbnail": True}},
        "youtube": {"quality": 480, "fps": None},
    }
    youtube = platforms.effective_download(base, "youtube", groups, settings)
    assert youtube == {"quality": 480, "fps": 30, "metadata": {"info_json": True, "thumbnail": True}}
    twitter = platforms.effective_download(base, "twitter", groups, settings)
    assert twitter["quality"] == 720 and twitter["metadata"]["thumbnail"] is True
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
                "quality": 720,
                "platforms": {
                    "allow": ["@popular", "generic"],
                    "groups": {"@news": ["youtube", "@social"], "@mine": ["@news", "@news"]},
                    "settings": {"x": {"quality": 480}, "@news": {"fps": 30}},
                },
            }
        },
    )
    assert cfg.download.platforms.allow == ["@popular", "generic"]
    assert cfg.download.platforms.groups["@news"] == ["youtube", "@social"]
    assert cfg.download.platforms.groups["@mine"] == ["@news", "@news"]  # a group visited twice is only checked once
    assert cfg.download.platforms.settings["x"] == {"quality": 480}


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
        ({"settings": {" ": {"quality": 720}}}, "empty platform name"),
        ({"settings": {"@nope": {"quality": 720}}}, "unknown group @nope"),
        ({"settings": {"x": {}, "twitter": {}}}, "name the same platform"),
        ({"settings": {"youtube": {"platforms": {}}}}, "platforms and js_runtime cannot be overridden"),
        ({"settings": {"youtube": {"js_runtime": {}}}}, "platforms and js_runtime cannot be overridden"),
        ({"settings": {"youtube": {"quality": 100}}}, r"settings.youtube: quality"),
        ({"settings": {"youtube": {"nope": 1}}}, r"settings.youtube: nope: Unexpected keyword"),
    ],
)
def test_platform_config_errors(platforms_cfg: dict[str, object], fragment: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(fragment) if ":" in fragment else fragment):
        parse(Config, {"download": {"platforms": platforms_cfg}})
