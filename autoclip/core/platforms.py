"""Platform policy for downloads: extractor families, built-in groups and allow/deny decisions.

This module is data and pure functions.  ``yt-dlp`` is imported inside the functions that need the
list of extractors, so a run with only local files never imports it.
"""

from __future__ import annotations

import difflib
import re
from functools import cache
from typing import Any
from urllib.parse import urlsplit

from .errors import ConfigError

#: User-facing aliases -> the extractor family yt-dlp reports.
ALIASES: dict[str, str] = {
    "x": "twitter",
    "vm.tiktok": "tiktok",
    "snapchat": "snapchatspotlight",
    "html5": "generic",  # the <video> scraper is the generic policy ("any web page")
}

#: The fallback extractor is never part of a group: it accepts every web page.
GENERIC = "generic"

#: Built-in groups.  ``@all`` is expanded at runtime (every family except ``generic``).
BUILTIN_GROUPS: dict[str, tuple[str, ...]] = {
    "@popular": (
        "youtube",
        "tiktok",
        "twitter",
        "instagram",
        "facebook",
        "reddit",
        "vimeo",
        "dailymotion",
        "twitch",
        "bilibili",
    ),
    "@social": ("tiktok", "twitter", "instagram", "facebook", "reddit"),
    "@video": ("youtube", "vimeo", "dailymotion", "twitch", "bilibili", "archive.org", "rumble"),
    "@all": (),  # every family except generic, computed from the installed yt-dlp
}


def canonical(name: str) -> str:
    """The extractor family a user wrote (case-insensitive, aliases resolved)."""
    key = name.strip().lower()
    return ALIASES.get(key, key)


@cache
def extractor_families() -> tuple[str, ...]:
    """The literal first segment of every extractor name (``vm.tiktok`` stays distinct)."""
    from yt_dlp.extractor import gen_extractor_classes

    return tuple(sorted({ie.IE_NAME.split(":")[0].lower() for ie in gen_extractor_classes()}))


@cache
def family_names() -> tuple[str, ...]:
    """Every canonical extractor family of the installed yt-dlp."""
    return tuple(sorted({canonical(name) for name in extractor_families()}))


@cache
def identify(url: str) -> str:
    """The canonical family of the first specific extractor that accepts ``url`` (offline).

    ``generic`` is returned when no specific extractor recognises the URL; the policy then decides
    whether the generic page scraper may be tried.  No request is made.
    """
    from yt_dlp.extractor import gen_extractor_classes

    for ie in gen_extractor_classes():
        name = ie.IE_NAME.split(":")[0].lower()
        if name != GENERIC and ie.suitable(url):
            return canonical(name)
    return GENERIC


def group_members(name: str, groups: dict[str, list[str]], stack: tuple[str, ...] = ()) -> set[str]:
    """Every canonical family named by a group; user groups may nest.

    Raises:
        KeyError: if the group does not exist.
        ValueError: if the groups contain a cycle.
    """
    if name == "@all":
        return {family for family in family_names() if family != GENERIC}
    members = BUILTIN_GROUPS.get(name) or groups.get(name)
    if members is None:
        raise KeyError(name)
    if name in stack:
        raise ValueError("group cycle: " + " -> ".join((*stack, name)))
    out: set[str] = set()
    for member in members:
        key = member.strip().lower()
        if key.startswith("@"):
            out |= group_members(key, groups, (*stack, name))
        else:
            out.add(canonical(key))
    return out


def expand_names(names: list[str], groups: dict[str, list[str]]) -> set[str]:
    """Expand a list of families and ``@groups`` into canonical families."""
    out: set[str] = set()
    for name in names:
        key = name.strip().lower()
        out |= group_members(key, groups) if key.startswith("@") else {canonical(key)}
    return out


def allowed(family: str, allow: list[str], deny: list[str], groups: dict[str, list[str]]) -> bool:
    """Whether a family passes the policy: deny wins, then the allow list decides."""
    fam = canonical(family)
    if fam in expand_names(deny, groups):
        return False
    return fam in expand_names(allow, groups)


def check_families(
    allow: list[str],
    deny: list[str],
    groups: dict[str, list[str]],
    settings: dict[str, dict[str, Any]],
) -> None:
    """Reject family names the installed yt-dlp does not know, before any request is made.

    The families come from yt-dlp, which is imported lazily, so the config file cannot validate them.
    Built-in groups are trusted (a test checks their members); user groups and per-platform settings
    are walked as well, because a typo there would otherwise be ignored silently.

    Raises:
        ConfigError: naming the field, the unknown name and the closest known family.
    """
    known = set(family_names())
    named = [("allow", name) for name in allow]
    named += [("deny", name) for name in deny]
    named += [(f"groups.{group}", member) for group, members in groups.items() for member in members]
    named += [("settings", name) for name in settings]
    checked: set[str] = set()
    for field, name in named:
        if name.startswith("@"):
            continue
        family = canonical(name)
        if family in known or family in checked:
            continue
        checked.add(family)
        close = difflib.get_close_matches(family, known, n=1)
        hint = f" (did you mean '{close[0]}'?)" if close else ""
        raise ConfigError(f"download.platforms.{field}: unknown platform '{name}'{hint}")


def extractor_patterns(families: set[str]) -> list[str]:
    """Regexes for yt-dlp's ``allowed_extractors`` (it full-matches lowercased extractor names).

    Every literal name of a family is included (``tiktok`` *and* ``vm.tiktok``), plus ``end`` so an
    unsupported URL produces yt-dlp's normal "Unsupported URL" error instead of a bare lookup error.
    """
    actual = extractor_families()
    patterns: list[str] = []
    for family in sorted(families):
        names = [name for name in actual if canonical(name) == family] or [family]
        patterns.extend(re.escape(name) + "(:.*)?" for name in names)
    patterns.append("end")
    return patterns


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``over`` into ``base``; ``null`` means inherit (it cannot clear a value)."""
    out = dict(base)
    for key, value in over.items():
        current = out.get(key)
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(current, dict):
            out[key] = deep_merge(current, value)
        else:
            out[key] = value
    return out


def settings_entry(family: str, settings: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """The settings entry written for ``family`` (its aliases accepted)."""
    fam = canonical(family)
    for name, entry in settings.items():
        if not name.startswith("@") and canonical(name) == fam:
            return entry
    return None


def effective_download(
    base: dict[str, Any], family: str, groups: dict[str, list[str]], settings: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """The download settings for one family: top-level, then matching groups, then the family."""
    merged = dict(base)
    fam = canonical(family)
    for name, group_entry in settings.items():
        if name.startswith("@") and fam in group_members(name, groups):
            merged = deep_merge(merged, group_entry)
    own = settings_entry(fam, settings)
    if own:
        merged = deep_merge(merged, own)
    return merged


def host_of(url: str) -> str:
    """The lowercased host of a URL (empty when it has none)."""
    return (urlsplit(url).hostname or "").lower()
