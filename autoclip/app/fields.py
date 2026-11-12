"""The catalogue of every configuration-file option: what it does, why you would change it, when to.

The desktop app builds its form from this list, so an option that exists in
:class:`autoclip.core.config.FileConfig` but is missing here would be invisible in the UI - the test
suite checks that the two stay in sync.

Each option carries three texts:

``what``
    what the option does, in one sentence.
``why``
    why you would ever change it away from its default.
``when``
    when in a project that is the right moment to do so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The kinds of control the form knows how to build.
KINDS = ("bool", "choice", "int", "float", "str", "path", "dir", "csv", "lines", "json")


@dataclass(frozen=True, slots=True)
class Option:
    """One configuration option and its explanation."""

    path: str  # the config path, e.g. "clips.preset"
    label: str  # the label in the form
    kind: str  # one of KINDS
    what: str
    why: str
    when: str
    choices: tuple[str, ...] = ()
    visible_when: str = ""  # "other.path=value": only shown when that option has that value
    advanced: bool = False  # shown in the "advanced" group of its section
    placeholder: str = ""  # the grey hint inside an empty entry
    csv_is_list: bool = True  # a csv field that may also be a single string ("best" is not ["b", "e", "s", "t"])
    numeric: bool = False  # a text field that takes a number or a word ("1080", "best")


@dataclass(frozen=True, slots=True)
class Section:
    """A group of options shown as one tab of the app."""

    key: str
    title: str
    subtitle: str
    options: tuple[Option, ...] = field(default_factory=tuple)


def _o(
    path: str,
    label: str,
    kind: str,
    what: str,
    why: str,
    when: str,
    choices: tuple[str, ...] = (),
    visible_when: str = "",
    advanced: bool = False,
    placeholder: str = "",
    csv_is_list: bool = True,
    numeric: bool = False,
) -> Option:
    """A short constructor for one option."""
    return Option(path, label, kind, what, why, when, choices, visible_when, advanced, placeholder, csv_is_list, numeric)


# --------------------------------------------------------------------------------------------
# The sections, in the order the app shows them. Only the seven file keys live here: everything
# else (output folder, quality, AI server, ...) is a run-time choice.
# --------------------------------------------------------------------------------------------

VIDEOS = Section(
    "videos",
    "Videos",
    "What to cut",
    (
        _o(
            "input",
            "Videos to process",
            "lines",
            "The videos to work on: full paths, a glob such as `talks/*.mp4`, or http(s) URLs to download.",
            "The batch is the unit of work: every video gets its own clips, and one broken file never stops the others.",
            "Fill this in first - it is the only option without a useful default.",
            placeholder="one path, glob or URL per line",
        ),
    ),
)

CLIPS = Section(
    "clips",
    "Clips",
    "Which moments become clips",
    (
        _o(
            "clips.preset",
            "What to produce",
            "choice",
            "`both` renders vertical shorts and horizontal long clips, `shorts` only the 20-60 s vertical ones, "
            "`long` only the 3-10 minute horizontal ones.",
            "The two families have their own length, shape and destination; most projects want one of them.",
            "`shorts` for TikTok/Reels/Shorts, `long` for YouTube compilations, `both` for a full package.",
            choices=("both", "shorts", "long"),
        ),
        _o(
            "clips.scenes",
            "Scene types",
            "csv",
            "The style to pick moments for: one name, a comma separated list, or `all`. "
            "`best` is the general-purpose choice; every name gets its own clips.",
            "A cooking channel wants `best`, a comedy channel `funny`, an explainer channel `educational`.",
            "Start with one scene type; add others once the clips look right.",
            placeholder="best",
            csv_is_list=False,
        ),
        _o(
            "clips.keywords",
            "Keywords",
            "csv",
            "Words or phrases (any language) that make a moment more interesting.",
            "Your audience cares about specific topics that the general scene weights cannot know.",
            "Add the names, products or topics that should appear in the clips.",
            placeholder="pricing, demo, the secret",
        ),
        _o(
            "clips.ranges",
            "Exact ranges",
            "lines",
            "Exact cuts, one per line: `00:01:00-00:01:45`. When this is filled in, the selected scene types "
            "render these ranges instead of choosing highlights.",
            "You already know the timestamps and want exactly those, or you want a quick test cut.",
            "Leave it empty to let autoclip pick the best moments.",
            placeholder="00:01:00-00:01:45",
        ),
    ),
)

DOWNLOAD = Section(
    "download",
    "Download",
    "Which platforms may be downloaded",
    (
        _o(
            "download.platforms.allow",
            "Allowed platforms",
            "csv",
            "The platforms that may be downloaded: platform names (`youtube`, `vimeo`, ...) and the groups "
            "`@popular`, `@social`, `@video`, `@all`.",
            "A URL is only downloaded when it is on this list - everything else is refused before a request is made.",
            "Start with `@popular` and add the platforms you use.",
        ),
        _o(
            "download.platforms.deny",
            "Denied platforms",
            "csv",
            "Platforms that are never downloaded, even when an allow entry covers them.",
            "Deny wins over allow, so a broad group can stay while one platform is excluded.",
            "Add a platform that you do not want to support.",
        ),
    ),
)


#: Every section, in the order the app shows them.
SECTIONS: tuple[Section, ...] = (VIDEOS, CLIPS, DOWNLOAD)

#: Every option, by its config path.
OPTIONS: dict[str, Option] = {o.path: o for s in SECTIONS for o in s.options}

#: Options that hold a whole JSON object: the form treats one as a single leaf, not as nested options.
JSON_PATHS: frozenset[str] = frozenset(path for path, option in OPTIONS.items() if option.kind == "json")


def option_paths() -> list[str]:
    """The config paths of every option, in the order the app shows them."""
    return [o.path for s in SECTIONS for o in s.options]
