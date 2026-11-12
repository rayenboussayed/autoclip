# All configuration options

Generated from `autoclip/app/fields.py`; do not edit by hand.

## Videos

What to cut

| Path | Kind | Vision | Details |
|---|---|---|---|
| `input` | `lines` | normal | **What:** The videos to work on: full paths, a glob such as `talks/*.mp4`, or http(s) URLs to download. **Why:** The batch is the unit of work: every video gets its own clips, and one broken file never stops the others. **When:** Fill this in first - it is the only option without a useful default. |

## Clips

Which moments become clips

| Path | Kind | Vision | Details |
|---|---|---|---|
| `clips.preset` | `choice` | normal | **What:** `both` renders vertical shorts and horizontal long clips, `shorts` only the 20-60 s vertical ones, `long` only the 3-10 minute horizontal ones. **Why:** The two families have their own length, shape and destination; most projects want one of them. **When:** `shorts` for TikTok/Reels/Shorts, `long` for YouTube compilations, `both` for a full package. Choices: `both, shorts, long`. |
| `clips.scenes` | `csv` | normal | **What:** The style to pick moments for: one name, a comma separated list, or `all`. `best` is the general-purpose choice; every name gets its own clips. **Why:** A cooking channel wants `best`, a comedy channel `funny`, an explainer channel `educational`. **When:** Start with one scene type; add others once the clips look right. |
| `clips.keywords` | `csv` | normal | **What:** Words or phrases (any language) that make a moment more interesting. **Why:** Your audience cares about specific topics that the general scene weights cannot know. **When:** Add the names, products or topics that should appear in the clips. |
| `clips.ranges` | `lines` | normal | **What:** Exact cuts, one per line: `00:01:00-00:01:45`. When this is filled in, the selected scene types render these ranges instead of choosing highlights. **Why:** You already know the timestamps and want exactly those, or you want a quick test cut. **When:** Leave it empty to let autoclip pick the best moments. |

## Download

Which platforms may be downloaded

| Path | Kind | Vision | Details |
|---|---|---|---|
| `download.platforms.allow` | `csv` | normal | **What:** The platforms that may be downloaded: platform names (`youtube`, `vimeo`, ...) and the groups `@popular`, `@social`, `@video`, `@all`. **Why:** A URL is only downloaded when it is on this list - everything else is refused before a request is made. **When:** Start with `@popular` and add the platforms you use. |
| `download.platforms.deny` | `csv` | normal | **What:** Platforms that are never downloaded, even when an allow entry covers them. **Why:** Deny wins over allow, so a broad group can stay while one platform is excluded. **When:** Add a platform that you do not want to support. |
