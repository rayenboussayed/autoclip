"""Generate GENERATED markdown reference from ``autoclip.app.fields``.

Run ``python tools/export_options.py --write docs/options-reference.md`` to refresh the
catalogue documentation. The catalogue stays the single source of truth; this tool only
mirrors it so users do not have to read the source or run the app.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoclip.app.fields import SECTIONS  # noqa: E402


def render() -> str:
    lines = ["# All configuration options", "", "Generated from `autoclip/app/fields.py`; do not edit by hand.", ""]
    for section in SECTIONS:
        lines.append(f"## {section.title}")
        lines.append("")
        if section.subtitle:
            lines.append(section.subtitle)
            lines.append("")
        lines.append("| Path | Kind | Vision | Details |")
        lines.append("|---|---|---|---|")
        for option in section.options:
            vision = "advanced" if option.advanced else "normal"
            extra = ""
            if option.choices:
                extra = f"Choices: `{', '.join(option.choices)}`."
            lines.append(
                f"| `{option.path}` | `{option.kind}` | {vision} | "
                f"**What:** {option.what} "
                f"**Why:** {option.why} "
                f"**When:** {option.when} {extra}".strip()
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", type=Path, help="write the generated reference to this file instead of stdout")
    args = parser.parse_args()
    text = render()
    if args.write is None:
        print(text, end="")
        return 0
    args.write.write_text(text, encoding="utf-8")
    print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
