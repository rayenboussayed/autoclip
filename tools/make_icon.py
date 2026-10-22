"""Draw the icon of the app.

    python tools/make_icon.py assets/autoclip.png

The icon is a dark rounded square with a white play triangle that is "cut" by a yellow bar - the
app cuts a video into clips. It is drawn here (instead of being shipped as a binary) so that it can
be changed and rebuilt with one command.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

BACKGROUND = (16, 16, 16, 255)
FOREGROUND = (255, 255, 255, 255)
ACCENT = (255, 212, 0, 255)


def draw(size: int = 512) -> Image.Image:
    """The icon as a square image with an alpha channel."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw_ = ImageDraw.Draw(image)
    m = size // 16  # a margin of 1/16
    draw_.rounded_rectangle((m, m, size - m, size - m), radius=size // 6, fill=BACKGROUND)
    # the play triangle
    left, top = size * 0.30, size * 0.28
    right, bottom = size * 0.66, size * 0.72
    draw_.polygon([(left, top), (left, bottom), (right, size / 2)], fill=FOREGROUND)
    # the cut
    bar = size * 0.055
    draw_.rounded_rectangle((size * 0.72, size * 0.22, size * 0.72 + bar, size * 0.78), radius=bar / 2, fill=ACCENT)
    return image


def main(argv: list[str]) -> int:
    """Write the icon to the file the command line names."""
    target = Path(argv[1] if len(argv) > 1 else "assets/autoclip.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    draw().save(target)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
