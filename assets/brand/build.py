"""genoxide's logo, wordmark, lockups, banner and social preview, as SVG.

Run from the repository's root: python assets/brand/build.py (needs fontTools). It downloads
Space Grotesk (SIL Open Font License 1.1) and converts the letters to paths, so the SVGs need no
font. The PNG exports are made from these SVGs by assets/brand/export.mjs.

The design: genes + oxide. The mark is a double helix in iron oxide's colors, from deep red to
copper and amber, in a hexagon, a bolt head and a chemical ring; the wordmark sets "oxide" in the
same gradient. It's rust-themed without Rust's own logo, a trademark of the Rust Foundation.
"""

import io
import math
import pathlib
import random
import re
import urllib.request

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

OUT = pathlib.Path(__file__).parent
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/spacegrotesk/SpaceGrotesk%5Bwght%5D.ttf"

# iron oxide, from deep red to amber, and the dark card behind the banners
DEEP = "#8a2a10"
RUST = "#b7410e"
ORANGE = "#d9611c"
COPPER = "#ee8a32"
AMBER = "#f7b955"
CREAM = "#fbe7c6"
CARD_TOP = "#2b1d17"
CARD_BOTTOM = "#161010"
INK = "#231a16"  # "gen" on light backgrounds
PAPER = "#f6ede5"  # "gen" on dark backgrounds
MUTED = "#d8c3ae"  # the tagline on the card


def font(weight):
    data = urllib.request.urlopen(FONT_URL, timeout=60).read()
    return instantiateVariableFont(TTFont(io.BytesIO(data)), {"wght": weight})


FONTS = {}


def text_path(text, weight, size, x, y, tracking=0.0):
    """The outline of `text` as one SVG path, its baseline's left end at (x, y); and its width."""
    if weight not in FONTS:
        FONTS[weight] = font(weight)
    f = FONTS[weight]
    scale = size / f["head"].unitsPerEm
    glyphs = f.getGlyphSet()
    cmap = f.getBestCmap()
    d, advance = [], 0.0
    for char in text:
        name = cmap[ord(char)]
        pen = SVGPathPen(glyphs)
        # font units are y-up: flip, scale and move to the baseline
        glyphs[name].draw(TransformPen(pen, (scale, 0, 0, -scale, x + advance, y)))
        # a tenth of a unit is invisible at these sizes, and halves the files
        d.append(re.sub(r"-?\d+\.\d+", lambda m: fmt(round(float(m.group()), 1)), pen.getCommands()))
        advance += glyphs[name].width * scale + tracking * size
    return " ".join(d), advance - tracking * size


def fmt(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


# ---- the mark --------------------------------------------------------------------------------


def hexagon(cx, cy, r):
    points = [(cx + r * math.cos(math.radians(-90 + 60 * k)), cy + r * math.sin(math.radians(-90 + 60 * k))) for k in range(6)]
    return "M" + " L".join(f"{fmt(x)} {fmt(y)}" for x, y in points) + " Z"


def strand(cx, top, bottom, amplitude, phase, turns, steps=64):
    """A helix strand from top to bottom: x = cx + amplitude · sin(2π turns t + phase)."""
    pts = []
    for i in range(steps + 1):
        t = i / steps
        pts.append((cx + amplitude * math.sin(2 * math.pi * turns * t + phase), top + (bottom - top) * t))
    return "M" + " L".join(f"{fmt(x)} {fmt(y)}" for x, y in pts)


def mark(id_prefix, size=128, x=0, y=0):
    """The mark in a `size` square at (x, y): the hexagon, the two strands and their rungs."""
    s = size / 128
    top, bottom, amp, turns = 26, 102, 22, 1.25
    rungs = []
    for k in range(9):
        t = (k + 0.5) / 9
        yy = top + (bottom - top) * t
        a = 64 + amp * math.sin(2 * math.pi * turns * t)
        b = 64 + amp * math.sin(2 * math.pi * turns * t + math.pi)
        if abs(a - b) > 14:  # not where the strands cross
            rungs.append(f'<line x1="{fmt(a)}" y1="{fmt(yy)}" x2="{fmt(b)}" y2="{fmt(yy)}"/>')
    p = id_prefix
    return f'''<g transform="translate({fmt(x)} {fmt(y)}) scale({fmt(s)})">
  <defs>
    <linearGradient id="{p}-face" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{CARD_TOP}"/><stop offset="1" stop-color="{CARD_BOTTOM}"/></linearGradient>
    <linearGradient id="{p}-rim" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{COPPER}"/><stop offset="1" stop-color="{DEEP}"/></linearGradient>
    <linearGradient id="{p}-a" x1="0" y1="{top}" x2="0" y2="{bottom}" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="{AMBER}"/><stop offset="1" stop-color="{ORANGE}"/></linearGradient>
    <linearGradient id="{p}-b" x1="0" y1="{top}" x2="0" y2="{bottom}" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="{RUST}"/><stop offset="1" stop-color="{COPPER}"/></linearGradient>
  </defs>
  <path d="{hexagon(64, 64, 58)}" fill="url(#{p}-face)" stroke="url(#{p}-rim)" stroke-width="6" stroke-linejoin="round"/>
  <g stroke="{CREAM}" stroke-width="3.5" stroke-linecap="round" opacity="0.8">{"".join(rungs)}</g>
  <path d="{strand(64, top, bottom, amp, math.pi, turns)}" fill="none" stroke="url(#{p}-b)" stroke-width="9" stroke-linecap="round"/>
  <path d="{strand(64, top, bottom, amp, 0, turns)}" fill="none" stroke="url(#{p}-a)" stroke-width="9" stroke-linecap="round"/>
</g>'''


def svg(width, height, body, title):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
<title>{title}</title>
{body}
</svg>
'''


# ---- the wordmark ------------------------------------------------------------------------------


def wordmark(id_prefix, gen_color, size, x, y, weight=700):
    """ "genoxide" with "oxide" in the rust gradient, its baseline's left end at (x, y); and its width."""
    tracking = -0.02
    gen, gen_width = text_path("gen", weight, size, x, y, tracking)
    oxide, oxide_width = text_path("oxide", weight, size, x + gen_width + tracking * size, y, tracking)
    width = gen_width + tracking * size + oxide_width
    p = id_prefix
    body = f'''<defs><linearGradient id="{p}-oxide" x1="{fmt(x + gen_width)}" y1="0" x2="{fmt(x + width)}" y2="0" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="{RUST}"/><stop offset="0.55" stop-color="{COPPER}"/><stop offset="1" stop-color="{AMBER}"/></linearGradient></defs>
<path d="{gen}" fill="{gen_color}"/>
<path d="{oxide}" fill="url(#{p}-oxide)"/>'''
    return body, width


# ---- the population on the banners -----------------------------------------------------------


def population(cx, cy, spread, count, seed, width, height):
    """Points of a population converging on (cx, cy): denser and brighter near it."""
    rng = random.Random(seed)
    colors = [DEEP, RUST, ORANGE, COPPER, AMBER]
    dots = []
    for _ in range(count):
        r = spread * math.sqrt(-2 * math.log(1 - rng.random() * 0.999))
        a = rng.random() * 2 * math.pi
        x, y = cx + r * math.cos(a) * 1.6, cy + r * math.sin(a)
        if not (8 < x < width - 8 and 8 < y < height - 8):
            continue
        closeness = math.exp(-((r / spread) ** 2) / 2)
        color = colors[min(4, int(closeness * 5))]
        radius = 1.2 + 2.4 * closeness
        dots.append(f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="{fmt(radius)}" fill="{color}" opacity="{fmt(0.25 + 0.6 * closeness)}"/>')
    return "".join(dots)


def contours(cx, cy, count, step, width, height):
    """Faint level sets of a fitness landscape around (cx, cy)."""
    rings = []
    for k in range(1, count + 1):
        rx, ry = k * step * 1.6, k * step
        rings.append(f'<ellipse cx="{fmt(cx)}" cy="{fmt(cy)}" rx="{fmt(rx)}" ry="{fmt(ry)}"/>')
    return f'<g fill="none" stroke="{COPPER}" stroke-opacity="0.09" stroke-width="1.2">{"".join(rings)}</g>'


def card(p, width, height, radius, glow_x, glow_y):
    """The dark card behind a banner; its ids start with `p`, and `url(#{p}-clip)` clips to it."""
    return f'''<defs>
  <linearGradient id="{p}-card" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{CARD_TOP}"/><stop offset="1" stop-color="{CARD_BOTTOM}"/></linearGradient>
  <radialGradient id="{p}-glow" cx="{fmt(glow_x / width)}" cy="{fmt(glow_y / height)}" r="0.6"><stop offset="0" stop-color="{RUST}" stop-opacity="0.35"/><stop offset="1" stop-color="{RUST}" stop-opacity="0"/></radialGradient>
  <clipPath id="{p}-clip"><rect width="{width}" height="{height}" rx="{radius}"/></clipPath>
</defs>
<rect width="{width}" height="{height}" rx="{radius}" fill="url(#{p}-card)"/>
<rect width="{width}" height="{height}" rx="{radius}" fill="url(#{p}-glow)"/>'''


# ---- the files ----------------------------------------------------------------------------------


def write(name, text):
    (OUT / name).write_text(text, encoding="utf-8", newline="\n")
    print(f"{name}: {len(text.encode()) / 1024:.1f} KB")


def main():
    # every id starts with the file's own prefix: the SVGs can be inlined on one page together
    write("logo.svg", svg(128, 128, mark("gx-logo"), "genoxide"))

    # the wordmark alone, for light and dark backgrounds
    for theme, color in [("light", INK), ("dark", PAPER)]:
        body, width = wordmark(f"gx-wordmark-{theme}", color, 96, 0, 76)
        write(f"wordmark-{theme}.svg", svg(math.ceil(width) + 4, 100, body, "genoxide"))

    # the horizontal lockup: the mark beside the wordmark
    for theme, color in [("light", INK), ("dark", PAPER)]:
        p = f"gx-lockup-{theme}"
        body, width = wordmark(p, color, 96, 140, 88)
        write(f"lockup-{theme}.svg", svg(math.ceil(140 + width) + 4, 128, mark(p, 128) + body, "genoxide"))

    # the banner: a dark card, the same on both themes, 4:1
    W, H, p = 1280, 320, "gx-banner"
    _, width = wordmark(p, PAPER, 112, 0, 0)
    left = (W - (200 + 28 + width)) / 2
    word, _ = wordmark(p, PAPER, 112, left + 228, 170)
    tagline, _ = text_path("Optimization for Rust and Python", 500, 30, left + 232, 226)
    banner = (
        card(p, W, H, 28, left + 100, 160)
        + f'<g clip-path="url(#{p}-clip)">{contours(1190, 70, 8, 22, W, H)}{population(1190, 70, 48, 120, 7, W, H)}'
        + f"{population(110, 270, 50, 55, 11, W, H)}</g>"
        + mark(p, 200, left, 60)
        + word
        + f'<path d="{tagline}" fill="{MUTED}"/>'
    )
    write("banner.svg", svg(W, H, banner, "genoxide: optimization for Rust and Python"))

    # the social preview, 2:1 (exported to PNG for GitHub)
    W, H, p = 1280, 640, "gx-social"
    _, width = wordmark(p, PAPER, 132, 0, 0)
    left = (W - (236 + 32 + width)) / 2
    word, _ = wordmark(p, PAPER, 132, left + 268, 285)
    line = "Optimization for Rust and Python"
    _, tagline_width = text_path(line, 500, 36, 0, 0)
    tagline, _ = text_path(line, 500, 36, (W - tagline_width) / 2, 440)
    line = "genetic algorithms · CMA-ES · differential evolution · multi-objective · L-BFGS-B · Adam · MMA"
    _, details_width = text_path(line, 400, 22, 0, 0)
    details, _ = text_path(line, 400, 22, (W - details_width) / 2, 490)
    social = (
        card(p, W, H, 0, left + 118, 260)
        + f'<g clip-path="url(#{p}-clip)">{contours(1130, 110, 9, 26, W, H)}{population(1130, 110, 70, 190, 3, W, H)}'
        + f"{population(110, 600, 50, 70, 5, W, H)}</g>"
        + mark(p, 236, left, 145)
        + word
        + f'<path d="{tagline}" fill="{MUTED}"/>'
        + f'<path d="{details}" fill="{MUTED}" opacity="0.7"/>'
    )
    write("social-preview.svg", svg(W, H, social, "genoxide: optimization for Rust and Python"))


if __name__ == "__main__":
    main()
