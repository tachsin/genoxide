// Drawing on a 2D canvas: the creatures (a double helix each), rings,
// sparkles and the gene strips of a crossover. Colours come from the
// active daisyUI theme, read from its CSS variables.

const TAU = Math.PI * 2;
// the second strand trails the first by less than half a turn, like DNA's grooves
const STRAND_OFFSET = Math.PI * 0.8;

let probe = null;

/** 0 (black) to 1 (white), for any CSS colour the canvas understands. */
function lightness(color) {
  if (!probe) {
    const canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;
    probe = canvas.getContext("2d", { willReadFrequently: true });
  }
  probe.clearRect(0, 0, 1, 1);
  probe.fillStyle = "#fff";
  probe.fillStyle = color;
  probe.fillRect(0, 0, 1, 1);
  const [r, g, b] = probe.getImageData(0, 0, 1, 1).data;
  return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
}

let themeVersion = 0;

/** The theme's colours, from the CSS variables daisyUI sets on the page. */
export function readTheme(element) {
  const style = getComputedStyle(element);
  const read = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
  const base = read("--color-base-100", "#ffffff");
  const ink = read("--color-base-content", "#1f2937");
  const dark = lightness(base) < 0.45;
  themeVersion += 1;
  return {
    version: themeVersion,
    dark,
    base,
    ink,
    primary: read("--color-primary", ink),
    // creature lightness in oklch: readable on the theme's background
    strand: dark ? 0.78 : 0.6,
    rung: dark ? 0.66 : 0.72,
  };
}

/** Calls `onChange` when the site's theme changes (data-theme or class on <html>, or the OS scheme). */
export function observeTheme(onChange) {
  const observer = new MutationObserver(onChange);
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class"] });
  const scheme = window.matchMedia("(prefers-color-scheme: dark)");
  scheme.addEventListener("change", onChange);
  return () => {
    observer.disconnect();
    scheme.removeEventListener("change", onChange);
  };
}

const fixed = (value) => Math.round(value * 1000) / 1000;

export function oklch(l, c, h) {
  return `oklch(${fixed(l)} ${fixed(c)} ${fixed(((h % 360) + 360) % 360)})`;
}

/** The creature's three colours, rebuilt only when its look or the theme changed. */
export function paint(creature, theme) {
  const { look, colors } = creature;
  colors.strand1 = oklch(theme.strand, look.chroma, look.hue);
  colors.strand2 = oklch(theme.strand - 0.04, look.chroma, look.hue2);
  colors.rung = oklch(theme.rung, look.chroma * 0.55, (look.hue + look.hue2) / 2);
  creature.painted = theme.version;
}

/** Where a strand is at u in [-0.5, 0.5] along the body, in the creature's frame. */
function axis(look, u) {
  return look.bend * look.length * (u * u - 0.25);
}

function envelope(look, u) {
  const t = 2 * Math.abs(u);
  return Math.max(0.22, 1 - look.taper * 0.75 * t * t);
}

function strandY(look, u, phase, strand, open) {
  const theta = phase + TAU * look.turns * (u + 0.5) + strand * STRAND_OFFSET;
  const apart = (strand ? 1 : -1) * open * look.radius * 0.9;
  return axis(look, u) + apart + look.radius * envelope(look, u) * (1 + open * 0.6) * Math.sin(theta);
}

/**
 * A creature: two strands, the rungs between them (lines, beads or two-tone
 * bars, from the pattern gene) and, while it breeds, the strands pulled
 * apart (`open`, 0 to 1).
 */
export function drawHelix(ctx, look, colors, x, y, angle, scale, phase, open, alpha, dpr) {
  const cos = Math.cos(angle) * scale * dpr;
  const sin = Math.sin(angle) * scale * dpr;
  ctx.setTransform(cos, sin, -sin, cos, x * dpr, y * dpr);
  ctx.globalAlpha = alpha;

  const length = look.length;
  const rungs = look.rungs;
  const gap = open * 0.5;

  // rungs, or the base pairs' two halves
  if (look.pattern === 2) {
    ctx.lineWidth = look.thickness * 1.35;
    for (let strand = 0; strand < 2; strand++) {
      ctx.strokeStyle = strand ? colors.strand2 : colors.strand1;
      ctx.beginPath();
      for (let r = 0; r < rungs; r++) {
        const u = (r + 0.5) / rungs - 0.5;
        const lx = u * length;
        const y1 = strandY(look, u, phase, 0, open);
        const y2 = strandY(look, u, phase, 1, open);
        const from = strand ? y2 : y1;
        const to = strand ? y1 : y2;
        ctx.moveTo(lx, from);
        ctx.lineTo(lx, from + (to - from) * (0.42 - gap));
      }
      ctx.stroke();
    }
  } else {
    ctx.lineWidth = Math.max(0.8, look.thickness * 0.55);
    ctx.strokeStyle = colors.rung;
    ctx.beginPath();
    for (let r = 0; r < rungs; r++) {
      const u = (r + 0.5) / rungs - 0.5;
      const lx = u * length;
      const y1 = strandY(look, u, phase, 0, open);
      const y2 = strandY(look, u, phase, 1, open);
      ctx.moveTo(lx, y1);
      ctx.lineTo(lx, y1 + (y2 - y1) * (0.5 - gap));
      ctx.moveTo(lx, y2);
      ctx.lineTo(lx, y2 + (y1 - y2) * (0.5 - gap));
    }
    ctx.stroke();
  }

  // the strands
  const samples = Math.max(16, Math.min(48, Math.round(look.turns * 12) + 8));
  ctx.lineWidth = look.thickness;
  for (let strand = 0; strand < 2; strand++) {
    ctx.strokeStyle = strand ? colors.strand2 : colors.strand1;
    ctx.beginPath();
    for (let i = 0; i < samples; i++) {
      const u = i / (samples - 1) - 0.5;
      const ly = strandY(look, u, phase, strand, open);
      if (i === 0) ctx.moveTo(u * length, ly);
      else ctx.lineTo(u * length, ly);
    }
    ctx.stroke();
  }

  // beads where the rungs meet the strands, larger on the near side
  if (look.pattern === 1) {
    for (let strand = 0; strand < 2; strand++) {
      ctx.fillStyle = strand ? colors.strand2 : colors.strand1;
      ctx.beginPath();
      for (let r = 0; r < rungs; r++) {
        const u = (r + 0.5) / rungs - 0.5;
        const theta = phase + TAU * look.turns * (u + 0.5) + strand * STRAND_OFFSET;
        const size = look.thickness * (0.75 + 0.45 * Math.cos(theta));
        const ly = strandY(look, u, phase, strand, open);
        ctx.moveTo(u * length + size, ly);
        ctx.arc(u * length, ly, size, 0, TAU);
      }
      ctx.fill();
    }
  }
}

/** The distance from the body's axis to its outline, for hit tests and rings. */
export function reach(look) {
  return look.length / 2 + look.radius;
}

/** Whether (x, y) in the creature's frame (unscaled) is on its body, with `slop` px to spare. */
export function onBody(look, x, y, slop) {
  const half = look.length / 2;
  if (Math.abs(x) > half + slop) return false;
  const u = Math.max(-0.5, Math.min(0.5, x / look.length));
  return Math.abs(y - axis(look, u)) <= look.radius * (1.6 + Math.abs(look.bend)) + slop;
}

export function ring(ctx, x, y, radius, color, width, alpha, dpr, dash) {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, TAU);
  ctx.stroke();
  if (dash) ctx.setLineDash(NO_DASH);
}

const NO_DASH = [];

/** A four-pointed sparkle. */
export function sparkle(ctx, x, y, size) {
  ctx.moveTo(x, y - size);
  ctx.lineTo(x + size * 0.28, y - size * 0.28);
  ctx.lineTo(x + size, y);
  ctx.lineTo(x + size * 0.28, y + size * 0.28);
  ctx.lineTo(x, y + size);
  ctx.lineTo(x - size * 0.28, y + size * 0.28);
  ctx.lineTo(x - size, y);
  ctx.lineTo(x - size * 0.28, y - size * 0.28);
  ctx.closePath();
}
