// What a moth looks like, decoded from its genes, how visible it is on the patch of bark under
// it, and how it and its DNA strip are drawn.

import { EYESPOTS, GENES, GENE_COUNT, SHADE, SIZE, SPECKLES, TINT, createRng } from "./ga";
import { lch, oklabToCss, oklchToOklab } from "./habitat";

const TAU = Math.PI * 2;
/** The wingspan of a moth of size 0.5, in CSS pixels. */
export const WINGSPAN = 40;

/** The moth that a genome codes for: colours in OKLab, sizes in CSS pixels. */
export function decode(genome) {
  const lightness = 0.2 + 0.72 * genome[SHADE];
  // from a warm grey-brown to a lichen green, more colourful toward green
  const hue = 40 + 110 * genome[TINT];
  const chroma = 0.012 + 0.07 * genome[TINT] ** 1.2;
  const wing = oklchToOklab(lightness, chroma, hue);
  return {
    wing,
    lightness,
    hue,
    chroma,
    speckles: genome[SPECKLES],
    // the spread of lightness the flecks give the wings, comparable to the bark's roughness
    roughness: 0.02 + 0.16 * genome[SPECKLES],
    eyespots: genome[EYESPOTS],
    span: WINGSPAN * (0.7 + 0.6 * genome[SIZE]),
  };
}

/**
 * How visible a moth is on a patch of bark, 0 (invisible) to 1 (stands out): the difference in
 * lightness counts the most, then in colour and roughness; eyespots catch the eye anywhere, and
 * a large moth is easier to see than a small one. A bird hunts the visible ones first.
 */
export function visibility(look, patch) {
  const [L, a, b] = look.wing;
  const [pl, pa, pb] = patch.color;
  const dL = Math.abs(L - pl) / 0.5;
  const dColor = Math.hypot(a - pa, b - pb) / 0.07;
  const dRough = Math.abs(look.roughness - patch.roughness) / 0.12;
  const raw = 1.6 * dL + 0.55 * dColor + 0.3 * dRough + 1.4 * look.eyespots ** 1.4;
  const size = 0.62 + 0.38 * ((look.span / WINGSPAN - 0.7) / 0.6);
  return (1 - Math.exp(-raw)) * size;
}

/**
 * What gave a moth away on its patch: the largest part of its visibility, in words, for the
 * player to learn what the bird saw.
 */
export function giveaway(look, patch) {
  const [L, a, b] = look.wing;
  const [pl, pa, pb] = patch.color;
  const parts = [
    [1.6 * (Math.abs(L - pl) / 0.5), L < pl ? "too dark for the bark" : "too pale for the bark"],
    [0.55 * (Math.hypot(a - pa, b - pb) / 0.07), "the wrong colour"],
    [0.3 * (Math.abs(look.roughness - patch.roughness) / 0.12), look.roughness > patch.roughness ? "too speckled" : "too plain"],
    [1.4 * look.eyespots ** 1.4, "its eyespots"],
  ];
  parts.sort((x, y) => y[0] - x[0]);
  return parts[0][1];
}

// the wing outline of the right side, in units of the half span: a forewing swept back from the
// shoulder to its tip, and a rounder hindwing under it
const FOREWING = [
  [0.06, -0.22],
  [0.45, -0.42, 0.95, -0.28, 1, -0.06],
  [0.86, 0.08, 0.5, 0.2, 0.06, 0.14],
];
const HINDWING = [
  [0.05, 0.06],
  [0.42, 0.06, 0.66, 0.26, 0.56, 0.46],
  [0.4, 0.62, 0.14, 0.5, 0.04, 0.3],
];

function wingPath(ctx, shape, side, half) {
  const [[x0, y0], ...curves] = shape;
  ctx.moveTo(side * x0 * half, y0 * half);
  for (const [c1x, c1y, c2x, c2y, x, y] of curves) {
    ctx.bezierCurveTo(side * c1x * half, c1y * half, side * c2x * half, c2y * half, side * x * half, y * half);
  }
  ctx.closePath();
}

/**
 * A moth at rest at (x, y), head up, turned by `angle`, its wings `open` (1 at rest, less while
 * it flutters), with `alpha`. `seed` places its flecks, so a moth always looks the same.
 */
export function drawMoth(ctx, look, seed, x, y, angle, open, alpha, dpr) {
  const half = look.span / 2;
  const cos = Math.cos(angle) * dpr;
  const sin = Math.sin(angle) * dpr;
  ctx.setTransform(cos, sin, -sin, cos, x * dpr, y * dpr);
  ctx.globalAlpha = alpha;
  const wing = oklabToCss(look.wing);
  const [L, a, b] = look.wing;
  const edge = oklabToCss([Math.max(0, L - 0.12), a, b]);
  for (const side of [-1, 1]) {
    ctx.save();
    ctx.scale(open, 1);
    // hindwing, then forewing over it
    for (const shape of [HINDWING, FOREWING]) {
      ctx.beginPath();
      wingPath(ctx, shape, side, half);
      ctx.fillStyle = shape === HINDWING ? oklabToCss([L - 0.05, a, b]) : wing;
      ctx.fill();
      ctx.save();
      ctx.clip();
      drawFlecks(ctx, look, seed + (side > 0 ? 7 : 0) + (shape === HINDWING ? 13 : 0), side, half);
      ctx.restore();
      ctx.lineWidth = 0.6;
      ctx.strokeStyle = edge;
      ctx.stroke();
    }
    if (look.eyespots > 0.08) {
      const r = half * 0.2 * look.eyespots;
      const ex = side * half * 0.55;
      const ey = -0.08 * half;
      ctx.beginPath();
      ctx.arc(ex, ey, r * 1.35, 0, TAU);
      ctx.fillStyle = "rgb(30 22 18)";
      ctx.fill();
      ctx.beginPath();
      ctx.arc(ex, ey, r, 0, TAU);
      ctx.fillStyle = lch(0.9, 0.16, 95);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(ex - r * 0.25, ey - r * 0.25, r * 0.35, 0, TAU);
      ctx.fillStyle = "rgb(255 255 255 / 0.85)";
      ctx.fill();
    }
    ctx.restore();
  }
  // the body and the antennae
  ctx.beginPath();
  ctx.ellipse(0, 0.02 * half, half * 0.07, half * 0.3, 0, 0, TAU);
  ctx.fillStyle = oklabToCss([Math.max(0.05, L - 0.2), a * 0.6, b * 0.6]);
  ctx.fill();
  ctx.beginPath();
  for (const side of [-1, 1]) {
    ctx.moveTo(side * half * 0.03, -half * 0.28);
    ctx.quadraticCurveTo(side * half * 0.12, -half * 0.5, side * half * 0.22, -half * 0.56);
  }
  ctx.lineWidth = 0.7;
  ctx.strokeStyle = edge;
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function drawFlecks(ctx, look, seed, side, half) {
  const count = Math.round(2 + 46 * look.speckles);
  if (count <= 2 && look.speckles < 0.05) return;
  const rng = createRng(seed);
  const [L, a, b] = look.wing;
  const dark = oklabToCss([Math.max(0, L - 0.32), a * 0.5, b * 0.5], 0.85);
  const light = oklabToCss([Math.min(1, L + 0.22), a * 0.5, b * 0.5], 0.75);
  for (let i = 0; i < count; i++) {
    ctx.beginPath();
    ctx.arc(side * rng() * half, (rng() - 0.45) * half * 0.9, 0.5 + rng() * 1.5, 0, TAU);
    ctx.fillStyle = rng() < 0.7 ? dark : light;
    ctx.fill();
  }
}

/** Whether (x, y) is on a moth resting at (mx, my). */
export function onMoth(look, mx, my, x, y, slop) {
  const half = look.span / 2 + slop;
  const dx = (x - mx) / half;
  const dy = (y - my) / (half * 0.62);
  return dx * dx + dy * dy <= 1;
}

// ---- the DNA strip ---------------------------------------------------------------------------

/**
 * One gene's block of a DNA strip, drawn as the moth shows the gene: the wing's shade, its tint,
 * flecks, an eyespot, and a bar for size. `x`, `y` is the block's top left, in CSS pixels.
 */
export function drawGene(ctx, k, value, x, y, w, h, ink) {
  ctx.save();
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, Math.min(5, h / 4));
  ctx.clip();
  if (k === SHADE) {
    ctx.fillStyle = lch(0.2 + 0.72 * value, 0.01, 70);
    ctx.fillRect(x, y, w, h);
  } else if (k === TINT) {
    // the hue at a fixed, readable lightness and a stronger colour than on the wing
    ctx.fillStyle = lch(0.62, 0.02 + 0.12 * value ** 1.2, 40 + 110 * value);
    ctx.fillRect(x, y, w, h);
  } else if (k === SPECKLES) {
    ctx.fillStyle = lch(0.82, 0.01, 80);
    ctx.fillRect(x, y, w, h);
    const rng = createRng(17);
    const count = Math.round(1 + 26 * value);
    ctx.fillStyle = lch(0.3, 0.01, 60);
    for (let i = 0; i < count; i++) {
      ctx.beginPath();
      ctx.arc(x + 2 + rng() * (w - 4), y + 2 + rng() * (h - 4), 0.9 + rng() * 0.9, 0, TAU);
      ctx.fill();
    }
  } else if (k === EYESPOTS) {
    ctx.fillStyle = lch(0.82, 0.01, 80);
    ctx.fillRect(x, y, w, h);
    const r = (Math.min(w, h) / 2 - 2) * value;
    if (r > 0.6) {
      ctx.beginPath();
      ctx.arc(x + w / 2, y + h / 2, r * 1.3, 0, TAU);
      ctx.fillStyle = "rgb(30 22 18)";
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x + w / 2, y + h / 2, r, 0, TAU);
      ctx.fillStyle = lch(0.9, 0.16, 95);
      ctx.fill();
    }
  } else {
    ctx.fillStyle = lch(0.82, 0.01, 80);
    ctx.fillRect(x, y, w, h);
    ctx.fillStyle = lch(0.4, 0.02, 60);
    const bar = (0.15 + 0.85 * value) * (w - 6);
    ctx.fillRect(x + (w - bar) / 2, y + h / 2 - 2.5, bar, 5);
  }
  ctx.restore();
  ctx.beginPath();
  ctx.roundRect(x + 0.5, y + 0.5, w - 1, h - 1, Math.min(5, h / 4));
  ctx.strokeStyle = ink;
  ctx.globalAlpha = 0.35;
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.globalAlpha = 1;
}

/**
 * A DNA strip: the genome's genes side by side, from (x, y), each block `w` x `h` with `gap`.
 * `outline(k)` may return a colour to frame gene k in (whose parent it came from), and
 * `mutated` the genes to mark with a spark.
 */
export function drawStrip(ctx, genome, x, y, w, h, gap, ink, outline, mutated = []) {
  for (let k = 0; k < GENE_COUNT; k++) {
    const gx = x + k * (w + gap);
    drawGene(ctx, k, genome[k], gx, y, w, h, ink);
    const color = outline?.(k);
    if (color) {
      ctx.beginPath();
      ctx.roundRect(gx - 1.5, y - 1.5, w + 3, h + 3, Math.min(6, h / 4 + 1.5));
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    if (mutated.includes(k)) spark(ctx, gx + w - 3, y + 3, 6, ink);
  }
}

export function stripWidth(w, gap) {
  return GENE_COUNT * w + (GENE_COUNT - 1) * gap;
}

/** A four-pointed spark, marking a mutation. */
export function spark(ctx, x, y, size, color) {
  ctx.beginPath();
  ctx.moveTo(x, y - size);
  ctx.quadraticCurveTo(x, y, x + size, y);
  ctx.quadraticCurveTo(x, y, x, y + size);
  ctx.quadraticCurveTo(x, y, x - size, y);
  ctx.quadraticCurveTo(x, y, x, y - size);
  ctx.fillStyle = color;
  ctx.fill();
}

export const GENE_LABELS = GENES.map((gene) => gene.label);
