// The bark the moths rest on, drawn once per habitat with seeded noise, and what a bird sees of
// it: the average colour and the roughness of every small patch, to measure how well a moth
// blends into the patch under it.

import { createRng } from "./ga";

// ---- colour: OKLab / OKLCH (Ottosson, 2020) to and from sRGB ---------------------------------

const toLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const toGamma = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

/** [L, a, b] of an sRGB colour with channels in 0..255. */
export function rgbToOklab(r8, g8, b8) {
  const r = toLinear(r8 / 255);
  const g = toLinear(g8 / 255);
  const b = toLinear(b8 / 255);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
}

/** [L, a, b] of an OKLCH colour, hue in degrees. */
export function oklchToOklab(l, c, h) {
  const rad = (h * Math.PI) / 180;
  return [l, c * Math.cos(rad), c * Math.sin(rad)];
}

/** A CSS rgb() colour for OKLab [L, a, b], clipped to sRGB, with an optional alpha. */
export function oklabToCss([L, a, bb], alpha = 1) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * bb) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * bb) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * bb) ** 3;
  const channel = (v) => Math.round(255 * Math.min(1, Math.max(0, toGamma(Math.min(1, Math.max(0, v))))));
  const r = channel(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s);
  const g = channel(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s);
  const b = channel(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s);
  return alpha >= 1 ? `rgb(${r} ${g} ${b})` : `rgb(${r} ${g} ${b} / ${alpha})`;
}

export const lch = (l, c, h, alpha) => oklabToCss(oklchToOklab(l, c, h), alpha);

// ---- the habitats ----------------------------------------------------------------------------

/**
 * Each habitat: its name, what it looks like, and how to paint it. `base` is the bark's colour
 * (OKLCH), `marks` the darker and lighter marks on it.
 */
export const HABITATS = {
  birch: {
    name: "Birch bark",
    base: [0.86, 0.018, 85],
    marks: [
      { kind: "lenticel", l: 0.32, c: 0.02, h: 60, count: 220 },
      { kind: "lichen", l: 0.74, c: 0.045, h: 125, count: 40 },
      { kind: "fleck", l: 0.5, c: 0.02, h: 70, count: 5200 },
    ],
  },
  soot: {
    name: "Sooty bark",
    base: [0.3, 0.012, 60],
    marks: [
      { kind: "streak", l: 0.22, c: 0.01, h: 50, count: 90 },
      { kind: "fleck", l: 0.45, c: 0.012, h: 70, count: 2200 },
      { kind: "lenticel", l: 0.16, c: 0.01, h: 40, count: 60 },
    ],
  },
  lichen: {
    name: "Lichen-covered bark",
    base: [0.66, 0.06, 135],
    marks: [
      { kind: "lichen", l: 0.56, c: 0.07, h: 140, count: 70 },
      { kind: "lichen", l: 0.76, c: 0.05, h: 115, count: 50 },
      { kind: "fleck", l: 0.46, c: 0.04, h: 120, count: 3600 },
    ],
  },
};

function paintMark(ctx, mark, rng, width, height) {
  ctx.fillStyle = lch(mark.l, mark.c, mark.h, mark.kind === "fleck" ? 0.55 : 0.85);
  for (let i = 0; i < mark.count; i++) {
    const x = rng() * width;
    const y = rng() * height;
    ctx.beginPath();
    if (mark.kind === "lenticel") {
      // birch's dark horizontal slits
      const w = 8 + rng() * 40;
      ctx.ellipse(x, y, w, 1.2 + rng() * 2.4, (rng() - 0.5) * 0.12, 0, Math.PI * 2);
    } else if (mark.kind === "streak") {
      // rough vertical furrows of old bark
      const h = 30 + rng() * 120;
      ctx.ellipse(x, y, 2 + rng() * 5, h, (rng() - 0.5) * 0.2, 0, Math.PI * 2);
    } else if (mark.kind === "lichen") {
      // round, lumpy patches
      const r = 6 + rng() * 26;
      for (let k = 0; k < 5; k++) {
        ctx.moveTo(x + r, y);
        ctx.ellipse(x + (rng() - 0.5) * r, y + (rng() - 0.5) * r, r * (0.4 + rng() * 0.5), r * (0.3 + rng() * 0.4), rng() * Math.PI, 0, Math.PI * 2);
      }
    } else {
      ctx.arc(x, y, 0.6 + rng() * 1.6, 0, Math.PI * 2);
    }
    ctx.fill();
  }
}

/**
 * A habitat painted on a canvas of `width` x `height` CSS pixels at `dpr`, and the patch
 * statistics of what it shows: for every cell of CELL pixels, its mean OKLab colour and the
 * spread of its lightness (how rough the bark is there).
 */
export const CELL = 12;

export function paintHabitat(key, width, height, seed, dpr = 1) {
  const habitat = HABITATS[key];
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(width * dpr));
  canvas.height = Math.max(1, Math.round(height * dpr));
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.scale(dpr, dpr);
  const rng = createRng(seed);
  const [l, c, h] = habitat.base;
  ctx.fillStyle = lch(l, c, h);
  ctx.fillRect(0, 0, width, height);
  // a soft, uneven tone across the trunk
  for (let i = 0; i < 18; i++) {
    const shade = (rng() - 0.5) * 0.08;
    ctx.fillStyle = lch(l + shade, c, h + (rng() - 0.5) * 12, 0.35);
    ctx.beginPath();
    ctx.ellipse(rng() * width, rng() * height, 60 + rng() * 220, 40 + rng() * 160, rng() * Math.PI, 0, Math.PI * 2);
    ctx.fill();
  }
  for (const mark of habitat.marks) paintMark(ctx, { ...mark, count: Math.round((mark.count * width * height) / 1.2e6) + 4 }, rng, width, height);
  return { key, canvas, stats: patchStats(ctx, canvas.width, canvas.height, width, height, dpr) };
}

function patchStats(ctx, pixelWidth, pixelHeight, width, height, dpr) {
  const columns = Math.ceil(width / CELL);
  const rows = Math.ceil(height / CELL);
  const data = ctx.getImageData(0, 0, pixelWidth, pixelHeight).data;
  const mean = new Float32Array(columns * rows * 3);
  const rough = new Float32Array(columns * rows);
  const step = Math.max(1, Math.round(dpr * 2));
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      let n = 0;
      let sl = 0;
      let sa = 0;
      let sb = 0;
      let sll = 0;
      const x0 = Math.floor(column * CELL * dpr);
      const y0 = Math.floor(row * CELL * dpr);
      const x1 = Math.min(pixelWidth, Math.floor((column + 1) * CELL * dpr));
      const y1 = Math.min(pixelHeight, Math.floor((row + 1) * CELL * dpr));
      for (let y = y0; y < y1; y += step) {
        for (let x = x0; x < x1; x += step) {
          const i = (y * pixelWidth + x) * 4;
          const [L, a, b] = rgbToOklab(data[i], data[i + 1], data[i + 2]);
          sl += L;
          sa += a;
          sb += b;
          sll += L * L;
          n += 1;
        }
      }
      const cell = row * columns + column;
      const meanL = n ? sl / n : 0;
      mean[cell * 3] = meanL;
      mean[cell * 3 + 1] = n ? sa / n : 0;
      mean[cell * 3 + 2] = n ? sb / n : 0;
      rough[cell] = n ? Math.sqrt(Math.max(0, sll / n - meanL * meanL)) : 0;
    }
  }
  return { columns, rows, mean, rough };
}

/** The mean colour [L, a, b] and roughness of the patch of `radius` around (x, y). */
export function patchAt(stats, x, y, radius) {
  const { columns, rows, mean, rough } = stats;
  const c0 = Math.max(0, Math.floor((x - radius) / CELL));
  const c1 = Math.min(columns - 1, Math.floor((x + radius) / CELL));
  const r0 = Math.max(0, Math.floor((y - radius) / CELL));
  const r1 = Math.min(rows - 1, Math.floor((y + radius) / CELL));
  let n = 0;
  const color = [0, 0, 0];
  let roughness = 0;
  for (let r = r0; r <= r1; r++) {
    for (let c = c0; c <= c1; c++) {
      const cell = r * columns + c;
      color[0] += mean[cell * 3];
      color[1] += mean[cell * 3 + 1];
      color[2] += mean[cell * 3 + 2];
      roughness += rough[cell];
      n += 1;
    }
  }
  if (!n) return { color: [0.5, 0, 0], roughness: 0 };
  return { color: color.map((v) => v / n), roughness: roughness / n };
}
