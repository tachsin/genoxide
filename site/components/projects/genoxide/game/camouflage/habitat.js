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

// ---- noise -----------------------------------------------------------------------------------

// a hash of integer coordinates to [0, 1), different for each seed
function hash(x, y, seed) {
  let h = Math.imul(x | 0, 0x27d4eb2d) ^ Math.imul(y | 0, 0x165667b1) ^ Math.imul(seed | 0, 0x9e3779b1);
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

const smooth = (t) => t * t * (3 - 2 * t);

// value noise in [0, 1], smooth between integer coordinates
function noise(x, y, seed) {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = smooth(x - ix);
  const fy = smooth(y - iy);
  const a = hash(ix, iy, seed);
  const b = hash(ix + 1, iy, seed);
  const c = hash(ix, iy + 1, seed);
  const d = hash(ix + 1, iy + 1, seed);
  return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
}

// fractal noise: `octaves` layers of value noise, each twice as fine and half as strong
function fbm(x, y, seed, octaves = 4) {
  let sum = 0;
  let amplitude = 0.5;
  let total = 0;
  let f = 1;
  for (let i = 0; i < octaves; i++) {
    sum += amplitude * noise(x * f, y * f, seed + i * 101);
    total += amplitude;
    amplitude *= 0.5;
    f *= 2;
  }
  return sum / total;
}

// cellular noise: the distances to the nearest and second-nearest of one jittered point per unit
// cell; F2 − F1 is small along the borders between cells, the cracks between plates of bark
function cells(x, y, seed) {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  let f1 = Infinity;
  let f2 = Infinity;
  let id = 0;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const cx = ix + dx;
      const cy = iy + dy;
      const px = cx + hash(cx, cy, seed);
      const py = cy + hash(cx, cy, seed + 1);
      const d = Math.hypot(px - x, py - y);
      if (d < f1) {
        f2 = f1;
        f1 = d;
        id = hash(cx, cy, seed + 2);
      } else if (d < f2) {
        f2 = d;
      }
    }
  }
  return [f1, f2, id];
}

/** sRGB bytes of an OKLab colour, clipped. */
function oklabToRgb8(L, a, bb, out, i) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * bb) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * bb) ** 3;
  const s = (L - 0.0894841775 * a - 1.291485548 * bb) ** 3;
  const channel = (v) => Math.round(255 * Math.min(1, Math.max(0, toGamma(Math.min(1, Math.max(0, v))))));
  out[i] = channel(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s);
  out[i + 1] = channel(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s);
  out[i + 2] = channel(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s);
  out[i + 3] = 255;
}

// the base of a habitat, pixel by pixel at a `scale` of its size: `color(x, y)` in CSS pixels
// returns [L, a, b]; drawn smoothed onto the full canvas
function paintPixels(ctx, width, height, scale, color) {
  const w = Math.max(1, Math.round(width * scale));
  const h = Math.max(1, Math.round(height * scale));
  const small = document.createElement("canvas");
  small.width = w;
  small.height = h;
  const sctx = small.getContext("2d");
  const image = sctx.createImageData(w, h);
  const data = image.data;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [L, a, b] = color(x / scale, y / scale);
      oklabToRgb8(L, a, b, data, (y * w + x) * 4);
    }
  }
  sctx.putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(small, 0, 0, width, height);
}

const lab = (l, c, h) => oklchToOklab(l, c, h);

// an irregular closed shape around (x, y): an ellipse of radii rx, ry whose radius wobbles by
// `rough` (0 to 1) with a few random harmonics
function blobPath(ctx, x, y, rx, ry, rng, rough, angle = 0) {
  const harmonics = [2, 3, 5, 7].map((k) => [k, rng() * Math.PI * 2, (rng() * rough) / k]);
  const points = 28;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  ctx.beginPath();
  for (let i = 0; i <= points; i++) {
    const t = (i / points) * Math.PI * 2;
    let r = 1;
    for (const [k, phase, amplitude] of harmonics) r += amplitude * Math.sin(k * t + phase);
    const px = Math.cos(t) * rx * r;
    const py = Math.sin(t) * ry * r;
    const X = x + px * cos - py * sin;
    const Y = y + px * sin + py * cos;
    if (i) ctx.lineTo(X, Y);
    else ctx.moveTo(X, Y);
  }
  ctx.closePath();
}

// a horizontal sliver from x - w/2 to x + w/2, thickest in the middle, with ragged edges: a
// lenticel, a crack, the edge of a peeling strip
function sliverPath(ctx, x, y, w, h, rng) {
  const steps = Math.max(6, Math.round(w / 3));
  const top = [];
  const bottom = [];
  const tilt = (rng() - 0.5) * 0.05;
  const wobble = rng() * 100;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const px = x - w / 2 + t * w;
    const profile = Math.sin(Math.PI * t) ** 0.55;
    const ragged = 0.6 + 0.4 * noise(i * 0.8, wobble, 3);
    const center = y + (px - x) * tilt + (noise(i * 0.3, wobble, 5) - 0.5) * h * 0.6;
    top.push([px, center - (h / 2) * profile * ragged]);
    bottom.push([px, center + (h / 2) * profile * (0.6 + 0.4 * noise(i * 0.8, wobble, 7))]);
  }
  ctx.beginPath();
  top.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
  for (let i = bottom.length - 1; i >= 0; i--) ctx.lineTo(bottom[i][0], bottom[i][1]);
  ctx.closePath();
}

// ---- the habitats ----------------------------------------------------------------------------

/** The habitats, by key: their names, and how each is painted. */
export const HABITATS = {
  birch: { name: "Birch bark", paint: paintBirch },
  soot: { name: "Sooty bark", paint: paintSoot },
  lichen: { name: "Lichen-covered bark", paint: paintLichen },
};

// the trunk is a cylinder seen up close: a little darker toward the left and right edges
function trunkShade(x, width) {
  const u = (x / width) * 2 - 1;
  return -0.06 * u * u * u * u;
}

/**
 * Silver birch (Betula pendula) up close: chalky white bark with fine horizontal grain and grey
 * banding, dark horizontal lenticels with pale lips, strips of paper bark peeling back over the
 * peach inner bark, dark scars where branches fell, and crusts of grey-green lichen.
 */
function paintBirch(ctx, width, height, rng, seed) {
  const white = lab(0.9, 0.012, 85);
  paintPixels(ctx, width, height, 0.5, (x, y) => {
    const mottle = fbm(x / 260, y / 160, seed);
    // grey horizontal bands, and finer streaks along the grain
    const band = Math.max(0, fbm(x / 400, y / 28, seed + 13) - 0.52) * 2.2;
    const grain = fbm(x / 140, y / 1.8, seed + 7, 3);
    const speck = noise(x / 1.3, y / 1.3, seed + 19) > 0.93 ? 0.06 : 0;
    const L = white[0] - 0.06 * (mottle - 0.5) - 0.045 * (grain - 0.5) - 0.13 * band - speck + trunkShade(x, width);
    return [L, white[1] - 0.004 * band, white[2] + 0.006 * band];
  });
  const area = width * height;
  // strips of paper bark peeling back: ragged peach inner bark, a pale curled lip above it
  for (let i = 0, n = Math.round(area / 70000); i < n; i++) {
    const x = rng() * width;
    const y = rng() * height;
    const w = 50 + rng() * 170;
    const h = 4 + rng() * 9;
    sliverPath(ctx, x, y, w, h, rng);
    ctx.fillStyle = lch(0.74 + rng() * 0.06, 0.055, 48, 0.92);
    ctx.fill();
    sliverPath(ctx, x, y - h * 0.55, w * 0.96, h * 0.45, rng);
    ctx.fillStyle = lch(0.97, 0.006, 90, 0.95);
    ctx.fill();
    ctx.shadowColor = "rgb(60 50 40 / 0.35)";
    ctx.shadowBlur = 3;
    ctx.shadowOffsetY = 1.5;
    ctx.fill();
    ctx.shadowColor = "transparent";
  }
  // lenticels: ragged dark dashes in loose horizontal rows, of every size and darkness
  for (let row = 0, rows = Math.round(height / 9); row < rows; row++) {
    const y0 = (row + rng()) * 9;
    const count = Math.round((width / 120) * rng() ** 1.5 * 2.2);
    for (let i = 0; i < count; i++) {
      const x = rng() * width;
      const w = 4 + rng() ** 2.2 * 80;
      const h = 1 + Math.min(5, w * 0.07) * (0.5 + rng() * 0.6);
      const dark = 0.18 + rng() ** 2 * 0.3;
      sliverPath(ctx, x, y0 - h * 0.7, w * 0.92, h * 0.55, rng);
      ctx.fillStyle = lch(0.95, 0.006, 90, 0.75);
      ctx.fill();
      sliverPath(ctx, x, y0, w, h, rng);
      ctx.fillStyle = lch(dark, 0.015, 55, 0.95);
      ctx.fill();
    }
  }
  // scars of fallen branches: a dark, rough, wide chevron with a halo of darker bark
  for (let i = 0, n = Math.round(area / 450000) + 1; i < n; i++) {
    const x = rng() * width;
    const y = rng() * height;
    const w = 26 + rng() * 50;
    const h = 12 + rng() * 18;
    blobPath(ctx, x, y + h * 0.3, w * 1.25, h * 1.1, rng, 0.25);
    ctx.fillStyle = lch(0.62, 0.02, 60, 0.35);
    ctx.fill();
    ctx.beginPath();
    const steps = 14;
    for (let k = 0; k <= steps; k++) {
      const t = k / steps;
      const px = x - w + 2 * w * t;
      const py = y - h * 0.15 + Math.abs(t - 0.5) * -h * 0.25 + (rng() - 0.5) * 2;
      if (k) ctx.lineTo(px, py);
      else ctx.moveTo(px, py);
    }
    for (let k = steps; k >= 0; k--) {
      const t = k / steps;
      const drop = (1 - Math.abs(t - 0.5) * 2) ** 1.6;
      ctx.lineTo(x - w + 2 * w * t, y + h * drop + (rng() - 0.5) * 3);
    }
    ctx.closePath();
    ctx.fillStyle = lch(0.17, 0.012, 50, 0.95);
    ctx.fill();
  }
  lichenCrusts(ctx, width, height, rng, Math.round(area / 45000), 0.7, 0.04, 125);
}

/**
 * Bark blackened by soot, as in industrial towns: rough plates split by deep vertical fissures
 * and shorter cross-cracks, lit from the upper left, under a film of grime.
 */
function paintSoot(ctx, width, height, rng, seed) {
  const base = lab(0.31, 0.013, 60);
  // the bark's height: plates (long cells of cellular noise, each a little tilted and rough),
  // with deep cracks along their borders
  const heightAt = (x, y) => {
    const wx = x + fbm(x / 90, y / 90, seed + 3) * 26;
    const wy = y + fbm(x / 90, y / 90, seed + 4) * 40;
    const [f1, f2, id] = cells(wx / 52, wy / 230, seed);
    // wide fissures with sloping walls, and finer cracks within the plates
    const crack = Math.min(1, (f2 - f1) / 0.38);
    const [g1, g2] = cells(wx / 18, wy / 60, seed + 31);
    const fine = Math.min(1, (g2 - g1) / 0.12);
    const plate = fbm(x / 9, y / 20, seed + 11, 3);
    return crack ** 1.4 * (0.7 + 0.3 * id) * (0.82 + 0.18 * fine) + 0.22 * plate * crack;
  };
  paintPixels(ctx, width, height, 0.5, (x, y) => {
    const h = heightAt(x, y);
    // light from the upper left: brighter where the bark faces it
    const slope = heightAt(x - 2, y - 2) - heightAt(x + 2, y + 2);
    const grime = fbm(x / 240, y / 240, seed + 17);
    const L = base[0] - 0.22 * (1 - h) + 0.35 * slope - 0.04 * (grime - 0.5) + trunkShade(x, width);
    return [L, base[1], base[2]];
  });
  // grit and flakes of soot
  for (let i = 0, n = Math.round((width * height) / 500); i < n; i++) {
    ctx.fillStyle = rng() < 0.65 ? lch(0.12, 0.008, 50, 0.55) : lch(0.42, 0.012, 65, 0.4);
    ctx.beginPath();
    ctx.arc(rng() * width, rng() * height, 0.4 + rng() * 1.2, 0, Math.PI * 2);
    ctx.fill();
  }
}

/**
 * Old grey bark under lichen: fissured bark spotted with pale grey-green crusts, leafy rosettes
 * whose lobes curl up at their pale edges, and a few orange cups of Xanthoria.
 */
function paintLichen(ctx, width, height, rng, seed) {
  const bark = lab(0.47, 0.018, 75);
  const crust = lab(0.7, 0.04, 120);
  const heightAt = (x, y) => {
    const warp = fbm(x / 120, y / 120, seed + 3) * 20;
    return 1 - Math.abs(2 * noise((x + warp) / 34, y / 220, seed + 9) - 1);
  };
  paintPixels(ctx, width, height, 0.5, (x, y) => {
    // crusts in round patches on the plates; bare bark in the fissures
    const h = heightAt(x, y);
    const cover = fbm(x / 45, y / 45, seed) * 0.75 + 0.25 * h;
    const t = Math.min(1, Math.max(0, (cover - 0.5) * 6));
    const grain = fbm(x / 3, y / 3, seed + 5, 2);
    const slope = heightAt(x - 2, y - 2) - heightAt(x + 2, y + 2);
    const L =
      bark[0] * (1 - t) + crust[0] * t + 0.08 * (grain - 0.5) - 0.1 * (1 - h) ** 2 + 0.1 * slope + trunkShade(x, width);
    return [L, bark[1] * (1 - t) + crust[1] * t, bark[2] * (1 - t) + crust[2] * t];
  });
  // leafy rosettes: a cluster of overlapping lobes, larger near the middle, each with a pale
  // curled rim, and dark fruiting cups in the centre
  for (let i = 0, n = Math.round((width * height) / 24000); i < n; i++) {
    const x = rng() * width;
    const y = rng() * height;
    const r = 8 + rng() * 20;
    const lobes = 10 + Math.floor(rng() * 16);
    for (let k = 0; k < lobes; k++) {
      const angle = rng() * Math.PI * 2;
      const d = r * Math.sqrt(rng());
      const size = r * (0.16 + 0.18 * (1 - d / r)) * (0.7 + rng() * 0.6);
      const lx = x + Math.cos(angle) * d;
      const ly = y + Math.sin(angle) * d * 0.85;
      blobPath(ctx, lx, ly, size * 1.2, size, rng, 0.45, angle);
      ctx.fillStyle = lch(0.64 + rng() * 0.06, 0.04, 125, 0.92);
      ctx.fill();
      ctx.strokeStyle = lch(0.82, 0.03, 115, 0.8);
      ctx.lineWidth = 0.9;
      ctx.stroke();
    }
    for (let k = 0, cups = Math.floor(rng() * 4); k < cups; k++) {
      ctx.fillStyle = lch(0.42, 0.04, 70, 0.9);
      ctx.beginPath();
      ctx.arc(x + (rng() - 0.5) * r * 0.5, y + (rng() - 0.5) * r * 0.5, 1.2 + rng() * 1.8, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  // Xanthoria: small orange cups
  for (let i = 0, n = Math.round((width * height) / 160000); i < n; i++) {
    const x = rng() * width;
    const y = rng() * height;
    for (let k = 0; k < 3 + rng() * 5; k++) {
      blobPath(ctx, x + (rng() - 0.5) * 10, y + (rng() - 0.5) * 10, 1.5 + rng() * 2.5, 1.5 + rng() * 2.2, rng, 0.4);
      ctx.fillStyle = lch(0.7, 0.12, 65, 0.9);
      ctx.fill();
    }
  }
  lichenCrusts(ctx, width, height, rng, Math.round((width * height) / 20000), 0.66, 0.035, 120);
}

// crustose lichen: small clusters of granules
function lichenCrusts(ctx, width, height, rng, count, l, c, h) {
  for (let i = 0; i < count; i++) {
    const x = rng() * width;
    const y = rng() * height;
    const r = 4 + rng() * 14;
    for (let k = 0, n = 8 + Math.floor(rng() * 20); k < n; k++) {
      const angle = rng() * Math.PI * 2;
      const d = r * Math.sqrt(rng());
      ctx.fillStyle = lch(l + (rng() - 0.5) * 0.08, c, h, 0.75);
      ctx.beginPath();
      ctx.arc(x + Math.cos(angle) * d, y + Math.sin(angle) * d, 0.7 + rng() * 1.6, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

/**
 * A habitat painted on a canvas of `width` x `height` CSS pixels at `dpr`, and the patch
 * statistics of what it shows: for every cell of CELL pixels, its mean OKLab colour and the
 * spread of its lightness (how rough the bark is there).
 */
export const CELL = 12;

export function paintHabitat(key, width, height, seed, dpr = 1) {
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(width * dpr));
  canvas.height = Math.max(1, Math.round(height * dpr));
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.scale(dpr, dpr);
  HABITATS[key].paint(ctx, width, height, createRng(seed), seed);
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
