"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";

/**
 * The player's small chart toolkit: scales, ticks, number formats, the
 * palettes, and the hooks every plot shares (size, color scheme, reduced
 * motion, eased extents). Hand-written so the page ships no chart library.
 */

// Categorical hues in a fixed order, a light and a dark step of each. The
// order is the colorblind-safety mechanism: adjacent slots stay apart under
// the common color vision deficiencies, and the first three stay apart as a
// set (use at most three for scatter series).
const CATEGORICAL = {
  light: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
  dark: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
};

// Diverging blue <-> red around a neutral gray, for values with a midpoint
// (the XOR network's output around 0.5).
const DIVERGING = {
  light: { low: [42, 120, 214], mid: [240, 239, 236], high: [227, 73, 72] },
  dark: { low: [57, 135, 229], mid: [56, 56, 53], high: [230, 103, 103] },
};

export const STATUS = { good: "#0ca30c", critical: "#d03b3b", warning: "#fab219" };

export function categorical(dark) {
  return dark ? CATEGORICAL.dark : CATEGORICAL.light;
}

/** Color for t in [0, 1] on the diverging scale, as an [r, g, b] triple. */
export function diverging(t, dark) {
  const { low, mid, high } = dark ? DIVERGING.dark : DIVERGING.light;
  const u = Math.min(1, Math.max(0, t));
  const [a, b, s] = u < 0.5 ? [low, mid, u * 2] : [mid, high, (u - 0.5) * 2];
  return [0, 1, 2].map((i) => Math.round(a[i] + (b[i] - a[i]) * s));
}

export function rgb([r, g, b]) {
  return `rgb(${r} ${g} ${b})`;
}

// ---------- Numbers ----------

const plain = new Intl.NumberFormat("en", { maximumSignificantDigits: 6 });
const short = new Intl.NumberFormat("en", { maximumSignificantDigits: 3 });

function exponent(value, digits) {
  const [m, e] = value.toExponential(digits - 1).split("e");
  const mantissa = String(Number(m));
  return `${mantissa === "1" ? "" : `${mantissa}×`}10${superscript(Number(e))}`;
}

const SUPERSCRIPTS = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };

function superscript(n) {
  return String(n)
    .split("")
    .map((c) => SUPERSCRIPTS[c] ?? c)
    .join("");
}

/** A value for a readout or tooltip: up to 6 significant digits. */
export function formatValue(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  const abs = Math.abs(value);
  if (abs !== 0 && (abs >= 1e9 || abs < 1e-4)) return exponent(value, 4);
  return plain.format(value);
}

/** A value for an axis tick: short. */
export function formatTick(value) {
  const abs = Math.abs(value);
  if (abs !== 0 && (abs >= 1e6 || abs < 1e-3)) return exponent(value, 2);
  return short.format(value);
}

/**
 * The tick format of a log axis: when any tick needs a power of ten, all of
 * them get one, so the labels don't switch from 0.001 to 10⁻⁵ halfway down.
 * @param {number[]} values the axis's ticks
 */
export function logTickFormat(values) {
  const needsExponent = values.some((v) => v !== 0 && (Math.abs(v) >= 1e6 || Math.abs(v) < 1e-3));
  return needsExponent ? (value) => (value === 0 ? "0" : exponent(value, 2)) : formatTick;
}

// ---------- Scales and ticks ----------

/** A linear map from `domain` to `range`, with `invert`. */
export function linear([d0, d1], [r0, r1]) {
  const span = d1 - d0 || 1;
  const k = (r1 - r0) / span;
  const scale = (v) => r0 + (v - d0) * k;
  scale.invert = (p) => d0 + (p - r0) / k;
  scale.domain = [d0, d1];
  scale.log = false;
  return scale;
}

/** A base-10 log map; values at or below zero clamp to the domain's low end. */
export function logarithmic([d0, d1], [r0, r1]) {
  const l0 = Math.log10(d0);
  const l1 = Math.log10(d1);
  const k = (r1 - r0) / (l1 - l0 || 1);
  const scale = (v) => r0 + (Math.log10(Math.max(v, d0)) - l0) * k;
  scale.invert = (p) => 10 ** (l0 + (p - r0) / k);
  scale.domain = [d0, d1];
  scale.log = true;
  return scale;
}

function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const power = 10 ** Math.floor(Math.log10(raw));
  const f = raw / power;
  return (f >= 7.5 ? 10 : f >= 3.5 ? 5 : f >= 1.5 ? 2 : 1) * power;
}

/** Round tick values covering [lo, hi], about `count` of them. */
export function ticks(lo, hi, count = 5) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  if (lo === hi) return [lo];
  const step = niceStep(hi - lo, count);
  const out = [];
  const start = Math.ceil(lo / step - 1e-9) * step;
  for (let v = start; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

/** Extends [lo, hi] outward to round values. */
export function nice(lo, hi, count = 5) {
  if (lo === hi) {
    const pad = Math.abs(lo) * 0.1 || 1;
    return [lo - pad, hi + pad];
  }
  const step = niceStep(hi - lo, count);
  return [Math.floor(lo / step) * step, Math.ceil(hi / step) * step];
}

/** Powers of ten across [lo, hi]; with few decades, 2 and 5 in between. */
export function logTicks(lo, hi) {
  const a = Math.floor(Math.log10(lo));
  const b = Math.ceil(Math.log10(hi));
  const decades = b - a;
  const every = Math.max(1, Math.ceil(decades / 6));
  const out = [];
  for (let e = a; e <= b; e += every) {
    const base = 10 ** e;
    if (base >= lo * 0.999 && base <= hi * 1.001) out.push(base);
    if (decades <= 2) {
      for (const m of [2, 5]) {
        const v = m * base;
        if (v >= lo && v <= hi) out.push(v);
      }
    }
  }
  return out.sort((x, y) => x - y);
}

/** [min, max] of the finite numbers in `values`, or null. */
export function extent(values) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo <= hi ? [lo, hi] : null;
}

export function pad([lo, hi], fraction = 0.05) {
  const span = hi - lo || Math.abs(hi) || 1;
  return [lo - span * fraction, hi + span * fraction];
}

// ---------- Hooks ----------

/** The content box of an element, kept up to date. */
export function useSize(ref) {
  const [size, setSize] = useState({ width: 0, height: 0 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const update = () => {
      const r = el.getBoundingClientRect();
      setSize((s) => (Math.abs(s.width - r.width) < 0.5 && Math.abs(s.height - r.height) < 0.5 ? s : { width: r.width, height: r.height }));
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, [ref]);
  return size;
}

function isDarkScheme() {
  return getComputedStyle(document.documentElement).colorScheme.includes("dark");
}

/** Whether the site's current theme is a dark one (daisyUI themes set color-scheme). */
export function useDarkScheme() {
  const [dark, setDark] = useState(false);
  useEffect(() => {
    const update = () => setDark(isDarkScheme());
    update();
    const observer = new MutationObserver(update);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class"] });
    return () => observer.disconnect();
  }, []);
  return dark;
}

export function useReducedMotion() {
  const [reduced, setReduced] = useState(() =>
    typeof window === "undefined" ? false : window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(query.matches);
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);
  return reduced;
}

/**
 * Follows `target` (an array of numbers, e.g. an axis extent) with a short
 * ease, so axes glide instead of jumping between frames. Jumps when
 * `instant` (reduced motion).
 */
export function useEased(target, instant = false, duration = 280) {
  const [value, setValue] = useState(target);
  const current = useRef(target);
  const key = target.join(",");

  // biome-ignore lint/correctness/useExhaustiveDependencies: `key` stands for `target`'s contents
  useEffect(() => {
    const from = current.current;
    if (instant || from.length !== target.length) {
      current.current = target;
      setValue(target);
      return undefined;
    }
    let raf = 0;
    const start = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const e = 1 - (1 - t) ** 3;
      const next = target.map((v, i) => from[i] + (v - from[i]) * e);
      current.current = next;
      setValue(next);
      if (t < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [key, instant, duration]);

  return value;
}

/**
 * Resolves any CSS color (oklch, color-mix, var()) to [r, g, b, a] by
 * letting the browser paint it, for canvas pixels.
 */
export function resolveColor(css, element) {
  const probe = document.createElement("span");
  probe.style.color = css;
  probe.style.display = "none";
  (element ?? document.body).appendChild(probe);
  const computed = getComputedStyle(probe).color;
  probe.remove();
  const canvas = document.createElement("canvas");
  canvas.width = 1;
  canvas.height = 1;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return [128, 128, 128, 255];
  ctx.fillStyle = computed;
  ctx.fillRect(0, 0, 1, 1);
  return Array.from(ctx.getImageData(0, 0, 1, 1).data);
}
