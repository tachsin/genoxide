"use client";

import { useRef } from "react";
import { formatTick, useSize } from "./chart-kit";

/**
 * Pieces every plot is made of: the measured SVG box, axes with hairline
 * gridlines, a legend, and a tooltip. Colors come from the theme (Tailwind's
 * daisyUI color utilities), so every site theme works.
 */

/**
 * A responsive plot box: measures its width, picks a height from `aspect`
 * (height / width, capped by `maxHeight`) or fills `fill` height, and renders
 * `children({ width, height })` in an SVG of exactly that many pixels, so
 * text and hairlines stay crisp at any size.
 */
export function PlotBox({ aspect = 0.75, height: fixed, minHeight = 200, maxHeight = 520, fill = false, label, children, overlay, className = "", svgProps }) {
  const ref = useRef(null);
  const { width, height: boxHeight } = useSize(ref);
  const height = fill
    ? Math.max(minHeight, boxHeight)
    : fixed ?? Math.round(Math.min(maxHeight, Math.max(minHeight, width * aspect)));

  return (
    <div
      ref={ref}
      className={`relative w-full ${fill ? "h-full" : ""} ${className}`}
      style={fill ? { minHeight } : { height: width || fixed ? height : minHeight }}
    >
      {width > 0 ? (
        <svg
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={label}
          // filling its box, the SVG stays out of the layout so the box can shrink again
          className={`block select-none overflow-visible ${fill ? "absolute inset-0" : ""}`}
          {...svgProps}
        >
          {children({ width, height })}
        </svg>
      ) : null}
      {width > 0 && overlay ? overlay({ width, height }) : null}
    </div>
  );
}

const TICK = "fill-base-content/55 text-[10px] tabular-nums";

/**
 * Hairline gridlines and tick labels for a plot area.
 * @param {object} props
 * @param {(v: number) => number} props.x  x scale (px)
 * @param {(v: number) => number} props.y  y scale (px)
 * @param {{ left: number, right: number, top: number, bottom: number }} props.area  plot area in px
 * @param {number[]} [props.xTicks]
 * @param {number[]} [props.yTicks]
 * @param {string} [props.xLabel]
 * @param {string} [props.yLabel]
 * @param {(v: number) => string} [props.formatX]
 * @param {(v: number) => string} [props.formatY]
 */
export function Axes({ x, y, area, xTicks = [], yTicks = [], xLabel, yLabel, formatX = formatTick, formatY = formatTick, xGrid = false }) {
  const { left, right, top, bottom } = area;
  return (
    <g aria-hidden>
      {yTicks.map((t) => (
        <g key={`y${t}`}>
          <line x1={left} x2={right} y1={y(t)} y2={y(t)} className="stroke-base-content/10" strokeWidth={1} shapeRendering="crispEdges" />
          <text x={left - 6} y={y(t)} dy="0.32em" textAnchor="end" className={TICK}>
            {formatY(t)}
          </text>
        </g>
      ))}
      {xTicks.map((t) => (
        <g key={`x${t}`}>
          {xGrid ? (
            <line x1={x(t)} x2={x(t)} y1={top} y2={bottom} className="stroke-base-content/10" strokeWidth={1} shapeRendering="crispEdges" />
          ) : null}
          <line x1={x(t)} x2={x(t)} y1={bottom} y2={bottom + 4} className="stroke-base-content/25" strokeWidth={1} shapeRendering="crispEdges" />
          <text x={x(t)} y={bottom + 15} textAnchor="middle" className={TICK}>
            {formatX(t)}
          </text>
        </g>
      ))}
      <line x1={left} x2={right} y1={bottom} y2={bottom} className="stroke-base-content/25" strokeWidth={1} shapeRendering="crispEdges" />
      {xLabel ? (
        <text x={right} y={bottom + 30} textAnchor="end" className="fill-base-content/60 text-[11px]">
          {xLabel}
        </text>
      ) : null}
      {yLabel ? (
        <text x={left - 6} y={top - 10} textAnchor="start" className="fill-base-content/60 text-[11px]">
          {yLabel}
        </text>
      ) : null}
    </g>
  );
}

/**
 * A mark's key beside its label; identity is never color alone.
 * @param {{ items: { label: string, color?: string, shape?: "dot" | "line" | "dash" | "square" | "ring" | "triangle" | "diamond", className?: string }[] }} props
 */
export function Legend({ items, className = "" }) {
  if (!items.length) return null;
  return (
    <ul className={`flex flex-wrap items-center gap-x-4 gap-y-1 text-base-content/70 text-xs ${className}`}>
      {items.map((item) => (
        <li key={item.label} className="inline-flex items-center gap-1.5">
          <Swatch {...item} />
          {item.label}
        </li>
      ))}
    </ul>
  );
}

function Swatch({ color, shape = "dot", className = "" }) {
  const style = color ? { color } : undefined;
  return (
    <svg width={shape === "line" || shape === "dash" ? 16 : 10} height={10} aria-hidden className={`flex-none ${className}`} style={style}>
      {shape === "line" ? <line x1={0} x2={16} y1={5} y2={5} stroke="currentColor" strokeWidth={2} strokeLinecap="round" /> : null}
      {shape === "dash" ? <line x1={0} x2={16} y1={5} y2={5} stroke="currentColor" strokeWidth={1.5} strokeDasharray="3 2" /> : null}
      {shape === "dot" ? <circle cx={5} cy={5} r={4} fill="currentColor" /> : null}
      {shape === "ring" ? <circle cx={5} cy={5} r={3.5} fill="none" stroke="currentColor" strokeWidth={1.5} /> : null}
      {shape === "square" ? <rect x={1} y={1} width={8} height={8} rx={1.5} fill="currentColor" /> : null}
      {shape === "triangle" ? <path d="M5 0.8 L9.3 8.6 H0.7 Z" fill="currentColor" /> : null}
      {shape === "diamond" ? <path d="M5 0.5 L9.5 5 L5 9.5 L0.5 5 Z" fill="currentColor" /> : null}
    </svg>
  );
}

/** A marker shape at (x, y) with radius r; the ring in the surface color keeps overlaps legible. */
export function Marker({ shape = "dot", x, y, r = 4, color, ring = true, hollow = false, className = "", opacity }) {
  const common = {
    fill: hollow ? "none" : color,
    stroke: hollow ? color : ring ? "var(--color-base-100)" : "none",
    strokeWidth: hollow ? 1.5 : ring ? 1.5 : 0,
    className,
    opacity,
  };
  if (shape === "square") return <rect x={x - r * 0.9} y={y - r * 0.9} width={r * 1.8} height={r * 1.8} rx={1.5} {...common} />;
  if (shape === "triangle") return <path d={`M${x} ${y - r * 1.1} L${x + r} ${y + r * 0.75} H${x - r} Z`} {...common} />;
  if (shape === "diamond") return <path d={`M${x} ${y - r * 1.2} L${x + r * 1.2} ${y} L${x} ${y + r * 1.2} L${x - r * 1.2} ${y} Z`} {...common} />;
  return <circle cx={x} cy={y} r={r} {...common} />;
}

export const SHAPES = ["dot", "square", "triangle", "diamond"];

/**
 * A tooltip over a plot, placed beside (x, y) and kept inside the box.
 * A `wide` one (with a table) takes the side with more room, up to 22rem,
 * or the box's whole width when neither side has enough.
 * @param {{ x: number, y: number, width: number, wide?: boolean, children: any }} props
 */
export function Tooltip({ x, y, width, wide = false, children }) {
  const top = Math.max(0, y - 12);
  let style;
  let size = "max-w-[16rem]";
  if (wide) {
    const room = Math.max(width - x, x) - 12;
    const right = x > width - x;
    size = room < 280 ? "" : "w-max";
    style =
      room < 280
        ? { left: 0, right: 0, top }
        : { left: right ? undefined : x + 12, right: right ? width - x + 12 : undefined, top, maxWidth: Math.min(352, room) };
  } else {
    const right = x > width * 0.6;
    style = { left: right ? undefined : x + 12, right: right ? width - x + 12 : undefined, top };
  }
  return (
    <div
      aria-hidden
      className={`pointer-events-none absolute z-10 ${size} rounded-lg border border-base-content/10 bg-base-100/95 px-2.5 py-1.5 text-base-content text-xs shadow-lg backdrop-blur-sm`}
      style={style}
    >
      {children}
    </div>
  );
}

/** Rows of "label value" inside a tooltip. */
export function TipRows({ rows }) {
  return (
    <dl className="grid grid-cols-[auto_auto] gap-x-3 gap-y-0.5">
      {rows.map(([label, value]) => (
        <div key={label} className="contents">
          <dt className="text-base-content/60">{label}</dt>
          <dd className="text-right font-medium tabular-nums">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

/** Nearest item to a pointer, within `radius` px; `items` are { px, py, ... }. */
export function nearest(items, px, py, radius = 18) {
  let best = null;
  let bestD = radius * radius;
  for (const item of items) {
    const d = (item.px - px) ** 2 + (item.py - py) ** 2;
    if (d <= bestD) {
      bestD = d;
      best = item;
    }
  }
  return best;
}

/** Pointer position inside the SVG, or null. */
export function pointerIn(event) {
  const el = event.currentTarget;
  const r = (el.ownerSVGElement ?? el).getBoundingClientRect();
  return { x: event.clientX - r.left, y: event.clientY - r.top };
}
