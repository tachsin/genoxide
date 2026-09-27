"use client";

import { useEffect, useId, useMemo, useState } from "react";
import { diverging, formatValue, linear, rgb } from "../chart-kit";
import { Axes, PlotBox, TipRows, Tooltip, nearest, pointerIn } from "../chart-parts";
import { gridImage, isoline } from "../contour";

/** The grid's value at (x, y) in [0, 1]², bilinear; grid[i][j] is at x = j / (n - 1), y = i / (n - 1). */
function valueAt(grid, x, y) {
  const n = grid.length;
  const gx = Math.min(n - 1, Math.max(0, x * (n - 1)));
  const gy = Math.min(n - 1, Math.max(0, y * (n - 1)));
  const j = Math.min(n - 2, Math.floor(gx));
  const i = Math.min(n - 2, Math.floor(gy));
  const u = gx - j;
  const v = gy - i;
  return (
    grid[i][j] * (1 - u) * (1 - v) + grid[i][j + 1] * u * (1 - v) + grid[i + 1][j] * (1 - u) * v + grid[i + 1][j + 1] * u * v
  );
}

/**
 * `surface`: the network's output over the unit square, blue (0) through
 * gray to red (1), the 0.5 decision boundary drawn, and the training inputs
 * colored by their target.
 */
export default function SurfacePlot({ trace, frame, dark }) {
  const [hover, setHover] = useState(null);
  const [image, setImage] = useState(null);
  const clip = `clip${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const grid = frame.state?.grid ?? null;
  const inputs = trace.problem?.inputs ?? [];
  const targets = trace.problem?.targets ?? [];
  const n = grid?.length ?? 0;

  useEffect(() => {
    if (!grid?.length) return;
    setImage(gridImage(grid, (t) => [...diverging(t, dark), 255]));
  }, [grid, dark]);

  const boundary = useMemo(() => (grid?.length > 1 ? isoline(grid, 0.5, (c) => c, (r) => r) : ""), [grid]);
  const gradient = `linear-gradient(to right, ${rgb(diverging(0, dark))}, ${rgb(diverging(0.5, dark))}, ${rgb(diverging(1, dark))})`;

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-base-content/70 text-xs">
        <span className="inline-flex items-center gap-2">
          output
          <span className="relative inline-block h-2.5 w-28 rounded-sm" style={{ background: gradient }} aria-hidden />
          <span className="tabular-nums">0 – 1</span>
        </span>
        <span className="inline-flex items-center gap-1.5">
          <svg width={16} height={10} aria-hidden>
            <line x1={0} x2={16} y1={5} y2={5} stroke="currentColor" strokeWidth={2} />
          </svg>
          0.5 boundary
        </span>
        <span className="inline-flex items-center gap-1.5">
          <svg width={12} height={12} aria-hidden>
            <circle cx={6} cy={6} r={4.5} fill="none" stroke="currentColor" strokeWidth={2} />
          </svg>
          training input, filled by its target
        </span>
      </div>
      <PlotBox
        aspect={1}
        minHeight={260}
        maxHeight={480}
        label={`The network's output: ${inputs.map((p, k) => `(${p.join(", ")}) → ${grid ? formatValue(Number(valueAt(grid, p[0], p[1]).toPrecision(3))) : "–"}, target ${targets[k]}`).join("; ")}`}
        overlay={({ width }) =>
          hover ? (
            <Tooltip x={hover.px} y={hover.py} width={width}>
              <TipRows
                rows={[
                  ["input", hover.p.map((v) => formatValue(Number(v.toPrecision(3)))).join(", ")],
                  ...(hover.target !== undefined ? [["target", hover.target]] : []),
                  ["output", grid ? formatValue(Number(valueAt(grid, hover.p[0], hover.p[1]).toPrecision(4))) : "–"],
                ]}
              />
            </Tooltip>
          ) : null
        }
      >
        {({ width, height }) => {
          const margin = { left: 40, right: 16, top: 20, bottom: 38 };
          const side = Math.min(width - margin.left - margin.right, height - margin.top - margin.bottom);
          const left = margin.left + (width - margin.left - margin.right - side) / 2;
          const area = { left, right: left + side, top: margin.top, bottom: margin.top + side };
          // A small inset, so the inputs at the corners clear the axes.
          const x = linear([-0.04, 1.04], [area.left, area.right]);
          const y = linear([-0.04, 1.04], [area.bottom, area.top]);
          const box = { left: x(0), top: y(1), size: x(1) - x(0) };
          const cell = n > 1 ? box.size / (n - 1) : box.size;
          const points = inputs.map((p, k) => ({ p, target: targets[k], px: x(p[0]), py: y(p[1]) }));
          return (
            <>
              <defs>
                <clipPath id={clip}>
                  <rect x={box.left} y={box.top} width={box.size} height={box.size} />
                </clipPath>
              </defs>
              <Axes x={x} y={y} area={area} xTicks={[0, 0.5, 1]} yTicks={[0, 0.5, 1]} xLabel="x₁" yLabel="x₂" />
              {image ? (
                <image
                  href={image}
                  x={box.left - cell / 2}
                  y={box.top - cell / 2}
                  width={box.size + cell}
                  height={box.size + cell}
                  preserveAspectRatio="none"
                  clipPath={`url(#${clip})`}
                />
              ) : null}
              <g clipPath={`url(#${clip})`}>
                <g transform={`translate(${box.left} ${box.top + box.size}) scale(${cell} ${-cell})`}>
                <path d={boundary} fill="none" className="stroke-base-content" strokeWidth={2} vectorEffect="non-scaling-stroke" />
                </g>
              </g>
              {points.map((q) => (
                <g key={q.p.join(",")}>
                  <circle cx={q.px} cy={q.py} r={9} fill={rgb(diverging(q.target, dark))} className="stroke-base-content" strokeWidth={2} />
                  <text x={q.px} y={q.py} dy="0.34em" textAnchor="middle" fill="#fff" className="font-semibold text-[10px]">
                    {q.target}
                  </text>
                </g>
              ))}
              <rect
                x={area.left}
                y={area.top}
                width={side}
                height={side}
                fill="transparent"
                onPointerMove={(event) => {
                  const p = pointerIn(event);
                  const hit = nearest(points, p.x, p.y, 14);
                  setHover(hit ?? { p: [x.invert(p.x), y.invert(p.y)], px: p.x, py: p.y });
                }}
                onPointerLeave={() => setHover(null)}
              />
            </>
          );
        }}
      </PlotBox>
    </div>
  );
}
