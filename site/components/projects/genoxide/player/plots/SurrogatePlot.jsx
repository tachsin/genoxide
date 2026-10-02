"use client";

import { useEffect, useId, useMemo, useState } from "react";
import { categorical, formatValue, linear, resolveColor, ticks } from "../chart-kit";
import { Axes, Legend, Marker, PlotBox, TipRows, Tooltip, nearest, pointerIn } from "../chart-parts";
import { gridImage, isoline } from "../contour";

const LEVELS = [0.12, 0.24, 0.36, 0.48, 0.6, 0.72, 0.84];

/**
 * `surrogate`: a step of Bayesian optimization in two panels. Left, the
 * model's posterior mean over the box (`state.mean`, a grid of thousandths of
 * its range on a log scale, row i along x₂, column j along x₁), darker where
 * it's higher, with the points evaluated so far (`state.population`), the
 * newest one (`state.newest`), the best (`state.best`), the known minima and,
 * in the last frame, the point where the model's mean is lowest
 * (`state.polished`). Right, the acquisition function that chose the newest
 * point (`state.acquisition`, thousandths of its shaded range), darker where
 * the point is more worth evaluating. Generation 0, the initial design, has no
 * model: the left panel shows the points alone. `problem.panels`, if given,
 * names what the two grids are (each `{ title, shading }`), e.g. a model's
 * probability of feasibility on the left; `problem.minima_label` names the
 * known minima.
 */
export default function SurrogatePlot({ trace, frame, dark }) {
  const problem = trace.problem ?? {};
  const bounds = problem.bounds ?? [
    [0, 1],
    [0, 1],
  ];
  const minima = problem.minima ?? [];
  const [left, right] = [
    { title: "The model's mean", shading: "shading: higher mean (log)", ...problem.panels?.[0] },
    { title: "The log expected improvement", shading: "shading: more worth evaluating", ...problem.panels?.[1] },
  ];
  const state = frame.state ?? {};
  const palette = categorical(dark);
  const legend = [
    { label: "evaluated", color: palette[1], shape: "dot" },
    { label: "newest", color: palette[2], shape: "square" },
    { label: "best", color: palette[0], shape: "diamond" },
    ...(state.polished ? [{ label: "the mean's minimum", color: palette[3], shape: "triangle" }] : []),
    ...(minima.length ? [{ label: problem.minima_label ?? "global minima", shape: "ring", className: "text-base-content" }] : []),
  ];
  return (
    <div>
      <Legend className="mb-2" items={legend} />
      <div className="grid gap-4 sm:grid-cols-2">
        <Panel
          title={left.title}
          shading={left.shading}
          grid={state.mean ?? null}
          bounds={bounds}
          minima={minima}
          state={state}
          palette={palette}
          dark={dark}
          lines
        />
        <Panel
          title={right.title}
          shading={right.shading}
          grid={state.acquisition ?? null}
          bounds={bounds}
          minima={[]}
          state={{ newest: state.mean ? state.newest : null }}
          palette={palette}
          dark={dark}
        />
      </div>
    </div>
  );
}

function Panel({ title, shading, grid, bounds, minima, state, palette, dark, lines = false }) {
  const [hover, setHover] = useState(null);
  const [image, setImage] = useState(null);
  const clip = `clip${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const n = grid?.length ?? 0;
  const key = JSON.stringify(grid);
  const values = useMemo(() => (key === "null" ? null : JSON.parse(key).map((row) => row.map((v) => v / 1000))), [key]);
  const isolines = useMemo(() => (values && lines ? LEVELS.map((level) => isoline(values, level, (c) => c, (r) => r)) : []), [values, lines]);

  useEffect(() => {
    if (!values) {
      setImage(null);
      return;
    }
    const [r, g, b] = resolveColor("var(--color-base-content)");
    const top = dark ? 0.42 : 0.36;
    setImage(gridImage(values, (t) => [r, g, b, Math.round(255 * (0.02 + top * t))]));
  }, [values, dark]);

  const population = state.population ?? [];
  const inBounds = (p) => p && p[0] >= bounds[0][0] && p[0] <= bounds[0][1] && p[1] >= bounds[1][0] && p[1] <= bounds[1][1];

  return (
    <section aria-label={title} className="min-w-0">
      <h4 className="mb-1 text-base-content/70 text-xs">
        {title}
        {values ? <span className="text-base-content/50"> · {shading}</span> : null}
      </h4>
      <PlotBox
        aspect={1}
        minHeight={220}
        maxHeight={420}
        label={`${title}, with ${population.length} evaluated points`}
        overlay={({ width }) =>
          hover ? (
            <Tooltip x={hover.px} y={hover.py} width={width}>
              <TipRows rows={[["x₁, x₂", `${formatValue(Number(hover.p[0].toPrecision(6)))}, ${formatValue(Number(hover.p[1].toPrecision(6)))}`]]} />
            </Tooltip>
          ) : null
        }
      >
        {({ width, height }) => {
          const margin = { left: 40, right: 12, top: 12, bottom: 38 };
          const side = Math.min(width - margin.left - margin.right, height - margin.top - margin.bottom);
          const left = margin.left + (width - margin.left - margin.right - side) / 2;
          const area = { left, right: left + side, top: margin.top, bottom: margin.top + side };
          const x = linear(bounds[0], [area.left, area.right]);
          const y = linear(bounds[1], [area.bottom, area.top]);
          const cell = n > 1 ? side / (n - 1) : side;
          const points = population.filter(inBounds).map((p) => ({ p, px: x(p[0]), py: y(p[1]) }));
          return (
            <>
              <defs>
                <clipPath id={clip}>
                  <rect x={area.left} y={area.top} width={side} height={side} />
                </clipPath>
              </defs>
              <Axes
                x={x}
                y={y}
                area={area}
                xTicks={ticks(bounds[0][0], bounds[0][1], 4)}
                yTicks={ticks(bounds[1][0], bounds[1][1], 4)}
                xLabel="x₁"
                yLabel="x₂"
              />
              {image ? (
                <image
                  href={image}
                  x={area.left - cell / 2}
                  y={area.top - cell / 2}
                  width={side + cell}
                  height={side + cell}
                  preserveAspectRatio="none"
                  clipPath={`url(#${clip})`}
                />
              ) : null}
              {isolines.length ? (
                <g clipPath={`url(#${clip})`}>
                  <g transform={`translate(${area.left} ${area.bottom}) scale(${cell} ${-cell})`}>
                    {isolines.map((d, k) => (
                      <path key={LEVELS[k]} d={d} fill="none" className="stroke-base-content/25" strokeWidth={0.75} vectorEffect="non-scaling-stroke" />
                    ))}
                  </g>
                </g>
              ) : null}
              <rect x={area.left} y={area.top} width={side} height={side} fill="none" className="stroke-base-content/20" />
              <g clipPath={`url(#${clip})`}>
                {minima.map((m) => (
                  <circle key={`${m[0]},${m[1]}`} cx={x(m[0])} cy={y(m[1])} r={7} fill="none" className="stroke-base-content" strokeWidth={1.5} />
                ))}
                {points.map((q, k) => (
                  <Marker key={k} x={q.px} y={q.py} r={3.5} color={palette[1]} />
                ))}
                {inBounds(state.best) ? <Marker shape="diamond" x={x(state.best[0])} y={y(state.best[1])} r={5.5} color={palette[0]} /> : null}
                {inBounds(state.newest) ? <Marker shape="square" x={x(state.newest[0])} y={y(state.newest[1])} r={4.5} color={palette[2]} /> : null}
                {inBounds(state.polished) ? <Marker shape="triangle" x={x(state.polished[0])} y={y(state.polished[1])} r={5.5} color={palette[3]} /> : null}
              </g>
              {hover ? <circle cx={hover.px} cy={hover.py} r={7} fill="none" className="stroke-base-content" strokeWidth={1.5} /> : null}
              <rect
                x={area.left}
                y={area.top}
                width={side}
                height={side}
                fill="transparent"
                onPointerMove={(event) => {
                  const p = pointerIn(event);
                  const hit = nearest(points, p.x, p.y, 14);
                  setHover(hit ? { p: hit.p, px: hit.px, py: hit.py } : null);
                }}
                onPointerLeave={() => setHover(null)}
              />
            </>
          );
        }}
      </PlotBox>
    </section>
  );
}
