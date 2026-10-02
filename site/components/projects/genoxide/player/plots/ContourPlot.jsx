"use client";

import { useEffect, useId, useMemo, useState } from "react";
import { categorical, formatValue, linear, resolveColor, ticks } from "../chart-kit";
import { Axes, Legend, Marker, PlotBox, TipRows, Tooltip, nearest, pointerIn } from "../chart-parts";
import { FUNCTIONS, contourLegend, gridImage, isoline, rotated, sample } from "../contour";

const GRID = 140;
const LEVELS = [0.12, 0.24, 0.36, 0.48, 0.6, 0.72, 0.84];

/**
 * `contour`: the population over the function's landscape (computed here),
 * darker where the function is higher, with iso-lines and the known minima.
 * `compact` (a panel of a grid) leaves the legend to the grid. Optionally,
 * `problem.labels` names the axes and the field (`{ x, y, f }`: a function
 * of more than 2 variables is drawn as a slice or a projection), and
 * `problem.minima_label` the minima, when they aren't proven global, and `problem.rotation`
 * (`{ matrix, center }`) a function rotated as genoxide's `problems::Rotated` rotates it.
 * A local method's frames can add `state.simplex` (its vertices, drawn as a
 * polygon: Nelder-Mead's triangle) and `state.ends` (where its runs before
 * converged), and `problem.population_label` names the points (e.g. "simplex").
 */
export default function ContourPlot({ trace, frame, dark, compact = false }) {
  const [hover, setHover] = useState(null);
  const [image, setImage] = useState(null);
  const clip = `clip${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const problem = trace.problem ?? {};
  const rotationKey = JSON.stringify(problem.rotation ?? null);
  const f = useMemo(
    () => rotated(FUNCTIONS[problem.function] ?? null, JSON.parse(rotationKey)),
    [problem.function, rotationKey],
  );
  const bounds = problem.bounds ?? [
    [-5, 5],
    [-5, 5],
  ];
  const minima = problem.minima ?? [];
  const labels = { x: "x₁", y: "x₂", f: "f", ...problem.labels };
  const population = frame.state?.population ?? [];
  const best = frame.state?.best ?? null;
  const simplex = frame.state?.simplex ?? null;
  const ends = frame.state?.ends ?? null;
  const palette = categorical(dark);

  // The landscape on a log scale, normalized to [0, 1].
  const boundsKey = JSON.stringify(bounds);
  const field = useMemo(() => {
    if (!f) return null;
    const raw = sample(f, JSON.parse(boundsKey), GRID);
    let lo = Infinity;
    let hi = -Infinity;
    for (const row of raw) {
      for (const v of row) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    const top = Math.log1p(hi - lo) || 1;
    const values = raw.map((row) => Array.from(row, (v) => Math.log1p(v - lo) / top));
    const lines = LEVELS.map((level) => isoline(values, level, (c) => c, (r) => r));
    return { values, lines };
  }, [f, boundsKey]);

  useEffect(() => {
    if (!field) return;
    const [r, g, b] = resolveColor("var(--color-base-content)");
    // a light ink on a dark theme needs a little more of it to read
    const top = dark ? 0.36 : 0.3;
    setImage(gridImage(field.values, (t) => [r, g, b, Math.round(255 * (0.02 + top * t))]));
  }, [field, dark]);

  const inBounds = (p) => p[0] >= bounds[0][0] && p[0] <= bounds[0][1] && p[1] >= bounds[1][0] && p[1] <= bounds[1][1];

  return (
    <div>
      {compact ? null : <Legend className="mb-2" items={contourLegend({
            best,
            minima: minima.length,
            palette,
            minimaLabel: problem.minima_label,
            populationLabel: problem.population_label,
            ends: ends !== null,
          })} />}
      <PlotBox
        aspect={1}
        minHeight={compact ? 220 : 280}
        maxHeight={540}
        label={`The population of ${population.length} on ${problem.function ?? "the function"}`}
        overlay={({ width }) =>
          hover ? (
            <Tooltip x={hover.px} y={hover.py} width={width}>
              <TipRows
                rows={[
                  [`${labels.x}, ${labels.y}`, `${formatValue(hover.p[0])}, ${formatValue(hover.p[1])}`],
                  ...(f ? [[labels.f, formatValue(Number(f(hover.p[0], hover.p[1]).toPrecision(6)))]] : []),
                ]}
              />
            </Tooltip>
          ) : null
        }
      >
        {({ width, height }) => {
          const margin = { left: 40, right: 12, top: 22, bottom: 38 };
          const side = Math.min(width - margin.left - margin.right, height - margin.top - margin.bottom);
          const left = margin.left + (width - margin.left - margin.right - side) / 2;
          const area = { left, right: left + side, top: margin.top, bottom: margin.top + side };
          const x = linear(bounds[0], [area.left, area.right]);
          const y = linear(bounds[1], [area.bottom, area.top]);
          const cell = side / (GRID - 1);
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
                xTicks={ticks(bounds[0][0], bounds[0][1], 5)}
                yTicks={ticks(bounds[1][0], bounds[1][1], 5)}
                xLabel={labels.x}
                yLabel={labels.y}
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
              {field ? (
                <g clipPath={`url(#${clip})`}>
                <g transform={`translate(${area.left} ${area.bottom}) scale(${cell} ${-cell})`}>
                  {field.lines.map((d, k) => (
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
                {(ends ?? []).map((p, k) => (
                  <Marker key={`end${k}`} x={x(p[0])} y={y(p[1])} r={4.5} color={palette[2]} hollow />
                ))}
                {simplex && simplex.length > 1 ? (
                  <polygon
                    points={simplex.map((p) => `${x(p[0])},${y(p[1])}`).join(" ")}
                    fill={palette[1]}
                    fillOpacity={0.12}
                    stroke={palette[1]}
                    strokeWidth={1.5}
                    strokeLinejoin="round"
                  />
                ) : null}
                {points.map((q, k) => (
                  <Marker key={k} x={q.px} y={q.py} r={3.5} color={palette[1]} />
                ))}
                {best && inBounds(best) ? <Marker shape="diamond" x={x(best[0])} y={y(best[1])} r={5.5} color={palette[0]} /> : null}
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
    </div>
  );
}
