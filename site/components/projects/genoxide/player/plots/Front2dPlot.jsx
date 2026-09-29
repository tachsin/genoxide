"use client";

import { useId, useMemo, useState } from "react";
import { categorical, extent, formatValue, linear, nice, pad, ticks, useEased } from "../chart-kit";
import { Axes, Legend, Marker, PlotBox, SHAPES, TipRows, Tooltip, nearest, pointerIn } from "../chart-parts";

/** The true front as a path, broken where the front itself is (a disconnected front's gaps). */
function frontPath(points, x, y) {
  if (!points?.length) return "";
  const sorted = [...points].sort((a, b) => a[0] - b[0]);
  const steps = [];
  for (let i = 1; i < sorted.length; i++) steps.push(Math.hypot(sorted[i][0] - sorted[i - 1][0], sorted[i][1] - sorted[i - 1][1]));
  const typical = [...steps].sort((a, b) => a - b)[Math.floor(steps.length / 2)] ?? 0;
  const first = sorted[0];
  const last = sorted[sorted.length - 1];
  const diagonal = Math.hypot(last[0] - first[0], Math.max(...sorted.map((p) => p[1])) - Math.min(...sorted.map((p) => p[1])));
  let d = "";
  sorted.forEach((p, i) => {
    // a gap: far longer than the usual step, and a visible share of the front
    const jump = i === 0 || (steps[i - 1] > typical * 6 && steps[i - 1] > diagonal * 0.08);
    d += `${jump ? "M" : "L"}${x(p[0]).toFixed(1)} ${y(p[1]).toFixed(1)}`;
  });
  return d;
}

/** One piece of a true front, sorted by its first objective, as a path. */
function piecePath(points, x, y) {
  return [...points]
    .sort((a, b) => a[0] - b[0])
    .map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)} ${y(p[1]).toFixed(1)}`)
    .join("");
}

/**
 * A feasible region, `{ x: [x0, x1], y: [y0, y1], columns, rows }`: a grid of
 * cells over that box, each row (from the bottom up) the pairs of columns
 * [start, end) where it's feasible. As an image, white where feasible, for an
 * SVG mask; null on the server.
 */
function regionImage(region) {
  const rows = region?.rows;
  if (!Array.isArray(rows) || !rows.length || typeof document === "undefined") return null;
  const canvas = document.createElement("canvas");
  canvas.width = region.columns;
  canvas.height = rows.length;
  const context = canvas.getContext("2d");
  if (!context) return null;
  context.fillStyle = "#fff";
  rows.forEach((row, i) => {
    for (let k = 0; k + 1 < row.length; k += 2) context.fillRect(row[k], rows.length - 1 - i, row[k + 1] - row[k], 1);
  });
  return canvas.toDataURL();
}

/**
 * `front-2d`: each algorithm's front in objective space over the true
 * front; infeasible points hollow, the true front's isolated points as rings.
 * A constrained problem's trace may give its feasible region
 * (`problem.feasible_region`, see `regionImage`), shaded behind the fronts.
 * The axes follow the points with a short glide, so the front's approach
 * reads at every scale.
 */
export default function Front2dPlot({ trace, frame, dark, reduced, compact = false }) {
  const [hover, setHover] = useState(null);
  const clip = `clip${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const problem = trace.problem ?? {};
  const fronts = frame.state?.fronts ?? {};
  const infeasible = frame.state?.infeasible ?? {};
  const feasible = frame.state?.feasible ?? null;
  const names = problem.series ?? Object.keys(fronts);
  const palette = categorical(dark);
  const objectives = problem.objectives ?? ["f1", "f2"];
  // the true front: its points, or a disconnected front given in pieces (a list of point lists),
  // each drawn on its own so no line crosses a gap
  const truePieces =
    Array.isArray(problem.true_front) && Array.isArray(problem.true_front[0]?.[0]) ? problem.true_front : null;
  const trueFront = truePieces ? truePieces.flat() : Array.isArray(problem.true_front) ? problem.true_front : null;
  // a front of separate points (e.g. ZDT5's, on a binary genome) is drawn as rings, not a line,
  // and so is a piece of a single point
  const discrete = problem.true_front_kind === "points";
  const lonePoints = truePieces ? truePieces.filter((piece) => piece.length === 1).map((piece) => piece[0]) : [];
  // the feasible region of a constrained problem, shaded behind everything
  const region = problem.feasible_region ?? null;
  const regionUrl = useMemo(() => regionImage(region), [region]);

  const target = useMemo(() => {
    const pts = [...names.flatMap((n) => fronts[n] ?? []), ...names.flatMap((n) => infeasible[n] ?? []), ...(trueFront ?? [])];
    const xs = extent(pts.map((p) => p[0])) ?? [0, 1];
    const ys = extent(pts.map((p) => p[1])) ?? [0, 1];
    return [...pad(xs, 0.04), ...pad(ys, 0.06)];
  }, [names, fronts, infeasible, trueFront]);
  const [x0, x1, y0, y1] = useEased(target, reduced);

  const hasInfeasible = names.some((n) => (infeasible[n] ?? []).length);
  const legend = [
    ...names.map((n, k) => ({ label: n, color: palette[k % palette.length], shape: SHAPES[k % SHAPES.length] })),
    ...(hasInfeasible ? [{ label: "infeasible", shape: "ring", className: "text-base-content/50" }] : []),
    ...(trueFront ? [{ label: "true front", shape: discrete ? "ring" : "line", className: "text-base-content/45" }] : []),
    ...(regionUrl ? [{ label: "feasible region", shape: "square", className: "text-base-content/20" }] : []),
  ];

  return (
    <div>
      <div className={`flex flex-wrap items-center justify-between gap-2 ${compact && !feasible ? "" : "mb-2"}`}>
        <Legend items={compact ? [] : legend} />
        {feasible && typeof feasible === "object" ? (
          <span className="text-base-content/65 text-xs tabular-nums">
            feasible:{" "}
            {names
              .filter((n) => typeof feasible[n] === "number")
              .map((n) => `${names.length > 1 ? `${n} ` : ""}${Math.round(feasible[n] * 100)}%`)
              .join(", ")}
          </span>
        ) : null}
      </div>
      <PlotBox
        aspect={0.85}
        minHeight={compact ? 220 : 280}
        label={`The front in ${objectives.join(" and ")}: ${names.map((n) => `${n} ${(fronts[n] ?? []).length} points`).join(", ")}`}
        overlay={({ width }) =>
          hover ? (
            <Tooltip x={hover.px} y={hover.py} width={width}>
              <TipRows
                rows={[
                  ...(names.length > 1 ? [["algorithm", hover.series]] : []),
                  [objectives[0], formatValue(hover.p[0])],
                  [objectives[1], formatValue(hover.p[1])],
                  ...(hover.infeasible ? [["", "infeasible"]] : []),
                ]}
              />
            </Tooltip>
          ) : null
        }
      >
        {({ width, height }) => {
          const area = { left: 48, right: width - 12, top: 24, bottom: height - 38 };
          const x = linear([x0, x1], [area.left, area.right]);
          const y = linear([y0, y1], [area.bottom, area.top]);
          const xt = nice(x0, x1, 5);
          const yt = nice(y0, y1, 5);
          const points = [];
          names.forEach((n, k) => {
            for (const p of infeasible[n] ?? []) points.push({ series: n, k, p, infeasible: true, px: x(p[0]), py: y(p[1]) });
          });
          names.forEach((n, k) => {
            for (const p of fronts[n] ?? []) points.push({ series: n, k, p, infeasible: false, px: x(p[0]), py: y(p[1]) });
          });
          return (
            <>
              <Axes
                x={x}
                y={y}
                area={area}
                xTicks={ticks(xt[0], xt[1], Math.max(3, Math.floor((area.right - area.left) / 70))).filter((t) => t >= x0 && t <= x1)}
                yTicks={ticks(yt[0], yt[1], 5).filter((t) => t >= y0 && t <= y1)}
                xLabel={objectives[0]}
                yLabel={objectives[1]}
              />
              <defs>
                <clipPath id={clip}>
                  <rect x={area.left - 6} y={area.top - 6} width={Math.max(0, area.right - area.left + 12)} height={Math.max(0, area.bottom - area.top + 12)} />
                </clipPath>
              </defs>
              <g clipPath={`url(#${clip})`}>
              {regionUrl ? (
                <>
                  <mask id={`${clip}region`}>
                    <image
                      href={regionUrl}
                      x={x(region.x[0])}
                      y={y(region.y[1])}
                      width={Math.max(0, x(region.x[1]) - x(region.x[0]))}
                      height={Math.max(0, y(region.y[0]) - y(region.y[1]))}
                      preserveAspectRatio="none"
                      style={{ imageRendering: "pixelated" }}
                    />
                  </mask>
                  <rect
                    x={x(region.x[0])}
                    y={y(region.y[1])}
                    width={Math.max(0, x(region.x[1]) - x(region.x[0]))}
                    height={Math.max(0, y(region.y[0]) - y(region.y[1]))}
                    mask={`url(#${clip}region)`}
                    className="fill-base-content/10"
                  />
                </>
              ) : null}
              {(discrete ? (trueFront ?? []) : lonePoints).map((p, i) => (
                <circle key={`t${i}`} cx={x(p[0])} cy={y(p[1])} r={5} fill="none" className="stroke-base-content/45" strokeWidth={1.5} />
              ))}
              {trueFront && !discrete ? (
                <path
                  d={truePieces ? truePieces.map((piece) => piecePath(piece, x, y)).join("") : frontPath(trueFront, x, y)}
                  fill="none"
                  className="stroke-base-content/45"
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              ) : null}
              {points.map((q, i) => (
                <Marker
                  key={i}
                  shape={q.infeasible ? "dot" : SHAPES[q.k % SHAPES.length]}
                  x={q.px}
                  y={q.py}
                  r={q.infeasible ? 3.5 : 4}
                  color={palette[q.k % palette.length]}
                  hollow={q.infeasible}
                  opacity={q.infeasible ? 0.7 : 1}
                />
              ))}
              </g>
              {hover ? <circle cx={hover.px} cy={hover.py} r={8} fill="none" className="stroke-base-content" strokeWidth={1.5} /> : null}
              <rect
                x={area.left}
                y={area.top}
                width={Math.max(0, area.right - area.left)}
                height={Math.max(0, area.bottom - area.top)}
                fill="transparent"
                onPointerMove={(event) => {
                  const p = pointerIn(event);
                  setHover(nearest(points, p.x, p.y, 14));
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
