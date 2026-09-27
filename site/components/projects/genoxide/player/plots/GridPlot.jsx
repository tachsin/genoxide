"use client";

import { Suspense, lazy, useMemo } from "react";
import { categorical } from "../chart-kit";
import { Legend, SHAPES } from "../chart-parts";
import { contourLegend } from "../contour";

// The plots a grid repeats, loaded on demand like the player's own.
const PANELS = {
  contour: lazy(() => import("./ContourPlot")),
  "front-2d": lazy(() => import("./Front2dPlot")),
  "front-3d": lazy(() => import("./Front3dPlot")),
};

/** One legend for all the panels: what any of them shows. */
function legendOf(kind, traces, frames, palette) {
  if (kind === "contour") {
    const best = frames.some((f) => f.state?.best);
    const minima = Math.max(0, ...traces.map((t) => t.problem?.minima?.length ?? 0));
    return contourLegend({ best, minima, palette });
  }
  if (kind === "front-2d") {
    const names = [...new Set(traces.flatMap((t) => t.problem?.series ?? []))];
    const infeasible = frames.some((f) => Object.values(f.state?.infeasible ?? {}).some((points) => points.length));
    const trueFront = traces.some((t) => Array.isArray(t.problem?.true_front));
    return [
      ...names.map((n, k) => ({ label: n, color: palette[k % palette.length], shape: SHAPES[k % SHAPES.length] })),
      ...(infeasible ? [{ label: "infeasible", shape: "ring", className: "text-base-content/50" }] : []),
      ...(trueFront ? [{ label: "true front", shape: "line", className: "text-base-content/45" }] : []),
    ];
  }
  return [{ label: "front", color: palette[0], shape: "dot" }];
}

/**
 * `grid`: small multiples of another plot kind, one panel per problem of the
 * run. `problem.panel_plot` names the kind, `problem.panels[k].problem` is
 * the k-th panel's own `problem`, and each frame's `state.panels[k]` its
 * `state`, exactly as that kind takes them in a trace of its own.
 */
export default function GridPlot({ trace, frame, index, dark, reduced }) {
  const kind = trace.problem?.panel_plot;
  const Panel = PANELS[kind] ?? null;
  const panels = trace.problem?.panels ?? [];
  const palette = categorical(dark);

  // each panel's own trace; the frames don't matter to the panels, only the current one
  const traces = useMemo(() => panels.map((p) => ({ ...trace, plot: kind, problem: p.problem ?? {} })), [trace, kind, panels]);
  const states = frame.state?.panels ?? [];
  const frames = panels.map((_, k) => ({ ...frame, state: states[k] ?? {} }));
  const legend = legendOf(kind, traces, frames, palette);

  if (!Panel) return null;
  const columns = panels.length % 3 === 0 ? "sm:grid-cols-2 lg:grid-cols-3" : "sm:grid-cols-2";
  return (
    <div>
      <Legend className="mb-3" items={legend} />
      <div className={`grid gap-x-6 gap-y-5 ${columns}`}>
        {panels.map((p, k) => (
          <figure key={p.title ?? k} className="min-w-0">
            <figcaption className="mb-1 font-medium text-base-content/75 text-xs">{p.title}</figcaption>
            <Suspense fallback={<div className="aspect-square animate-pulse rounded-lg bg-base-content/5 motion-reduce:animate-none" />}>
              <Panel trace={traces[k]} frame={frames[k]} index={index} dark={dark} reduced={reduced} compact />
            </Suspense>
          </figure>
        ))}
      </div>
    </div>
  );
}
