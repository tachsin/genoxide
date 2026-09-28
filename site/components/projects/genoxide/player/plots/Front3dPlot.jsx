"use client";

import { RotateCcw } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { categorical, formatTick, formatValue, useEased } from "../chart-kit";
import { Legend, PlotBox, TipRows, Tooltip, nearest, pointerIn } from "../chart-parts";

const DEG = Math.PI / 180;
const VIEW = { yaw: 225, pitch: 24 };

/** The unit sphere's positive octant as wireframe arcs (the true front of DTLZ2-like problems). */
function sphereArcs() {
  const arcs = [];
  const steps = 24;
  for (const lon of [0, 22.5, 45, 67.5, 90]) {
    const arc = [];
    for (let k = 0; k <= steps; k++) {
      const lat = (90 * k) / steps;
      arc.push([Math.cos(lat * DEG) * Math.cos(lon * DEG), Math.cos(lat * DEG) * Math.sin(lon * DEG), Math.sin(lat * DEG)]);
    }
    arcs.push(arc);
  }
  for (const lat of [0, 22.5, 45, 67.5]) {
    const arc = [];
    for (let k = 0; k <= steps; k++) {
      const lon = (90 * k) / steps;
      arc.push([Math.cos(lat * DEG) * Math.cos(lon * DEG), Math.cos(lat * DEG) * Math.sin(lon * DEG), Math.sin(lat * DEG)]);
    }
    arcs.push(arc);
  }
  return arcs;
}

const ARCS = sphereArcs();

/**
 * The plane f1 + f2 + f3 = s in the positive octant (the true front of DTLZ1): its edges and
 * the lines where one objective is a quarter, a half and three quarters of s.
 */
function simplexLines(s) {
  const lines = [];
  for (let axis = 0; axis < 3; axis++) {
    const [a, b] = [0, 1, 2].filter((k) => k !== axis);
    for (const level of [0, 0.25, 0.5, 0.75]) {
      const c = level * s;
      const from = [0, 0, 0];
      const to = [0, 0, 0];
      from[axis] = c;
      to[axis] = c;
      from[a] = s - c;
      to[b] = s - c;
      lines.push([from, to]);
    }
  }
  return lines;
}

/** Orthographic projection: f1, f2 on the floor, f3 up; returns screen x, y (up) and depth (toward the viewer). */
function projector(yaw, pitch) {
  const ct = Math.cos(yaw * DEG);
  const st = Math.sin(yaw * DEG);
  const cp = Math.cos(pitch * DEG);
  const sp = Math.sin(pitch * DEG);
  return ([x, y, z]) => ({
    x: x * ct - y * st,
    y: x * st * sp + y * ct * sp + z * cp,
    depth: -x * st * cp - y * ct * cp + z * sp,
  });
}

/**
 * `front-3d`: the three-objective front, rotatable (drag, or the arrow keys
 * when focused), over the true front's wireframe. Orthographic, drawn in SVG.
 * Over a sphere, every axis runs from 0 to one maximum; otherwise each axis
 * spans its own objective's values (they can be negative, and far apart in
 * scale). `compact` (a panel of a grid) leaves the legend to the grid.
 */
export default function Front3dPlot({ trace, frame, dark, reduced, compact = false }) {
  const [view, setView] = useState(VIEW);
  const [hover, setHover] = useState(null);
  const drag = useRef(null);
  const color = categorical(dark)[0];
  const objectives = trace.problem?.objectives ?? ["f1", "f2", "f3"];
  const sphere = trace.problem?.true_front === "sphere";
  // the plane f1 + f2 + f3 = simplex (DTLZ1's front, with simplex 0.5)
  const simplex = trace.problem?.true_front === "simplex" ? Number(trace.problem?.simplex ?? 1) : null;
  // a true front drawn from the origin: every axis from 0 to one maximum
  const shared = sphere || simplex !== null;
  const front = frame.state?.front ?? [];

  // [lo1, lo2, lo3, hi1, hi2, hi3]
  const target = useMemo(() => {
    if (shared) {
      let m = sphere ? 1 : simplex;
      for (const p of front) for (const c of p) if (c > m) m = c;
      return [0, 0, 0, m * 1.05, m * 1.05, m * 1.05];
    }
    if (!front.length) return [0, 0, 0, 1, 1, 1];
    const lo = [0, 1, 2].map((k) => Math.min(...front.map((p) => p[k])));
    const hi = [0, 1, 2].map((k) => Math.max(...front.map((p) => p[k])));
    const span = [0, 1, 2].map((k) => hi[k] - lo[k] || Math.abs(hi[k]) || 1);
    // from each objective's least value, as the sphere's cube starts at 0, with room above
    return [...lo, ...hi.map((v, k) => v + span[k] * 0.05)];
  }, [front, sphere, shared, simplex]);
  const eased = useEased(target, reduced);
  const lo = eased.slice(0, 3);
  const hi = eased.slice(3);

  const rotate = (dYaw, dPitch) =>
    setView((v) => ({ yaw: (v.yaw + dYaw + 360) % 360, pitch: Math.max(-10, Math.min(89, v.pitch + dPitch)) }));

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <Legend
          items={
            compact
              ? []
              : [
                  { label: "front", color, shape: "dot" },
                  ...(sphere ? [{ label: "true front (sphere)", shape: "line", className: "text-base-content/35" }] : []),
                  ...(simplex !== null ? [{ label: `true front (f1 + f2 + f3 = ${simplex})`, shape: "line", className: "text-base-content/35" }] : []),
                ]
          }
        />
        <button
          type="button"
          className="btn btn-ghost btn-xs gap-1 font-normal text-base-content/70"
          onClick={() => setView(VIEW)}
          // aria-disabled, not disabled: clicking it would otherwise drop the focus to the page
          aria-disabled={view.yaw === VIEW.yaw && view.pitch === VIEW.pitch}
        >
          <RotateCcw size={12} aria-hidden />
          Reset view
        </button>
      </div>
      <PlotBox
        aspect={0.9}
        minHeight={compact ? 220 : 280}
        maxHeight={540}
        label={`The front of ${front.length} points in ${objectives.join(", ")}. Drag or use the arrow keys to rotate.`}
        svgProps={{
          tabIndex: 0,
          className: "block select-none overflow-visible cursor-grab rounded-lg outline-none focus-visible:ring-2 focus-visible:ring-primary active:cursor-grabbing",
          style: { touchAction: "pan-y" },
          onKeyDown: (event) => {
            const keys = { ArrowLeft: [-10, 0], ArrowRight: [10, 0], ArrowUp: [0, 8], ArrowDown: [0, -8] };
            const d = keys[event.key];
            if (!d) return;
            event.preventDefault();
            event.stopPropagation();
            rotate(d[0], d[1]);
          },
          onPointerDown: (event) => {
            drag.current = { x: event.clientX, y: event.clientY };
            setHover(null);
            event.currentTarget.setPointerCapture(event.pointerId);
          },
          onPointerMove: (event) => {
            if (!drag.current) return;
            const dx = event.clientX - drag.current.x;
            const dy = event.clientY - drag.current.y;
            drag.current = { x: event.clientX, y: event.clientY };
            rotate(-dx * 0.5, dy * 0.4);
          },
          onPointerUp: () => {
            drag.current = null;
          },
          onPointerCancel: () => {
            drag.current = null;
          },
        }}
        overlay={({ width }) =>
          hover ? (
            <Tooltip x={hover.px} y={hover.py} width={width}>
              <TipRows
                rows={[
                  ...objectives.map((o, k) => [o, formatValue(hover.p[k])]),
                  ...(sphere ? [["distance to origin", formatValue(Number(Math.hypot(...hover.p).toPrecision(4)))]] : []),
                  ...(simplex !== null ? [["f1 + f2 + f3", formatValue(Number((hover.p[0] + hover.p[1] + hover.p[2]).toPrecision(4)))]] : []),
                ]}
              />
            </Tooltip>
          ) : null
        }
      >
        {({ width, height }) => {
          const project = projector(view.yaw, view.pitch);
          const unit = (Math.min(width, height) / 2) * 0.98;
          const cx = width / 2;
          const cy = height / 2 + unit * 0.05;
          // Centre the box of the ranges, scaled to a unit cube, on the plot.
          const at = (p) => {
            const q = project([0, 1, 2].map((k) => (p[k] - lo[k]) / (hi[k] - lo[k] || 1) - 0.5));
            return { x: cx + q.x * unit, y: cy - q.y * unit, depth: q.depth };
          };
          const origin = at(lo);
          const axes = [0, 1, 2].map((k) => {
            const end = [...lo];
            end[k] = hi[k];
            const label = [...lo];
            label[k] = hi[k] + (hi[k] - lo[k]) * 0.12;
            return { k, end: at(end), label: at(label) };
          });
          const arcs = sphere
            ? ARCS.map((arc) =>
                arc
                  .map((p, i) => {
                    const s = at(p);
                    return `${i ? "L" : "M"}${s.x.toFixed(1)} ${s.y.toFixed(1)}`;
                  })
                  .join(""),
              )
            : simplex !== null
              ? simplexLines(simplex).map(([from, to]) => {
                  const a = at(from);
                  const b = at(to);
                  return `M${a.x.toFixed(1)} ${a.y.toFixed(1)}L${b.x.toFixed(1)} ${b.y.toFixed(1)}`;
                })
              : [];
          const points = front.map((p) => ({ p, ...at(p) }));
          points.sort((a, b) => a.depth - b.depth);
          const depths = points.map((q) => q.depth);
          const dMin = Math.min(...depths);
          const dMax = Math.max(...depths);
          const hits = points.map((q) => ({ ...q, px: q.x, py: q.y }));

          return (
            <>
              {axes.map((a) => (
                <g key={a.k}>
                  <line x1={origin.x} y1={origin.y} x2={a.end.x} y2={a.end.y} className="stroke-base-content/40" strokeWidth={1} />
                  <text x={a.label.x} y={a.label.y} dy="0.32em" textAnchor="middle" className="fill-base-content/70 text-[11px]">
                    {objectives[a.k]}
                  </text>
                </g>
              ))}
              {arcs.map((d, i) => (
                <path key={i} d={d} fill="none" className="stroke-base-content/25" strokeWidth={1} />
              ))}
              {points.map((q, i) => {
                const t = dMax > dMin ? (q.depth - dMin) / (dMax - dMin) : 1;
                return (
                  <circle
                    key={i}
                    cx={q.x}
                    cy={q.y}
                    r={3 + 1.5 * t}
                    fill={color}
                    fillOpacity={0.45 + 0.55 * t}
                    stroke="var(--color-base-100)"
                    strokeWidth={1.2}
                  />
                );
              })}
              {hover ? <circle cx={hover.px} cy={hover.py} r={8} fill="none" className="stroke-base-content" strokeWidth={1.5} /> : null}
              <rect
                x={0}
                y={0}
                width={width}
                height={height}
                fill="transparent"
                onPointerMove={(event) => {
                  if (drag.current) return;
                  const p = pointerIn(event);
                  setHover(nearest(hits, p.x, p.y, 12));
                }}
                onPointerLeave={() => setHover(null)}
              />
            </>
          );
        }}
      </PlotBox>
      <p className="mt-1 text-base-content/50 text-xs">
        {shared
          ? `Each axis runs from 0 to ${formatTick(Number(hi[0].toPrecision(2)))}.`
          : `${objectives.map((o, k) => `${o} ${formatTick(Number(lo[k].toPrecision(2)))} to ${formatTick(Number(hi[k].toPrecision(2)))}`).join(", ")}.`}{" "}
        {compact ? "Drag to rotate." : "Drag to rotate, or focus the plot and use the arrow keys."}
      </p>
    </div>
  );
}
