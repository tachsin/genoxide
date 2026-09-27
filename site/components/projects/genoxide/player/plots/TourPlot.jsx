"use client";

import { useMemo, useState } from "react";
import { categorical, extent, formatValue } from "../chart-kit";
import { Legend, PlotBox, TipRows, Tooltip, nearest, pointerIn } from "../chart-parts";

function tourPath(order, at) {
  if (!order?.length) return "";
  let d = "";
  order.forEach((city, i) => {
    const p = at(city);
    if (p) d += `${i ? "L" : "M"}${p.x.toFixed(1)} ${p.y.toFixed(1)}`;
  });
  return `${d}Z`;
}

/** `tour`: the best tour through the points, the previous frame's tour faint behind it. */
export default function TourPlot({ trace, frame, index, dark }) {
  const [hover, setHover] = useState(null);
  const points = trace.problem?.points ?? [];
  const order = frame.state?.best ?? [];
  const previous = index > 0 ? trace.frames[index - 1]?.state?.best : null;
  const color = categorical(dark)[0];
  const bounds = useMemo(() => {
    const xs = extent(points.map((p) => p[0])) ?? [0, 1];
    const ys = extent(points.map((p) => p[1])) ?? [0, 1];
    return { xs, ys, ratio: (ys[1] - ys[0] || 1) / (xs[1] - xs[0] || 1) };
  }, [points]);
  const position = useMemo(() => {
    const m = new Map();
    order.forEach((city, i) => m.set(city, i));
    return m;
  }, [order]);

  return (
    <div>
      <Legend
        className="mb-2"
        items={[
          { label: "best tour", color, shape: "line" },
          ...(previous ? [{ label: "previous frame", shape: "line", className: "text-base-content/25" }] : []),
          { label: "location", shape: "dot", className: "text-base-content/70" },
        ]}
      />
      <PlotBox
        aspect={Math.min(1.1, Math.max(0.5, bounds.ratio))}
        minHeight={260}
        label={`A tour through ${points.length} locations`}
        overlay={({ width }) =>
          hover ? (
            <Tooltip x={hover.px} y={hover.py} width={width}>
              <TipRows
                rows={[
                  ["location", hover.city + 1],
                  ["x, y", `${formatValue(points[hover.city][0])}, ${formatValue(points[hover.city][1])}`],
                  ["stop", `${(position.get(hover.city) ?? 0) + 1} of ${order.length}`],
                ]}
              />
            </Tooltip>
          ) : null
        }
      >
        {({ width, height }) => {
          const padding = 10;
          const w = width - padding * 2;
          const h = height - padding * 2;
          const scale = Math.min(w / (bounds.xs[1] - bounds.xs[0] || 1), h / (bounds.ys[1] - bounds.ys[0] || 1));
          const ox = padding + (w - (bounds.xs[1] - bounds.xs[0]) * scale) / 2;
          const oy = padding + (h - (bounds.ys[1] - bounds.ys[0]) * scale) / 2;
          // y grows upward, as on a map
          const at = (city) => {
            const p = points[city];
            return p ? { x: ox + (p[0] - bounds.xs[0]) * scale, y: oy + (bounds.ys[1] - p[1]) * scale } : null;
          };
          const located = points.map((_, city) => ({ city, px: at(city).x, py: at(city).y }));
          return (
            <>
              {previous ? (
                <path d={tourPath(previous, at)} fill="none" className="stroke-base-content/15" strokeWidth={3} strokeLinejoin="round" />
              ) : null}
              <path d={tourPath(order, at)} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
              {located.map((p) => (
                <circle key={p.city} cx={p.px} cy={p.py} r={3} className="fill-base-content/75" stroke="var(--color-base-100)" strokeWidth={1.5} />
              ))}
              {hover ? <circle cx={hover.px} cy={hover.py} r={7} fill="none" className="stroke-base-content" strokeWidth={1.5} /> : null}
              <rect
                x={0}
                y={0}
                width={width}
                height={height}
                fill="transparent"
                onPointerMove={(event) => {
                  const p = pointerIn(event);
                  setHover(nearest(located, p.x, p.y, 14));
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
