"use client";

import { useState } from "react";
import { STATUS, categorical, extent, formatValue, linear, nice, ticks } from "../chart-kit";
import { Axes, Legend, PlotBox, TipRows, Tooltip, nearest, pointerIn } from "../chart-parts";

/**
 * `knapsack`: the packed items laid end to end against the capacity, and
 * every item by weight and value, the packed ones filled.
 */
export default function KnapsackPlot({ trace, frame, dark }) {
  const [hover, setHover] = useState(null);
  const items = trace.problem?.items ?? [];
  const capacity = trace.problem?.capacity ?? 0;
  const packed = frame.state?.best ?? [];
  const color = categorical(dark)[0];

  let weight = 0;
  let value = 0;
  items.forEach((item, i) => {
    if (packed[i]) {
      weight += item.weight;
      value += item.value;
    }
  });

  const wExtent = extent(items.map((i) => i.weight)) ?? [0, 1];
  const vExtent = extent(items.map((i) => i.value)) ?? [0, 1];
  const wDomain = nice(0, wExtent[1]);
  const vDomain = nice(0, vExtent[1]);
  const totalMax = Math.max(capacity, weight) * 1.04;

  return (
    <div>
      <Legend
        className="mb-2"
        items={[
          { label: "packed", color, shape: "dot" },
          { label: "left out", shape: "ring", className: "text-base-content/45" },
          ...(weight > capacity ? [{ label: "over capacity", color: STATUS.critical, shape: "square" }] : []),
        ]}
      />
      <PlotBox
        aspect={0.78}
        minHeight={300}
        label={`Packed ${packed.filter(Boolean).length} of ${items.length} items: weight ${formatValue(weight)} of ${formatValue(capacity)}, value ${formatValue(value)}`}
        overlay={({ width }) =>
          hover ? (
            <Tooltip x={hover.px} y={hover.py} width={width}>
              <TipRows
                rows={[
                  ["item", hover.i + 1],
                  ["weight", formatValue(items[hover.i].weight)],
                  ["value", formatValue(items[hover.i].value)],
                  ["value / weight", formatValue(Number((items[hover.i].value / items[hover.i].weight).toPrecision(3)))],
                  ["packed", packed[hover.i] ? "yes" : "no"],
                ]}
              />
            </Tooltip>
          ) : null
        }
      >
        {({ width, height }) => {
          // The knapsack: packed weights end to end, the capacity marked.
          const bar = { left: 8, right: width - 8, top: 20, height: 22 };
          const bx = linear([0, totalMax], [bar.left, bar.right]);
          const segments = [];
          let at = 0;
          items.forEach((item, i) => {
            if (!packed[i]) return;
            segments.push({ i, from: at, to: at + item.weight });
            at += item.weight;
          });

          const area = { left: 48, right: width - 14, top: bar.top + bar.height + 44, bottom: height - 38 };
          const x = linear(wDomain, [area.left, area.right]);
          const y = linear(vDomain, [area.bottom, area.top]);
          const points = items.map((item, i) => ({ i, px: x(item.weight), py: y(item.value) }));

          return (
            <>
              <text x={bar.left} y={bar.top - 7} className="fill-base-content/60 text-[11px] tabular-nums">
                weight {formatValue(weight)} of {formatValue(capacity)} · value {formatValue(value)}
              </text>
              <rect x={bar.left} y={bar.top} width={bx(capacity) - bar.left} height={bar.height} rx={4} className="fill-base-content/[0.06]" />
              {segments.map((s) => {
                const x0 = bx(s.from);
                const x1 = bx(s.to);
                const over = s.to > capacity;
                return (
                  <rect
                    key={s.i}
                    x={x0 + 1}
                    y={bar.top}
                    width={Math.max(0.5, x1 - x0 - 2)}
                    height={bar.height}
                    rx={2}
                    fill={over ? STATUS.critical : color}
                    opacity={hover && hover.i !== s.i ? 0.45 : 1}
                  />
                );
              })}
              <line x1={bx(capacity)} x2={bx(capacity)} y1={bar.top - 4} y2={bar.top + bar.height + 4} className="stroke-base-content" strokeWidth={1.5} />
              <text x={bx(capacity)} y={bar.top + bar.height + 15} textAnchor="end" className="fill-base-content/60 text-[10px]">
                capacity
              </text>

              <Axes
                x={x}
                y={y}
                area={area}
                xTicks={ticks(wDomain[0], wDomain[1], 5)}
                yTicks={ticks(vDomain[0], vDomain[1], 4)}
                xLabel="weight"
                yLabel="value"
              />
              {items.map((item, i) =>
                packed[i] ? null : (
                  <circle
                    key={i}
                    cx={x(item.weight)}
                    cy={y(item.value)}
                    r={4}
                    fill="none"
                    className="stroke-base-content/40"
                    strokeWidth={1.5}
                  />
                ),
              )}
              {items.map((item, i) =>
                packed[i] ? (
                  <circle key={i} cx={x(item.weight)} cy={y(item.value)} r={4.5} fill={color} stroke="var(--color-base-100)" strokeWidth={1.5} />
                ) : null,
              )}
              {hover ? (
                <circle cx={x(items[hover.i].weight)} cy={y(items[hover.i].value)} r={8} fill="none" className="stroke-base-content" strokeWidth={1.5} />
              ) : null}
              <rect
                x={0}
                y={0}
                width={width}
                height={height}
                fill="transparent"
                onPointerMove={(event) => {
                  const p = pointerIn(event);
                  if (p.y >= bar.top && p.y <= bar.top + bar.height) {
                    const w = bx.invert(p.x);
                    const s = segments.find((seg) => w >= seg.from && w < seg.to);
                    setHover(s ? { i: s.i, px: p.x, py: bar.top + bar.height } : null);
                    return;
                  }
                  const hit = nearest(points, p.x, p.y);
                  setHover(hit ? { i: hit.i, px: hit.px, py: hit.py } : null);
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
