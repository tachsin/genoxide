"use client";

import { useState } from "react";
import { categorical } from "../chart-kit";
import { Legend, PlotBox, TipRows, Tooltip, pointerIn } from "../chart-parts";

function countOnes(bits = "") {
  let count = 0;
  for (const c of bits) if (c === "1") count++;
  return count;
}

/** `bits`: the population's bit strings, a row per individual, ones filled. */
export default function BitsPlot({ trace, frame, dark }) {
  const [hover, setHover] = useState(null);
  const rows = frame.state?.population ?? [];
  const n = trace.problem?.length ?? rows[0]?.length ?? 1;
  const color = categorical(dark)[0];

  return (
    <div>
      <Legend
        className="mb-2"
        items={[
          { label: "1", color, shape: "square" },
          { label: "0", shape: "square", className: "text-base-content/15" },
        ]}
      />
      <PlotBox
        aspect={Math.max(0.3, Math.min(1.1, (rows.length / n) * 1.6))}
        minHeight={160}
        label={`The population's ${rows.length} bit strings of length ${n}`}
        overlay={({ width }) =>
          hover ? (
            <Tooltip x={hover.px} y={hover.py} width={width}>
              <TipRows
                rows={[
                  ["individual", hover.row + 1],
                  ["ones", `${countOnes(rows[hover.row])} of ${n}`],
                ]}
              />
            </Tooltip>
          ) : null
        }
      >
        {({ width, height }) => {
          const cw = width / n;
          const rh = height / Math.max(1, rows.length);
          const gap = rh >= 5 ? 1 : 0;
          let ones = "";
          rows.forEach((bits, r) => {
            const y = (r * rh).toFixed(2);
            const h = (rh - gap).toFixed(2);
            let start = -1;
            for (let i = 0; i <= bits.length; i++) {
              const on = bits[i] === "1";
              if (on && start < 0) start = i;
              if (!on && start >= 0) {
                ones += `M${(start * cw).toFixed(2)} ${y}h${((i - start) * cw).toFixed(2)}v${h}h${(-(i - start) * cw).toFixed(2)}z`;
                start = -1;
              }
            }
          });
          return (
            <>
              {rows.map((_, r) => (
                <rect key={r} x={0} y={r * rh} width={width} height={Math.max(0.5, rh - gap)} className="fill-base-content/[0.07]" />
              ))}
              <path d={ones} fill={color} shapeRendering="crispEdges" />
              {hover ? (
                <rect x={0} y={hover.row * rh} width={width} height={Math.max(0.5, rh - gap)} fill="none" className="stroke-base-content" strokeWidth={1.5} />
              ) : null}
              <rect
                x={0}
                y={0}
                width={width}
                height={height}
                fill="transparent"
                onPointerMove={(event) => {
                  const p = pointerIn(event);
                  const row = Math.min(rows.length - 1, Math.max(0, Math.floor(p.y / rh)));
                  setHover({ row, px: p.x, py: row * rh });
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
