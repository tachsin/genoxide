"use client";

import { useState } from "react";
import { STATUS, categorical } from "../chart-kit";
import { Legend, Marker, PlotBox, TipRows, Tooltip, nearest, pointerIn } from "../chart-parts";

/**
 * `board`: the best placement of the queens, attacking pairs joined by a line. On a large board
 * (128×128 has squares of 2 to 4 pixels), the squares would be only a texture: the board is plain
 * then, and the attacked queens larger.
 */
export default function BoardPlot({ trace, frame, dark }) {
  const [hover, setHover] = useState(null);
  const n = trace.problem?.n ?? frame.state?.best?.length ?? 8;
  const cols = frame.state?.best ?? [];
  const attacks = frame.state?.attacks ?? [];
  const color = categorical(dark)[0];

  const attackedBy = new Map();
  for (const [a, b] of attacks) {
    attackedBy.set(a, (attackedBy.get(a) ?? 0) + 1);
    attackedBy.set(b, (attackedBy.get(b) ?? 0) + 1);
  }

  return (
    <div>
      <Legend
        className="mb-2"
        items={[
          { label: "queen", color, shape: "dot" },
          { label: "attacked queen", color: STATUS.critical, shape: "diamond" },
          { label: "attacking pair", color: STATUS.critical, shape: "line" },
        ]}
      />
      <PlotBox
        aspect={1}
        minHeight={260}
        maxHeight={540}
        label={`${n} queens on a ${n} by ${n} board, ${attacks.length} attacking pairs`}
        overlay={({ width }) =>
          hover ? (
            <Tooltip x={hover.px} y={hover.py} width={width}>
              <TipRows
                rows={[
                  ["row", hover.row + 1],
                  ["column", (cols[hover.row] ?? 0) + 1],
                  ["attacked by", attackedBy.get(hover.row) ?? 0],
                ]}
              />
            </Tooltip>
          ) : null
        }
      >
        {({ width, height }) => {
          const size = Math.min(width, height);
          const ox = (width - size) / 2;
          const cell = size / n;
          const center = (r) => ({ x: ox + (cols[r] + 0.5) * cell, y: (r + 0.5) * cell });
          const squares = cell >= 6;
          let dark = "";
          for (let r = 0; squares && r < n; r++) {
            for (let c = (r + 1) % 2; c < n; c += 2) dark += `M${(ox + c * cell).toFixed(2)} ${(r * cell).toFixed(2)}h${cell.toFixed(2)}v${cell.toFixed(2)}h${(-cell).toFixed(2)}z`;
          }
          const r = squares ? Math.max(2, cell * (cell < 12 ? 0.38 : 0.3)) : Math.max(1.5, cell * 0.45);
          // an attacked queen stands out among many
          const attackedR = squares ? r * 0.95 : r * 1.3;
          const queens = cols.map((_, row) => ({ row, ...center(row) }));
          return (
            <>
              <rect x={ox} y={0} width={size} height={size} className="fill-base-content/[0.03]" />
              {squares ? <path d={dark} className="fill-base-content/[0.09]" shapeRendering="crispEdges" /> : null}
              <rect x={ox} y={0} width={size} height={size} fill="none" className="stroke-base-content/25" strokeWidth={1} />
              {attacks.map(([a, b]) => {
                const p = center(a);
                const q = center(b);
                return <line key={`${a}-${b}`} x1={p.x} y1={p.y} x2={q.x} y2={q.y} stroke={STATUS.critical} strokeOpacity={0.6} strokeWidth={Math.max(1, Math.min(2, cell / 8))} />;
              })}
              {queens.map((q) =>
                attackedBy.has(q.row) ? (
                  <Marker key={q.row} shape="diamond" x={q.x} y={q.y} r={attackedR} color={STATUS.critical} ring={cell > 8} />
                ) : (
                  <Marker key={q.row} x={q.x} y={q.y} r={r} color={color} ring={cell > 8} />
                ),
              )}
              {hover ? <circle cx={hover.px} cy={hover.py} r={r + 4} fill="none" className="stroke-base-content" strokeWidth={1.5} /> : null}
              <rect
                x={0}
                y={0}
                width={width}
                height={height}
                fill="transparent"
                onPointerMove={(event) => {
                  const p = pointerIn(event);
                  const hit = nearest(
                    queens.map((q) => ({ row: q.row, px: q.x, py: q.y })),
                    p.x,
                    p.y,
                    Math.max(10, cell * 0.6),
                  );
                  setHover(hit);
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
