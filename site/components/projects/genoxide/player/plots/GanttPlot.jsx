"use client";

import { useMemo, useState } from "react";
import { categorical, formatValue, linear, ticks } from "../chart-kit";
import { Axes, Legend, PlotBox, TipRows, Tooltip, pointerIn } from "../chart-parts";

/** Black or white text on a fill, whichever reads. */
function inkOn(hex) {
  const n = Number.parseInt(hex.slice(1), 16);
  const [r, g, b] = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return l > 0.36 ? "#111" : "#fff";
}

const ROW = 30;
const MARGIN = { left: 40, right: 14, top: 26, bottom: 38 };

/** `gantt`: the best schedule, a row per machine, a color and label per job. */
export default function GanttPlot({ trace, frame, dark }) {
  const [hover, setHover] = useState(null);
  const machines = trace.problem?.machines ?? 0;
  const jobCount = trace.problem?.jobs?.length ?? 0;
  const ops = frame.state?.best ?? [];
  const palette = categorical(dark);
  const makespan = ops.reduce((m, o) => Math.max(m, o[4]), 0);
  const optimum = typeof trace.optimum === "number" ? trace.optimum : null;
  // One time axis for the whole run, so the schedule visibly shrinks.
  const horizon = useMemo(
    () =>
      trace.frames.reduce((m, f) => Math.max(m, ...(f.state?.best ?? []).map((o) => o[4])), optimum ?? 0) || 1,
    [trace, optimum],
  );
  const colorOf = (job) => palette[job % palette.length];

  return (
    <div>
      <Legend
        className="mb-2"
        items={[
          ...Array.from({ length: jobCount }, (_, j) => ({ label: `job ${j + 1}`, color: colorOf(j), shape: "square" })),
          { label: "makespan", shape: "line", className: "text-base-content" },
          ...(optimum !== null ? [{ label: `optimum ${formatValue(optimum)}`, shape: "dash", className: "text-base-content/60" }] : []),
        ]}
      />
      <PlotBox
        height={machines * ROW + MARGIN.top + MARGIN.bottom}
        label={`A schedule of ${jobCount} jobs on ${machines} machines, makespan ${formatValue(makespan)}`}
        overlay={({ width }) =>
          hover ? (
            <Tooltip x={hover.px} y={hover.py} width={width}>
              <TipRows
                rows={[
                  ["job", hover.op[0] + 1],
                  ["operation", hover.op[1] + 1],
                  ["machine", hover.op[2] + 1],
                  ["start – end", `${formatValue(hover.op[3])} – ${formatValue(hover.op[4])}`],
                ]}
              />
            </Tooltip>
          ) : null
        }
      >
        {({ width, height }) => {
          const area = { left: MARGIN.left, right: width - MARGIN.right, top: MARGIN.top, bottom: height - MARGIN.bottom };
          const x = linear([0, horizon], [area.left, area.right]);
          const rowY = (m) => area.top + m * ROW;
          const y = (v) => v;
          return (
            <>
              <Axes x={x} y={y} area={area} xTicks={ticks(0, horizon, Math.max(3, Math.floor((area.right - area.left) / 70)))} xLabel="time" xGrid />
              {Array.from({ length: machines }, (_, m) => (
                <text key={m} x={area.left - 8} y={rowY(m) + ROW / 2} dy="0.32em" textAnchor="end" className="fill-base-content/60 text-[10px]">
                  M{m + 1}
                </text>
              ))}
              {ops.map((op) => {
                const [job, , machine, start, end] = op;
                const x0 = x(start);
                const w = Math.max(1, x(end) - x0 - 2);
                const fill = colorOf(job);
                const dim = hover && hover.op[0] !== job;
                return (
                  <g key={`${job}-${op[1]}`} opacity={dim ? 0.35 : 1}>
                    <rect x={x0 + 1} y={rowY(machine) + 3} width={w} height={ROW - 6} rx={3} fill={fill} />
                    {w > 16 ? (
                      <text x={x0 + 1 + w / 2} y={rowY(machine) + ROW / 2} dy="0.34em" textAnchor="middle" fill={inkOn(fill)} className="font-medium text-[10px]">
                        {job + 1}
                      </text>
                    ) : null}
                  </g>
                );
              })}
              {optimum !== null ? (
                <line x1={x(optimum)} x2={x(optimum)} y1={area.top - 6} y2={area.bottom} className="stroke-base-content/55" strokeDasharray="4 3" strokeWidth={1} />
              ) : null}
              <line x1={x(makespan)} x2={x(makespan)} y1={area.top - 6} y2={area.bottom} className="stroke-base-content" strokeWidth={1.5} />
              <text
                x={x(makespan) > area.right - 90 ? x(makespan) - 4 : x(makespan) + 4}
                y={area.top - 10}
                textAnchor={x(makespan) > area.right - 90 ? "end" : "start"}
                className="fill-base-content font-medium text-[10px] tabular-nums"
              >
                makespan {formatValue(makespan)}
              </text>
              <rect
                x={area.left}
                y={area.top}
                width={Math.max(0, area.right - area.left)}
                height={Math.max(0, area.bottom - area.top)}
                fill="transparent"
                onPointerMove={(event) => {
                  const p = pointerIn(event);
                  const machine = Math.floor((p.y - area.top) / ROW);
                  const t = x.invert(p.x);
                  const op = ops.find((o) => o[2] === machine && t >= o[3] && t <= o[4]);
                  setHover(op ? { op, px: p.x, py: rowY(machine) + ROW } : null);
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
