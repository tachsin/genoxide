"use client";

import { useMemo, useState } from "react";
import { categorical, extent, formatValue, linear, ticks, useEased } from "../chart-kit";
import { Axes, Legend, PlotBox, TipRows, Tooltip, pointerIn } from "../chart-parts";

const MARGIN = { left: 40, right: 14, top: 22, bottom: 38 };
// the frames on each side whose evaluations can fill the window while the axis catches up: the
// axis lags about 4 frames behind at 4× speed
const NEIGHBORS = 8;

// a worker runs one evaluation at a time: its worker and start name it, in every frame
const eventKey = (e) => `${e[0]}:${e[1]}`;

/**
 * The evaluations to draw in the window [t0, t1]: the frame's own, the last ones to end, and the
 * neighbouring frames' that ended before them or started after them. The frame's window starts where the
 * earliest of its own started, so earlier evaluations end in it too. And while the axis eases to
 * the next frame's window, which moves on by most of its width, the earlier frames' evaluations
 * fill what it still shows of the last one: with the frame's own alone, the plot would empty and
 * fill again every frame of the playback.
 */
function shownEvents(frames, index, events, t0, t1) {
  if (!events.length) return events;
  const ends = extent(events.map((e) => e[2]));
  const seen = new Set(events.map(eventKey));
  // in time order, as the bars already drawn are, so that React adds and removes bars only
  const earlier = [];
  const later = [];
  for (const [step, chunks] of [
    [-1, earlier],
    [1, later],
  ]) {
    for (let j = index + step, n = 0; j >= 0 && j < frames.length && n < NEIGHBORS; j += step, n++) {
      const chunk = [];
      // past the window: so are the frames further on (the nearer ones can be too, when the axis
      // lags several frames behind at a high speed)
      let past = true;
      for (const e of frames[j].state?.events ?? []) {
        if (step < 0 ? e[2] >= t0 : e[1] <= t1) past = false;
        if (e[2] < t0 || e[1] > t1) continue;
        // ended before the frame's own, or (scrubbing back) started after the last of them ended
        if ((e[2] < ends[0] || e[1] >= ends[1]) && !seen.has(eventKey(e))) {
          seen.add(eventKey(e));
          chunk.push(e);
        }
      }
      if (past) break;
      chunks.push(chunk);
    }
  }
  return [...earlier.reverse().flat(), ...events, ...later.flat()];
}

/**
 * `timeline`: each evaluation as a bar on its worker's row, over wall-clock
 * time; older evaluations fade. The time axis follows the recorded window.
 */
export default function TimelinePlot({ trace, frame, index, dark, reduced }) {
  const [hover, setHover] = useState(null);
  const workers = Math.max(1, trace.problem?.workers ?? 1);
  const events = frame.state?.events ?? [];
  const color = categorical(dark)[0];
  const row = workers > 16 ? 12 : workers > 8 ? 18 : workers > 2 ? 26 : 44;

  const target = useMemo(() => {
    const starts = extent(events.map((e) => e[1])) ?? [0, 1];
    const ends = extent(events.map((e) => e[2])) ?? [0, 1];
    return [starts[0], Math.max(ends[1], starts[0] + 1e-9)];
  }, [events]);
  const [t0, t1] = useEased(target, reduced);
  const span = target[1] - target[0];
  const rate = span > 0 ? events.length / span : null;
  const now = target[1];
  const shown = shownEvents(trace.frames ?? [], index ?? 0, events, t0, t1);

  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <Legend
          items={[
            { label: "evaluation call", color, shape: "square" },
            { label: "earlier", shape: "square", className: "text-base-content/25" },
          ]}
        />
        {rate !== null ? (
          <span className="text-base-content/65 text-xs tabular-nums">
            {formatValue(Number(rate.toPrecision(3)))} calls/s over the last {events.length}
          </span>
        ) : null}
      </div>
      <PlotBox
        height={workers * row + MARGIN.top + MARGIN.bottom}
        label={`${events.length} evaluations on ${workers} workers`}
        overlay={({ width }) =>
          hover ? (
            <Tooltip x={hover.px} y={hover.py} width={width}>
              <TipRows
                rows={[
                  ["worker", hover.e[0] + 1],
                  ["start", `${formatValue(hover.e[1])} s`],
                  ["duration", `${formatValue(Number((hover.e[2] - hover.e[1]).toPrecision(4)))} s`],
                ]}
              />
            </Tooltip>
          ) : null
        }
      >
        {({ width, height }) => {
          const area = { left: MARGIN.left, right: width - MARGIN.right, top: MARGIN.top, bottom: height - MARGIN.bottom };
          const x = linear([t0, t1], [area.left, area.right]);
          const every = Math.ceil(workers / Math.max(1, Math.floor((area.bottom - area.top) / 14)));
          // the right edge once the axis has caught up; in the plot while it catches up
          const nowX = Math.min(area.right, Math.max(area.left, x(now)));
          return (
            <>
              <Axes
                x={x}
                y={(v) => v}
                area={area}
                xTicks={ticks(t0, t1, Math.max(3, Math.floor((area.right - area.left) / 80))).filter((t) => t >= t0 && t <= t1)}
                xLabel="seconds"
                xGrid
              />
              {Array.from({ length: workers }, (_, w) =>
                w % every === 0 ? (
                  <text key={w} x={area.left - 8} y={area.top + w * row + row / 2} dy="0.32em" textAnchor="end" className="fill-base-content/60 text-[10px]">
                    W{w + 1}
                  </text>
                ) : null,
              )}
              <g>
                {shown.map((e) => {
                  const x0 = Math.max(area.left, x(e[1]));
                  const x1 = Math.min(area.right, x(e[2]));
                  if (x1 <= area.left || x0 >= area.right) return null;
                  // an evaluation keeps its bar from frame to frame, and fades with its time in the
                  // window, not its place in the list: the bars slide with the axis instead of
                  // being drawn again
                  const age = t1 > t0 ? Math.min(1, Math.max(0, (e[2] - t0) / (t1 - t0))) : 1;
                  return (
                    <rect
                      key={eventKey(e)}
                      x={x0}
                      y={area.top + e[0] * row + 2}
                      width={Math.max(1, x1 - x0 - 1)}
                      height={row - 4}
                      rx={Math.min(3, (row - 4) / 3)}
                      fill={color}
                      fillOpacity={0.3 + 0.7 * age}
                    />
                  );
                })}
              </g>
              <line x1={nowX} x2={nowX} y1={area.top - 6} y2={area.bottom} className="stroke-base-content" strokeWidth={1.5} />
              <text x={nowX - 4} y={area.top - 9} textAnchor="end" className="fill-base-content/70 text-[10px] tabular-nums">
                {formatValue(Number(now.toPrecision(4)))} s
              </text>
              <rect
                x={area.left}
                y={area.top}
                width={Math.max(0, area.right - area.left)}
                height={Math.max(0, area.bottom - area.top)}
                fill="transparent"
                onPointerMove={(event) => {
                  const p = pointerIn(event);
                  const w = Math.floor((p.y - area.top) / row);
                  const t = x.invert(p.x);
                  const e = shown.find((ev) => ev[0] === w && t >= ev[1] && t <= ev[2]);
                  setHover(e ? { e, px: p.x, py: area.top + (w + 1) * row } : null);
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
