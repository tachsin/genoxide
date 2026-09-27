"use client";

import { useMemo, useState } from "react";
import { Axes, Legend, PlotBox, TipRows, Tooltip, pointerIn } from "./chart-parts";
import { categorical, extent, formatValue, linear, logTickFormat, logTicks, logarithmic, nice, ticks } from "./chart-kit";

/**
 * The curve beside every solution plot: `best` (and `median`) per recorded
 * frame, or the hypervolume of each series for multi-objective runs. The
 * part already played is drawn in full, the rest faint, with a marker at the
 * current frame. Click or drag on it to seek.
 */

/** The curve's series from a trace: what's plotted, and its labels. */
export function curveOf(trace) {
  const frames = trace.frames;
  // A timeline (asynchronous or GPU evaluation) is about time, so its curve runs over the
  // recorded seconds; the others over their trace's x_label.
  const overTime = trace.plot === "timeline" && frames.every((f) => typeof f.seconds === "number");
  const xKey = overTime ? "seconds" : trace.x_label === "evaluations" ? "evaluations" : "generation";
  const xs = frames.map((f) => f[xKey] ?? f.generation);
  const xLabel = xKey;
  // for the curve's accessible name: "best value over time"
  const over = { generation: "generations", evaluations: "evaluations", seconds: "time" }[xKey];

  if (frames.some((f) => typeof f.best === "number")) {
    const series = [{ key: "best", label: "best", values: frames.map((f) => f.best), slot: 0 }];
    if (frames.some((f) => typeof f.median === "number")) {
      series.push({ key: "median", label: "median", values: frames.map((f) => f.median), muted: true });
    }
    return {
      xs,
      xLabel,
      over,
      yLabel: trace.y_label ?? "best",
      logY: !!trace.log_y,
      optimum: numberOrNull(trace.optimum),
      minimize: trace.objective !== "maximize",
      series,
    };
  }

  const hv = frames.map((f) => f.state?.hypervolume);
  if (hv.some((v) => typeof v === "number")) {
    return {
      xs,
      xLabel,
      over,
      yLabel: "hypervolume",
      logY: !!trace.log_y,
      optimum: numberOrNull(trace.optimum),
      minimize: false,
      series: [{ key: "hv", label: "hypervolume", values: hv.map(numberOrNull), slot: 0 }],
    };
  }
  if (hv.some((v) => v && typeof v === "object")) {
    const names = trace.problem?.series ?? Object.keys(hv.find((v) => v && typeof v === "object"));
    return {
      xs,
      xLabel,
      over,
      yLabel: "hypervolume",
      logY: !!trace.log_y,
      optimum: numberOrNull(trace.optimum),
      minimize: false,
      series: names.map((name, i) => ({ key: name, label: name, values: hv.map((v) => numberOrNull(v?.[name])), slot: i })),
    };
  }
  return null;
}

function numberOrNull(v) {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function pathOf(xs, values, x, y, logY, upTo = Infinity) {
  let d = "";
  let pen = false;
  const end = Math.min(values.length - 1, upTo);
  for (let i = 0; i <= end; i++) {
    const v = values[i];
    // on a log scale, 0 (an error that reached the optimum) sits on the floor
    if (v === null || v === undefined) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${x(xs[i]).toFixed(1)} ${y(v).toFixed(1)}`;
    pen = true;
  }
  return d;
}

const MARGIN = { left: 52, right: 14, top: 12, bottom: 38 };

export default function FitnessCurve({ curve, index, onSeek, dark }) {
  const [hover, setHover] = useState(null);
  const palette = categorical(dark);

  const domain = useMemo(() => {
    const all = curve.series.flatMap((s) => s.values);
    if (curve.optimum !== null && !(curve.logY && curve.optimum <= 0)) all.push(curve.optimum);
    const xr = extent(curve.xs) ?? [0, 1];
    if (curve.logY) {
      const positive = all.filter((v) => typeof v === "number" && v > 0);
      const e = extent(positive) ?? [1e-6, 1];
      // values at or below 0 are drawn on a floor a decade under the smallest
      const floorNeeded = all.some((v) => typeof v === "number" && v <= 0);
      const lo = 10 ** (Math.floor(Math.log10(e[0])) - (floorNeeded ? 1 : 0));
      const hi = 10 ** Math.ceil(Math.log10(e[1]));
      return { x: xr, y: [lo, hi === lo ? lo * 10 : hi] };
    }
    const e = extent(all) ?? [0, 1];
    return { x: xr, y: nice(e[0], e[1], 5) };
  }, [curve]);

  const seriesColor = (s) => (s.muted ? "currentColor" : palette[s.slot % palette.length]);
  // A log scale has no place for an optimum of 0: the curve's approach to it is the story.
  const showOptimum = curve.optimum !== null && !(curve.logY && curve.optimum <= 0);
  const legend =
    curve.series.length > 1 || showOptimum
      ? [
          ...curve.series.map((s) => ({
            label: s.label,
            color: s.muted ? undefined : seriesColor(s),
            shape: "line",
            className: s.muted ? "text-base-content/50" : "",
          })),
          ...(showOptimum ? [{ label: "optimum", shape: "dash", className: "text-base-content/70" }] : []),
        ]
      : [];

  const current = curve.series.map((s) => s.values[index]);

  return (
    <div className="flex h-full flex-col">
      <Legend items={legend} className="mb-2 min-h-5" />
      <PlotBox
        fill
        minHeight={220}
        label={`${curve.yLabel} over ${curve.over}`}
        className="flex-1"
        overlay={({ width }) =>
          hover ? (
            <Tooltip x={hover.px} y={hover.py} width={width}>
              <TipRows
                rows={[
                  [curve.xLabel, formatValue(curve.xs[hover.i])],
                  ...curve.series.map((s) => [s.label, formatValue(s.values[hover.i])]),
                ]}
              />
            </Tooltip>
          ) : null
        }
      >
        {({ width, height }) => {
          const area = { left: MARGIN.left, right: width - MARGIN.right, top: MARGIN.top, bottom: height - MARGIN.bottom };
          const x = linear(domain.x, [area.left, area.right]);
          const y = curve.logY
            ? logarithmic(domain.y, [area.bottom, area.top])
            : linear(domain.y, [area.bottom, area.top]);
          const yTicks = curve.logY ? logTicks(domain.y[0], domain.y[1]) : ticks(domain.y[0], domain.y[1], Math.max(3, Math.floor((area.bottom - area.top) / 45)));
          const xTicks = ticks(domain.x[0], domain.x[1], Math.max(2, Math.floor((area.right - area.left) / 80)));
          const cx = x(curve.xs[index]);

          const frameAt = (px) => {
            const v = x.invert(px);
            let best = 0;
            for (let i = 1; i < curve.xs.length; i++) {
              if (Math.abs(curve.xs[i] - v) < Math.abs(curve.xs[best] - v)) best = i;
            }
            return best;
          };
          const move = (event) => {
            const p = pointerIn(event);
            const i = frameAt(p.x);
            const main = curve.series[0].values[i];
            setHover({ i, px: x(curve.xs[i]), py: main === null || main === undefined ? p.y : y(main) });
            if (event.buttons === 1) onSeek(i);
          };

          return (
            <>
              <Axes x={x} y={y} area={area} xTicks={xTicks} yTicks={yTicks} xLabel={curve.xLabel} formatY={curve.logY ? logTickFormat(yTicks) : undefined} />
              {curve.optimum !== null && !(curve.logY && curve.optimum <= 0)
                ? (() => {
                    const oy = y(curve.optimum);
                    // below the line when the curve comes down to it, unless that's the x axis
                    const below = curve.minimize ? oy < area.bottom - 16 : oy < area.top + 14;
                    return (
                      <g aria-hidden>
                        <line x1={area.left} x2={area.right} y1={oy} y2={oy} className="stroke-base-content/55" strokeWidth={1} strokeDasharray="4 3" />
                        <text x={area.left + 4} y={below ? oy + 12 : oy - 5} className="fill-base-content/65 text-[10px] tabular-nums">
                          optimum {formatValue(curve.optimum)}
                        </text>
                      </g>
                    );
                  })()
                : null}
              {curve.series.map((s) => {
                const color = seriesColor(s);
                const cls = s.muted ? "text-base-content/45" : "";
                return (
                  <g key={s.key} className={cls} aria-hidden>
                    <path
                      d={pathOf(curve.xs, s.values, x, y, curve.logY)}
                      fill="none"
                      stroke={color}
                      strokeOpacity={0.22}
                      strokeWidth={s.muted ? 1.5 : 2}
                      strokeLinejoin="round"
                      strokeLinecap="round"
                    />
                    <path
                      d={pathOf(curve.xs, s.values, x, y, curve.logY, index)}
                      fill="none"
                      stroke={color}
                      strokeWidth={s.muted ? 1.5 : 2}
                      strokeLinejoin="round"
                      strokeLinecap="round"
                    />
                  </g>
                );
              })}
              <line x1={cx} x2={cx} y1={area.top} y2={area.bottom} className="stroke-base-content/30" strokeWidth={1} aria-hidden />
              {curve.series.map((s, i) => {
                const v = current[i];
                if (v === null || v === undefined) return null;
                return (
                  <circle
                    key={s.key}
                    cx={cx}
                    cy={y(v)}
                    r={s.muted ? 3 : 4.5}
                    fill={s.muted ? "currentColor" : seriesColor(s)}
                    className={s.muted ? "text-base-content/50" : ""}
                    stroke="var(--color-base-100)"
                    strokeWidth={2}
                    aria-hidden
                  />
                );
              })}
              {hover ? (
                <line x1={hover.px} x2={hover.px} y1={area.top} y2={area.bottom} className="stroke-base-content/40" strokeWidth={1} strokeDasharray="2 2" aria-hidden />
              ) : null}
              <rect
                x={area.left}
                y={area.top}
                width={Math.max(0, area.right - area.left)}
                height={Math.max(0, area.bottom - area.top)}
                fill="transparent"
                className="cursor-crosshair"
                style={{ touchAction: "pan-y" }}
                onPointerMove={move}
                onPointerDown={(event) => {
                  const i = frameAt(pointerIn(event).x);
                  onSeek(i);
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
