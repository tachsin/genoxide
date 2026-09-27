"use client";

import { useMemo, useState } from "react";
import { categorical, extent, formatValue, linear, logTickFormat, logTicks, logarithmic, nice, ticks } from "../chart-kit";
import { Axes, Legend, PlotBox, TipRows, Tooltip, pointerIn } from "../chart-parts";

/** Splits "sphere/CMA-ES" into its panel and its line; a name without "/" is a line of one panel. */
function split(name) {
  const at = name.indexOf("/");
  return at === -1 ? ["", name] : [name.slice(0, at), name.slice(at + 1)];
}

const MARGIN = { left: 44, right: 10, top: 24, bottom: 34 };

function Panel({ title, lines, xs, index, logY, colors, xLabel }) {
  const [hover, setHover] = useState(null);
  const domain = useMemo(() => {
    const all = lines.flatMap((l) => l.values);
    if (logY) {
      const e = extent(all.filter((v) => v > 0)) ?? [1e-6, 1];
      // errors clamped at 0 (the minimum reached) are drawn on a floor a decade under the smallest
      const floorNeeded = all.some((v) => typeof v === "number" && v <= 0);
      const lo = 10 ** (Math.floor(Math.log10(e[0])) - (floorNeeded ? 1 : 0));
      const hi = 10 ** Math.ceil(Math.log10(e[1]));
      return [lo, hi === lo ? lo * 10 : hi];
    }
    const e = extent(all) ?? [0, 1];
    return nice(e[0], e[1], 4);
  }, [lines, logY]);
  // up to the panel's last value: a run that ended early (a smaller budget) fills its own panel
  const end = useMemo(() => {
    let last = 0;
    for (let i = 0; i < xs.length; i++) if (lines.some((l) => typeof l.values[i] === "number")) last = i;
    return last || xs.length - 1;
  }, [lines, xs]);
  const xDomain = extent(xs.slice(0, end + 1)) ?? [0, 1];
  const ended = index > end;

  return (
    <figure className="min-w-0">
      {title ? <figcaption className="mb-1 font-medium text-base-content/75 text-xs">{title}</figcaption> : null}
      <PlotBox
        aspect={0.72}
        minHeight={150}
        maxHeight={260}
        label={`${title}: ${lines.map((l) => `${l.label} ${formatValue(l.values[index])}`).join(", ")}`}
        overlay={({ width }) =>
          hover !== null ? (
            <Tooltip x={hover.px} y={hover.py} width={width}>
              <TipRows rows={[[xLabel, formatValue(xs[hover.i])], ...lines.map((l) => [l.label, formatValue(l.values[hover.i])])]} />
            </Tooltip>
          ) : null
        }
      >
        {({ width, height }) => {
          const area = { left: MARGIN.left, right: width - MARGIN.right, top: MARGIN.top, bottom: height - MARGIN.bottom };
          const x = linear(xDomain, [area.left, area.right]);
          const y = logY ? logarithmic(domain, [area.bottom, area.top]) : linear(domain, [area.bottom, area.top]);
          const yTicks = logY ? logTicks(domain[0], domain[1]).filter((_, k, a) => a.length <= 4 || k % 2 === 0) : ticks(domain[0], domain[1], 3);
          const path = (values, upTo) => {
            let d = "";
            let pen = false;
            for (let i = 0; i <= Math.min(upTo, values.length - 1); i++) {
              const v = values[i];
              if (typeof v !== "number") {
                pen = false;
                continue;
              }
              d += `${pen ? "L" : "M"}${x(xs[i]).toFixed(1)} ${y(v).toFixed(1)}`;
              pen = true;
            }
            return d;
          };
          const cx = x(xs[index]);
          return (
            <>
              <Axes x={x} y={y} area={area} xTicks={ticks(xDomain[0], xDomain[1], Math.max(2, Math.floor((area.right - area.left) / 70)))} yTicks={yTicks} formatY={logY ? logTickFormat(yTicks) : undefined} />
              {lines.map((l) => (
                <g key={l.label}>
                  <path d={path(l.values, Infinity)} fill="none" stroke={colors[l.slot]} strokeOpacity={0.2} strokeWidth={1.5} />
                  <path d={path(l.values, index)} fill="none" stroke={colors[l.slot]} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
                </g>
              ))}
              {ended ? null : <line x1={cx} x2={cx} y1={area.top} y2={area.bottom} className="stroke-base-content/30" strokeWidth={1} />}
              {lines.map((l) => {
                const v = l.values[index];
                if (typeof v !== "number") return null;
                return <circle key={l.label} cx={cx} cy={y(v)} r={3.5} fill={colors[l.slot]} stroke="var(--color-base-100)" strokeWidth={1.5} />;
              })}
              {hover !== null ? <line x1={hover.px} x2={hover.px} y1={area.top} y2={area.bottom} className="stroke-base-content/40" strokeDasharray="2 2" /> : null}
              <rect
                x={area.left}
                y={area.top}
                width={Math.max(0, area.right - area.left)}
                height={Math.max(0, area.bottom - area.top)}
                fill="transparent"
                onPointerMove={(event) => {
                  const p = pointerIn(event);
                  const v = x.invert(p.x);
                  let i = 0;
                  for (let k = 1; k <= end; k++) if (Math.abs(xs[k] - v) < Math.abs(xs[i] - v)) i = k;
                  setHover({ i, px: x(xs[i]), py: p.y });
                }}
                onPointerLeave={() => setHover(null)}
              />
            </>
          );
        }}
      </PlotBox>
    </figure>
  );
}

/** `multi-curve`: small multiples, one per function, a line per algorithm, drawn up to the current frame. */
export default function MultiCurvePlot({ trace, index, dark }) {
  const colors = categorical(dark);
  const xKey = trace.x_label === "evaluations" ? "evaluations" : "generation";
  const xs = useMemo(() => trace.frames.map((f) => f[xKey] ?? f.generation), [trace, xKey]);

  const { panels, algorithms } = useMemo(() => {
    const names = trace.problem?.series ?? Object.keys(trace.frames[0]?.state?.values ?? {});
    const algs = [];
    const byPanel = new Map();
    for (const name of names) {
      const [panel, line] = split(name);
      if (!algs.includes(line)) algs.push(line);
      if (!byPanel.has(panel)) byPanel.set(panel, []);
      byPanel.get(panel).push({
        label: line,
        slot: algs.indexOf(line) % colors.length,
        values: trace.frames.map((f) => f.state?.values?.[name] ?? null),
      });
    }
    return { panels: [...byPanel.entries()], algorithms: algs };
  }, [trace, colors.length]);

  return (
    <div>
      <Legend className="mb-2" items={algorithms.map((a, k) => ({ label: a, color: colors[k % colors.length], shape: "line" }))} />
      <p className="mb-2 text-base-content/55 text-xs">
        {trace.y_label ?? "value"} over {xKey === "evaluations" ? "evaluations" : "generations"}
        {trace.log_y ? ", log scale" : ""}
      </p>
      <div className="@container">
      <div className={`grid gap-x-4 gap-y-3 ${panels.length > 1 ? "grid-cols-1 @md:grid-cols-2 @3xl:grid-cols-3 @5xl:grid-cols-4" : ""}`}>
        {panels.map(([title, lines]) => (
          <Panel key={title || "all"} title={title} lines={lines} xs={xs} index={index} logY={!!trace.log_y} colors={colors} xLabel={xKey} />
        ))}
      </div>
      </div>
    </div>
  );
}
