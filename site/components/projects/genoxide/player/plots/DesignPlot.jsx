"use client";

import { CircleCheck, CircleDot, CircleX } from "lucide-react";
import { useMemo } from "react";
import { STATUS, categorical, formatTick, formatValue, linear } from "../chart-kit";
import { Legend, PlotBox } from "../chart-parts";

const ROW = 46;
const TRAIL = 12;
// |g| at or below this counts as on the boundary: an active constraint.
const ACTIVE = 1e-6;

/**
 * A constraint's state. `signed`: the trace records g itself (g <= 0 is
 * satisfied, g near 0 is active); otherwise it records the violation
 * max(0, g), and 0 only says satisfied.
 */
function status(g, signed) {
  if (typeof g !== "number") return { label: "–", tone: "muted" };
  if (g > ACTIVE) return { label: "violated", tone: "critical" };
  if (signed && g >= -ACTIVE) return { label: "active", tone: "active" };
  return { label: "satisfied", tone: "good" };
}

/**
 * `design`: each design variable on its range, the best design's value
 * marked (the last frames' values faint behind it), and every constraint's
 * state: violated, active (on its boundary) or satisfied.
 */
export default function DesignPlot({ trace, frame, index, dark }) {
  const variables = trace.problem?.variables ?? [];
  const constraints = trace.problem?.constraints ?? [];
  const best = frame.state?.best ?? [];
  const violations = frame.state?.violations ?? [];
  const color = categorical(dark)[0];
  const trail = trace.frames.slice(Math.max(0, index - TRAIL), index);
  const signed = useMemo(() => trace.frames.some((f) => (f.state?.violations ?? []).some((g) => g < 0)), [trace]);

  return (
    <div>
      <Legend
        className="mb-2"
        items={[
          { label: "best design", color, shape: "dot" },
          { label: `the ${TRAIL} frames before`, shape: "dot", className: "text-base-content/25" },
        ]}
      />
      <PlotBox
        height={variables.length * ROW + 8}
        label={`The best design: ${variables.map((v, i) => `${v.name} ${formatValue(best[i])}${v.unit ? ` ${v.unit}` : ""}`).join(", ")}`}
      >
        {({ width }) => {
          const left = Math.min(120, width * 0.3);
          const right = width - 12;
          return variables.map((v, i) => {
            const [lo, hi] = v.bounds ?? [0, 1];
            const x = linear([lo, hi], [left, right]);
            const y = 8 + i * ROW + 16;
            const clamp = (value) => x(Math.min(hi, Math.max(lo, value)));
            const value = best[i];
            return (
              <g key={v.name}>
                <text x={0} y={y} dy="0.32em" className="fill-base-content text-[12px] font-medium">
                  {v.name}
                </text>
                <text x={0} y={y + 16} dy="0.32em" className="fill-base-content/70 text-[11px] tabular-nums">
                  {formatValue(value)}
                  {v.unit ? <tspan className="fill-base-content/55"> {v.unit}</tspan> : null}
                </text>
                <line x1={left} x2={right} y1={y} y2={y} className="stroke-base-content/20" strokeWidth={4} strokeLinecap="round" />
                <text x={left} y={y + 16} dy="0.32em" className="fill-base-content/50 text-[10px] tabular-nums">
                  {formatTick(lo)}
                </text>
                <text x={right} y={y + 16} dy="0.32em" textAnchor="end" className="fill-base-content/50 text-[10px] tabular-nums">
                  {formatTick(hi)}
                </text>
                {trail.map((f, k) =>
                  typeof f.state?.best?.[i] === "number" ? (
                    <circle key={k} cx={clamp(f.state.best[i])} cy={y} r={3} className="fill-base-content" opacity={0.06 + (0.2 * (k + 1)) / trail.length} />
                  ) : null,
                )}
                {typeof value === "number" ? (
                  <circle cx={clamp(value)} cy={y} r={6} fill={color} stroke="var(--color-base-100)" strokeWidth={2} />
                ) : null}
              </g>
            );
          });
        }}
      </PlotBox>

      {constraints.length ? (
        <div className="mt-4">
          <h4 className="mb-2 font-medium text-base-content/75 text-xs">Constraints</h4>
          <ul className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {constraints.map((name, i) => {
              const g = violations[i];
              const s = status(g, signed);
              const Icon = s.tone === "critical" ? CircleX : s.tone === "active" ? CircleDot : CircleCheck;
              const iconColor = s.tone === "critical" ? STATUS.critical : s.tone === "good" ? STATUS.good : s.tone === "active" ? color : undefined;
              return (
                <li key={name} className="flex items-center gap-2 rounded-lg border border-base-content/10 px-2.5 py-1.5 text-xs">
                  <Icon size={15} aria-hidden style={{ color: iconColor }} className="flex-none" />
                  <span className="min-w-0">
                    <span className="block font-medium">{name}</span>
                    <span className="block text-base-content/60 tabular-nums">
                      {s.label}
                      {s.tone === "critical" || (s.tone === "good" && signed) ? ` · ${formatValue(g)}` : ""}
                    </span>
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
