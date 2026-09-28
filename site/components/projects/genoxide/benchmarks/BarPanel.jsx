"use client";

import { useId, useMemo, useState } from "react";
import { linear, logarithmic, logTicks, ticks } from "../player/chart-kit";
import { Axes, PlotBox, TipRows, Tooltip } from "../player/chart-parts";
import { barNotes, exactValue, QUANTITY_NAMES, tickFormat } from "./format";
import { sameRun, useHighlight } from "./Highlight";

/**
 * One panel of a benchmark chart: a horizontal bar per library (the
 * scenario's one method), in the harness's order (best first; missing values,
 * then the runs the time cap stopped, last), on a log axis, or a linear one
 * when the panel says so. Every number comes from charts.json: a bar's label
 * is its `text`, as in the SVG chart.
 *
 * Hover a bar, or focus the panel and use the arrow keys, for its numbers.
 * The highlighted library (the legend's) dims the others.
 *
 * A bar of a method in a scenario (a `solver`, in a panel whose `key` is the
 * scenario's) selects that run on a click, or on Enter when the arrow keys are
 * on it: the details panel below the charts shows its runs, output and code.
 */

const ROW = 18;
const BAR = 12;
const TOP = 4;
const BOTTOM = 24;
const CHAR = 5.9; // average width of a character of the 11px labels
const GENOXIDE = new Set(["genoxide", "genoxide_python"]);

function fit(text, room) {
  const chars = Math.max(3, Math.floor(room / CHAR));
  return text.length <= chars ? text : `${text.slice(0, chars - 1)}…`;
}

export default function BarPanel({ panel, quantity, libraries, limit = Infinity, label, room }) {
  const { focus, details, run, select } = useHighlight();
  const [expanded, setExpanded] = useState(false);
  const [active, setActive] = useState(null);
  const patterns = useId().replace(/[^a-zA-Z0-9_-]/g, "");

  const all = panel.bars;
  const collapsible = all.length > limit + 2;
  // collapsed, the first `limit` bars, and every bar of the highlighted library
  const bars = useMemo(
    () =>
      all
        .map((bar, index) => ({ ...bar, index }))
        .filter((bar) => !collapsible || expanded || bar.index < limit || bar.library === focus),
    [all, collapsible, expanded, limit, focus],
  );

  // the axis covers every bar, shown or not, so it doesn't move when the panel expands
  const domain = useMemo(() => {
    const values = all.map((b) => b.value).filter((v) => typeof v === "number");
    const positive = values.filter((v) => v > 0);
    if (panel.log && positive.length) {
      const lo = 10 ** Math.floor(Math.log10(Math.min(...positive) / 2));
      return [lo, Math.max(...positive) * 1.05];
    }
    // a fixed end, if the panel gives one
    if (typeof panel.axis_to === "number") return [0, panel.axis_to];
    const hi = values.length ? Math.max(...values) : 1;
    return [0, hi * 1.06 || 1];
  }, [all, panel.log, panel.axis_to]);

  const format = tickFormat(quantity);
  const height = TOP + bars.length * ROW + BOTTOM;
  const summary = `${panel.title ?? label}: ${all.length} bars, ${panel.better} is better. ${all
    .slice(0, 3)
    .filter((b) => b.value !== null)
    .map((b) => `${b.label} ${b.text}`)
    .join(", ")}${all.length > 3 ? ", …" : ""}`;

  const activeBar = active !== null ? bars.find((b) => b.index === active) : null;
  // the run a bar selects: its library and method in this panel's scenario
  const runOf = (bar) =>
    details && bar?.solver && bar.library && panel.key
      ? { scenario: panel.key, library: bar.library, solver: bar.solver }
      : null;
  const selectable = all.some((bar) => runOf(bar));
  const move = (step) => {
    const at = bars.findIndex((b) => b.index === active);
    const next = at === -1 ? (step > 0 ? 0 : bars.length - 1) : Math.min(bars.length - 1, Math.max(0, at + step));
    setActive(bars[next]?.index ?? null);
  };

  return (
    <div>
      <div
        // biome-ignore lint/a11y/noNoninteractiveTabindex: the arrow keys read the bars one by one
        tabIndex={0}
        role="group"
        aria-label={`${panel.title ?? label}: use the up and down arrow keys to read each bar${
          selectable ? ", and Enter to show its runs, output and code below the charts" : ""
        }`}
        className="rounded-md outline-none focus-visible:ring-2 focus-visible:ring-primary/60"
        onKeyDown={(event) => {
          if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            move(event.key === "ArrowDown" ? 1 : -1);
          } else if (event.key === "Home" || event.key === "End") {
            event.preventDefault();
            setActive(bars[event.key === "Home" ? 0 : bars.length - 1]?.index ?? null);
          } else if (event.key === "Escape") {
            setActive(null);
          } else if ((event.key === "Enter" || event.key === " ") && runOf(activeBar)) {
            event.preventDefault();
            select(runOf(activeBar));
          }
        }}
        onBlur={() => setActive(null)}
      >
        <PlotBox
          height={height}
          minHeight={height}
          label={summary}
          overlay={({ width, height: box }) => {
            if (!activeBar) return null;
            const geometry = layout(width, box, domain, panel.log, room);
            const row = bars.indexOf(activeBar);
            const end = activeBar.value === null ? geometry.area.left : geometry.x(Math.max(activeBar.value, domain[0]));
            const y = TOP + row * ROW + ROW / 2;
            return (
              <Tooltip x={Math.min(end, width - 40)} y={y} width={width}>
                <p className="mb-1 font-semibold">{activeBar.label}</p>
                <TipRows
                  rows={[
                    [QUANTITY_NAMES[quantity] ?? quantity, exactValue(activeBar.value, quantity)],
                    ...(libraries[activeBar.library]
                      ? [["version", `${libraries[activeBar.library].version} (${libraries[activeBar.library].language})`]]
                      : []),
                    ...(typeof activeBar.runs === "number" ? [["runs", String(activeBar.runs)]] : []),
                  ]}
                />
                {barNotes(activeBar).map((note) => (
                  <p key={note} className="mt-1 text-base-content/70">
                    {note}
                  </p>
                ))}
              </Tooltip>
            );
          }}
        >
          {({ width, height: box }) => {
            const { area, x } = layout(width, box, domain, panel.log, room);
            const xTicks = panel.log
              ? thin(logTicks(domain[0], domain[1]), area)
              : ticks(domain[0], domain[1], Math.max(2, Math.floor((area.right - area.left) / 70)));
            const firstCapped = bars.findIndex((b) => b.ended_on_cap);
            return (
              <>
                <defs>
                  <pattern id={`${patterns}-cap`} width={4} height={4} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                    <line x1={0} x2={0} y1={0} y2={4} stroke="var(--color-base-100)" strokeWidth={1.2} />
                    <line x1={0} x2={4} y1={0} y2={0} stroke="var(--color-base-100)" strokeWidth={1.2} />
                  </pattern>
                </defs>
                <Axes x={x} y={() => 0} area={area} xTicks={xTicks} formatX={format} xGrid />
                {typeof panel.target === "number" && panel.target >= domain[0] ? (
                  <line x1={x(panel.target)} x2={x(panel.target)} y1={area.top} y2={area.bottom} className="stroke-base-content/60" strokeDasharray="4 3" aria-hidden />
                ) : null}
                {firstCapped > 0 ? (
                  <line
                    x1={4}
                    x2={width}
                    y1={TOP + firstCapped * ROW}
                    y2={TOP + firstCapped * ROW}
                    className="stroke-base-content/35"
                    strokeDasharray="1 3"
                    aria-hidden
                  />
                ) : null}
                <g aria-hidden>
                  {bars.map((bar, row) => {
                    const y = TOP + row * ROW;
                    const color = libraries[bar.library]?.color ?? "#888888";
                    const dim = focus && focus !== bar.library;
                    const isActive = activeBar === bar;
                    const barRun = runOf(bar);
                    const isSelected = sameRun(barRun, run);
                    const labelText = fit(bar.label, area.left - 10);
                    const labelClass = `${GENOXIDE.has(bar.library) ? "font-semibold fill-base-content" : "fill-base-content/75"} text-[11px]`;
                    let mark;
                    let end = area.left;
                    if (bar.value === null) {
                      const cx = area.left + 7;
                      mark = (
                        <path d={`M${cx - 3.5} ${y + ROW / 2 - 3.5} l7 7 m0 -7 l-7 7`} stroke={color} strokeWidth={1.8} strokeLinecap="round" />
                      );
                      end = cx + 4;
                    } else {
                      const stub = bar.value <= domain[0];
                      end = stub ? area.left + Math.max(4, (area.right - area.left) * 0.04) : Math.max(area.left + 2, x(bar.value));
                      const hatch = bar.ended_on_cap ? "cap" : null;
                      mark = (
                        <g opacity={hatch ? 0.6 : 1}>
                          <rect x={area.left} y={y + (ROW - BAR) / 2} width={end - area.left} height={BAR} rx={2} fill={color} />
                          {hatch ? (
                            <rect x={area.left} y={y + (ROW - BAR) / 2} width={end - area.left} height={BAR} rx={2} fill={`url(#${patterns}-${hatch})`} />
                          ) : null}
                        </g>
                      );
                    }
                    const valueText = bar.value === null ? bar.missing : bar.text;
                    const noteText = bar.value === null ? "" : bar.note ?? "";
                    const room = width - end - 6;
                    const withNote = noteText && (valueText.length + noteText.length + 3) * CHAR <= room;
                    return (
                      <g key={`${bar.library}/${bar.solver}/${bar.index}`} opacity={dim ? 0.2 : 1} className="transition-opacity duration-150 motion-reduce:transition-none">
                        {isActive ? <rect x={0} y={y} width={width} height={ROW} rx={3} className="fill-base-content/[0.07]" /> : null}
                        {isSelected ? (
                          <rect x={0.75} y={y + 0.75} width={width - 1.5} height={ROW - 1.5} rx={3} fill="none" className="stroke-primary" strokeWidth={1.5} />
                        ) : null}
                        <text x={area.left - 6} y={y + ROW / 2} dy="0.34em" textAnchor="end" className={labelClass}>
                          {labelText}
                        </text>
                        {mark}
                        <text x={end + 4} y={y + ROW / 2} dy="0.34em" className="fill-base-content/85 text-[10.5px] tabular-nums">
                          {withNote ? `${valueText}  ` : valueText}
                          {withNote ? <tspan className="fill-base-content/55">{noteText}</tspan> : noteText ? <tspan className="fill-base-content/55"> *</tspan> : null}
                        </text>
                        <rect
                          x={0}
                          y={y}
                          width={width}
                          height={ROW}
                          fill="transparent"
                          className={barRun ? "cursor-pointer" : undefined}
                          onClick={barRun ? () => select(barRun) : undefined}
                          onPointerEnter={() => setActive(bar.index)}
                          onPointerLeave={() => setActive((current) => (current === bar.index ? null : current))}
                        />
                      </g>
                    );
                  })}
                </g>
              </>
            );
          }}
        </PlotBox>
      </div>
      <p className="sr-only" aria-live="polite">
        {activeBar
          ? [
              `${activeBar.label}: ${activeBar.value === null ? "no value" : `${QUANTITY_NAMES[quantity] ?? quantity} ${exactValue(activeBar.value, quantity)}`}`,
              ...barNotes(activeBar),
              ...(sameRun(runOf(activeBar), run) ? ["its runs are shown below the charts"] : []),
            ].join(". ")
          : ""}
      </p>
      {collapsible ? (
        <button
          type="button"
          onClick={() => setExpanded((e) => !e)}
          aria-expanded={expanded}
          className="mt-1 rounded text-primary text-xs underline-offset-2 hover:underline"
        >
          {expanded ? `Show the first ${limit}` : `Show all ${all.length}`}
        </button>
      ) : null}
    </div>
  );
}

/**
 * The plot area and value scale of a panel: labels on the left, room for the
 * values on the right (`room` px at most, for longer ones, up to 30% of the width).
 */
function layout(width, height, domain, log, room) {
  const left = Math.round(Math.min(180, Math.max(96, width * 0.4)));
  const right = width - (room ? Math.min(room, width * 0.3) : Math.min(64, Math.max(52, width * 0.12)));
  const area = { left, right: Math.max(left + 20, right), top: TOP, bottom: height - BOTTOM };
  return { area, x: (log ? logarithmic : linear)(domain, [area.left, area.right]) };
}

/** Fewer ticks when they would crowd: every other one, and so on. */
function thin(values, area) {
  const most = Math.max(2, Math.floor((area.right - area.left) / 42));
  const every = Math.ceil(values.length / most);
  return values.filter((_, k) => k % every === 0);
}
