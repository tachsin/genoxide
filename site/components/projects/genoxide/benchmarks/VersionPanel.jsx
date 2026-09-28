"use client";

import { useMemo, useState } from "react";
import { linear, logarithmic, logTicks, ticks } from "../player/chart-kit";
import { Axes, Legend, Marker, nearest, PlotBox, pointerIn, TipRows, Tooltip } from "../player/chart-parts";
import { barNotes, exactValue, QUANTITY_NAMES, tickFormat } from "./format";

/**
 * One panel of genoxide's versions chart (rule 10 of the benchmark rules):
 * the versions on the x axis, in the harness's order (`versions`), and a line
 * per method through the instructions of its run in each version. Each point
 * is one of the panel's `bars` of charts.json (a method and a version), with
 * its evaluations and whether it reached the target: hollow when it didn't.
 * The value axis is logarithmic where the harness's chart is (`log`).
 *
 * Hover a point, or focus the panel and use the arrow keys, for its numbers
 * and the change from the method's previous version.
 */

const HEIGHT = 200;
const TOP = 10;
const BOTTOM = 24;
const LEFT = 46;
const RIGHT = 12;
const exact = new Intl.NumberFormat("en");

function change(value, previous) {
  const ratio = value / previous - 1;
  if (!Number.isFinite(ratio)) return "–";
  const text = `${Math.abs(ratio * 100).toFixed(Math.abs(ratio) < 0.1 ? 1 : 0)}%`;
  return ratio === 0 ? "the same" : `${ratio < 0 ? "−" : "+"}${text}`;
}

export default function VersionPanel({ panel, quantity, versions, label }) {
  const [active, setActive] = useState(null);

  // the versions of the chart, or of this panel's points
  const order = useMemo(
    () => (Array.isArray(versions) && versions.length ? versions : [...new Set(panel.bars.map((bar) => bar.version))]),
    [versions, panel.bars],
  );
  // a series per method, and every point in reading order: by version, then by method
  const { series, points } = useMemo(() => {
    const methods = new Map();
    const all = [];
    panel.bars.forEach((bar, index) => {
      const at = order.indexOf(bar.version);
      if (at === -1 || typeof bar.value !== "number") return;
      if (!methods.has(bar.solver)) {
        methods.set(bar.solver, { solver: bar.solver, label: bar.method ?? bar.solver, color: bar.color ?? "#888888", points: [] });
      }
      const method = methods.get(bar.solver);
      const point = { ...bar, index, at, previous: method.points.at(-1) ?? null };
      method.points.push(point);
      all.push(point);
    });
    all.sort((a, b) => a.at - b.at || a.index - b.index);
    return { series: [...methods.values()], points: all };
  }, [panel.bars, order]);

  const domain = useMemo(() => {
    const values = points.map((point) => point.value).filter((value) => value > 0);
    if (!values.length) return [0, 1];
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    if (panel.log) return [lo / 1.8, hi * 1.8];
    // from 0, as the harness draws it: a change of a fraction of a percent stays as small as it is
    return [0, hi * 1.25];
  }, [points, panel.log]);

  const format = tickFormat(quantity);
  const activePoint = active !== null ? points.find((point) => point.index === active) : null;
  const summary = `${panel.title ?? label}: ${series.length} methods over ${order.length} versions, lower is better. ${series
    .map((method) => {
      const last = method.points.at(-1);
      return `${method.label} ${last.version}: ${last.text}`;
    })
    .join(", ")}`;

  const geometry = (width, height) => {
    const area = { left: LEFT, right: Math.max(LEFT + 20, width - RIGHT), top: TOP, bottom: height - BOTTOM };
    const step = (area.right - area.left) / Math.max(1, order.length);
    const x = (at) => area.left + step * (at + 0.5);
    const y = (panel.log ? logarithmic : linear)(domain, [area.bottom, area.top]);
    return { area, x, y };
  };

  const move = (step) => {
    const at = points.findIndex((point) => point.index === active);
    const next = at === -1 ? (step > 0 ? 0 : points.length - 1) : Math.min(points.length - 1, Math.max(0, at + step));
    setActive(points[next]?.index ?? null);
  };

  return (
    <div>
      <div
        // biome-ignore lint/a11y/noNoninteractiveTabindex: the arrow keys read the points one by one
        tabIndex={0}
        role="group"
        aria-label={`${panel.title ?? label}: use the arrow keys to read each point`}
        className="rounded-md outline-none focus-visible:ring-2 focus-visible:ring-primary/60"
        onKeyDown={(event) => {
          if (["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp"].includes(event.key)) {
            event.preventDefault();
            move(event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : -1);
          } else if (event.key === "Escape") {
            setActive(null);
          }
        }}
        onBlur={() => setActive(null)}
      >
        <PlotBox
          height={HEIGHT}
          minHeight={HEIGHT}
          label={summary}
          overlay={({ width, height }) => {
            if (!activePoint) return null;
            const { x, y } = geometry(width, height);
            const previous = activePoint.previous;
            return (
              <Tooltip x={x(activePoint.at)} y={y(activePoint.value)} width={width}>
                <p className="mb-1 font-semibold">
                  {activePoint.method ?? activePoint.solver} · genoxide {activePoint.version}
                </p>
                <TipRows
                  rows={[
                    [QUANTITY_NAMES[quantity] ?? quantity, exact.format(activePoint.value)],
                    ...(previous ? [[`vs ${previous.version}`, change(activePoint.value, previous.value)]] : []),
                  ]}
                />
                {barNotes(activePoint).map((note) => (
                  <p key={note} className="mt-1 text-base-content/70">
                    {note}
                  </p>
                ))}
              </Tooltip>
            );
          }}
          svgProps={{
            onPointerMove: (event) => {
              const pointer = pointerIn(event);
              const box = event.currentTarget.getBoundingClientRect();
              const { x, y } = geometry(box.width, box.height);
              const hit = nearest(
                points.map((point) => ({ px: x(point.at), py: y(point.value), index: point.index })),
                pointer.x,
                pointer.y,
                22,
              );
              setActive(hit ? hit.index : null);
            },
            onPointerLeave: () => setActive(null),
          }}
        >
          {({ width, height }) => {
            const { area, x, y } = geometry(width, height);
            const yTicks = panel.log
              ? logTicks(domain[0], domain[1])
              : ticks(domain[0], domain[1], Math.max(2, Math.floor((area.bottom - area.top) / 40)));
            return (
              <>
                <Axes
                  x={x}
                  y={y}
                  area={area}
                  xTicks={order.map((_, at) => at)}
                  yTicks={yTicks}
                  formatX={(at) => order[at] ?? ""}
                  formatY={format}
                />
                <g aria-hidden>
                  {series.map((method) => (
                    <polyline
                      key={method.solver}
                      points={method.points.map((point) => `${x(point.at)},${y(point.value)}`).join(" ")}
                      fill="none"
                      stroke={method.color}
                      strokeWidth={1.8}
                      strokeLinejoin="round"
                    />
                  ))}
                  {points.map((point) => {
                    const on = activePoint === point;
                    const hollow = point.reached === 0;
                    return (
                      <Marker
                        key={`${point.solver}/${point.version}`}
                        x={x(point.at)}
                        y={y(point.value)}
                        r={on ? 5.5 : 4}
                        color={point.color ?? "#888888"}
                        hollow={hollow}
                        className={hollow ? "fill-base-100" : ""}
                      />
                    );
                  })}
                </g>
              </>
            );
          }}
        </PlotBox>
      </div>
      <Legend
        className="mt-1"
        items={[
          ...series.map((method) => ({ label: method.label, color: method.color, shape: "line" })),
          ...(points.some((point) => point.reached === 0) ? [{ label: "target not reached", shape: "ring", className: "text-base-content/60" }] : []),
        ]}
      />
      <p className="sr-only" aria-live="polite">
        {activePoint
          ? [
              `${activePoint.method ?? activePoint.solver}, genoxide ${activePoint.version}: ${exactValue(activePoint.value, quantity)} instructions`,
              ...barNotes(activePoint),
            ].join(". ")
          : ""}
      </p>
    </div>
  );
}
