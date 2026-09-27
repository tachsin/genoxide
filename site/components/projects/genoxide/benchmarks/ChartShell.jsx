"use client";

import { ExternalLink, X } from "lucide-react";
import { useState } from "react";
import BarPanel from "./BarPanel";
import { barNotes, pointsText, QUANTITY_NAMES, tickFormat } from "./format";
import { useHighlight } from "./Highlight";

/**
 * What every benchmark chart has around its panels: a switch between its
 * views (time or evaluations, hypervolume or time), the legend of its
 * libraries, which highlights one across every chart of the page, the panels,
 * their numbers as tables, and a link to the harness's own chart.
 *
 * @param {object} props
 * @param {{ key: string, label: string, chart: object, href: string }[]} props.views  the charts it can show, from charts.json
 * @param {Record<string, { name: string, version: string, language: string, color: string }>} props.libraries
 * @param {number} [props.limit]  bars per panel before "show all"
 * @param {string} [props.grid]  the panels' grid classes, by the chart's container width
 * @param {number} [props.room]  px beside the bars for their labels, when they're long
 * @param {string} props.name  the chart's name, for its controls' labels
 */
export default function ChartShell({ views, libraries, limit, grid = "", name, room }) {
  const [current, setCurrent] = useState(0);
  const view = views[current] ?? views[0];
  const { chart } = view;
  const quantity = chart.quantity;

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <p className="text-base-content/65 text-xs">{chart.title}</p>
        {views.length > 1 ? (
          <div role="group" aria-label={`${name}: what the bars show`} className="join">
            {views.map((v, k) => (
              <button
                key={v.key}
                type="button"
                aria-pressed={k === current}
                onClick={() => setCurrent(k)}
                className={`join-item btn btn-xs ${k === current ? "btn-primary" : "btn-ghost border-base-content/15"}`}
              >
                {v.label}
              </button>
            ))}
          </div>
        ) : null}
      </div>

      <LibraryLegend ids={chart.libraries} libraries={libraries} name={name} />

      <div className="@container mt-4">
        <div className={`grid gap-x-8 gap-y-6 ${grid}`}>
          {chart.panels.map((panel) => (
            <figure key={`${view.key}/${panel.key}`} className="min-w-0">
              {panel.title ? (
                <figcaption className="mb-1.5">
                  <span className="font-medium text-base-content/85 text-sm">{panel.title}</span>
                  {panel.detail ? <span className="block text-base-content/55 text-xs">{panel.detail}</span> : null}
                </figcaption>
              ) : null}
              <BarPanel panel={panel} quantity={quantity} libraries={libraries} limit={limit} label={chart.title} room={room} />
              {panel.note ? <p className="mt-1 text-base-content/60 text-xs">{panel.note}</p> : null}
            </figure>
          ))}
        </div>
      </div>

      <KeyNote panels={chart.panels} />

      <details className="group mt-4 text-sm">
        <summary className="cursor-pointer text-base-content/70 hover:text-primary">The numbers as tables</summary>
        <div className="mt-3 space-y-5">
          {chart.panels.map((panel) => (
            <div key={`${view.key}/${panel.key}`} className="space-y-5">
              <NumbersTable panel={panel} quantity={quantity} fallbackTitle={chart.title} />
              <RatiosTable panel={panel} />
            </div>
          ))}
        </div>
      </details>

      <p className="mt-3 text-base-content/60 text-xs">
        <a
          href={view.href}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 underline decoration-base-content/25 underline-offset-2 hover:text-primary hover:decoration-primary"
        >
          The harness's chart
          <ExternalLink size={11} aria-hidden />
        </a>
        {` (${view.href.split("/").pop()}), with the same numbers`}
      </p>
    </div>
  );
}

/** The libraries of a chart, each a button: hover or focus to highlight it, click to keep it highlighted. */
function LibraryLegend({ ids, libraries, name }) {
  const { focus, pinned, pin, preview } = useHighlight();
  return (
    <div className="mt-3 flex flex-wrap items-start gap-x-1 gap-y-1">
      <ul className="contents" aria-label={`${name}: libraries; select one to highlight it in every chart`}>
        {ids.map((id) => {
          const library = libraries[id];
          if (!library) return null;
          const on = focus === id;
          return (
            <li key={id}>
              <button
                type="button"
                aria-pressed={pinned === id}
                onClick={() => pin(id)}
                onPointerEnter={() => preview(id)}
                onPointerLeave={() => preview(null)}
                onFocus={() => preview(id)}
                onBlur={() => preview(null)}
                className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs transition-colors motion-reduce:transition-none ${
                  on ? "border-base-content/40 bg-base-content/10" : "border-transparent hover:border-base-content/20"
                } ${focus && !on ? "text-base-content/45" : "text-base-content/80"}`}
              >
                <span aria-hidden className="size-2.5 flex-none rounded-sm" style={{ background: library.color }} />
                <span className={id.startsWith("genoxide") ? "font-semibold" : ""}>{library.name}</span>
                <span className="text-base-content/50">
                  {library.version} ({library.language})
                </span>
              </button>
            </li>
          );
        })}
      </ul>
      {pinned ? (
        <button
          type="button"
          onClick={() => pin(pinned)}
          className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-base-content/60 text-xs hover:text-primary"
        >
          <X size={12} aria-hidden />
          Clear the highlight
        </button>
      ) : null}
    </div>
  );
}

/** What the marks beyond plain bars mean, for those the chart has. */
function KeyNote({ panels }) {
  const bars = panels.flatMap((panel) => panel.bars);
  const parts = [
    bars.some((bar) => bar.value === null) ? "×: too few runs reached the target for a value" : null,
    bars.some((bar) => bar.note) ? "*: a note that doesn't fit beside the bar" : null,
    bars.some((bar) => bar.ended_on_cap) ? "cross-hatched, past a dotted line: the runs the time cap stopped" : null,
    bars.some((bar) => bar.below_axis) ? "hatched stub: a value below the axis's start" : null,
  ].filter(Boolean);
  if (!parts.length) return null;
  return (
    <p className="mt-4 text-base-content/60 text-xs">
      {parts.join(" · ")}. Every bar's numbers and notes are in its tooltip and in the tables.
    </p>
  );
}

const seconds = tickFormat("seconds");

/**
 * The points behind an overall score (bars with `ratios`): a row per
 * library, a column per scenario of the panel's `scenarios`; nothing for
 * other panels.
 */
function RatiosTable({ panel }) {
  const scenarios = panel.scenarios ?? [];
  const bars = panel.bars.filter((bar) => Array.isArray(bar.ratios));
  if (!scenarios.length || !bars.length) return null;
  return (
    <div className="max-w-full overflow-x-auto">
      <table className="w-full text-left text-xs">
        <caption className="mb-1 text-left font-medium text-base-content/80">
          Points per scenario
          <span className="font-normal text-base-content/55"> · 100: the fastest · 0: unsolved within the time cap · –: can't run it</span>
        </caption>
        <thead className="text-base-content/55">
          <tr>
            <th scope="col" className="py-1 pr-3 font-normal">
              Library
            </th>
            {scenarios.map((scenario) => (
              <th key={scenario.key} scope="col" className="min-w-[4.5rem] py-1 pr-3 text-right align-bottom font-normal">
                {scenario.title}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {bars.map((bar, k) => {
            const ratios = Object.fromEntries(bar.ratios.map((ratio) => [ratio.scenario, ratio]));
            return (
              <tr key={`${bar.library}/${k}`} className="border-base-content/10 border-t">
                <th scope="row" className="whitespace-nowrap py-1 pr-3 font-normal">
                  {bar.label}
                </th>
                {scenarios.map((scenario) => {
                  const ratio = ratios[scenario.key];
                  return (
                    <td
                      key={scenario.key}
                      title={ratio && !ratio.penalized ? `${ratio.method ?? ratio.solver}, ${seconds(ratio.time)}` : undefined}
                      className={`whitespace-nowrap py-1 pr-3 text-right tabular-nums ${ratio?.penalized ? "text-base-content/50" : ""}`}
                    >
                      {ratio ? pointsText(ratio.points) : "–"}
                      {ratio?.penalized ? <span className="block text-[10px]">unsolved</span> : null}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** A panel's bars as a table, for screen readers and anyone who wants the list. */
function NumbersTable({ panel, quantity, fallbackTitle }) {
  return (
    <div className="max-w-full overflow-x-auto">
      <table className="w-full min-w-[18rem] text-left text-xs">
        <caption className="mb-1 text-left font-medium text-base-content/80">
          {panel.title ?? fallbackTitle}
          {panel.detail ? <span className="font-normal text-base-content/55"> · {panel.detail}</span> : null}
        </caption>
        <thead className="text-base-content/55">
          <tr>
            <th scope="col" className="py-1 pr-3 font-normal">
              {panel.bars.some((bar) => bar.solver) ? "Library and method" : "Library"}
            </th>
            <th scope="col" className="py-1 pr-3 text-right font-normal">
              {QUANTITY_NAMES[quantity] ?? quantity}
            </th>
            <th scope="col" className="py-1 font-normal">
              Notes
            </th>
          </tr>
        </thead>
        <tbody>
          {panel.bars.map((bar, k) => (
            <tr key={`${bar.library}/${bar.solver}/${k}`} className="border-base-content/10 border-t">
              <th scope="row" className="py-1 pr-3 font-normal">
                {bar.label}
              </th>
              <td className="py-1 pr-3 text-right tabular-nums">{bar.value === null ? "–" : bar.text}</td>
              <td className="py-1 text-base-content/65">{barNotes(bar).join("; ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {panel.note ? <p className="mt-1 text-base-content/60 text-xs">{panel.note}</p> : null}
    </div>
  );
}
