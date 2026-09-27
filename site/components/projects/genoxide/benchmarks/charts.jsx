"use client";

import ChartShell from "./ChartShell";

/**
 * The benchmark page's interactive charts, one per kind, each drawing the
 * charts of charts.json it's given (`views`) with the harness's library
 * colors (`libraries`). BenchmarkCharts picks them by BENCHMARK_CHARTS's
 * `kind`.
 */

// the panels side by side as the chart's width allows
const PANELS = "@2xl:grid-cols-2 @5xl:grid-cols-3";
// long panels, two at most, so their labels and notes fit
const LONG_PANELS = "@2xl:grid-cols-2";

/**
 * Each library's overall score (rule 8.5): one panel, a bar per library,
 * with room for its coverage beside it; a bar's tooltip and the tables have
 * its points, time and method in each scenario.
 */
export function OverallChart({ views, libraries }) {
  return <ChartShell views={views} libraries={libraries} grid="max-w-3xl" room={150} name="Overall score" />;
}

/** Each library's fastest method per scenario: a panel per scenario, its libraries by time to target. */
export function SummaryChart({ views, libraries }) {
  return <ChartShell views={views} libraries={libraries} grid={PANELS} name="Each library's fastest method" />;
}

/**
 * Every method's expected time, or evaluations, to target: a panel per
 * scenario, the first 12 methods until "show all".
 */
export function ToTargetChart({ views, libraries }) {
  return <ChartShell views={views} libraries={libraries} grid={LONG_PANELS} limit={12} name="To target" />;
}

/** The CPU instructions of one evaluation, framework and fitness function together: one panel. */
export function InstructionsChart({ views, libraries }) {
  return <ChartShell views={views} libraries={libraries} grid="max-w-2xl" name="Instructions per evaluation" />;
}

/**
 * The multi-objective scenarios: the final front's hypervolume, or the time
 * of the budget, per scenario; the runs the time cap stopped apart, hatched.
 */
export function FrontChart({ views, libraries }) {
  return <ChartShell views={views} libraries={libraries} grid={LONG_PANELS} limit={12} name="Multi-objective" />;
}
