"use client";

import ChartShell from "./ChartShell";
import VersionPanel from "./VersionPanel";

/**
 * The benchmark page's interactive charts, one per kind, each drawing the
 * charts of charts.json it's given (`views`) with the harness's library
 * colors (`libraries`). BenchmarkCharts picks them by BENCHMARK_CHARTS's
 * `kind`.
 */

// the panels side by side as the chart's width allows
const PANELS = "@2xl:grid-cols-2 @5xl:grid-cols-3";

/**
 * One problem of the matched suite: its panel of each chart, a bar per
 * library that runs the problem's method, switching between the time to
 * target, the evaluations to target and the distance to the optimum at the
 * end.
 */
export function ProblemChart({ views, libraries }) {
  return <ChartShell views={views} libraries={libraries} grid="max-w-3xl" name="Results" />;
}

/**
 * genoxide's versions: the CPU instructions of the same runs in each version,
 * a panel per scenario, a line per method across the versions.
 */
export function VersionsChart({ views, libraries }) {
  return <ChartShell views={views} libraries={libraries} grid={PANELS} name="genoxide's versions" Panel={VersionPanel} />;
}
