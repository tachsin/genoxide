import { formatTick, formatValue } from "../player/chart-kit";

/**
 * Number formats of the benchmark charts, by the quantity charts.json gives
 * each chart ("seconds", "evaluations", "instructions", "hypervolume",
 * "distance", "score"). The bars' own labels are the harness's (`text`), so they read
 * as in its SVG charts; these are for axes and tooltips.
 */

const exact = new Intl.NumberFormat("en", { maximumSignificantDigits: 6 });
const short = new Intl.NumberFormat("en", { maximumSignificantDigits: 3 });

/** What the quantity is called, e.g. in a table's header. */
export const QUANTITY_NAMES = {
  seconds: "Time",
  evaluations: "Evaluations",
  // genoxide's versions chart: the CPU instructions of a whole run
  instructions: "Instructions",
  hypervolume: "Hypervolume",
  distance: "Distance to the optimum",
  score: "Score",
};

/** A value in full (6 significant digits, as stored), with its unit. */
export function exactValue(value, quantity) {
  if (value === null || value === undefined) return "–";
  // an overall score is given to one decimal (rule 8.5)
  if (quantity === "score") return value.toFixed(1);
  if (quantity === "seconds") {
    if (value < 1e-3) return `${exact.format(value * 1e6)} µs`;
    if (value < 1) return `${exact.format(value * 1e3)} ms`;
    return `${exact.format(value)} s`;
  }
  return formatValue(value);
}

/** A tick of a chart's value axis. */
export function tickFormat(quantity) {
  if (quantity === "seconds") {
    return (value) => {
      // rounded first, so that 999.7 µs reads 1 ms, not 1,000 µs
      const v = Number(value.toPrecision(3));
      return v < 1e-3 ? `${short.format(v * 1e6)} µs` : v < 1 ? `${short.format(v * 1e3)} ms` : `${short.format(v)} s`;
    };
  }
  if (quantity === "evaluations" || quantity === "instructions") {
    return (v) => {
      for (const [limit, suffix] of [
        [1e9, "G"],
        [1e6, "M"],
        [1e3, "k"],
      ]) {
        if (v >= limit) return `${short.format(v / limit)}${suffix}`;
      }
      return short.format(v);
    };
  }
  return formatTick;
}

/** A library's points in a scenario of the overall score, 0 to 100: one decimal, e.g. 96.8, 100 or 0. */
export function pointsText(points) {
  if (typeof points !== "number") return "–";
  return points >= 100 ? "100" : points <= 0 ? "0" : points.toFixed(1);
}

/** A share of the budget, e.g. 0.07 as "7%". */
export function percent(share) {
  return `${Math.round(share * 100)}%`;
}

const count = new Intl.NumberFormat("en");

/** The notes of a bar, in words: why it's missing, how many runs reached the target, were stopped or failed. */
export function barNotes(bar) {
  // a point of genoxide's versions chart: one run of a method in a version
  if (typeof bar.version === "string") {
    const notes = [];
    if (typeof bar.evaluations === "number") notes.push(`${count.format(bar.evaluations)} evaluations`);
    if (typeof bar.reached === "number") notes.push(bar.reached ? "reached the target" : "didn't reach the target within the budget");
    if (typeof bar.hypervolume === "number") notes.push(`hypervolume ${bar.hypervolume.toFixed(4)}, for the whole budget`);
    if (bar.invalid) notes.push("failed a check of the rules");
    return notes;
  }
  // a missing value's label says it all, as the harness's chart writes it
  if (bar.value === null) return [bar.missing ? `too few runs reached the target: ${bar.missing}` : "no value"];
  const notes = [];
  if (typeof bar.reached === "number" && typeof bar.runs === "number" && bar.reached < bar.runs) {
    notes.push(`reached the target in ${bar.reached} of ${bar.runs} runs`);
  }
  if (bar.errors) notes.push(`${bar.errors} of ${bar.runs} runs failed`);
  if (bar.capped) {
    notes.push(
      `${bar.capped} run${bar.capped === 1 ? "" : "s"} stopped by the time cap` +
        (typeof bar.capped_share === "number" ? `, after a median ${percent(bar.capped_share)} of the budget` : ""),
    );
  }
  if (bar.ended_on_cap) notes.push("these are the runs the time cap stopped");
  // an overall score: how many scenarios the library runs and solves
  if (typeof bar.of === "number" && typeof bar.scenarios === "number") {
    notes.push(
      `runs ${bar.scenarios} of the ${bar.of} scenarios` + (typeof bar.solved === "number" ? `, solves ${bar.solved}` : ""),
    );
  }
  if (bar.below_axis) notes.push("below the axis's start");
  return notes;
}
