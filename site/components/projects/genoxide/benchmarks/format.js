import { formatTick, formatValue } from "../player/chart-kit";

/**
 * Number formats of the benchmark charts, by the quantity charts.json gives
 * each chart ("seconds", "evaluations", "instructions", "distance"). The
 * bars' own labels are the harness's (`text`), so they read
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
  distance: "Distance to the optimum",
};

/** A value in full (6 significant digits, as stored), with its unit. */
export function exactValue(value, quantity) {
  if (value === null || value === undefined) return "–";
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

/** A share of the budget, e.g. 0.07 as "7%". */
export function percent(share) {
  return `${Math.round(share * 100)}%`;
}

const count = new Intl.NumberFormat("en");

/** The notes of a bar, in words: why it's missing, how many runs reached the target or were stopped. */
export function barNotes(bar) {
  // a point of genoxide's versions chart: one run of a method in a version
  if (typeof bar.version === "string") {
    const notes = [];
    if (typeof bar.evaluations === "number") notes.push(`${count.format(bar.evaluations)} evaluations`);
    if (typeof bar.reached === "number") notes.push(bar.reached ? "reached the target" : "didn't reach the target within the budget");
    if (bar.invalid) notes.push("failed a check of the rules");
    return notes;
  }
  // a missing value's label says it all, as the harness's chart writes it
  // (a bar of a fixed budget has no target, so no "reached")
  if (bar.value === null) {
    if (!bar.missing) return ["no value"];
    return [typeof bar.reached === "number" ? `too few runs reached the target: ${bar.missing}` : bar.missing];
  }
  const notes = [];
  if (typeof bar.reached === "number" && typeof bar.runs === "number" && bar.reached < bar.runs) {
    notes.push(`reached the target in ${bar.reached} of ${bar.runs} runs`);
  }
  if (bar.capped) {
    notes.push(
      `${bar.capped} run${bar.capped === 1 ? "" : "s"} stopped by the time cap` +
        (typeof bar.capped_share === "number" ? `, after a median ${percent(bar.capped_share)} of the budget` : ""),
    );
  }
  if (bar.ended_on_cap) notes.push("these are the runs the time cap stopped");
  if (bar.ended_early) notes.push(`${bar.ended_early} run${bar.ended_early === 1 ? "" : "s"} ended early by the library`);
  return notes;
}
