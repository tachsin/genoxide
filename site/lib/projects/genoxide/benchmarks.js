import { BLOB_BASE, getRepoFile, getRepoFiles } from "./github";

/**
 * The benchmark page's data.
 *
 * The charts are drawn from docs/benchmarks/charts.json, which the benchmark
 * harness (benchmarks/run.py) writes beside its SVG charts of the published
 * run, from the same summaries: every number, label and note the SVGs show,
 * and the run's date, machine and versions. It's read at the pinned commit,
 * like everything else here, so the page shows that commit's run. A pin from
 * before the file existed gets the SVG charts as images instead
 * (BENCHMARK_IMAGES). No number here is ever made up.
 */
export const BENCHMARK_CHARTS_FILE = "docs/benchmarks/charts.json";

/** What the overall score is, under its title: rule 8.5 of docs/benchmarks/rules.md, in short. */
const OVERALL_CAPTION =
  "How fast each library solves the 14 scenarios, in one number. Per scenario, the fastest library gets 100 points and a library that doesn't solve it within the time cap 0, " +
  "evenly per order of magnitude of time in between; the score is the mean over the scenarios a library runs. " +
  "Its time is its fastest method's expected time to the target or, with several objectives, its fastest method's time for the budget among those within 1% of the best hypervolume. " +
  "Hover a bar for its points in each scenario.";

/**
 * The interactive charts, in order: each a `kind` of component
 * (components/projects/genoxide/benchmarks/charts.jsx) and its `views`, the
 * charts of charts.json it can switch between, each with its SVG `file`.
 */
export const BENCHMARK_CHARTS = [
  {
    id: "overall",
    kind: "overall",
    title: "Overall score",
    caption: OVERALL_CAPTION,
    views: [{ data: "overall", label: "Score", file: "docs/benchmarks/overall.svg" }],
  },
  {
    id: "summary",
    kind: "summary",
    title: "Time to target: each library's fastest method",
    caption: "Each bar is a library's fastest method on that problem, single-threaded on the same machine, 10 seeds.",
    views: [{ data: "summary", label: "Time", file: "docs/benchmarks/summary.svg" }],
  },
  {
    id: "to-target",
    kind: "to-target",
    title: "Time and evaluations to target",
    caption:
      "Every method's expected running time to each single-objective target, per scenario, or the fitness evaluations it took: what counts when the fitness function is expensive.",
    views: [
      { data: "time_to_target", label: "Time", file: "docs/benchmarks/time_to_target.svg" },
      { data: "evaluations_to_target", label: "Evaluations", file: "docs/benchmarks/evaluations_to_target.svg" },
    ],
  },
  {
    id: "instructions",
    kind: "instructions",
    title: "Instructions per evaluation",
    caption: "The framework's own cost around one evaluation.",
    views: [{ data: "instructions", label: "Instructions", file: "docs/benchmarks/instructions.svg" }],
  },
  {
    id: "hypervolume",
    kind: "front",
    title: "Multi-objective fronts",
    caption: "Median hypervolume of the final front on ZDT and DTLZ problems, or the time its evaluation budget took.",
    views: [
      { data: "hypervolume", label: "Hypervolume", file: "docs/benchmarks/hypervolume.svg" },
      { data: "front_time", label: "Time", file: "docs/benchmarks/front_time.svg" },
    ],
  },
];

/** Without charts.json (a pin from before it): the harness's SVG charts, as images. */
export const BENCHMARK_IMAGES = [
  {
    id: "overall",
    title: "Overall score",
    caption: OVERALL_CAPTION.replace(" Hover a bar for its points in each scenario.", ""),
    file: "docs/benchmarks/overall.svg",
    wide: true,
  },
  {
    id: "summary",
    title: "Time to target: each library's fastest method",
    caption: "Each bar is a library's fastest method on that problem, single-threaded on the same machine, 10 seeds.",
    file: "docs/benchmarks/summary.svg",
    wide: true,
  },
  {
    id: "time-to-target",
    title: "Time to target",
    caption: "Expected running time to reach each single-objective target, per scenario.",
    file: "docs/benchmarks/time_to_target.svg",
  },
  {
    id: "evaluations-to-target",
    title: "Evaluations to target",
    caption: "Fitness evaluations to reach the target: what counts when the fitness function is expensive.",
    file: "docs/benchmarks/evaluations_to_target.svg",
  },
  {
    id: "instructions",
    title: "Instructions per evaluation",
    caption: "The framework's own cost around one evaluation.",
    file: "docs/benchmarks/instructions.svg",
  },
  {
    id: "hypervolume",
    title: "Hypervolume",
    caption: "Median hypervolume of the final front on ZDT and DTLZ problems.",
    file: "docs/benchmarks/hypervolume.svg",
  },
];

/**
 * The SVG charts of BENCHMARK_IMAGES that the pinned commit has (all of them
 * when GitHub can't be reached): overall.svg is newer than the others.
 */
export async function getBenchmarkImages() {
  const files = await getRepoFiles();
  return files ? BENCHMARK_IMAGES.filter((image) => files.has(image.file)) : BENCHMARK_IMAGES;
}

/** The problems, as in benchmarks/README.md. */
export const BENCHMARK_PROBLEMS = {
  single: ["OneMax", "N-Queens", "Rastrigin (shifted)", "Rosenbrock", "Ackley (shifted)"],
  multi: ["ZDT1", "ZDT2", "ZDT3", "DTLZ1", "DTLZ2"],
};

const LIBRARY_NAMES = {
  genoxide: "genoxide",
  genoxide_python: "genoxide (Python)",
  genetic_algorithm: "genetic_algorithm",
  radiate: "radiate",
  moors: "moors",
  openga: "openGA",
  pygmo: "pygmo",
  deap: "DEAP",
  pymoo: "pymoo",
  pygad: "PyGAD",
  pycma: "pycma",
  nevergrad: "Nevergrad",
  scipy: "SciPy",
  jenetics: "Jenetics",
  jmetal: "jMetal",
  evolutionary_jl: "Evolutionary.jl",
  metaheuristics_jl: "Metaheuristics.jl",
};

const LIBRARIES_DIR = "docs/benchmarks/libraries";

/**
 * One page per library, from the repository's docs/benchmarks/libraries/.
 * Falls back to the list known when this page was written if GitHub can't
 * be reached, so the links never disappear.
 * @returns {Promise<{ name: string, url: string }[]>}
 */
export async function getBenchmarkLibraryPages() {
  const files = await getRepoFiles();
  const stems = files
    ? [...files]
        .map((p) => new RegExp(`^${LIBRARIES_DIR}/([^/]+)\\.md$`).exec(p)?.[1])
        .filter(Boolean)
    : Object.keys(LIBRARY_NAMES);

  const known = Object.keys(LIBRARY_NAMES);
  const rank = (stem) => {
    const i = known.indexOf(stem);
    return i === -1 ? known.length : i;
  };
  return stems
    .sort((a, b) => rank(a) - rank(b) || a.localeCompare(b))
    .map((stem) => ({
      name: LIBRARY_NAMES[stem] ?? stem.replace(/_/g, " "),
      url: `${BLOB_BASE}${LIBRARIES_DIR}/${stem}.md`,
    }));
}

/**
 * The published run's charts.json at the pinned commit, parsed, or null when
 * the commit has none (or it can't be read, or it isn't a format this page
 * knows): the page then shows the SVG charts.
 * @returns {Promise<{ format: 1, run: object, libraries: object[], charts: Record<string, object> } | null>}
 */
export async function getBenchmarkChartData() {
  const files = await getRepoFiles();
  if (files && !files.has(BENCHMARK_CHARTS_FILE)) return null;
  const text = await getRepoFile(BENCHMARK_CHARTS_FILE);
  if (typeof text !== "string") return null;
  try {
    const data = JSON.parse(text);
    return data?.format === 1 && data.charts && Array.isArray(data.libraries) ? data : null;
  } catch {
    return null;
  }
}
