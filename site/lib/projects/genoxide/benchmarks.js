import { fetchCached } from "@/lib/projects/fetch-cached";
import { BLOB_BASE, getRepoFiles } from "./github";

/**
 * The benchmark page's data.
 *
 * RESULTS: interactive charts will read a results JSON published by the
 * benchmark harness. Until that file exists, BENCHMARK_RESULTS_URL stays
 * null and the page shows the charts the harness drew for the published run
 * (docs/benchmarks/*.svg, at the pinned commit), each with its `file`: no
 * number here is ever made up.
 */
export const BENCHMARK_RESULTS_URL = null;

/** The charts the page will have, in order. */
export const BENCHMARK_CHARTS = [
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
 * The published results, or null until BENCHMARK_RESULTS_URL is set (or when
 * it can't be fetched). The shape is the harness's to define.
 */
export async function getBenchmarkResults() {
  if (!BENCHMARK_RESULTS_URL) return null;
  return fetchCached(BENCHMARK_RESULTS_URL, { as: "json", tags: ["genoxide"] });
}
