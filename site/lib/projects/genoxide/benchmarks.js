import { highlightCode } from "@/lib/projects/highlight";
import { BLOB_BASE, getRepoFile, getRepoFiles } from "./github";
import { GENOXIDE_COMMIT, GENOXIDE_PATH, GENOXIDE_REPO } from "./meta";

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

/** What genoxide's versions chart is (rule 10 of docs/benchmarks/rules.md), in short. */
const VERSIONS_CAPTION =
  "How genoxide's releases compare on the same benchmark runs: the CPU instructions Callgrind counts in one seeded run of each method, " +
  "to its target or its evaluation budget. The counts are exact whatever the machine's load, so a change between versions is genoxide's. " +
  "Hover a point for its evaluations, whether it reached the target and the change from the previous version.";

/** The charts of charts.json a problem's card switches between, each with its SVG. */
const PROBLEM_VIEWS = [
  { data: "time_to_target", label: "Time", file: "docs/benchmarks/time_to_target.svg" },
  { data: "evaluations_to_target", label: "Evaluations", file: "docs/benchmarks/evaluations_to_target.svg" },
  { data: "distance_to_optimum", label: "Distance", file: "docs/benchmarks/distance_to_optimum.svg" },
];

const PROBLEM_CAPTION =
  "Each bar is a library's own implementation of the method, set to the same definition, single-threaded on the same machine, 10 seeds. " +
  "Time and evaluations are the expected running time to the target; the distance is how far the runs ended from the optimum, the measure that counts where few runs reach the target.";

/**
 * The problems of the matched suite (benchmarks/README.md, rule 6 of
 * docs/benchmarks/rules.md): each its scenario of charts.json, and the one
 * method every library runs on it, by name and in short.
 */
export const BENCHMARK_PROBLEMS = [
  {
    scenario: "onemax-1000-matched",
    name: "OneMax 1000",
    method: "GA",
    settings: "DEAP's eaSimple: 300 individuals, tournament of 3, two-point crossover, bit flip, no elitism",
  },
  {
    scenario: "rastrigin-30-matched",
    name: "Rastrigin 30, shifted",
    method: "DE/rand/1/bin",
    settings: "100 individuals, F 0.5, CR 0.9, no adaptation or restarts; no target, a fixed budget of 300,000 evaluations",
    // no target: the time for the budget and the error at the end, which should agree across
    // libraries within the seeds' spread, since they run the same algorithm
    views: [
      { data: "time_to_target", label: "Time for the budget", file: "docs/benchmarks/time_to_target.svg" },
      { data: "distance_to_optimum", label: "Error at the end", file: "docs/benchmarks/distance_to_optimum.svg" },
    ],
    caption:
      "Each bar is a library's own implementation of the method, set to the same definition, single-threaded on the same machine, 10 seeds. " +
      "There's no target: every run uses the same budget, so the bars show the median time for it and the error the runs end at. " +
      "The libraries run the same algorithm, so their errors should agree within the seeds' spread; one far off points to a difference or a bug.",
  },
  {
    scenario: "rosenbrock-10-matched",
    name: "Rosenbrock 10",
    method: "CMA-ES",
    settings: "Hansen's defaults: 10 samples, the best 5 recombined, an initial step of 0.3 of the range, no restarts",
  },
];

/**
 * The interactive charts, in order: a card per problem (`scenario`: its
 * panel of each chart), then genoxide's versions. Each is a `kind` of
 * component (components/projects/genoxide/benchmarks/charts.jsx) and its
 * `views`, the charts of charts.json it can switch between, each with its SVG
 * `file`.
 */
export const BENCHMARK_CHARTS = [
  ...BENCHMARK_PROBLEMS.map((problem) => ({
    id: problem.scenario,
    kind: "problem",
    scenario: problem.scenario,
    title: `${problem.name}: ${problem.method}`,
    caption: `${problem.method}, ${problem.settings}. ${problem.caption ?? PROBLEM_CAPTION}`,
    views: problem.views ?? PROBLEM_VIEWS,
  })),
  {
    // genoxide only (rule 10): a pin whose charts.json doesn't have it leaves it out
    id: "genoxide-versions",
    kind: "versions",
    title: "genoxide's versions",
    caption: VERSIONS_CAPTION,
    views: [{ data: "genoxide_versions", label: "Instructions", file: "docs/benchmarks/genoxide_versions.svg" }],
  },
];

/** Without charts.json (a pin from before it): the harness's SVG charts, as images. */
export const BENCHMARK_IMAGES = [
  {
    id: "time-to-target",
    title: "Time to target",
    caption: "Expected running time to reach each target, a panel per problem.",
    file: "docs/benchmarks/time_to_target.svg",
  },
  {
    id: "evaluations-to-target",
    title: "Evaluations to target",
    caption: "Fitness evaluations to reach the target: what counts when the fitness function is expensive.",
    file: "docs/benchmarks/evaluations_to_target.svg",
  },
  {
    id: "genoxide-versions",
    title: "genoxide's versions",
    caption: "The CPU instructions of one seeded run of each of genoxide's methods, in each of its releases.",
    file: "docs/benchmarks/genoxide_versions.svg",
    wide: true,
  },
];

/**
 * The SVG charts of BENCHMARK_IMAGES that the pinned commit has (all of them
 * when GitHub can't be reached).
 */
export async function getBenchmarkImages() {
  const files = await getRepoFiles();
  return files ? BENCHMARK_IMAGES.filter((image) => files.has(image.file)) : BENCHMARK_IMAGES;
}

const LIBRARY_NAMES = {
  genoxide: "genoxide",
  genoxide_python: "genoxide (Python)",
  radiate: "radiate",
  pygmo: "pygmo",
  deap: "DEAP",
  pymoo: "pymoo",
  pygad: "PyGAD",
  pycma: "pycma",
  scipy: "SciPy",
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

/**
 * The run details: docs/benchmarks/runs/<scenario>.json, which the harness
 * writes beside charts.json from the same run (benchmarks/run.py,
 * run_details). Per scenario, its settings and, per library and method, its
 * summary, every run with the JSON line its adapter printed, and the blocks
 * of the adapter's code that set the method up. The page's details panel
 * reads one method at a time through BENCHMARK_RUN_ENDPOINT, so the browser
 * never downloads a whole scenario.
 */
export const BENCHMARK_RUNS_DIR = "docs/benchmarks/runs";

/**
 * Whether the pinned commit has the run details (true when GitHub can't be
 * reached: the panel then says so if a selection can't be read).
 */
export async function hasBenchmarkRuns() {
  const files = await getRepoFiles();
  return !files || [...files].some((file) => file.startsWith(`${BENCHMARK_RUNS_DIR}/`));
}

/** The route that answers getBenchmarkRun (app/projects/genoxide/benchmarks/run/route.js). */
export const BENCHMARK_RUN_ENDPOINT = `${GENOXIDE_PATH}/benchmarks/run`;

const SCENARIO_KEY = /^[a-z0-9]+-\d+-[a-z]+$/;
const NAME = /^[a-z0-9_]+$/;

/** What each language of the adapters is called above its code. */
const LANGUAGE_NAMES = { rust: "Rust", python: "Python", java: "Java", julia: "Julia", cpp: "C++", text: "Code" };

function blobAt(commit, path) {
  return `https://github.com/${GENOXIDE_REPO}/blob/${commit}/${path.split("/").map(encodeURIComponent).join("/")}`;
}

/** The lines without the indentation they all share. */
function dedent(text) {
  const lines = text.split("\n");
  const indents = lines.filter((line) => line.trim()).map((line) => line.length - line.trimStart().length);
  const common = indents.length ? Math.min(...indents) : 0;
  return lines.map((line) => line.slice(common)).join("\n");
}

/**
 * Where a block of the adapter's code is at the pinned commit: its lines
 * there if the file has the excerpt (at the harness's line, or moved), so the
 * link is to the same commit as the page's other files; otherwise at the
 * commit the harness read it at (`commit`, when the file was unchanged from
 * it); otherwise the file at the pinned commit, without lines.
 */
async function placeBlock(block) {
  const text = await getRepoFile(block.path);
  const excerpt = block.excerpt.split("\n");
  const length = block.end - block.start;
  if (typeof text === "string") {
    const lines = text.split(/\r?\n/);
    const at = (i) => excerpt.every((line, k) => lines[i + k] === line);
    let found = at(block.start - 1) ? block.start - 1 : -1;
    for (let i = 0; found === -1 && i + excerpt.length <= lines.length; i++) if (at(i)) found = i;
    if (found !== -1) return { commit: GENOXIDE_COMMIT, start: found + 1, end: found + 1 + length, current: true };
  }
  if (block.commit) return { commit: block.commit, start: block.start, end: block.end, current: false };
  return { commit: GENOXIDE_COMMIT, start: null, end: null, current: false };
}

/**
 * One library's method in one scenario, for the details panel: the scenario,
 * the method's summary and runs, their output and its code, highlighted here
 * on the server as the site's other code is, with links to the code and the
 * library's page at the pinned commit. `{ error, status }` when it isn't a
 * run of the published results, or its details can't be read.
 * @param {string | null} scenario  e.g. "rastrigin-30-matched"
 * @param {string | null} library  e.g. "genoxide"
 * @param {string | null} solver  e.g. "de"
 */
export async function getBenchmarkRun(scenario, library, solver) {
  if (!SCENARIO_KEY.test(scenario ?? "") || !NAME.test(library ?? "") || !NAME.test(solver ?? "")) {
    return { error: "Not a run of the benchmarks.", status: 400 };
  }
  const path = `${BENCHMARK_RUNS_DIR}/${scenario}.json`;
  const files = await getRepoFiles();
  if (files && !files.has(path)) return { error: "The published run has no details for this scenario.", status: 404 };
  const text = await getRepoFile(path);
  if (typeof text !== "string") return { error: "The run details couldn't be read from GitHub right now.", status: 502 };
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    return { error: "The run details couldn't be read.", status: 502 };
  }
  if (data?.format !== 1 || !Array.isArray(data.methods)) return { error: "The run details couldn't be read.", status: 502 };
  const method = data.methods.find((m) => m.library === library && m.solver === solver);
  if (!method) return { error: "This method has no runs in this scenario.", status: 404 };

  const { runs = [], code = [], page, adapter, ...rest } = method;
  const [output, blocks] = await Promise.all([
    // as plain text: a token per number of a 1,000-bit solution would make it hundreds of kB
    highlightCode(runs.map((run) => run.output).join("\n"), "text"),
    Promise.all(
      code.map(async (block) => {
        const place = await placeBlock(block);
        return {
          path: block.path,
          filename: block.path.replace(/^benchmarks\/adapters\//, ""),
          language: LANGUAGE_NAMES[block.language] ?? block.language,
          start: place.start,
          end: place.end,
          // the block's length, and how many of its lines the excerpt has
          lines: block.end - block.start + 1,
          shown: block.excerpt.split("\n").length,
          current: place.current,
          commit: place.commit,
          href: `${blobAt(place.commit, block.path)}${place.start ? `#L${place.start}-L${place.end}` : ""}`,
          html: await highlightCode(dedent(block.excerpt), block.language),
        };
      }),
    ),
  ]);

  return {
    run: data.run,
    scenario: data.scenario,
    method: { ...rest, runs: runs.map(({ output: _output, ...run }) => run) },
    output,
    code: blocks,
    links: {
      page: page && (!files || files.has(page)) ? blobAt(GENOXIDE_COMMIT, page) : null,
      adapter: adapter ? blobAt(GENOXIDE_COMMIT, adapter) : null,
      commit: GENOXIDE_COMMIT,
    },
  };
}
