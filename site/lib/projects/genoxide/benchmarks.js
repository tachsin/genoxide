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

/** What the overall score is, under its title: rule 8.5 of docs/benchmarks/rules.md, in short. */
const OVERALL_CAPTION =
  "How fast each library solves the 14 scenarios, in one number. Per scenario, the fastest library gets 100 points and a library that doesn't solve it within the time cap 0, " +
  "evenly per order of magnitude of time in between; the score is the mean over the scenarios a library runs. " +
  "Its time is its fastest method's expected time to the target or, with several objectives, its fastest method's time for the budget among those within 1% of the best hypervolume. " +
  "Hover a bar for its points in each scenario.";

/** What genoxide's versions chart is (rule 10 of docs/benchmarks/rules.md), in short. */
const VERSIONS_CAPTION =
  "How genoxide's releases compare on the same benchmark runs: the CPU instructions Callgrind counts in one seeded run of each method, " +
  "to its target or its evaluation budget. The counts are exact whatever the machine's load, so a change between versions is genoxide's. " +
  "Hover a point for its evaluations, whether it reached the target and the change from the previous version.";

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
    id: "hypervolume",
    kind: "front",
    title: "Multi-objective fronts",
    caption: "Median hypervolume of the final front on ZDT and DTLZ problems, or the time its evaluation budget took.",
    views: [
      { data: "hypervolume", label: "Hypervolume", file: "docs/benchmarks/hypervolume.svg" },
      { data: "front_time", label: "Time", file: "docs/benchmarks/front_time.svg" },
    ],
  },
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
    id: "hypervolume",
    title: "Hypervolume",
    caption: "Median hypervolume of the final front on ZDT and DTLZ problems.",
    file: "docs/benchmarks/hypervolume.svg",
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
 * @param {string | null} scenario  e.g. "rastrigin-30-idiomatic"
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
