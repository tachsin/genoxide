import { ExternalLink, FileText, Scale } from "lucide-react";
import BenchmarkCharts from "@/components/projects/genoxide/BenchmarkCharts";
import Breadcrumbs from "@/components/projects/Breadcrumbs";
import JsonLd from "@/components/projects/JsonLd";
import {
  BENCHMARK_CHARTS,
  BENCHMARK_PROBLEMS,
  getBenchmarkChartData,
  getBenchmarkImages,
  getBenchmarkLibraryPages,
  hasBenchmarkRuns,
} from "@/lib/projects/genoxide/benchmarks";
import { GENOXIDE_LINKS, GENOXIDE_OG_IMAGE, GENOXIDE_PATH } from "@/lib/projects/genoxide/meta";
import { breadcrumbList } from "@/lib/projects/json-ld";
import { projectsMetadata } from "@/lib/projects/metadata";

const PATH = `${GENOXIDE_PATH}/benchmarks`;

const DESCRIPTION =
  "How genoxide and its Python package are benchmarked: three problems, one method each, run by every library with its own implementation of that method, under public rules.";

export const metadata = projectsMetadata({
  title: "Benchmarks",
  fullTitle: "Benchmarks — genoxide",
  description: DESCRIPTION,
  path: PATH,
  image: GENOXIDE_OG_IMAGE,
});

const CRUMBS = [
  { name: "Projects", path: "/projects" },
  { name: "genoxide", path: GENOXIDE_PATH },
  { name: "Benchmarks", path: PATH },
];

const DOCS = [
  {
    title: "Rules",
    body: "The rules every adapter follows, and which ones the harness checks before any timed run.",
    href: GENOXIDE_LINKS.benchmarkRules,
    Icon: Scale,
  },
  {
    title: "Methodology",
    body: "The libraries, the protocol, the three problems and their methods, and how to run it.",
    href: GENOXIDE_LINKS.benchmarkMethodology,
    Icon: FileText,
  },
  {
    title: "Notes",
    body: "What each library can't run, where it differs from a method's definition, and the bugs found.",
    href: GENOXIDE_LINKS.benchmarkNotes,
    Icon: FileText,
  },
  {
    title: "Results",
    body: "The full numbers of the latest run.",
    href: GENOXIDE_LINKS.benchmarkResults,
    Icon: FileText,
  },
];

export default async function BenchmarksPage() {
  const [libraries, data, images, details] = await Promise.all([
    getBenchmarkLibraryPages(),
    getBenchmarkChartData(),
    getBenchmarkImages(),
    hasBenchmarkRuns(),
  ]);

  return (
    <main className="proj-container pt-10 pb-8 sm:pt-14">
      <JsonLd data={breadcrumbList(CRUMBS)} />
      <Breadcrumbs items={CRUMBS} />

      <header className="proj-rise max-w-3xl">
        <p className="proj-eyebrow">genoxide</p>
        <h1 className="mt-2 font-semibold text-4xl tracking-tight sm:text-5xl">Benchmarks</h1>
        <div className="proj-lead mt-5 space-y-3 text-lg">
          <p>
            Three problems, one method each: a genetic algorithm on OneMax, differential evolution on Rastrigin and
            CMA-ES on Rosenbrock. genoxide, its Python package and the other libraries in Rust, Python, Java and Julia
            run a problem only with their own implementation of its method, set to the same written definition, with
            the same fitness functions, evaluation budgets and time cap (60 seconds to a target).
          </p>
          <p>
            The suite is small and matched on purpose: comparing each library's best pick among many methods says
            little about any of them, while the same algorithm in every library compares the implementations, their
            speed and whether they do what the definition says. More problems, and multi-objective ones, come back
            after these.
          </p>
          <p>
            Every run is validated against a Python reference, each library is timed single-threaded on fixed
            cores, and every setting and every bug found is documented per library.
          </p>
        </div>
      </header>

      <section aria-labelledby="bench-problems" className="proj-rise-1 mt-12">
        <h2 id="bench-problems" className="font-semibold text-xl tracking-tight">
          Problems
        </h2>
        <ul className="mt-4 grid gap-4 md:grid-cols-3">
          {BENCHMARK_PROBLEMS.map((problem) => (
            <li key={problem.scenario} className="proj-card p-5">
              <p className="text-base-content/55 text-xs uppercase tracking-wide">{problem.name}</p>
              <p className="mt-2 font-semibold">{problem.method}</p>
              <p className="proj-lead mt-1 text-sm">{problem.settings}</p>
            </li>
          ))}
        </ul>
        <p className="proj-lead mt-3 text-sm">
          Each method's exact definition, which every library's configuration is checked against, is in the{" "}
          <a
            href={`${GENOXIDE_LINKS.benchmarkRules}#6-the-methods`}
            target="_blank"
            rel="noopener noreferrer"
            className="font-medium text-primary underline-offset-2 hover:underline"
          >
            rules
          </a>
          ; where a library can't be set exactly to it, its page says how it differs.
        </p>
      </section>

      <section aria-labelledby="bench-charts" className="proj-rise-2 mt-12">
        <h2 id="bench-charts" className="font-semibold text-xl tracking-tight">
          Results
        </h2>
        {data ? (
          <p className="proj-lead mt-2 mb-5 text-sm">
            The published run of {data.run.date}: {data.run.platform}, single-threaded, {data.run.seeds} seeds per
            scenario. A card per problem, a bar per library: switch between the time and the evaluations to the
            target and the distance to the optimum. Hover a bar, or focus a chart and use the arrow keys, for its
            numbers; select a library to highlight it in every chart.
            {details
              ? " Click a method's bar, or press Enter on it, for its runs, what they printed and the adapter's code, with a link to suggest a better way to run it."
              : null}
          </p>
        ) : (
          <p className="proj-lead mt-2 mb-5 text-sm">
            The charts of the published run, as the benchmark harness drew them. Their numbers are in{" "}
            <a
              href={GENOXIDE_LINKS.benchmarkResults}
              target="_blank"
              rel="noopener noreferrer"
              className="font-medium text-primary underline-offset-2 hover:underline"
            >
              results.md
            </a>
            .
          </p>
        )}
        <BenchmarkCharts
          charts={BENCHMARK_CHARTS}
          images={images}
          data={data}
          resultsUrl={GENOXIDE_LINKS.benchmarkResults}
          details={details}
        />
      </section>

      <section aria-labelledby="bench-docs" className="mt-12">
        <h2 id="bench-docs" className="font-semibold text-xl tracking-tight">
          How it's run
        </h2>
        <ul className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {DOCS.map(({ title, body, href, Icon }) => (
            <li key={title}>
              <a href={href} target="_blank" rel="noopener noreferrer" className="proj-card group flex h-full flex-col p-5">
                <span className="proj-icon-tile">
                  <Icon size={18} aria-hidden />
                </span>
                <span className="mt-4 inline-flex items-center gap-1.5 font-semibold">
                  {title}
                  <ExternalLink size={13} aria-hidden className="text-base-content/40 group-hover:text-primary" />
                </span>
                <span className="proj-lead mt-1 text-sm">{body}</span>
              </a>
            </li>
          ))}
        </ul>
      </section>

      <section aria-labelledby="bench-libraries" className="mt-12">
        <h2 id="bench-libraries" className="font-semibold text-xl tracking-tight">
          One page per library
        </h2>
        <p className="proj-lead mt-2 text-sm">
          How its own implementation of each method is set to the definition, where it differs, what it can't run,
          its separate test runs and its bugs.
        </p>
        <ul className="mt-5 flex flex-wrap gap-2">
          {libraries.map((lib) => (
            <li key={lib.url}>
              <a href={lib.url} target="_blank" rel="noopener noreferrer" className="proj-pill">
                {lib.name}
              </a>
            </li>
          ))}
        </ul>
        <p className="mt-6 text-base-content/55 text-sm">
          A better way to run a library is welcome: the{" "}
          <a
            href={GENOXIDE_LINKS.benchmarksDir}
            target="_blank"
            rel="noopener noreferrer"
            className="underline decoration-base-content/25 underline-offset-2 hover:decoration-primary"
          >
            benchmark documentation
          </a>{" "}
          says how.
        </p>
      </section>
    </main>
  );
}
