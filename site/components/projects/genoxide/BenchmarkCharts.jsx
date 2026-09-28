import { ExternalLink } from "lucide-react";
import { HighlightProvider } from "@/components/projects/genoxide/benchmarks/Highlight";
import { ProblemChart, VersionsChart } from "@/components/projects/genoxide/benchmarks/charts";
import RunDetails from "@/components/projects/genoxide/benchmarks/RunDetails";
import { BENCHMARK_RUN_ENDPOINT } from "@/lib/projects/genoxide/benchmarks";
import { BLOB_BASE, RAW_BASE } from "@/lib/projects/genoxide/github";
import { GENOXIDE_COMMIT, GENOXIDE_LINKS } from "@/lib/projects/genoxide/meta";

const KINDS = {
  problem: ProblemChart,
  versions: VersionsChart,
};

const path = (file) => file.split("/").map(encodeURIComponent).join("/");

/**
 * A chart of charts.json as a card of `scenario` shows it: its panel of that
 * scenario only, and only the libraries of its bars; without a scenario, the
 * chart as it is. Null when it has no bars there (the scenario awaits the
 * next run).
 */
function forScenario(chart, scenario) {
  if (!scenario) return chart;
  const panels = chart.panels.filter((panel) => panel.key === scenario && panel.bars.length);
  if (!panels.length) return null;
  const used = new Set(panels.flatMap((panel) => panel.bars.map((bar) => bar.library)));
  return { ...chart, panels, libraries: chart.libraries.filter((library) => used.has(library)) };
}

/**
 * The benchmark page's charts. With the published run's charts.json
 * (`data`), interactive ones, one per `charts` entry, each linking to the
 * harness's SVG chart; a card of a problem (`scenario`) shows that
 * scenario's panel of each chart, or that it awaits the next run. Without
 * charts.json (a pin from before the file), the SVG charts themselves
 * (`images`), read at the pinned commit. Both have the same numbers.
 *
 * @param {object} props
 * @param {{ id: string, kind: string, scenario?: string, title: string, caption: string, views: { data: string, label: string, file: string }[] }[]} props.charts
 * @param {{ id: string, title: string, caption: string, file: string, wide?: boolean }[]} props.images
 * @param {{ run: object, libraries: object[], charts: Record<string, object> } | null} props.data  charts.json, or null
 * @param {string} props.resultsUrl  the tables of the published run
 * @param {boolean} [props.details]  whether the pinned commit has the run details (docs/benchmarks/runs/):
 *   then a bar of a method selects its runs, shown below the charts
 */
export default function BenchmarkCharts({ charts, images, data, resultsUrl, details = false }) {
  const tables = (
    <a
      href={resultsUrl}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 font-medium text-primary underline-offset-2 hover:underline"
    >
      results tables
      <ExternalLink size={12} aria-hidden />
    </a>
  );

  if (data) {
    const libraries = Object.fromEntries(data.libraries.map((library) => [library.id, library]));
    const cards = charts
      .map((chart) => ({
        ...chart,
        Kind: KINDS[chart.kind],
        views: chart.views
          .map((view) => ({ view, chart: data.charts[view.data] && forScenario(data.charts[view.data], chart.scenario) }))
          .filter(({ chart: shown }) => shown)
          .map(({ view, chart: shown }) => ({ key: view.data, label: view.label, chart: shown, href: `${BLOB_BASE}${path(view.file)}` })),
      }))
      // a problem's card stays when its scenario awaits the next run; another chart needs its data
      .filter((chart) => chart.Kind && (chart.views.length || chart.scenario));
    return (
      <HighlightProvider details={details}>
        <div className="grid gap-4">
          {cards.map(({ id, title, caption, Kind, views }) => {
            // only the libraries these charts show go to the browser
            const used = Object.fromEntries(
              [...new Set(views.flatMap((view) => view.chart.libraries))]
                .filter((library) => libraries[library])
                .map((library) => [library, libraries[library]]),
            );
            return (
              <section key={id} id={id} aria-labelledby={`${id}-title`} className="proj-card min-w-0 p-4 sm:p-5">
                <h3 id={`${id}-title`} className="font-semibold tracking-tight">
                  {title}
                </h3>
                <p className="proj-lead mt-1 mb-3 text-sm">{caption}</p>
                {views.length ? (
                  <Kind views={views} libraries={used} />
                ) : (
                  <p className="rounded-lg border border-base-content/15 border-dashed px-4 py-6 text-center text-base-content/65 text-sm">
                    Awaiting the next run: the published run has no runs of this problem's method yet.
                  </p>
                )}
              </section>
            );
          })}
          <p className="text-base-content/70 text-sm">The same numbers, with every library's range and throughput, are in the {tables}.</p>
          {/* the runs of the bar selected in any chart above; the answers name the commit, so a new pin is new URLs */}
          {details ? (
            <RunDetails
              endpoint={`${BENCHMARK_RUN_ENDPOINT}?commit=${encodeURIComponent(GENOXIDE_COMMIT ?? "")}`}
              issuesUrl={`${GENOXIDE_LINKS.github}/issues/new`}
              methodologyUrl={GENOXIDE_LINKS.benchmarkMethodology}
            />
          ) : null}
        </div>
      </HighlightProvider>
    );
  }

  return (
    <div className="grid gap-4 md:grid-cols-2">
      {images.map((chart) => (
        <figure key={chart.id} id={chart.id} className={`proj-card flex flex-col p-5 ${chart.wide ? "md:col-span-2" : ""}`}>
          <figcaption className="font-semibold tracking-tight">{chart.title}</figcaption>
          <p className="proj-lead mt-1 text-sm">{chart.caption}</p>
          <a
            href={`${BLOB_BASE}${path(chart.file)}`}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-4 block overflow-hidden rounded-lg bg-white p-2"
          >
            <img
              src={`${RAW_BASE}${path(chart.file)}`}
              alt={`${chart.title}: the chart of the published benchmark run`}
              loading="lazy"
              decoding="async"
              className="mx-auto h-auto w-full"
            />
          </a>
        </figure>
      ))}
      <p className="text-base-content/70 text-sm md:col-span-2">Every number in these charts is in the {tables}.</p>
    </div>
  );
}
