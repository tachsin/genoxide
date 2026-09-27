import { ExternalLink } from "lucide-react";
import { HighlightProvider } from "@/components/projects/genoxide/benchmarks/Highlight";
import {
  FrontChart,
  InstructionsChart,
  SummaryChart,
  ToTargetChart,
} from "@/components/projects/genoxide/benchmarks/charts";
import { BLOB_BASE, RAW_BASE } from "@/lib/projects/genoxide/github";

const KINDS = { summary: SummaryChart, "to-target": ToTargetChart, instructions: InstructionsChart, front: FrontChart };

const path = (file) => file.split("/").map(encodeURIComponent).join("/");

/**
 * The benchmark page's charts. With the published run's charts.json
 * (`data`), interactive ones, one per `charts` entry, each linking to the
 * harness's SVG chart; without it (a pin from before the file), the SVG
 * charts themselves (`images`), read at the pinned commit. Both have the
 * same numbers.
 *
 * @param {object} props
 * @param {{ id: string, kind: string, title: string, caption: string, views: { data: string, label: string, file: string }[] }[]} props.charts
 * @param {{ id: string, title: string, caption: string, file: string, wide?: boolean }[]} props.images
 * @param {{ run: object, libraries: object[], charts: Record<string, object> } | null} props.data  charts.json, or null
 * @param {string} props.resultsUrl  the tables of the published run
 */
export default function BenchmarkCharts({ charts, images, data, resultsUrl }) {
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
          .filter((view) => data.charts[view.data])
          .map((view) => ({ key: view.data, label: view.label, chart: data.charts[view.data], href: `${BLOB_BASE}${path(view.file)}` })),
      }))
      .filter((chart) => chart.Kind && chart.views.length);
    return (
      <HighlightProvider>
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
                <Kind views={views} libraries={used} />
              </section>
            );
          })}
          <p className="text-base-content/70 text-sm">The same numbers, with every method's range and throughput, are in the {tables}.</p>
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
