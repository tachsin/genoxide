import { ExternalLink } from "lucide-react";
import { BLOB_BASE, RAW_BASE } from "@/lib/projects/genoxide/github";

/**
 * The benchmark page's charts: the ones the harness drew for the published
 * run (docs/benchmarks/*.svg), read at the pinned commit, each linking to
 * the file. They're drawn on white, so they sit on a white panel in the dark
 * themes too.
 *
 * When the harness publishes its results JSON, `results` arrives here
 * non-null: interactive charts can then replace these images (a client
 * component with recharts, already a dependency), one per `charts` entry.
 *
 * @param {object} props
 * @param {{ id: string, title: string, caption: string, file: string, wide?: boolean }[]} props.charts
 * @param {unknown} props.results  the parsed results JSON, or null
 * @param {string} props.resultsUrl  the tables of the published run
 */
export default function BenchmarkCharts({ charts, resultsUrl }) {
  const path = (file) => file.split("/").map(encodeURIComponent).join("/");
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {charts.map((chart) => (
        <figure
          key={chart.id}
          id={chart.id}
          className={`proj-card flex flex-col p-5 ${chart.wide ? "md:col-span-2" : ""}`}
        >
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
      <p className="text-base-content/70 text-sm md:col-span-2">
        Every number in these charts is in the{" "}
        <a
          href={resultsUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 font-medium text-primary underline-offset-2 hover:underline"
        >
          results tables
          <ExternalLink size={12} aria-hidden />
        </a>
        .
      </p>
    </div>
  );
}
