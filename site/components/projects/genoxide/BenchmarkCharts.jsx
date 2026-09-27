import { ChartLine } from "lucide-react";

// Decorative bar heights for the empty frames (percent). Not data.
const SKELETON = [38, 62, 47, 80, 55, 70, 44, 66];

/**
 * The benchmark page's charts.
 *
 * Today there is no results file to read (lib/projects/genoxide/benchmarks.js
 * BENCHMARK_RESULTS_URL is null), so each chart renders as an empty frame
 * with a pointer to the published numbers. When the harness publishes its
 * results JSON, `results` arrives here non-null: render the interactive
 * charts from it (a client component with recharts, already a dependency),
 * one per `charts` entry, keeping this server wrapper for the frames.
 *
 * @param {object} props
 * @param {{ id: string, title: string, caption: string }[]} props.charts
 * @param {unknown} props.results  the parsed results JSON, or null
 * @param {string} props.resultsUrl  where the numbers are today
 */
export default function BenchmarkCharts({ charts, results, resultsUrl }) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {charts.map((chart) => (
        <figure key={chart.id} id={chart.id} className="proj-card flex flex-col p-5">
          <div className="flex items-center justify-between gap-3">
            <figcaption className="font-semibold tracking-tight">{chart.title}</figcaption>
            <ChartLine size={16} aria-hidden className="text-base-content/40" />
          </div>
          <p className="proj-lead mt-1 text-sm">{chart.caption}</p>

          <div className="relative mt-5">
            <div className="proj-chart-skeleton" aria-hidden>
              {SKELETON.map((h) => (
                <span key={h} style={{ height: `${h}%` }} />
              ))}
            </div>
            <div className="absolute inset-0 flex items-center justify-center p-4 text-center">
              <p className="rounded-lg bg-base-100/85 px-3 py-2 text-base-content/70 text-sm backdrop-blur-sm">
                {results ? "Interactive chart coming with the next results file." : "Interactive chart in preparation."}{" "}
                <a
                  href={resultsUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-medium text-primary underline-offset-2 hover:underline"
                >
                  Latest numbers
                </a>
              </p>
            </div>
          </div>
        </figure>
      ))}
    </div>
  );
}
