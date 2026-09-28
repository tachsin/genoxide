"use client";

import { ExternalLink, FileText, MessageSquarePlus, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import CopyButton from "@/components/projects/CopyButton";
import { formatValue } from "../player/chart-kit";
import { exactValue, percent } from "./format";
import { runHash, useHighlight } from "./Highlight";

/**
 * The page's one details panel, below the benchmark charts: the run selected
 * by a click on a bar (or Enter on it), or by the URL's `#run=` hash. It
 * reads that library's method in that scenario from `endpoint` (the page's
 * route, which reads the run details of the pinned commit and highlights the
 * code on the server) and shows the scenario's settings, the summary the bar
 * shows, a row per seed, the JSON line each run printed, and the adapter's
 * code that set the method up, with a link to suggest an improvement to it as
 * a GitHub issue. A selection scrolls here (smoothly unless reduced motion is
 * asked for) and moves the focus to the heading.
 *
 * @param {object} props
 * @param {string} props.endpoint  the route's URL, with its query string started (`?commit=…`)
 * @param {string} props.issuesUrl  where a new issue of the repository is opened
 * @param {string} props.methodologyUrl  benchmarks/README.md, for the matched and idiomatic settings
 */

const count = new Intl.NumberFormat("en");
const seconds = (value) => exactValue(value, "seconds");
const HEADING = "Runs, output and code";

export default function RunDetails({ endpoint, issuesUrl, methodologyUrl }) {
  const { run, select, scrollRequest } = useHighlight();
  const [state, setState] = useState({ key: null, status: "idle", data: null, error: null });
  const cache = useRef(new Map());
  const section = useRef(null);
  const heading = useRef(null);
  const key = run ? `${run.scenario}/${run.library}/${run.solver}` : null;

  useEffect(() => {
    if (!run) {
      setState({ key: null, status: "idle", data: null, error: null });
      return undefined;
    }
    const cached = cache.current.get(key);
    if (cached) {
      setState({ key, status: "ready", data: cached, error: null });
      return undefined;
    }
    const controller = new AbortController();
    setState({ key, status: "loading", data: null, error: null });
    const query = new URLSearchParams({ scenario: run.scenario, library: run.library, solver: run.solver });
    fetch(`${endpoint}${endpoint.includes("?") ? "&" : "?"}${query}`, { signal: controller.signal })
      .then(async (response) => {
        const body = await response.json().catch(() => null);
        if (!response.ok || !body || body.error) throw new Error(body?.error ?? "The runs couldn't be loaded.");
        cache.current.set(key, body);
        setState({ key, status: "ready", data: body, error: null });
      })
      .catch((error) => {
        if (error.name !== "AbortError") setState({ key, status: "error", data: null, error: error.message });
      });
    return () => controller.abort();
  }, [key, run, endpoint]);

  // a selection brings the panel into view, and the focus to its heading
  useEffect(() => {
    if (!scrollRequest || !section.current) return;
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    section.current.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
    heading.current?.focus({ preventScroll: true });
  }, [scrollRequest]);

  const data = state.key === key ? state.data : null;
  const title = data
    ? `${data.method.name} ${data.method.method} on ${data.scenario.title}`
    : run
      ? `${run.library} ${run.solver} on ${run.scenario}`
      : HEADING;

  return (
    <section
      ref={section}
      id="run-details"
      aria-labelledby="run-details-title"
      aria-busy={state.status === "loading"}
      className="proj-card min-w-0 scroll-mt-24 p-4 sm:p-5"
    >
      <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
        <div className="min-w-0">
          {run ? <p className="text-base-content/55 text-xs uppercase tracking-wide">{HEADING}</p> : null}
          <h3
            ref={heading}
            id="run-details-title"
            tabIndex={-1}
            className="break-words font-semibold tracking-tight outline-none focus-visible:ring-2 focus-visible:ring-primary/60"
          >
            {title}
          </h3>
          {data ? (
            <p className="mt-0.5 text-base-content/60 text-xs">
              {data.method.version} ({data.method.language}) · method <code>{data.method.solver}</code> · the published
              run of {data.run.date}
            </p>
          ) : null}
        </div>
        {run ? (
          <button
            type="button"
            onClick={() => {
              select(null);
              // the button goes: the focus stays in the panel
              heading.current?.focus();
            }}
            className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-base-content/60 text-xs hover:text-primary"
          >
            <X size={12} aria-hidden />
            Close
          </button>
        ) : null}
      </div>

      {!run ? (
        <p className="proj-lead mt-1 text-sm">
          Select a bar of a method in any chart above, or focus a chart, move to a bar with the arrow keys and press
          Enter: its runs in that scenario, what they printed and the adapter's code that set the method up show here.
        </p>
      ) : state.status === "error" && state.key === key ? (
        <p className="mt-3 text-sm" role="alert">
          {state.error}
        </p>
      ) : !data ? (
        <p className="mt-3 text-base-content/60 text-sm">Loading the runs…</p>
      ) : (
        <Details data={data} run={run} issuesUrl={issuesUrl} methodologyUrl={methodologyUrl} />
      )}
    </section>
  );
}

function Details({ data, run, issuesUrl, methodologyUrl }) {
  const { scenario, method, code, links } = data;
  const [pageUrl, setPageUrl] = useState(null);
  useEffect(() => {
    setPageUrl(`${window.location.origin}${window.location.pathname}${runHash(run)}`);
  }, [run]);

  return (
    <div className="mt-4 space-y-6">
      <div className="flex flex-wrap gap-2">
        <a
          href={issueUrl({ issuesUrl, data, pageUrl })}
          target="_blank"
          rel="noopener noreferrer"
          className="btn btn-sm btn-primary"
        >
          <MessageSquarePlus size={15} aria-hidden />
          Suggest an improvement
        </a>
        {links.page ? (
          <a
            href={links.page}
            target="_blank"
            rel="noopener noreferrer"
            title="Its settings, versions and known issues in the benchmarks"
            className="btn btn-sm btn-ghost border-base-content/15"
          >
            <FileText size={15} aria-hidden />
            {method.name}'s page
            <ExternalLink size={12} aria-hidden className="text-base-content/45" />
          </a>
        ) : null}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <h4 className="font-medium text-sm">The scenario</h4>
          <Facts rows={scenarioRows(scenario, methodologyUrl)} />
        </div>
        <div>
          <h4 className="font-medium text-sm">The summary</h4>
          <Facts rows={summaryRows(method, scenario)} />
          {method.parts?.length ? <Parts parts={method.parts} /> : null}
        </div>
      </div>

      <div>
        <h4 className="font-medium text-sm">Each seed</h4>
        <SeedsTable runs={method.runs} caption={`${method.label}, ${scenario.title}: each seed's run`} />
      </div>

      <details className="group">
        <summary className="cursor-pointer text-base-content/75 text-sm hover:text-primary">
          The output: the JSON line each run printed ({method.runs.length})
        </summary>
        <div className="proj-code-group mt-3">
          <div className="proj-code-meta">
            <span className="proj-code-file">stdout of the adapter, a line per run</span>
          </div>
          <div className="proj-code-body" data-copy-root>
            <CopyButton label="Copy the output" />
            {/* highlighted on the server from the run details (getBenchmarkRun) */}
            <div className="proj-code [&_pre]:max-h-[24rem]" dangerouslySetInnerHTML={{ __html: data.output }} />
          </div>
        </div>
      </details>

      <div>
        <h4 className="font-medium text-sm">The code</h4>
        <p className="mt-1 mb-3 text-base-content/60 text-xs">
          Where the adapter sets {method.method} up and runs it
          {links.commit ? (
            <>
              , at commit <code>{links.commit.slice(0, 7)}</code>
            </>
          ) : null}
          . The runs above are of {method.name} {method.version}.
        </p>
        {code.length ? (
          <div className="space-y-4">
            {code.map((block, k) => (
              <CodeBlock key={`${block.path}/${k}`} block={block} />
            ))}
          </div>
        ) : (
          <p className="text-sm">
            The harness has no code map for this method in this scenario
            {links.adapter ? (
              <>
                :{" "}
                <a href={links.adapter} target="_blank" rel="noopener noreferrer" className="text-primary underline-offset-2 hover:underline">
                  the adapter
                </a>{" "}
                has it
              </>
            ) : null}
            .
          </p>
        )}
      </div>
    </div>
  );
}

/** A definition list of [label, value, note?] rows. */
function Facts({ rows }) {
  return (
    <dl className="mt-2 grid grid-cols-[minmax(0,auto)_minmax(0,1fr)] gap-x-4 gap-y-1 text-sm">
      {rows.map(([label, value, note]) => (
        <div key={label} className="contents">
          <dt className="text-base-content/60">{label}</dt>
          <dd className="min-w-0 break-words tabular-nums">
            {value}
            {note ? <span className="block text-base-content/55 text-xs">{note}</span> : null}
          </dd>
        </div>
      ))}
    </dl>
  );
}

const MODES = {
  matched: "matched: the same algorithm and settings in every library that has them",
  idiomatic: "idiomatic: what the library's documentation recommends",
};

function scenarioRows(scenario, methodologyUrl) {
  const size =
    scenario.problem === "onemax"
      ? `${count.format(scenario.size)} bits`
      : scenario.problem === "nqueens"
        ? `${scenario.size} queens`
        : `${scenario.size} variables`;
  return [
    ["Problem", `${scenario.problem_name}, ${size}`],
    [
      "Mode",
      MODES[scenario.mode] ?? scenario.mode,
      methodologyUrl ? (
        <a href={`${methodologyUrl}#scenarios`} target="_blank" rel="noopener noreferrer" className="underline-offset-2 hover:text-primary hover:underline">
          the scenarios and matched settings
        </a>
      ) : null,
    ],
    ["Budget", typeof scenario.budget === "number" ? `${count.format(scenario.budget)} evaluations` : "–"],
    ["Time cap", typeof scenario.cap === "number" ? `${formatValue(scenario.cap)} s` : "–"],
    ["Target", typeof scenario.target === "number" ? (scenario.problem === "onemax" ? `${scenario.target} (all ones)` : `≤ ${formatValue(scenario.target)}`) : "–"],
    ["Seeds", scenario.seeds?.length ? `${scenario.seeds.length} (${scenario.seeds[0]} to ${scenario.seeds.at(-1)})` : "–"],
  ];
}

function cappedText(summary) {
  if (!summary?.capped) return "none";
  return (
    `${summary.capped} of ${summary.runs}` +
    (typeof summary.capped_share === "number" ? `, after a median ${percent(summary.capped_share)} of the budget` : "")
  );
}

function summaryRows(method, scenario) {
  const s = method.summary;
  if (!s) return [["Runs", "every run failed a check of the rules: none is summarized"]];
  const tooFew = `too few runs reached the target (${s.reached}/${s.runs})`;
  const gap = (value) => (scenario.problem === "onemax" ? count.format(value) : formatValue(value));
  return [
    ["Reached the target", `${s.reached} of ${s.runs} runs`],
    ["Expected time to target", typeof s.ert_time === "number" ? seconds(s.ert_time) : tooFew],
    ["Expected evaluations", typeof s.ert_evaluations === "number" ? count.format(Math.round(s.ert_evaluations)) : tooFew],
    ["Median evaluations", count.format(Math.round(s.median_evaluations))],
    [
      "Distance to the optimum",
      `${gap(s.median_gap)} median`,
      `${gap(s.best_gap)} best, ${gap(s.worst_gap)} worst, at the end of the runs`,
    ],
    ["Evaluations per second", count.format(Math.round(s.evaluations_per_second))],
    ["Stopped by the time cap", cappedText(s)],
  ];
}

/** The runs the time cap stopped and the others, summarized apart, as the charts show them. */
function Parts({ parts }) {
  return (
    <ul className="mt-2 space-y-0.5 text-base-content/65 text-xs">
      {parts.map((part) => (
        <li key={String(part.ended_on_cap)}>
          {part.ended_on_cap ? "Stopped by the time cap" : "The others"} ({part.runs}): reached {part.reached} of {part.runs},
          median distance {formatValue(part.median_gap)}
        </li>
      ))}
    </ul>
  );
}

function SeedsTable({ runs, caption }) {
  return (
    <div className="mt-2 max-w-full overflow-x-auto">
      <table className="w-full min-w-[30rem] text-left text-xs">
        <caption className="sr-only">{caption}</caption>
        <thead className="text-base-content/55">
          <tr>
            <th scope="col" className="py-1 pr-3 font-normal">
              Seed
            </th>
            <th scope="col" className="py-1 pr-3 text-right font-normal">
              Time
            </th>
            <th scope="col" className="py-1 pr-3 text-right font-normal">
              Evaluations
            </th>
            <th scope="col" className="py-1 pr-3 font-normal">
              Outcome
            </th>
            <th scope="col" className="py-1 pr-3 text-right font-normal">
              Best
            </th>
            <th scope="col" className="py-1 font-normal">
              Notes
            </th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run, k) => (
            <tr key={`${run.seed}/${k}`} className={`border-base-content/10 border-t ${run.invalid ? "text-base-content/55" : ""}`}>
              <th scope="row" className="py-1 pr-3 font-normal tabular-nums">
                {run.seed ?? "–"}
              </th>
              <td className="whitespace-nowrap py-1 pr-3 text-right tabular-nums">{seconds(run.time_s)}</td>
              <td className="whitespace-nowrap py-1 pr-3 text-right tabular-nums">
                {typeof run.evaluations === "number" ? count.format(run.evaluations) : "–"}
              </td>
              <td className="whitespace-nowrap py-1 pr-3 tabular-nums">{outcome(run)}</td>
              <td className="whitespace-nowrap py-1 pr-3 text-right tabular-nums">{typeof run.best === "number" ? formatValue(run.best) : "–"}</td>
              <td className="py-1 text-base-content/65">{notes(run).join("; ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function outcome(run) {
  if (run.reached) {
    const hit = run.first_hit;
    return hit ? `reached at ${count.format(hit.evaluations)} evals, ${seconds(hit.time_s)}` : "reached";
  }
  return "not reached";
}

function notes(run) {
  const out = [];
  if (run.capped) out.push("stopped by the time cap");
  if (run.invalid) out.push(`invalid, left out: ${run.invalid.join("; ")}`);
  return out;
}

function CodeBlock({ block }) {
  const range = block.start ? `lines ${block.start}–${block.end}` : null;
  const partial = block.shown < block.lines;
  return (
    <div className="proj-code-group">
      <div className="proj-code-head">
        <span className="proj-lang-single">{block.language}</span>
        {range ? <span className="proj-code-note">{range}</span> : null}
      </div>
      <div className="proj-code-meta">
        <span className="proj-code-file">{block.filename}</span>
        <a href={block.href} className="proj-code-source" target="_blank" rel="noopener noreferrer">
          View on GitHub
          <ExternalLink size={12} aria-hidden />
        </a>
      </div>
      <div className="proj-code-body" data-copy-root>
        <CopyButton />
        {/* highlighted on the server from the run details (getBenchmarkRun) */}
        <div className="proj-code [&_pre]:max-h-[32rem]" dangerouslySetInnerHTML={{ __html: block.html }} />
      </div>
      {partial || !block.current ? (
        <p className="border-base-content/10 border-t px-3.5 py-2 text-base-content/60 text-xs">
          {partial ? `The first ${block.shown} of its ${block.lines} lines: the rest is on GitHub. ` : ""}
          {!block.current
            ? `The adapter has changed since the run details were written: this is the code at commit ${block.commit.slice(0, 7)}.`
            : ""}
        </p>
      ) : null}
    </div>
  );
}

/**
 * A new issue about this method's code, prefilled: the title, the label and
 * a body with the scenario, the version, the numbers and the links (the code
 * at its commit, the library's page, these runs on the page), and headings
 * for the reporter to fill in. No code in the URL, only links, so it stays
 * well under GitHub's limit on its length.
 */
function issueUrl({ issuesUrl, data, pageUrl }) {
  const { scenario, method, code, links, run } = data;
  const s = method.summary;
  const settings = [
    typeof scenario.budget === "number" ? `budget ${count.format(scenario.budget)} evaluations` : null,
    typeof scenario.cap === "number" ? `time cap ${formatValue(scenario.cap)} s` : null,
    typeof scenario.target === "number" ? `target ${formatValue(scenario.target)}` : null,
    `${scenario.seeds?.length ?? method.runs.length} seeds`,
  ].filter(Boolean);
  let numbers = "none summarized: every run failed a check of the rules";
  if (s) {
    numbers =
      `reached the target in ${s.reached} of ${s.runs} runs` +
      (typeof s.ert_time === "number" ? `, expected time to target ${seconds(s.ert_time)}` : "") +
      (typeof s.ert_evaluations === "number" ? `, expected evaluations ${count.format(Math.round(s.ert_evaluations))}` : "") +
      `, median evaluations ${count.format(Math.round(s.median_evaluations))}, median distance to the optimum ${formatValue(s.median_gap)}` +
      (s.capped ? `, ${s.capped} of ${s.runs} runs stopped by the time cap` : "");
  }
  const lines = [
    `- Scenario: ${scenario.title} (\`${scenario.key}\`): ${settings.join(", ")}`,
    `- Library: ${method.name} ${method.version} (${method.language})`,
    `- Method: ${method.method} (\`${method.solver}\`)`,
    `- Results, the published run of ${run.date}: ${numbers}`,
    ...(code.length ? code.map((block, k) => `- Code${code.length > 1 ? ` (${k + 1} of ${code.length})` : ""}: ${block.href}`) : links.adapter ? [`- Adapter: ${links.adapter}`] : []),
    ...(links.page ? [`- Library page: ${links.page}`] : []),
    ...(pageUrl ? [`- The runs on the benchmarks page: ${pageUrl}`] : []),
    "",
    "## What could be better",
    "",
    "",
    "## Evidence (a run, a reference, the library's docs)",
    "",
    "",
  ];
  const query = new URLSearchParams({
    title: `Benchmarks: ${method.name} ${method.method} on ${scenario.title}`,
    labels: "benchmarks",
    body: lines.join("\n"),
  });
  return `${issuesUrl}?${query}`;
}
