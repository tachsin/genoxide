"use client";

import { ExternalLink } from "lucide-react";
import { Suspense, lazy, useEffect, useRef, useState } from "react";

// The player's code (controls, curve, plots) loads with the trace, only once
// the player is about to scroll into view: a reader who never gets there
// downloads neither.
const PlayerView = lazy(() => import("./PlayerView"));

/** A trace this player can show: the shared format of examples/<name>/trace.json. */
function isTrace(value) {
  return (
    value &&
    typeof value === "object" &&
    value.format === 1 &&
    typeof value.plot === "string" &&
    Array.isArray(value.frames) &&
    value.frames.length > 0
  );
}

function Placeholder({ children }) {
  return (
    <div className="flex min-h-[26rem] items-center justify-center rounded-xl border border-base-content/10 border-dashed p-6 text-center text-base-content/60 text-sm">
      {children}
    </div>
  );
}

/**
 * An example's recorded run, played back: the solution plot and the
 * fitness curve, with play, pause, a scrubber and the speed. One component
 * for every example; the trace's `plot` picks the solution plot.
 *
 * @param {object} props
 * @param {string} props.traceUrl   the example's trace.json (raw GitHub URL)
 * @param {string} [props.sourceUrl]  the same file on GitHub, for the error message
 */
export default function ExamplePlayer({ traceUrl, sourceUrl }) {
  const ref = useRef(null);
  const [near, setNear] = useState(false);
  const [state, setState] = useState({ status: "idle", trace: null });

  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setNear(true);
          observer.disconnect();
        }
      },
      { rootMargin: "300px 0px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!near) return undefined;
    const controller = new AbortController();
    setState({ status: "loading", trace: null });
    fetch(traceUrl, { signal: controller.signal })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((trace) => {
        if (!isTrace(trace)) throw new Error("not a trace");
        setState({ status: "ready", trace });
      })
      .catch((error) => {
        if (error?.name !== "AbortError") setState({ status: "error", trace: null });
      });
    return () => controller.abort();
  }, [near, traceUrl]);

  return (
    <div ref={ref} className="proj-card p-4 sm:p-5">
      {state.status === "ready" ? (
        <Suspense fallback={<Placeholder>Loading the player…</Placeholder>}>
          <PlayerView trace={state.trace} />
        </Suspense>
      ) : state.status === "error" ? (
        <Placeholder>
          <span>
            The recorded run couldn't be loaded right now.
            {sourceUrl ? (
              <>
                {" "}
                <a href={sourceUrl} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 font-medium text-primary underline-offset-2 hover:underline">
                  trace.json on GitHub
                  <ExternalLink size={12} aria-hidden />
                </a>
              </>
            ) : null}
          </span>
        </Placeholder>
      ) : (
        <Placeholder>
          <span className="animate-pulse motion-reduce:animate-none">Loading the recorded run…</span>
        </Placeholder>
      )}
    </div>
  );
}
