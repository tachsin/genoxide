"use client";

import { Pause, Play, RotateCcw, StepBack, StepForward } from "lucide-react";
import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from "react";
import FitnessCurve, { curveOf } from "./FitnessCurve";
import { formatValue, useDarkScheme, useReducedMotion } from "./chart-kit";

// One module per plot kind, loaded on demand: a page downloads only its own.
const PLOTS = {
  bits: lazy(() => import("./plots/BitsPlot")),
  knapsack: lazy(() => import("./plots/KnapsackPlot")),
  board: lazy(() => import("./plots/BoardPlot")),
  tour: lazy(() => import("./plots/TourPlot")),
  gantt: lazy(() => import("./plots/GanttPlot")),
  contour: lazy(() => import("./plots/ContourPlot")),
  "multi-curve": lazy(() => import("./plots/MultiCurvePlot")),
  design: lazy(() => import("./plots/DesignPlot")),
  "front-2d": lazy(() => import("./plots/Front2dPlot")),
  "front-3d": lazy(() => import("./plots/Front3dPlot")),
  surface: lazy(() => import("./plots/SurfacePlot")),
  timeline: lazy(() => import("./plots/TimelinePlot")),
};

const TITLES = {
  bits: "The population, a row per individual",
  knapsack: "The best packing",
  board: "The best placement",
  tour: "The best tour",
  gantt: "The best schedule",
  contour: "The population on the function",
  "multi-curve": "Error per function",
  design: "The best design",
  "front-2d": "The front",
  "front-3d": "The front",
  surface: "The network's output",
  timeline: "Evaluations over time",
};

const SPEEDS = [0.5, 1, 2, 4];

/** Seconds a whole run takes at 1×: about 20 frames a second, between 4 and 10 seconds. */
function runSeconds(frames) {
  return Math.min(10, Math.max(4, frames / 20));
}

function PlotFallback() {
  return <div className="h-72 animate-pulse rounded-lg bg-base-content/5 motion-reduce:animate-none" />;
}

export default function PlayerView({ trace }) {
  const frames = trace.frames;
  const last = frames.length - 1;
  const reduced = useReducedMotion();
  const dark = useDarkScheme();
  const curve = useMemo(() => curveOf(trace), [trace]);
  const Plot = PLOTS[trace.plot] ?? null;

  // Reduced motion: no autoplay, the last frame (the result) shows.
  const [index, setIndex] = useState(() =>
    window.matchMedia("(prefers-reduced-motion: reduce)").matches ? last : 0,
  );
  const [playing, setPlaying] = useState(() => last > 0 && !window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  const [speed, setSpeed] = useState(1);
  // the slider announces its own value: the readout stays quiet while it has focus
  const [sliderFocused, setSliderFocused] = useState(false);
  const position = useRef(index);
  const rootRef = useRef(null);
  const autoPaused = useRef(false);

  const seek = useCallback(
    (i) => {
      const next = Math.min(last, Math.max(0, Math.round(i)));
      position.current = next;
      setIndex(next);
    },
    [last],
  );

  const playingRef = useRef(playing);
  playingRef.current = playing;

  const toggle = useCallback(() => {
    autoPaused.current = false;
    if (playingRef.current) {
      setPlaying(false);
      return;
    }
    if (position.current >= last) seek(0);
    setPlaying(true);
  }, [last, seek]);

  // The clock: advances a fractional position every animation frame and
  // renders only when it reaches a new recorded frame.
  useEffect(() => {
    if (!playing) return undefined;
    let raf = 0;
    let previous = performance.now();
    const perSecond = (frames.length / runSeconds(frames.length)) * speed;
    const tick = (now) => {
      // rAF's timestamp can precede `previous` on the first frame
      const dt = Math.min(0.1, Math.max(0, (now - previous) / 1000));
      previous = now;
      position.current = Math.min(last, Math.max(0, position.current + dt * perSecond));
      setIndex(Math.floor(position.current));
      if (position.current >= last) {
        setPlaying(false);
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, last, frames.length]);

  // Pause while scrolled out of view, and carry on when back.
  useEffect(() => {
    const el = rootRef.current;
    if (!el) return undefined;
    const observer = new IntersectionObserver((entries) => {
      const visible = entries.some((e) => e.isIntersecting);
      if (!visible && playingRef.current) {
        autoPaused.current = true;
        setPlaying(false);
      } else if (visible && autoPaused.current) {
        autoPaused.current = false;
        setPlaying(true);
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Reduced motion switched on mid-run: stop.
  useEffect(() => {
    if (reduced) setPlaying(false);
  }, [reduced]);

  function onKeyDown(event) {
    if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
    const target = event.target;
    const onRange = target instanceof HTMLInputElement && target.type === "range";
    const onButton = target instanceof HTMLButtonElement;
    if ((event.key === " " && !onButton) || event.key === "k" || event.key === "K") {
      event.preventDefault();
      toggle();
    } else if (!onRange && (event.key === "ArrowLeft" || event.key === ",")) {
      event.preventDefault();
      setPlaying(false);
      seek(position.current - 1);
    } else if (!onRange && (event.key === "ArrowRight" || event.key === ".")) {
      event.preventDefault();
      setPlaying(false);
      seek(position.current + 1);
    } else if (!onRange && event.key === "Home") {
      event.preventDefault();
      seek(0);
    } else if (!onRange && event.key === "End") {
      event.preventDefault();
      seek(last);
    }
  }

  const frame = frames[index];
  const atEnd = index >= last;
  // A frame is a generation, or (runs without one, e.g. several algorithms
  // side by side) just a recorded point of the run.
  const byGeneration = typeof frame.generation === "number" && typeof frames[last].generation === "number";
  const unit = byGeneration ? "generation" : "frame";
  const where = byGeneration
    ? `Generation ${formatValue(frame.generation)} of ${formatValue(frames[last].generation)}`
    : `Frame ${index + 1} of ${frames.length}`;
  const details = [
    where,
    typeof frame.evaluations === "number" ? `${formatValue(frame.evaluations)} evaluations` : null,
    typeof frame.seconds === "number" ? `${formatValue(Number(frame.seconds.toPrecision(3)))} s` : null,
    typeof frame.best === "number" ? `${trace.y_label ?? "best"} ${formatValue(frame.best)}` : null,
    // multi-objective: the curve's own values (hypervolume per algorithm)
    ...(typeof frame.best !== "number" && curve
      ? curve.series.map((s) => (typeof s.values[index] === "number" ? `${s.label} ${formatValue(s.values[index])}` : null))
      : []),
  ].filter(Boolean);

  const plotPanel = Plot ? (
    <section aria-label={TITLES[trace.plot]} className="min-w-0">
      <h3 className="mb-2 font-medium text-base-content/80 text-sm">{TITLES[trace.plot]}</h3>
      <Suspense fallback={<PlotFallback />}>
        <Plot trace={trace} frame={frame} index={index} dark={dark} reduced={reduced} />
      </Suspense>
    </section>
  ) : null;

  const curvePanel = curve ? (
    <section aria-label={`${curve.yLabel} over the run`} className="flex min-w-0 flex-col">
      <h3 className="mb-2 font-medium text-base-content/80 text-sm first-letter:uppercase">{curve.yLabel}</h3>
      <div className="min-h-[15rem] flex-1">
        <FitnessCurve
          curve={curve}
          index={index}
          dark={dark}
          onSeek={(i) => {
            setPlaying(false);
            seek(i);
          }}
        />
      </div>
    </section>
  ) : null;

  return (
    <div ref={rootRef} onKeyDown={onKeyDown}>
      <div className={plotPanel && curvePanel ? "grid gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]" : "grid"}>
        {plotPanel}
        {curvePanel}
      </div>

      <div className="mt-5 flex flex-wrap items-center gap-x-3 gap-y-2 border-base-content/10 border-t pt-4">
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={toggle}
            className="btn btn-primary btn-sm btn-circle"
            aria-label={playing ? "Pause" : atEnd ? "Replay" : "Play"}
            title={playing ? "Pause (Space)" : atEnd ? "Replay (Space)" : "Play (Space)"}
          >
            {playing ? <Pause size={16} aria-hidden /> : atEnd ? <RotateCcw size={16} aria-hidden /> : <Play size={16} aria-hidden />}
          </button>
          <button
            type="button"
            onClick={() => {
              if (index === 0) return;
              setPlaying(false);
              seek(position.current - 1);
            }}
            // aria-disabled, not disabled: a focused button that disables itself would drop the
            // focus to the page, and with it the player's keyboard shortcuts
            aria-disabled={index === 0}
            className={`btn btn-ghost btn-sm btn-square ${index === 0 ? "opacity-40" : ""}`}
            aria-label={`Previous ${unit}`}
            title={`Previous ${unit} (←)`}
          >
            <StepBack size={16} aria-hidden />
          </button>
          <button
            type="button"
            onClick={() => {
              if (atEnd) return;
              setPlaying(false);
              seek(position.current + 1);
            }}
            aria-disabled={atEnd}
            className={`btn btn-ghost btn-sm btn-square ${atEnd ? "opacity-40" : ""}`}
            aria-label={`Next ${unit}`}
            title={`Next ${unit} (→)`}
          >
            <StepForward size={16} aria-hidden />
          </button>
        </div>

        <input
          type="range"
          min={0}
          max={last}
          step={1}
          value={index}
          onChange={(event) => {
            setPlaying(false);
            seek(Number(event.target.value));
          }}
          onFocus={() => setSliderFocused(true)}
          onBlur={() => setSliderFocused(false)}
          aria-label={byGeneration ? "Generation" : "Frame"}
          aria-valuetext={where}
          className="h-1.5 min-w-40 flex-1 cursor-pointer accent-primary"
        />

        <div role="group" aria-label="Speed" className="join">
          {SPEEDS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setSpeed(s)}
              aria-pressed={speed === s}
              className={`btn join-item btn-xs tabular-nums ${speed === s ? "btn-active" : "btn-ghost"}`}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>

      <p className="mt-2 text-base-content/65 text-xs tabular-nums" aria-live={playing || sliderFocused ? "off" : "polite"}>
        {details.join(" · ")}
      </p>
    </div>
  );
}
