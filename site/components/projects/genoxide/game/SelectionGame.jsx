"use client";

import { Bot, RotateCcw, Trophy, X } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { MATCH, randomSeed } from "./genome";
import { observeTheme, readTheme } from "./render";
import { World } from "./world";

const BEST_KEY = "genoxide-selection-best";
const START = 0.6;

function readBest() {
  try {
    const value = Number(window.localStorage.getItem(BEST_KEY));
    return Number.isFinite(value) && value > 0 ? value : 0;
  } catch {
    return 0;
  }
}

function writeBest(value) {
  try {
    window.localStorage.setItem(BEST_KEY, String(value));
  } catch {
    // private mode or storage blocked: the best score just isn't kept
  }
}

function Stat({ label, value }) {
  return (
    <div className="min-w-[3.25rem]">
      <dt className="text-[0.65rem] text-base-content/55 uppercase tracking-wider">{label}</dt>
      <dd className="font-mono font-semibold text-base tabular-nums">{value}</dd>
    </div>
  );
}

/**
 * The game: a full-screen dialog over the page. The visitor clicks the
 * creatures that look most like the target; each pair breeds two children
 * (two-point crossover and mutation, drawn as their genes), and the two
 * least like the target fade out. "Let genoxide select" runs tournament
 * selection instead, at a speed you can follow.
 */
export default function SelectionGame({ reduced, onClose }) {
  const dialogRef = useRef(null);
  const canvasRef = useRef(null);
  const targetRef = useRef(null);
  const hudRef = useRef(null);
  const hintRef = useRef(null);
  const worldRef = useRef(null);
  const resultRef = useRef(null);
  const winnerRef = useRef(null);
  const resultTargetRef = useRef(null);
  const records = useRef(new Map());
  const titleId = useId();
  const helpId = useId();
  const resultTitleId = useId();
  const [stats, setStats] = useState(null);
  const [result, setResult] = useState(null);
  const [announce, setAnnounce] = useState("");

  useEffect(() => {
    const canvas = canvasRef.current;
    const onRoundEnd = (round) => {
      const record = { ...records.current.get(round.seed) };
      if (!round.assisted) record.you = round.generations;
      else if (round.clicks === 0) record.genoxide = round.generations;
      records.current.set(round.seed, record);
      let best = readBest();
      let newBest = false;
      if (!round.assisted && round.score > best) {
        best = round.score;
        newBest = true;
        writeBest(best);
      }
      setResult({ ...round, best, newBest, record });
      setAnnounce(
        round.assisted
          ? `genoxide matched the target in ${round.generations} generations.`
          : `Matched! Evolved in ${round.generations} generations and ${round.clicks} clicks, score ${round.score}.`,
      );
    };
    const world = new World(canvas, { mode: "play", reduced, onStats: setStats, onRoundEnd });
    worldRef.current = world;
    world.setTheme(readTheme(canvas));
    world.setTargetCanvas(targetRef.current);

    const measure = () => {
      const rect = canvas.getBoundingClientRect();
      world.resize(rect.width, rect.height, Math.min(window.devicePixelRatio || 1, 2));
      const hud = hudRef.current?.getBoundingClientRect();
      const hint = hintRef.current?.getBoundingClientRect();
      world.setInsets(hud ? hud.bottom : 0, hint ? rect.bottom - hint.top : 0);
    };
    measure();
    world.newRound(randomSeed());
    const resizeObserver = new ResizeObserver(measure);
    resizeObserver.observe(canvas);
    if (hudRef.current) resizeObserver.observe(hudRef.current);
    const sync = () => (document.hidden ? world.stop() : world.start());
    document.addEventListener("visibilitychange", sync);
    const unobserveTheme = observeTheme(() => world.setTheme(readTheme(canvas)));
    sync();
    return () => {
      world.stop();
      worldRef.current = null;
      resizeObserver.disconnect();
      document.removeEventListener("visibilitychange", sync);
      unobserveTheme();
    };
  }, [reduced]);

  // the page behind doesn't scroll, and the focus starts on the creatures
  useEffect(() => {
    const root = document.documentElement;
    const overflow = root.style.overflow;
    root.style.overflow = "hidden";
    canvasRef.current?.focus({ preventScroll: true });
    return () => {
      root.style.overflow = overflow;
    };
  }, []);

  useEffect(() => {
    worldRef.current?.setResultCanvases(resultTargetRef.current, winnerRef.current);
    if (result) resultRef.current?.querySelector("button")?.focus();
  }, [result]);

  useEffect(() => {
    const onEscape = (event) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      onClose();
    };
    document.addEventListener("keydown", onEscape);
    return () => document.removeEventListener("keydown", onEscape);
  }, [onClose]);

  function onKeyDown(event) {
    if (event.key !== "Tab") return;
    // keep the focus inside the dialog
    const focusable = dialogRef.current.querySelectorAll("button:not([disabled]), [tabindex='0']");
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  function onCanvasKeyDown(event) {
    const world = worldRef.current;
    if (!world) return;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") world.moveFocus(1);
    else if (event.key === "ArrowLeft" || event.key === "ArrowUp") world.moveFocus(-1);
    else if (event.key === "Enter" || event.key === " ") world.chooseFocus();
    else return;
    event.preventDefault();
  }

  function onPointerDown(event) {
    const world = worldRef.current;
    if (!world || event.button > 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    world.clickAt(event.clientX - rect.left, event.clientY - rect.top, event.pointerType === "touch" ? 16 : 6);
  }

  function onPointerMove(event) {
    const world = worldRef.current;
    if (!world || event.pointerType !== "mouse") return;
    const rect = event.currentTarget.getBoundingClientRect();
    const over = !world.auto && !world.over && world.pick(event.clientX - rect.left, event.clientY - rect.top, 6);
    event.currentTarget.style.cursor = over ? "pointer" : "default";
  }

  function newRound(seed, auto) {
    const world = worldRef.current;
    if (!world) return;
    setResult(null);
    world.newRound(seed);
    if (auto) world.setAuto(true);
    setAnnounce(auto ? "genoxide is selecting." : "A new target.");
    canvasRef.current?.focus({ preventScroll: true });
  }

  function toggleAuto() {
    const world = worldRef.current;
    if (!world) return;
    world.setAuto(!world.auto);
    setAnnounce(world.auto ? "genoxide is selecting." : "Your turn to select.");
  }

  const auto = stats?.auto ?? false;
  const best = stats?.best ?? START;
  const progress = Math.max(0, Math.min(1, (best - START) / (MATCH - START)));
  const hint = auto
    ? "genoxide is selecting: in each tournament of 3, the creature most like the target wins."
    : stats?.selected
      ? "Now a second parent. Their two children replace the two creatures least like the target."
      : "Click or tap the two creatures that look most like the target to breed them.";

  // on <body>: the hero is its own stacking context, under the sticky header
  return createPortal(
    <div
      ref={dialogRef}
      role="dialog"
      aria-modal="true"
      aria-labelledby={titleId}
      aria-describedby={helpId}
      onKeyDown={onKeyDown}
      lang="en"
      className="fixed inset-0 z-[80] overflow-hidden bg-base-100 text-base-content"
      style={{ fontFamily: "var(--font-geist-sans), ui-sans-serif, system-ui, sans-serif" }}
    >
      <div className="proj-atmosphere" aria-hidden />
      <canvas
        ref={canvasRef}
        tabIndex={0}
        aria-label="Creatures whose looks come from their genes. Arrow keys move between them, Enter selects one."
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onKeyDown={onCanvasKeyDown}
        className="absolute inset-0 size-full touch-none select-none outline-none"
      />

      <div className="pointer-events-none absolute inset-x-0 top-0 p-2 sm:p-4">
        <div
          ref={hudRef}
          className="pointer-events-auto mx-auto flex max-w-5xl flex-wrap items-center gap-x-4 gap-y-2 rounded-2xl border border-base-content/10 bg-base-100/85 p-2 shadow-lg backdrop-blur-md sm:p-3"
        >
          <div className="flex items-center gap-2.5">
            <canvas
              ref={targetRef}
              aria-hidden
              className="h-14 w-24 rounded-xl border border-primary/30 bg-base-content/[0.03] sm:h-16 sm:w-32"
            />
            <div className="text-xs leading-tight">
              <h2 id={titleId} className="font-semibold text-primary uppercase tracking-wider">
                Be the selection
              </h2>
              <p className="mt-0.5 text-base-content/60">Evolve toward this target</p>
            </div>
          </div>

          <div className="min-w-[10rem] flex-1">
            <div className="flex items-baseline justify-between gap-2 whitespace-nowrap text-xs">
              <span className="text-base-content/60">Closest match</span>
              <span className="font-mono font-semibold tabular-nums">
                {Math.floor(best * 100)}% <span className="font-normal text-base-content/50">/ {Math.round(MATCH * 100)}%</span>
              </span>
            </div>
            <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-base-content/10" aria-hidden>
              <div
                className="h-full rounded-full bg-primary transition-[width] duration-500"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
          </div>

          <dl className="flex gap-3 text-center">
            <Stat label="Gen" value={stats?.generation ?? 0} />
            <Stat label="Clicks" value={stats?.clicks ?? 0} />
            <Stat label="Score" value={stats?.score ?? 0} />
          </dl>

          <div className="flex items-center gap-1.5">
            <button
              type="button"
              aria-pressed={auto}
              onClick={toggleAuto}
              disabled={stats?.over}
              className={`btn btn-sm ${auto ? "btn-primary" : "btn-ghost border-base-content/15"}`}
            >
              <Bot size={15} aria-hidden />
              <span className="hidden sm:inline">Let genoxide select</span>
              <span className="sm:hidden">genoxide</span>
            </button>
            <button
              type="button"
              onClick={() => newRound(randomSeed(), false)}
              className="btn btn-sm btn-ghost btn-square border-base-content/15"
              aria-label="New target"
              title="New target"
            >
              <RotateCcw size={15} aria-hidden />
            </button>
            <button
              type="button"
              onClick={onClose}
              className="btn btn-sm btn-ghost btn-square border-base-content/15"
              aria-label="Exit the game (Esc)"
              title="Exit (Esc)"
            >
              <X size={16} aria-hidden />
            </button>
          </div>
        </div>
      </div>

      <p
        ref={hintRef}
        id={helpId}
        className="pointer-events-none absolute inset-x-0 bottom-0 mx-auto max-w-xl px-4 pb-4 text-center text-base-content/65 text-xs sm:text-sm"
      >
        {hint}
      </p>

      {result ? (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center p-4">
          <section
            ref={resultRef}
            aria-labelledby={resultTitleId}
            className="pointer-events-auto w-full max-w-sm rounded-2xl border border-base-content/10 bg-base-100/95 p-6 text-center shadow-2xl backdrop-blur-md"
          >
            <h3 id={resultTitleId} className="inline-flex items-center gap-2 font-semibold text-2xl tracking-tight">
              <Trophy size={20} aria-hidden className="text-primary" />
              {result.assisted ? "genoxide matched it" : "Matched!"}
            </h3>
            <div className="mt-4 grid grid-cols-2 gap-2 text-base-content/60 text-xs" aria-hidden>
              <div>
                <canvas ref={resultTargetRef} className="h-16 w-full rounded-xl border border-base-content/10" />
                <p className="mt-1">Target</p>
              </div>
              <div>
                <canvas ref={winnerRef} className="h-16 w-full rounded-xl border border-primary/40" />
                <p className="mt-1">Evolved, {Math.floor(result.match * 100)}% alike</p>
              </div>
            </div>
            <p className="mt-2 text-base-content/70 text-sm">
              Evolved in <strong className="text-base-content">{result.generations} generations</strong>
              {result.clicks ? ` · ${result.clicks} clicks` : ""} · {Math.round(result.seconds)} s
            </p>
            {result.assisted ? null : (
              <p className="mt-3 font-mono text-lg">
                {result.score} points
                <span className="ml-2 text-base-content/55 text-xs">
                  {result.newBest ? "new best" : `best ${result.best}`}
                </span>
              </p>
            )}
            {result.record.you != null && result.record.genoxide != null ? (
              <p className="mt-3 rounded-xl bg-base-content/5 px-3 py-2 text-sm">
                This target: you in {result.record.you}, genoxide in {result.record.genoxide} generations
              </p>
            ) : null}
            <div className="mt-5 flex flex-wrap justify-center gap-2">
              <button type="button" className="btn btn-sm btn-primary" onClick={() => newRound(randomSeed(), false)}>
                Next target
              </button>
              {!result.assisted && result.record.genoxide == null ? (
                <button
                  type="button"
                  className="btn btn-sm btn-ghost border-base-content/15"
                  onClick={() => newRound(result.seed, true)}
                >
                  Let genoxide try this one
                </button>
              ) : result.assisted && result.record.you == null ? (
                <button
                  type="button"
                  className="btn btn-sm btn-ghost border-base-content/15"
                  onClick={() => newRound(result.seed, false)}
                >
                  Try this one yourself
                </button>
              ) : null}
            </div>
            <p className="mt-4 text-base-content/55 text-xs leading-relaxed">
              A steady-state GA: tournament selection, two-point crossover, Gaussian mutation, the two least
              fit replaced each generation.
            </p>
          </section>
        </div>
      ) : null}

      <p className="sr-only" aria-live="polite">
        {announce}
      </p>
    </div>,
    document.body,
  );
}
