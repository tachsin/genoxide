"use client";

import { Bird, Bot, ChevronRight, RotateCcw, X } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { observeTheme, readTheme } from "../render";
import { randomSeed } from "./ga";
import { LEVELS, CamouflageWorld } from "./world";

const PROGRESS_KEY = "genoxide-camouflage";

function readProgress() {
  try {
    const value = JSON.parse(window.localStorage.getItem(PROGRESS_KEY) ?? "{}");
    return value && typeof value === "object" ? value : {};
  } catch {
    return {};
  }
}

function writeProgress(progress) {
  try {
    window.localStorage.setItem(PROGRESS_KEY, JSON.stringify(progress));
  } catch {
    // private mode or storage blocked: progress just isn't kept
  }
}

let probe = null;

/** `color` with `alpha`, as an rgb() string the canvas can use. */
function withAlpha(color, alpha) {
  if (!probe) {
    const canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;
    probe = canvas.getContext("2d", { willReadFrequently: true });
  }
  probe.clearRect(0, 0, 1, 1);
  probe.fillStyle = "#000";
  probe.fillStyle = color;
  probe.fillRect(0, 0, 1, 1);
  const [r, g, b] = probe.getImageData(0, 0, 1, 1).data;
  return `rgb(${r} ${g} ${b} / ${alpha})`;
}

function gameTheme(element) {
  const theme = readTheme(element);
  return {
    ...theme,
    panel: withAlpha(theme.base, 0.92),
    line: withAlpha(theme.ink, 0.15),
    mother: "rgb(219 39 119)",
    father: "rgb(37 99 235)",
  };
}

const percent = (value) => `${Math.round(value * 100)}%`;

/** The camouflage of every generation, as a small line chart. */
function Chart({ values, goal, label }) {
  if (!values?.length) return null;
  const width = 280;
  const height = 72;
  const x = (i) => (values.length < 2 ? width / 2 : (i / (values.length - 1)) * (width - 8) + 4);
  const y = (v) => height - 6 - v * (height - 12);
  const path = values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  return (
    <figure className="mt-3">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-20 w-full" role="img" aria-label={label}>
        {goal != null ? (
          <line
            x1="0"
            x2={width}
            y1={y(goal)}
            y2={y(goal)}
            stroke="var(--color-base-content)"
            strokeOpacity="0.3"
            strokeDasharray="4 4"
          />
        ) : null}
        <path d={path} fill="none" stroke="var(--color-primary)" strokeWidth="2" strokeLinejoin="round" />
      </svg>
      <figcaption className="text-base-content/55 text-xs">{label}</figcaption>
    </figure>
  );
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
 * The camouflage game, a full-screen dialog over the page. You are a bird: catch the moths you
 * can see, and the survivors breed. Five levels, each showing one idea of genoxide's.
 */
export default function CamouflageGame({ reduced, onClose }) {
  const dialogRef = useRef(null);
  const canvasRef = useRef(null);
  const hudRef = useRef(null);
  const hintRef = useRef(null);
  const worldRef = useRef(null);
  const cardRef = useRef(null);
  const carry = useRef(null);
  const titleId = useId();
  const helpId = useId();
  const cardTitleId = useId();
  const [stats, setStats] = useState(null);
  const [index, setIndex] = useState(() => Math.min(readProgress().unlocked ?? 0, LEVELS.length - 1));
  const [card, setCard] = useState("intro");
  const [result, setResult] = useState(null);
  const [announce, setAnnounce] = useState("");
  const level = LEVELS[index];
  const firstIndex = useRef(index);

  useEffect(() => {
    const canvas = canvasRef.current;
    const onLevelEnd = (end) => {
      const progress = readProgress();
      if (end.reached) progress.unlocked = Math.max(progress.unlocked ?? 0, end.index + 1);
      if (end.id === "selection" && !end.assisted && end.reached) progress.selection = end.generations;
      writeProgress(progress);
      carry.current = end.genomes;
      setResult({ ...end, yours: progress.selection ?? null });
      setCard("result");
      setAnnounce(
        end.reached
          ? `Level complete in ${end.generations} generations.`
          : `The level ended after ${end.generations} generations.`,
      );
    };
    const world = new CamouflageWorld(canvas, { reduced, onStats: setStats, onLevelEnd });
    worldRef.current = world;
    world.setTheme(gameTheme(canvas));
    const measure = () => {
      const rect = canvas.getBoundingClientRect();
      const hud = hudRef.current?.getBoundingClientRect();
      const hint = hintRef.current?.getBoundingClientRect();
      world.setInsets(hud ? hud.bottom - rect.top : 0, hint ? rect.bottom - hint.top : 0);
      world.resize(rect.width, rect.height, Math.min(window.devicePixelRatio || 1, 2));
    };
    measure();
    world.startLevel(firstIndex.current, randomSeed(), null);
    const resizeObserver = new ResizeObserver(measure);
    resizeObserver.observe(canvas);
    if (hudRef.current) resizeObserver.observe(hudRef.current);
    const sync = () => (document.hidden ? world.stop() : world.start());
    document.addEventListener("visibilitychange", sync);
    const unobserveTheme = observeTheme(() => world.setTheme(gameTheme(canvas)));
    sync();
    return () => {
      world.stop();
      worldRef.current = null;
      resizeObserver.disconnect();
      document.removeEventListener("visibilitychange", sync);
      unobserveTheme();
    };
  }, [reduced]);

  // the page behind doesn't scroll
  useEffect(() => {
    const root = document.documentElement;
    const overflow = root.style.overflow;
    root.style.overflow = "hidden";
    return () => {
      root.style.overflow = overflow;
    };
  }, []);

  useEffect(() => {
    if (card) cardRef.current?.querySelector("button")?.focus();
    else canvasRef.current?.focus({ preventScroll: true });
  }, [card, index]);

  useEffect(() => {
    const onEscape = (event) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      onClose();
    };
    document.addEventListener("keydown", onEscape);
    return () => document.removeEventListener("keydown", onEscape);
  }, [onClose]);

  function play(levelIndex) {
    const world = worldRef.current;
    if (!world) return;
    setCard(null);
    world.begin();
    setAnnounce(`${LEVELS[levelIndex].title}. ${LEVELS[levelIndex].auto ? "genoxide is hunting." : "Catch the moths you can see."}`);
  }

  // a level's intro, over its bark and moths, paused
  function showIntro(levelIndex) {
    setIndex(levelIndex);
    setResult(null);
    setCard("intro");
    const carried = LEVELS[levelIndex].id === "change" ? carry.current : null;
    worldRef.current?.startLevel(levelIndex, randomSeed(), carried);
  }

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
    if (!world || card || event.button > 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    world.clickAt(event.clientX - rect.left, event.clientY - rect.top, event.pointerType === "touch" ? 10 : 2);
  }

  function toggleAuto() {
    const world = worldRef.current;
    if (!world) return;
    world.setAuto(!world.auto);
    setAnnounce(world.auto ? "genoxide is hunting." : "Your turn to hunt.");
  }

  const playing = !card && stats;
  const auto = stats?.auto ?? false;
  const goal = stats?.goal;
  const camouflage = stats?.camouflage ?? 0;
  const hint = !playing
    ? ""
    : stats.phase === "breed"
      ? "The survivors breed: each child takes its genes from two parents, cut at one point, and a few mutate."
      : auto
        ? "genoxide is the bird: of 3 random moths, it catches the most visible."
        : level.split
          ? `Catch the moths you can see, on either tree: ${stats.catchesLeft} more this generation.`
          : `Catch the moths you can see: ${stats.catchesLeft} more this generation.`;

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
      <canvas
        ref={canvasRef}
        tabIndex={0}
        aria-label="Moths resting on bark. Arrow keys move between them, Enter catches one."
        onPointerDown={onPointerDown}
        onKeyDown={onCanvasKeyDown}
        className="absolute inset-0 size-full touch-none select-none outline-none"
        style={{ cursor: playing && !auto && stats.phase === "hunt" ? "crosshair" : "default" }}
      />

      <div className="pointer-events-none absolute inset-x-0 top-0 p-2 sm:p-4">
        <div
          ref={hudRef}
          className="pointer-events-auto mx-auto flex max-w-5xl flex-wrap items-center gap-x-4 gap-y-2 rounded-2xl border border-base-content/10 bg-base-100/90 p-2 shadow-lg backdrop-blur-md sm:p-3"
        >
          <div className="text-xs leading-tight">
            <h2 id={titleId} className="font-semibold text-primary uppercase tracking-wider">
              Level {index + 1} of {LEVELS.length}: {level.title}
            </h2>
            <p className="mt-0.5 text-base-content/60">{level.concept}</p>
          </div>

          <div className="min-w-[10rem] flex-1">
            {stats?.islands ? (
              <div className="grid grid-cols-2 gap-3">
                {stats.islands.map((value, i) => (
                  <Meter key={i} label={i ? "Sooty tree" : "Birch tree"} value={value} goal={goal} />
                ))}
              </div>
            ) : (
              <Meter
                label={stats?.generations ? `Generation ${stats.generation} of ${stats.generations}` : "Camouflage"}
                value={stats?.generations ? stats.generation / stats.generations : camouflage}
                goal={stats?.generations ? null : goal}
                shown={stats?.generations ? `${stats.generation}/${stats.generations}` : percent(camouflage)}
              />
            )}
          </div>

          <dl className="flex gap-3 text-center">
            <Stat label="Gen" value={stats?.generation ?? 0} />
            <Stat label="Catches" value={stats ? `${stats.catches - stats.catchesLeft}/${stats.catches}` : "–"} />
          </dl>

          <div className="flex items-center gap-1.5">
            {level.auto ? null : (
              <button
                type="button"
                aria-pressed={auto}
                onClick={toggleAuto}
                disabled={!playing}
                className={`btn btn-sm ${auto ? "btn-primary" : "btn-ghost border-base-content/15"}`}
              >
                <Bot size={15} aria-hidden />
                <span className="hidden sm:inline">genoxide hunts</span>
              </button>
            )}
            <button
              type="button"
              onClick={() => showIntro(index)}
              className="btn btn-sm btn-ghost btn-square border-base-content/15"
              aria-label="Restart the level"
              title="Restart the level"
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
        className="pointer-events-none absolute inset-x-0 bottom-0 mx-auto max-w-xl px-4 pb-4 text-center text-xs sm:text-sm"
      >
        <span className="rounded-full bg-base-100/85 px-3 py-1 text-base-content/75 backdrop-blur">
          {hint || "Five levels, one idea of genoxide's each."}
        </span>
      </p>

      {card ? (
        <div className="absolute inset-0 flex items-center justify-center bg-base-100/40 p-4 backdrop-blur-[2px]">
          <section
            ref={cardRef}
            aria-labelledby={cardTitleId}
            className="w-full max-w-md rounded-2xl border border-base-content/10 bg-base-100/95 p-6 shadow-2xl backdrop-blur-md"
          >
            {card === "intro" ? (
              <Intro index={index} titleId={cardTitleId} onPlay={() => play(index)} onPick={showIntro} />
            ) : (
              <Result
                result={result}
                titleId={cardTitleId}
                onAgain={() => showIntro(result.index)}
                onNext={() => showIntro(Math.min(LEVELS.length - 1, result.index + 1))}
              />
            )}
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

function Meter({ label, value, goal, shown }) {
  const width = Math.max(0, Math.min(1, value));
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2 whitespace-nowrap text-xs">
        <span className="text-base-content/60">{label}</span>
        <span className="font-mono font-semibold tabular-nums">
          {shown ?? percent(value)}
          {goal != null ? <span className="font-normal text-base-content/50"> / {percent(goal)}</span> : null}
        </span>
      </div>
      <div className="relative mt-1.5 h-1.5 overflow-hidden rounded-full bg-base-content/10" aria-hidden>
        <div className="h-full rounded-full bg-primary transition-[width] duration-500" style={{ width: `${width * 100}%` }} />
        {goal != null ? <div className="absolute inset-y-0 w-0.5 bg-base-content/50" style={{ left: `${goal * 100}%` }} /> : null}
      </div>
    </div>
  );
}

function Intro({ index, titleId, onPlay, onPick }) {
  const level = LEVELS[index];
  const unlocked = readProgress().unlocked ?? 0;
  return (
    <>
      <p className="text-base-content/55 text-xs uppercase tracking-wider">
        Level {index + 1} of {LEVELS.length}
      </p>
      <h3 id={titleId} className="mt-1 inline-flex items-center gap-2 font-semibold text-2xl tracking-tight">
        <Bird size={20} aria-hidden className="text-primary" />
        {level.title}
      </h3>
      <p className="mt-3 text-base-content/80 text-sm leading-relaxed">{level.intro}</p>
      <div className="mt-4 rounded-xl bg-base-content/5 p-3">
        <p className="text-base-content/60 text-xs">In genoxide: {level.concept}</p>
        <pre className="mt-2 overflow-x-auto text-[0.7rem] leading-snug">
          <code>{level.code}</code>
        </pre>
      </div>
      <div className="mt-5 flex flex-wrap items-center justify-between gap-2">
        <div className="flex gap-1" role="group" aria-label="Levels">
          {LEVELS.map((l, i) => (
            <button
              key={l.id}
              type="button"
              disabled={i > unlocked}
              aria-current={i === index ? "step" : undefined}
              onClick={() => onPick(i)}
              className={`btn btn-xs btn-circle ${i === index ? "btn-primary" : "btn-ghost border-base-content/15"}`}
              title={i > unlocked ? "Finish the levels before it" : l.title}
            >
              {i + 1}
            </button>
          ))}
        </div>
        <button type="button" className="btn btn-sm btn-primary" onClick={onPlay}>
          {level.auto ? "Watch genoxide" : "Start hunting"}
          <ChevronRight size={15} aria-hidden />
        </button>
      </div>
    </>
  );
}

function Result({ result, titleId, onAgain, onNext }) {
  const level = LEVELS[result.index];
  const last = result.index === LEVELS.length - 1;
  let summary;
  if (level.id === "tradeoff") {
    summary = `After ${result.generations} generations, ${result.front.front.length} moths form the front: none of them is beaten on both camouflage and display by another. Better hidden costs display, more display costs camouflage.`;
  } else if (level.split) {
    summary = `The birch moths reached ${percent(result.islands[0])} camouflage, the soot moths ${percent(result.islands[1])}, from one kind of moth: each tree selected its own.`;
  } else if (level.id === "machine") {
    summary =
      result.yours != null
        ? `genoxide reached ${percent(result.camouflage)} camouflage in ${result.generations} generations. On birch, you needed ${result.yours}.`
        : `genoxide reached ${percent(result.camouflage)} camouflage in ${result.generations} generations, in ${Math.round(result.seconds)} seconds.`;
  } else {
    summary = result.reached
      ? `The moths reached ${percent(result.camouflage)} camouflage in ${result.generations} generations${result.assisted ? ", with genoxide's help" : ""}.`
      : `After ${result.generations} generations the moths reached ${percent(result.camouflage)}: catch the most visible ones first.`;
  }
  return (
    <>
      <p className="text-base-content/55 text-xs uppercase tracking-wider">
        Level {result.index + 1}: {level.title}
      </p>
      <h3 id={titleId} className="mt-1 font-semibold text-2xl tracking-tight">
        {result.reached ? "Evolved!" : "Not yet hidden"}
      </h3>
      <p className="mt-3 text-base-content/80 text-sm leading-relaxed">{summary}</p>
      <Chart
        values={result.history.map((h) => h.camouflage)}
        goal={level.goal}
        label="Average camouflage of each generation"
      />
      <div className="mt-5 flex flex-wrap justify-end gap-2">
        <button type="button" className="btn btn-sm btn-ghost border-base-content/15" onClick={onAgain}>
          Play again
        </button>
        {!last && result.reached ? (
          <button type="button" className="btn btn-sm btn-primary" onClick={onNext}>
            Next level
            <ChevronRight size={15} aria-hidden />
          </button>
        ) : null}
      </div>
    </>
  );
}
