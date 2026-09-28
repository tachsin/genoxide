"use client";

import { Dna } from "lucide-react";
import { Suspense, lazy, useCallback, useEffect, useRef, useState } from "react";
import { observeTheme, readTheme } from "./render";
import { World } from "./world";

// The game's HUD loads when someone presses Play.
const SelectionGame = lazy(() => import("./SelectionGame"));

function useReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(query.matches);
    const onChange = () => setReduced(query.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);
  return reduced;
}

/**
 * The hero's backdrop: little double helices whose looks are decoded from
 * their genomes, drifting and breeding on their own. Decorative (hidden from
 * assistive technology, no pointer events); under reduced motion one still
 * frame. Its button opens the game, where the visitor is the selection.
 * Mount it inside a positioned hero: the canvas fills it.
 */
export default function HeroSwarm() {
  const canvasRef = useRef(null);
  const buttonRef = useRef(null);
  const reopenFocus = useRef(false);
  const playingRef = useRef(false);
  const syncRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const reduced = useReducedMotion();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const world = new World(canvas, { mode: "ambient", reduced });
    world.setTheme(readTheme(canvas));
    let visible = false;
    // runs only while on screen, in a visible tab, and not behind the game
    const sync = () => {
      if (visible && !document.hidden && !playingRef.current && !reduced) world.start();
      else world.stop();
    };
    syncRef.current = sync;
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      world.resize(rect.width, rect.height, Math.min(window.devicePixelRatio || 1, 1.5));
    };
    resize();
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
    const intersection = new IntersectionObserver((entries) => {
      visible = entries.some((e) => e.isIntersecting);
      sync();
    });
    intersection.observe(canvas);
    document.addEventListener("visibilitychange", sync);
    const unobserveTheme = observeTheme(() => world.setTheme(readTheme(canvas)));
    sync();
    return () => {
      world.stop();
      resizeObserver.disconnect();
      intersection.disconnect();
      document.removeEventListener("visibilitychange", sync);
      unobserveTheme();
      syncRef.current = null;
    };
  }, [reduced]);

  useEffect(() => {
    playingRef.current = playing;
    syncRef.current?.();
    if (!playing && reopenFocus.current) {
      reopenFocus.current = false;
      buttonRef.current?.focus();
    }
  }, [playing]);

  const close = useCallback(() => {
    reopenFocus.current = true;
    setPlaying(false);
  }, []);

  return (
    <>
      <canvas ref={canvasRef} aria-hidden className="pointer-events-none absolute inset-0 -z-10 size-full" />
      <div className="proj-rise-2 mt-7 flex justify-center">
        <button
          ref={buttonRef}
          type="button"
          onClick={() => setPlaying(true)}
          aria-haspopup="dialog"
          className="group inline-flex cursor-pointer items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-4 py-1.5 font-medium text-primary text-sm transition-colors hover:border-primary/55 hover:bg-primary/15 focus-visible:outline-2 focus-visible:outline-primary focus-visible:outline-offset-2"
        >
          <Dna size={15} aria-hidden className="transition-transform group-hover:rotate-12" />
          Play: be the selection
        </button>
      </div>
      {playing ? (
        <Suspense fallback={null}>
          <SelectionGame reduced={reduced} onClose={close} />
        </Suspense>
      ) : null}
    </>
  );
}
