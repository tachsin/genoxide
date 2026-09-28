"use client";

import { useEffect, useState } from "react";

const LETTERS = "abcdefghijklmnopqrstuvwxyz";
const OFFSPRING = 100;
const RATE = 0.1;
const STEP_MS = 45;
const MAX_GENERATIONS = 30;

const randomLetter = () => LETTERS[Math.floor(Math.random() * LETTERS.length)];

function matches(text, target) {
  let n = 0;
  for (let i = 0; i < target.length; i++) if (text[i] === target[i]) n += 1;
  return n;
}

/**
 * The title, evolved from random letters once on load: Dawkins' weasel, a
 * real mutation and selection loop (each generation, the best of 100
 * mutated copies). The server renders the plain text, and the heading keeps
 * it as its label while the letters change.
 */
export default function EvolvingTitle({ text, className }) {
  const [shown, setShown] = useState(text);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return undefined;
    let current = Array.from(text, randomLetter).join("");
    let generation = 0;
    let last = 0;
    let frame = 0;
    setShown(current);
    const tick = (now) => {
      if (now - last >= STEP_MS) {
        last = now;
        generation += 1;
        let best = null;
        let bestScore = -1;
        for (let i = 0; i < OFFSPRING; i++) {
          let child = "";
          for (const letter of current) child += Math.random() < RATE ? randomLetter() : letter;
          const score = matches(child, text);
          if (score > bestScore) {
            best = child;
            bestScore = score;
          }
        }
        current = best;
        // past the time budget, the stragglers settle one per step
        if (generation > MAX_GENERATIONS) {
          const at = Array.from(current).findIndex((letter, i) => letter !== text[i]);
          if (at >= 0) current = current.slice(0, at) + text[at] + current.slice(at + 1);
        }
        setShown(current);
        if (current === text) return;
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame);
      setShown(text);
    };
  }, [text]);

  return (
    <h1 className={className} aria-label={shown === text ? undefined : text}>
      {shown}
    </h1>
  );
}
