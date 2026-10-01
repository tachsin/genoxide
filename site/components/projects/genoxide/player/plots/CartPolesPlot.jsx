"use client";

import { useEffect, useRef, useState } from "react";
import { STATUS, categorical } from "../chart-kit";
import { Legend, PlotBox } from "../chart-parts";

// seconds a failure stays on screen before its episode starts again
const HOLD = 0.9;
// seconds of the long pole's tip left behind as a trail
const TRAIL = 0.6;
// the cart's drawn size, m (the task gives it a mass only)
const CART = { width: 0.5, height: 0.22, wheel: 0.06 };

/** How an episode ends: balanced, or the first limit it crossed at its last step. */
function outcome(episode, track, failureAngle) {
  const last = episode[episode.length - 1];
  if (!last) return { failed: false };
  if (Math.abs(last[0]) > track) return { failed: true, cart: true, poles: [] };
  const poles = last.slice(1).map((angle) => Math.abs(angle) > failureAngle);
  return poles.some(Boolean) ? { failed: true, cart: false, poles } : { failed: false };
}

/**
 * A clock in seconds that runs while `running`, from the moment the plot mounts: generations
 * change under it without restarting it, so a run played fast shows one cart moving while its
 * controller improves.
 */
function useClock(running) {
  const [seconds, setSeconds] = useState(0);
  const elapsed = useRef(0);
  useEffect(() => {
    if (!running) return undefined;
    let raf = 0;
    let previous = performance.now();
    const tick = (now) => {
      elapsed.current += Math.min(0.1, Math.max(0, (now - previous) / 1000));
      previous = now;
      setSeconds(elapsed.current);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [running]);
  return seconds;
}

/**
 * A dial in the scene's top left corner, the poles' angles as needles (the longer pole's
 * longer): beyond `failureAngle`, in red, a pole has fallen. The short pole is a few pixels at
 * the scene's true scale; its needle shows its angle as clearly as the long one's.
 */
function Gauge({ angles, failureAngle, colors, lengths, width }) {
  const r = Math.max(26, Math.min(48, width * 0.07));
  const cx = 10 + r;
  const cy = 12 + r;
  // ±range maps to ±90°: the failure angle sits at four fifths of the way down
  const range = failureAngle * 1.25;
  const polar = (angle, radius) => {
    const a = ((Math.max(-range, Math.min(range, angle)) / range) * Math.PI) / 2;
    return [cx + radius * Math.sin(a), cy - radius * Math.cos(a)];
  };
  const arc = (from, to, radius) => {
    const [x0, y0] = polar(from, radius);
    const [x1, y1] = polar(to, radius);
    return `M${x0.toFixed(1)} ${y0.toFixed(1)} A${radius} ${radius} 0 0 1 ${x1.toFixed(1)} ${y1.toFixed(1)}`;
  };
  const longest = Math.max(...lengths);
  return (
    <g aria-hidden>
      <path d={`${arc(-range, range, r)} Z`} className="fill-base-100 stroke-base-content/25" strokeWidth={1} />
      <path d={arc(-range, -failureAngle, r - 2.5)} fill="none" stroke={STATUS.critical} strokeOpacity={0.6} strokeWidth={5} />
      <path d={arc(failureAngle, range, r - 2.5)} fill="none" stroke={STATUS.critical} strokeOpacity={0.6} strokeWidth={5} />
      {[-failureAngle, 0, failureAngle].map((angle) => {
        const [x0, y0] = polar(angle, r - 7);
        const [x1, y1] = polar(angle, r);
        return <line key={angle} x1={x0} y1={y0} x2={x1} y2={y1} className="stroke-base-content/45" strokeWidth={1} />;
      })}
      {angles.map((angle, i) => {
        const [x, y] = polar(angle, (r - 6) * (lengths[i] === longest ? 1 : 0.68));
        return <line key={i} x1={cx} y1={cy} x2={x} y2={y} stroke={colors[i]} strokeWidth={i === 0 ? 2.5 : 3} strokeLinecap="round" />;
      })}
      <circle cx={cx} cy={cy} r={2.5} className="fill-base-content/70" />
      <text x={cx} y={cy + 13} textAnchor="middle" className="fill-base-content/55 text-[10px]">
        {angles.length > 1 ? "angles" : "angle"}
      </text>
    </g>
  );
}

function signed(value, digits) {
  const text = Math.abs(value).toFixed(digits);
  return Number(text) === 0 ? text : `${value < 0 ? "−" : "+"}${text}`;
}

/**
 * `cart_poles`: the cart and its poles, animated. Each frame's `state.episode` is the best
 * network's first steps from the initial state, an `[x, θ₁, θ₂, ...]` per step (m, degrees,
 * positive angles leaning towards positive x); `problem` has the geometry (`track`, the poles'
 * `half_lengths`, `failure_angle`, the `step` in s) and the solution's `episode`, played at the
 * last frame. An episode plays in real time and starts again; one that fails holds its last state,
 * the limit it crossed in red.
 */
export default function CartPolesPlot({ trace, frame, index, dark, reduced }) {
  const problem = trace.problem ?? {};
  const track = problem.track ?? 2.4;
  const halfLengths = problem.half_lengths ?? [0.5];
  const failureAngle = problem.failure_angle ?? 12;
  const step = problem.step ?? 0.02;
  const atEnd = index === trace.frames.length - 1;
  const solution = atEnd && Array.isArray(problem.episode) && problem.episode.length > 0;
  const episode = (solution ? problem.episode : frame.state?.episode) ?? [];
  const ending = outcome(episode, track, failureAngle);

  const seconds = useClock(!reduced && episode.length > 1);
  const duration = episode.length * step;
  const cycle = duration + (ending.failed ? HOLD : 0);
  // reduced motion: the episode's end, how it fails or where it is after balancing
  const at = reduced || cycle === 0 ? episode.length - 1 : Math.min(episode.length - 1, Math.floor((seconds % cycle) / step));
  const state = episode[Math.max(0, at)] ?? [0, ...halfLengths.map(() => 0)];
  const showFailure = ending.failed && at === episode.length - 1;

  const palette = categorical(dark);
  const colors = halfLengths.map((_, i) => palette[i % palette.length]);
  const poleColor = (i) => (showFailure && !ending.cart && ending.poles[i] ? STATUS.critical : colors[i]);
  const trailFrom = reduced ? 0 : Math.max(0, at - Math.round(TRAIL / step));

  const generation = typeof frame.generation === "number" ? frame.generation : index;
  const caption = solution
    ? `The solution, its first ${(duration).toFixed(0)} s`
    : `The best network of generation ${generation}, its first ${duration.toFixed(duration < 10 ? 2 : 0)} s${ending.failed ? ", until it fails" : ""}`;
  const status = showFailure
    ? ending.cart
      ? "the cart left the track"
      : `${halfLengths.length > 1 ? `pole ${ending.poles.indexOf(true) + 1}` : "the pole"} passed ${failureAngle}°`
    : null;

  return (
    <div>
      <Legend
        className="mb-2"
        items={[
          ...halfLengths.map((h, i) => ({
            label: halfLengths.length > 1 ? `pole ${i + 1}, ${(2 * h).toFixed(2).replace(/\.?0+$/, "")} m` : `pole, ${(2 * h).toFixed(2).replace(/\.?0+$/, "")} m`,
            color: colors[i],
            shape: "line",
          })),
          { label: `limits: ±${failureAngle}°, ±${track} m`, color: STATUS.critical, shape: "dash" },
        ]}
      />
      <PlotBox
        aspect={0.46}
        minHeight={220}
        maxHeight={420}
        label={`${caption}. ${status ? `It fails: ${status}.` : "Balanced."}`}
      >
        {({ width, height }) => {
          const longest = 2 * Math.max(...halfLengths);
          const margin = 0.35;
          const spanX = 2 * (track + margin);
          const spanY = longest + CART.height + CART.wheel * 2 + 0.45;
          const scale = Math.min(width / spanX, height / spanY);
          const ox = width / 2;
          const ground = height - 26;
          const sx = (x) => ox + x * scale;
          const cartTop = ground - (CART.wheel * 2 + CART.height) * scale;
          const pivot = (x) => ({ x: sx(x), y: cartTop });
          const tip = (x, angle, half) => {
            const radians = (angle * Math.PI) / 180;
            const p = pivot(x);
            return { x: p.x + 2 * half * Math.sin(radians) * scale, y: p.y - 2 * half * Math.cos(radians) * scale };
          };
          const [x, ...angles] = state;
          const p = pivot(x);
          const limit = (failureAngle * Math.PI) / 180;
          const reach = longest * scale * 1.04;
          const trail = episode
            .slice(trailFrom, at + 1)
            .map(([cx, a]) => {
              const t = tip(cx, a, halfLengths[0]);
              return `${t.x.toFixed(1)},${t.y.toFixed(1)}`;
            })
            .join(" ");
          const poleWidth = Math.max(3, 0.035 * scale);
          const meters = [];
          for (let m = -Math.floor(track); m <= Math.floor(track); m++) meters.push(m);

          return (
            <>
              {/* the track, its end stops and a tick per metre */}
              <line x1={sx(-track - margin)} y1={ground} x2={sx(track + margin)} y2={ground} className="stroke-base-content/25" strokeWidth={1} />
              <line x1={sx(-track)} y1={ground} x2={sx(track)} y2={ground} className="stroke-base-content/55" strokeWidth={2.5} strokeLinecap="round" />
              {[-track, track].map((end) => (
                <line
                  key={end}
                  x1={sx(end)}
                  y1={ground + 1}
                  x2={sx(end)}
                  y2={ground - 22}
                  stroke={STATUS.critical}
                  strokeOpacity={ending.cart && showFailure ? 1 : 0.55}
                  strokeWidth={2.5}
                  strokeLinecap="round"
                />
              ))}
              {meters.map((m) => (
                <g key={m}>
                  <line x1={sx(m)} y1={ground} x2={sx(m)} y2={ground + 4} className="stroke-base-content/40" strokeWidth={1} />
                  <text x={sx(m)} y={ground + 16} textAnchor="middle" className="fill-base-content/55 text-[10px] tabular-nums">
                    {m === 0 ? "0" : `${m < 0 ? "−" : ""}${Math.abs(m)} m`}
                  </text>
                </g>
              ))}

              {/* the failure angle, either side of the upright */}
              <path
                d={`M${p.x + reach * Math.sin(-limit)} ${p.y - reach * Math.cos(limit)} L${p.x} ${p.y} L${p.x + reach * Math.sin(limit)} ${p.y - reach * Math.cos(limit)}`}
                fill="none"
                stroke={STATUS.critical}
                strokeOpacity={0.45}
                strokeWidth={1}
                strokeDasharray="4 4"
              />
              <line x1={p.x} y1={p.y} x2={p.x} y2={p.y - reach} className="stroke-base-content/15" strokeWidth={1} strokeDasharray="2 4" />

              {/* where the long pole's tip has been */}
              {trail ? <polyline points={trail} fill="none" stroke={colors[0]} strokeOpacity={reduced ? 0.25 : 0.35} strokeWidth={1.5} strokeLinejoin="round" /> : null}

              {/* the cart */}
              <rect
                x={sx(x - CART.width / 2)}
                y={cartTop}
                width={CART.width * scale}
                height={CART.height * scale}
                rx={Math.min(6, 0.04 * scale)}
                className="fill-base-content/15"
                stroke={ending.cart && showFailure ? STATUS.critical : "currentColor"}
                strokeOpacity={ending.cart && showFailure ? 1 : 0.55}
                strokeWidth={1.5}
              />
              {[-1, 1].map((side) => (
                <circle
                  key={side}
                  cx={sx(x + side * CART.width * 0.3)}
                  cy={ground - CART.wheel * scale}
                  r={CART.wheel * scale}
                  className="fill-base-100 stroke-base-content/60"
                  strokeWidth={1.5}
                />
              ))}

              {/* the hinge, then the poles, the long one behind */}
              <circle cx={p.x} cy={p.y} r={Math.max(2.5, poleWidth * 0.55)} className="fill-base-100 stroke-base-content/70" strokeWidth={1.5} />
              {angles.map((angle, i) => {
                const t = tip(x, angle, halfLengths[i] ?? halfLengths[0]);
                const color = poleColor(i);
                return (
                  <g key={i}>
                    <line x1={p.x} y1={p.y} x2={t.x} y2={t.y} stroke={color} strokeWidth={poleWidth} strokeLinecap="round" />
                    <circle cx={t.x} cy={t.y} r={poleWidth * 0.9} fill={color} />
                  </g>
                );
              })}

              <Gauge angles={angles} failureAngle={failureAngle} colors={angles.map((_, i) => poleColor(i))} lengths={halfLengths} width={width} />
            </>
          );
        }}
      </PlotBox>
      <p className="mt-1 text-base-content/65 text-xs tabular-nums">
        {caption} · t {(Math.max(0, at + 1) * step).toFixed(2)} s · x {signed(state[0], 2)} m
        {state.slice(1).map((angle, i) => ` · θ${state.length > 2 ? ["₁", "₂", "₃"][i] : ""} ${signed(angle, 1)}°`)}
        {status ? <span style={{ color: STATUS.critical }}> · {status}</span> : null}
      </p>
    </div>
  );
}
