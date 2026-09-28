// The creatures on a canvas, in one of two modes. "ambient": they drift,
// now and then two neighbours have a child and the oldest fades out, a GA
// with nobody watching. "play": the player (or genoxide's tournament
// selection) picks the parents, each pair's crossover is shown, and the two
// creatures least like the target are replaced: a steady-state GA.

import {
  MATCH,
  createRng,
  crossover,
  decode,
  easeGenes,
  mutate,
  randomGenome,
  randomSeed,
  similarity,
  tournament,
} from "./genome";
import { drawHelix, oklch, onBody, paint, reach, ring, sparkle } from "./render";

const TAU = Math.PI * 2;
const PARTICLES = 180;
const LABELS = 8;
// the share of a crossover's animation before its children appear
const BIRTH = 0.62;
const CELL = 9;
const DASH = [3, 5];

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const easeInOut = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
const easeOutBack = (t) => 1 + 2.2 * (t - 1) ** 3 + 1.2 * (t - 1) ** 2;
const wrapAngle = (a) => a - TAU * Math.round(a / TAU);

function bits(mask) {
  let n = 0;
  for (let m = mask; m; m &= m - 1) n += 1;
  return n;
}

function makeCreature(genome, x, y) {
  const shown = Float64Array.from(genome);
  return {
    genome,
    shown,
    look: decode(shown, {}),
    colors: { strand1: "", strand2: "", rung: "" },
    painted: 0,
    x,
    y,
    vx: 0,
    vy: 0,
    angle: (Math.random() - 0.5) * 1.2,
    heading: Math.random() * TAU,
    phase: Math.random() * TAU,
    fitness: 0,
    grow: 1, // 0 to 1 as it's born
    fade: 0, // above 0 while it fades out, gone at 1
    hold: 0, // held still (selected or breeding) while above 0
    open: 0, // strands pulled apart while it breeds
    pulse: 0,
    selected: false,
    born: 0,
  };
}

// gene value to lightness, so a strip of genes reads as a barcode
function cellColor(value, hue, theme) {
  return oklch((theme.dark ? 0.5 : 0.42) + value * 0.42, 0.14, hue);
}

export class World {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} options
   * @param {"ambient" | "play"} options.mode
   * @param {boolean} [options.reduced]  prefers-reduced-motion: nothing drifts
   * @param {(stats: object) => void} [options.onStats]  play: after every change of the counters
   * @param {(result: object) => void} [options.onRoundEnd]  play: a creature matched the target
   */
  constructor(canvas, { mode, reduced = false, onStats, onRoundEnd }) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.play = mode === "play";
    this.reduced = reduced;
    this.onStats = onStats;
    this.onRoundEnd = onRoundEnd;
    this.creatures = [];
    this.events = [];
    this.width = 0;
    this.height = 0;
    this.dpr = 1;
    this.scale = 1;
    this.insetTop = 0;
    this.insetBottom = 0;
    this.theme = null;
    this.rng = createRng(randomSeed());
    this.running = false;
    this.raf = 0;
    this.last = 0;
    this.time = 0;
    this.clock = 1.5;
    this.shimmer = 3;
    this.born = 0;

    this.px = new Float32Array(PARTICLES);
    this.py = new Float32Array(PARTICLES);
    this.pvx = new Float32Array(PARTICLES);
    this.pvy = new Float32Array(PARTICLES);
    this.plife = new Float32Array(PARTICLES);
    this.pmax = new Float32Array(PARTICLES);
    this.psize = new Float32Array(PARTICLES);
    this.pcolor = new Array(PARTICLES).fill("");
    this.pnext = 0;
    this.labels = Array.from({ length: LABELS }, () => ({ text: "", x: 0, y: 0, life: 0 }));
    this.lnext = 0;

    this.target = null;
    this.targetCanvas = null;
    this.targetCtx = null;
    this.resultTarget = null;
    this.resultTargetCtx = null;
    this.winnerCanvas = null;
    this.winnerCtx = null;
    this.selected = null;
    this.focus = null;
    this.winner = null;
    this.over = false;
    this.auto = false;
    this.autoStage = 0;
    this.autoClock = 0;
    this.contendersA = [];
    this.contendersB = [];
    this.winnerA = null;
    this.winnerB = null;
    this.stats = { generation: 0, clicks: 0, score: 0, best: 0, assisted: false };
    this.elapsed = 0;
    this.seed = 0;

    this.frame = this.frame.bind(this);
  }

  // ---------- lifecycle ----------

  setTheme(theme) {
    this.theme = theme;
    if (!this.running) this.draw();
  }

  resize(width, height, dpr) {
    if (width < 1 || height < 1) return;
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    if (this.width && this.creatures.length) {
      const sx = width / this.width;
      const sy = height / this.height;
      for (const c of this.creatures) {
        c.x *= sx;
        c.y *= sy;
      }
    }
    this.width = width;
    this.height = height;
    this.dpr = dpr;
    const small = width < 640;
    this.scale = this.play ? (small ? 0.95 : 1.35) : small ? 0.7 : 0.85;
    if (!this.play && !this.creatures.length) this.populate();
    if (this.play && this.reduced) this.layout(this.creatures.filter((c) => c.fade === 0));
    if (!this.running) this.draw();
  }

  /** Play: the heights the HUD covers at the top and bottom, kept clear. */
  setInsets(top, bottom) {
    this.insetTop = top;
    this.insetBottom = bottom;
    if (this.reduced && this.creatures.length) this.layout(this.creatures.filter((c) => c.fade === 0));
  }

  setTargetCanvas(canvas) {
    this.targetCanvas = canvas;
    this.targetCtx = canvas ? canvas.getContext("2d") : null;
  }

  /** Play: the result card's portraits of the target and the creature that matched it. */
  setResultCanvases(target, winner) {
    this.resultTarget = target;
    this.resultTargetCtx = target ? target.getContext("2d") : null;
    this.winnerCanvas = winner;
    this.winnerCtx = winner ? winner.getContext("2d") : null;
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.last = performance.now();
    this.raf = requestAnimationFrame(this.frame);
  }

  stop() {
    this.running = false;
    cancelAnimationFrame(this.raf);
  }

  frame(now) {
    if (!this.running) return;
    const dt = Math.min(0.05, Math.max(0, (now - this.last) / 1000));
    this.last = now;
    this.update(dt);
    this.draw();
    this.raf = requestAnimationFrame(this.frame);
  }

  bounds() {
    const margin = 36 * this.scale;
    if (!this.play) return { left: margin, right: this.width - margin, top: margin, bottom: this.height - margin };
    const top = this.insetTop + margin + 10;
    const bottom = Math.max(top + 60, this.height - this.insetBottom - margin);
    return { left: margin + 8, right: this.width - margin - 8, top, bottom };
  }

  // ---------- setting up ----------

  populate() {
    const count = Math.max(8, Math.min(40, Math.round((this.width * this.height) / 26000)));
    const genomes = Array.from({ length: count }, () => randomGenome(this.rng));
    this.creatures = genomes.map((g) => {
      const c = makeCreature(g, 0, 0);
      c.born = this.born++;
      // they fade in one after another on load
      c.grow = this.reduced ? 1 : -Math.random() * 1.4;
      return c;
    });
    this.layout(this.creatures);
  }

  /** Spreads creatures over the free area: a grid when nothing moves, else random with some spacing. */
  layout(list) {
    const { left, right, top, bottom } = this.bounds();
    const w = right - left;
    const h = bottom - top;
    if (this.reduced && this.play) {
      const cols = Math.max(2, Math.round(Math.sqrt((list.length * w) / h)));
      const rows = Math.ceil(list.length / cols);
      list.forEach((c, i) => {
        const col = i % cols;
        const row = Math.floor(i / cols);
        c.x = left + ((col + 0.5) * w) / cols;
        c.y = top + ((row + 0.5) * h) / rows;
        c.angle = ((i % 3) - 1) * 0.22;
      });
      return;
    }
    const placed = [];
    for (const c of list) {
      let bestX = 0;
      let bestY = 0;
      let bestGap = -1;
      for (let attempt = 0; attempt < 24; attempt++) {
        const x = left + Math.random() * w;
        const y = top + Math.random() * h;
        let gap = Number.POSITIVE_INFINITY;
        for (const p of placed) gap = Math.min(gap, Math.hypot(p.x - x, p.y - y));
        if (gap > bestGap) {
          bestGap = gap;
          bestX = x;
          bestY = y;
        }
      }
      c.x = bestX;
      c.y = bestY;
      placed.push(c);
    }
  }

  /** Play: a new target and population, from `seed` (the same seed, the same round). */
  newRound(seed) {
    this.seed = seed;
    const rng = createRng(seed);
    this.rng = rng;
    const count = this.width < 640 ? 12 : 16;
    let target;
    let genomes;
    // a target that nobody starts close to
    for (let attempt = 0; attempt < 40; attempt++) {
      target = randomGenome(rng);
      genomes = Array.from({ length: count }, () => randomGenome(rng));
      if (Math.max(...genomes.map((g) => similarity(g, target))) < 0.8) break;
    }
    this.target = makeCreature(target, 0, 0);
    this.creatures = genomes.map((g, i) => {
      const c = makeCreature(g, 0, 0);
      c.fitness = similarity(g, target);
      c.born = this.born++;
      c.grow = this.reduced ? 1 : -i * 0.04;
      return c;
    });
    this.layout(this.creatures);
    this.events.length = 0;
    this.selected = null;
    this.focus = null;
    this.winner = null;
    this.over = false;
    this.auto = false;
    this.autoStage = 0;
    this.resetAuto();
    this.plife.fill(0);
    for (const label of this.labels) label.life = 0;
    this.elapsed = 0;
    this.stats = {
      generation: 0,
      clicks: 0,
      score: 0,
      best: Math.max(...this.creatures.map((c) => c.fitness)),
      assisted: false,
    };
    this.emit();
  }

  emit() {
    this.onStats?.({ ...this.stats, selected: Boolean(this.selected), auto: this.auto, over: this.over });
  }

  // ---------- the player's side ----------

  /** The creature at (x, y), if any; `slop` px of tolerance (more for touch). */
  pick(x, y, slop) {
    let hit = null;
    let hitDistance = Number.POSITIVE_INFINITY;
    let near = null;
    let nearDistance = (slop * 2.5) ** 2;
    for (const c of this.creatures) {
      if (c.fade > 0 || c.grow < 0.6) continue;
      const dx = x - c.x;
      const dy = y - c.y;
      const d = dx * dx + dy * dy;
      const cos = Math.cos(c.angle);
      const sin = Math.sin(c.angle);
      const lx = (dx * cos + dy * sin) / this.scale;
      const ly = (-dx * sin + dy * cos) / this.scale;
      if (onBody(c.look, lx, ly, slop / this.scale) && d < hitDistance) {
        hit = c;
        hitDistance = d;
      }
      if (d < nearDistance) {
        near = c;
        nearDistance = d;
      }
    }
    return hit ?? near;
  }

  /** A click or tap at (x, y): selects a parent, or breeds the pair. */
  clickAt(x, y, slop) {
    if (this.over || this.auto) return false;
    const c = this.pick(x, y, slop);
    if (!c) return false;
    this.focus = null;
    this.choose(c);
    return true;
  }

  choose(c) {
    this.stats.clicks += 1;
    c.pulse = 1;
    if (!this.selected) {
      this.selected = c;
      c.selected = true;
      c.hold += 1;
    } else if (this.selected === c) {
      c.selected = false;
      c.hold -= 1;
      this.selected = null;
    } else {
      const a = this.selected;
      a.selected = false;
      a.hold -= 1;
      this.selected = null;
      this.breed(a, c, true);
    }
    this.emit();
  }

  /** Keyboard: moves the focus ring to the next creature from left to right (`step` 1) or back (-1). */
  moveFocus(step) {
    const list = this.creatures.filter((c) => c.fade === 0 && c.grow >= 0.6).sort((a, b) => a.x - b.x);
    if (!list.length) return;
    const at = list.indexOf(this.focus);
    this.focus = at < 0 ? list[step > 0 ? 0 : list.length - 1] : list[(at + step + list.length) % list.length];
    if (!this.running) this.draw();
  }

  chooseFocus() {
    if (this.over || this.auto || !this.focus || this.focus.fade > 0) return false;
    this.choose(this.focus);
    return true;
  }

  setAuto(on) {
    if (this.over || on === this.auto) return;
    this.auto = on;
    if (this.selected) {
      this.selected.selected = false;
      this.selected.hold -= 1;
      this.selected = null;
    }
    if (on) this.stats.assisted = true;
    this.resetAuto();
    this.emit();
  }

  resetAuto() {
    if (this.autoStage === 2 && this.winnerA) {
      this.winnerA.selected = false;
      this.winnerA.hold -= 1;
    }
    this.autoStage = 0;
    this.autoClock = 0;
    this.contendersA.length = 0;
    this.contendersB.length = 0;
    this.winnerA = null;
    this.winnerB = null;
  }

  // ---------- the GA ----------

  eligible(except) {
    return this.creatures.filter((c) => c.fade === 0 && c.grow >= 1 && c !== except);
  }

  /** The least fit creature that isn't selected, breeding or being born. */
  worst(except) {
    let worst = null;
    for (const c of this.creatures) {
      if (c.fade > 0 || c.hold > 0 || c.grow < 1 || c === except) continue;
      if (!worst || c.fitness < worst.fitness) worst = c;
    }
    return worst;
  }

  breed(a, b, byPlayer) {
    const { first, second, from, to } = crossover(a.genome, b.genome, this.rng);
    const mask1 = mutate(first, this.rng);
    const mask2 = mutate(second, this.rng);
    this.stats.generation += 1;
    a.hold += 1;
    b.hold += 1;
    const event = {
      a,
      b,
      first,
      second,
      from,
      to,
      mask1,
      mask2,
      byPlayer,
      t: 0,
      duration: this.reduced ? 0.25 : 1.15,
      x: (a.x + b.x) / 2,
      y: (a.y + b.y) / 2,
      born: false,
      sparked: false,
      cells: null,
    };
    if (!this.reduced && this.theme) {
      const theme = this.theme;
      const inside = (k) => k >= from && k < to;
      event.cells = {
        a: Array.from(a.genome, (v) => cellColor(v, a.look.hue, theme)),
        b: Array.from(b.genome, (v) => cellColor(v, b.look.hue, theme)),
        first: Array.from(first, (v, k) => cellColor(v, inside(k) ? b.look.hue : a.look.hue, theme)),
        second: Array.from(second, (v, k) => cellColor(v, inside(k) ? a.look.hue : b.look.hue, theme)),
      };
    }
    this.events.push(event);
  }

  birth(e) {
    e.born = true;
    const gone1 = this.worst(null);
    const gone2 = this.worst(gone1);
    let nx = -(e.b.y - e.a.y);
    let ny = e.b.x - e.a.x;
    const n = Math.hypot(nx, ny) || 1;
    nx /= n;
    ny /= n;
    const before = this.stats.best;
    let bestChild = null;
    let points = 0;
    for (let i = 0; i < 2; i++) {
      const genome = i ? e.second : e.first;
      const gone = i ? gone2 : gone1;
      const side = i ? 1 : -1;
      const x = this.reduced && gone ? gone.x : e.x + nx * 24 * side;
      const y = this.reduced && gone ? gone.y : e.y + ny * 24 * side;
      const child = makeCreature(genome, x, y);
      child.fitness = similarity(genome, this.target.genome);
      child.born = this.born++;
      child.grow = 0;
      if (this.reduced) child.angle = gone ? gone.angle : 0;
      else {
        child.angle = Math.atan2(e.b.y - e.a.y, e.b.x - e.a.x);
        child.vx = nx * 40 * side;
        child.vy = ny * 40 * side;
        child.heading = Math.atan2(child.vy, child.vx);
      }
      if (gone) gone.fade = 0.001;
      if (this.theme) paint(child, this.theme);
      this.creatures.push(child);
      this.burst(x, y, 3 + 2 * bits(i ? e.mask2 : e.mask1), child.colors.strand1);
      if (child.fitness > e.a.fitness && child.fitness > e.b.fitness) points += 10;
      if (!bestChild || child.fitness > bestChild.fitness) bestChild = child;
    }
    if (bestChild.fitness > before) {
      this.stats.best = bestChild.fitness;
      points += Math.round((bestChild.fitness - before) * 5000);
    }
    if (e.byPlayer && points > 0) {
      this.stats.score += points;
      this.label(`+${points}`, e.x, e.y - 18);
    }
    if (bestChild.fitness >= MATCH) this.finish(bestChild);
    this.emit();
  }

  finish(winner) {
    this.over = true;
    this.winner = winner;
    winner.hold += 1;
    this.auto = false;
    this.resetAuto();
    let bonus = 0;
    if (!this.stats.assisted) {
      bonus = 200 + Math.max(0, 60 - this.stats.generation) * 20;
      this.stats.score += bonus;
    }
    const color = this.theme?.primary ?? "#888";
    for (let i = 0; i < 4; i++) this.burst(winner.x, winner.y, 10, i % 2 ? winner.colors.strand2 : color);
    this.onRoundEnd?.({
      seed: this.seed,
      generations: this.stats.generation,
      clicks: this.stats.clicks,
      score: this.stats.score,
      bonus,
      seconds: this.elapsed,
      assisted: this.stats.assisted,
      match: winner.fitness,
    });
  }

  // ---------- updating ----------

  update(dt) {
    this.time += dt;
    const ease = this.reduced ? 1 : 1 - Math.exp(-dt * 5);
    for (const c of this.creatures) {
      c.open = 0;
      if (easeGenes(c.shown, c.genome, ease)) {
        decode(c.shown, c.look);
        c.painted = 0;
      }
      if (c.grow < 1) c.grow = Math.min(1, c.grow + dt / (this.reduced ? 0.25 : 0.7));
      if (c.fade > 0) c.fade = Math.min(1, c.fade + dt / (this.reduced ? 0.3 : 0.8));
      if (c.pulse > 0) c.pulse = Math.max(0, c.pulse - dt * 1.8);
      if (!this.reduced) c.phase += c.look.wobble * dt;
    }
    if (this.target && !this.reduced) this.target.phase += this.target.look.wobble * dt;
    if (!this.reduced) this.move(dt);
    for (let i = this.creatures.length - 1; i >= 0; i--) {
      const c = this.creatures[i];
      if (c.fade >= 1) {
        if (this.focus === c) this.focus = null;
        this.creatures.splice(i, 1);
      }
    }
    this.updateParticles(dt);
    if (this.play) {
      this.updateEvents(dt);
      if (this.auto && !this.over) this.updateAuto(dt);
      if (!this.over) this.elapsed += dt;
    } else {
      this.updateEcology(dt);
    }
  }

  move(dt) {
    const list = this.creatures;
    const { left, right, top, bottom } = this.bounds();
    const speed = this.play ? 16 : 11;
    const cx = (left + right) / 2;
    const cy = (top + bottom) / 2;
    for (const c of list) {
      if (c.hold > 0) {
        c.vx *= 1 - Math.min(1, dt * 8);
        c.vy *= 1 - Math.min(1, dt * 8);
        continue;
      }
      c.heading += (Math.random() - 0.5) * dt * 1.6;
      if (c.x < left || c.x > right || c.y < top || c.y > bottom) {
        c.heading += wrapAngle(Math.atan2(cy - c.y, cx - c.x) - c.heading) * dt;
      }
      c.vx += (Math.cos(c.heading) * speed - c.vx) * dt * 0.8;
      c.vy += (Math.sin(c.heading) * speed - c.vy) * dt * 0.8;
      if (c.x < left) c.vx += (left - c.x) * dt * 2;
      else if (c.x > right) c.vx -= (c.x - right) * dt * 2;
      if (c.y < top) c.vy += (top - c.y) * dt * 2;
      else if (c.y > bottom) c.vy -= (c.y - bottom) * dt * 2;
    }
    // keep a little room between them
    for (let i = 0; i < list.length; i++) {
      const a = list[i];
      if (a.fade > 0) continue;
      const ra = reach(a.look) * this.scale;
      for (let j = i + 1; j < list.length; j++) {
        const b = list[j];
        if (b.fade > 0) continue;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const min = (ra + reach(b.look) * this.scale) * 0.55;
        const d2 = dx * dx + dy * dy;
        if (d2 >= min * min || d2 < 1e-6) continue;
        const d = Math.sqrt(d2);
        const push = ((min - d) / min) * 40 * dt;
        const ux = dx / d;
        const uy = dy / d;
        if (a.hold === 0) {
          a.vx -= ux * push;
          a.vy -= uy * push;
        }
        if (b.hold === 0) {
          b.vx += ux * push;
          b.vy += uy * push;
        }
      }
    }
    for (const c of list) {
      c.x += c.vx * dt;
      c.y += c.vy * dt;
      const moving = c.vx * c.vx + c.vy * c.vy > 4;
      if (moving && c.hold === 0) {
        // along the direction of travel, whichever end first
        let turn = wrapAngle(Math.atan2(c.vy, c.vx) - c.angle);
        if (turn > Math.PI / 2) turn -= Math.PI;
        else if (turn < -Math.PI / 2) turn += Math.PI;
        c.angle += turn * Math.min(1, dt * 0.7);
      }
    }
  }

  updateEcology(dt) {
    this.clock -= dt;
    if (this.clock <= 0) {
      this.clock = 2 + Math.random() * 2;
      this.ambientBirth();
    }
    this.shimmer -= dt;
    if (this.shimmer <= 0) {
      this.shimmer = 3 + Math.random() * 3;
      const alive = this.eligible(null);
      if (alive.length) {
        const c = alive[Math.floor(Math.random() * alive.length)];
        if (mutate(c.genome, this.rng, 0.35, 0.2) && this.theme) this.burst(c.x, c.y, 4, c.colors.strand1);
      }
    }
  }

  ambientBirth() {
    const alive = this.eligible(null);
    if (alive.length < 2) return;
    const a = alive[Math.floor(Math.random() * alive.length)];
    let b = null;
    let near = Number.POSITIVE_INFINITY;
    for (const c of alive) {
      const d = (c.x - a.x) ** 2 + (c.y - a.y) ** 2;
      if (c !== a && d < near) {
        near = d;
        b = c;
      }
    }
    let genome;
    // now and then a newcomer, so the swarm doesn't all end up alike
    if (Math.random() < 0.2) genome = randomGenome(this.rng);
    else {
      genome = crossover(a.genome, b.genome, this.rng).first;
      mutate(genome, this.rng);
    }
    const child = makeCreature(genome, (a.x + b.x) / 2, (a.y + b.y) / 2);
    child.born = this.born++;
    child.grow = 0;
    this.creatures.push(child);
    if (this.theme) {
      paint(child, this.theme);
      this.burst(child.x, child.y, 5, child.colors.strand1);
    }
    let oldest = null;
    for (const c of alive) if (c !== a && c !== b && (!oldest || c.born < oldest.born)) oldest = c;
    if (oldest) oldest.fade = 0.001;
  }

  updateEvents(dt) {
    const events = this.events;
    for (let i = events.length - 1; i >= 0; i--) {
      const e = events[i];
      e.t += dt / e.duration;
      if (!this.reduced && e.t < BIRTH) {
        const open = Math.sin(Math.PI * clamp01(e.t / BIRTH));
        e.a.open = Math.max(e.a.open, open);
        e.b.open = Math.max(e.b.open, open);
      }
      if (!e.sparked && e.t >= 0.45) {
        e.sparked = true;
        this.mutationSparks(e);
      }
      if (!e.born && e.t >= BIRTH) this.birth(e);
      if (e.t >= 1) {
        e.a.hold -= 1;
        e.b.hold -= 1;
        events.splice(i, 1);
      }
    }
  }

  updateAuto(dt) {
    this.autoClock += dt;
    const step = this.reduced ? 0.25 : 0.28;
    if (this.autoStage === 0) {
      const candidates = this.eligible(null);
      if (candidates.length < 2) return;
      this.winnerA = tournament(candidates, this.rng, this.contendersA);
      this.contendersB.length = 0;
      this.winnerB = null;
      this.autoStage = 1;
      this.autoClock = 0;
    } else if (this.autoStage === 1 && this.autoClock >= step) {
      const a = this.winnerA;
      if (a.fade > 0) {
        this.autoStage = 0;
        return;
      }
      a.pulse = 1;
      a.selected = true;
      a.hold += 1;
      this.winnerB = tournament(this.eligible(a), this.rng, this.contendersB);
      this.autoStage = this.winnerB ? 2 : 0;
      this.autoClock = 0;
    } else if (this.autoStage === 2 && this.autoClock >= step) {
      const a = this.winnerA;
      const b = this.winnerB;
      a.selected = false;
      a.hold -= 1;
      if (b.fade === 0) {
        b.pulse = 1;
        this.breed(a, b, false);
        this.emit();
      }
      this.autoStage = 3;
      this.autoClock = 0;
    } else if (this.autoStage === 3 && this.autoClock >= step * 1.1) {
      this.contendersA.length = 0;
      this.contendersB.length = 0;
      this.winnerA = null;
      this.winnerB = null;
      this.autoStage = 0;
    }
  }

  // ---------- particles and labels ----------

  burst(x, y, count, color) {
    const still = this.reduced;
    for (let i = 0; i < count; i++) {
      const p = this.pnext;
      this.pnext = (p + 1) % PARTICLES;
      const angle = Math.random() * TAU;
      const speed = still ? 0 : 30 + Math.random() * 80;
      this.px[p] = x + (still ? (Math.random() - 0.5) * 30 : 0);
      this.py[p] = y + (still ? (Math.random() - 0.5) * 30 : 0);
      this.pvx[p] = Math.cos(angle) * speed;
      this.pvy[p] = Math.sin(angle) * speed;
      this.pmax[p] = 0.5 + Math.random() * 0.5;
      this.plife[p] = this.pmax[p];
      this.psize[p] = 2.5 + Math.random() * 3;
      this.pcolor[p] = color;
    }
  }

  updateParticles(dt) {
    const drag = 1 - Math.min(1, dt * 2.5);
    for (let p = 0; p < PARTICLES; p++) {
      if (this.plife[p] <= 0) continue;
      this.plife[p] -= dt;
      this.px[p] += this.pvx[p] * dt;
      this.py[p] += this.pvy[p] * dt;
      this.pvx[p] *= drag;
      this.pvy[p] *= drag;
    }
    for (const label of this.labels) {
      if (label.life <= 0) continue;
      label.life -= dt;
      if (!this.reduced) label.y -= dt * 22;
    }
  }

  label(text, x, y) {
    const label = this.labels[this.lnext];
    this.lnext = (this.lnext + 1) % LABELS;
    label.text = text;
    label.x = x;
    label.y = y;
    label.life = 1.2;
  }

  /** Where gene k of a pair's strips is drawn, and the rows' y. */
  stripGeometry(e) {
    const width = CELL * e.first.length;
    const x0 = Math.max(8, Math.min(this.width - width - 8, e.x - width / 2));
    return { x0, rowA: e.y - CELL - 2, rowB: e.y + 2 };
  }

  mutationSparks(e) {
    if (this.reduced || !this.theme) return;
    const { x0, rowA, rowB } = this.stripGeometry(e);
    for (let k = 0; k < e.first.length; k++) {
      if (e.mask1 & (1 << k)) this.burst(x0 + k * CELL + CELL / 2, rowA + CELL / 2, 3, this.theme.primary);
      if (e.mask2 & (1 << k)) this.burst(x0 + k * CELL + CELL / 2, rowB + CELL / 2, 3, this.theme.primary);
    }
  }

  // ---------- drawing ----------

  /** Ambient: fainter behind the hero's text and toward the bottom edge. */
  shade(x, y) {
    const w = this.width;
    const h = this.height;
    const column = Math.min(w * 0.3, 330);
    const dx = Math.abs(x - w / 2);
    let f = dx < column ? 0.35 : dx < column + 140 ? 0.35 + (0.65 * (dx - column)) / 140 : 1;
    if (y > h * 0.75) f *= Math.max(0, 1 - (y - h * 0.75) / (h * 0.25));
    return f;
  }

  draw() {
    const { ctx, dpr, theme } = this;
    if (!theme || !this.width) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const base = this.play ? 1 : theme.dark ? 0.36 : 0.32;

    for (const c of this.creatures) {
      if (c.grow <= 0) continue;
      if (c.painted !== theme.version) paint(c, theme);
      let alpha = base * (1 - c.fade) * Math.min(1, c.grow * 1.6);
      if (!this.play) alpha *= this.shade(c.x, c.y);
      else if (this.over && c !== this.winner) alpha *= 0.3;
      if (alpha <= 0.01) continue;
      const scale = this.scale * (0.3 + 0.7 * easeOutBack(c.grow)) * (1 + c.pulse * 0.12);
      drawHelix(ctx, c.look, c.colors, c.x, c.y, c.angle, scale, c.phase, c.open, alpha, dpr);
    }

    if (this.play) this.drawPlay();
    this.drawParticles();
    if (this.targetCtx && this.target) this.drawAlone(this.targetCanvas, this.targetCtx, this.target);
    if (this.winnerCtx && this.winner) {
      this.drawAlone(this.winnerCanvas, this.winnerCtx, this.winner);
      if (this.resultTargetCtx) this.drawAlone(this.resultTarget, this.resultTargetCtx, this.target);
    }
  }

  drawPlay() {
    const { ctx, dpr, theme } = this;
    const ringOf = (c, extra) => reach(c.look) * this.scale * 0.8 + 8 + extra;

    for (const c of this.contendersA) ring(ctx, c.x, c.y, ringOf(c, 0), theme.ink, 1.5, 0.45, dpr, DASH);
    for (const c of this.contendersB) ring(ctx, c.x, c.y, ringOf(c, 0), theme.ink, 1.5, 0.45, dpr, DASH);
    for (const c of this.creatures) {
      if (c.selected) ring(ctx, c.x, c.y, ringOf(c, 2), theme.primary, 2.5, 0.95, dpr);
      if (c.pulse > 0) ring(ctx, c.x, c.y, ringOf(c, 4 + (1 - c.pulse) * 26), theme.primary, 2, c.pulse * 0.8, dpr);
    }
    if (this.focus && this.focus.fade === 0) {
      ring(ctx, this.focus.x, this.focus.y, ringOf(this.focus, 6), theme.primary, 2, 0.9, dpr, DASH);
    }
    if (this.winner) {
      const w = this.winner;
      const beat = this.reduced ? 0 : Math.sin(this.time * 4) * 4;
      ring(ctx, w.x, w.y, ringOf(w, 8 + beat), theme.primary, 3, 0.9, dpr);
      ring(ctx, w.x, w.y, ringOf(w, 20 - beat), theme.primary, 1.5, 0.4, dpr);
    }

    for (const e of this.events) if (e.cells && e.t < BIRTH + 0.02) this.drawStrips(e);

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.font = "600 15px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillStyle = theme.primary;
    for (const label of this.labels) {
      if (label.life <= 0) continue;
      ctx.globalAlpha = Math.min(1, label.life * 1.5);
      ctx.fillText(label.text, label.x, label.y);
    }
  }

  /**
   * A crossover, drawn as the parents' genes: a strip of cells from each
   * parent meets in the middle, the cells between the two cut points change
   * rows, the mutated ones flash, then the strips become the two children.
   */
  drawStrips(e) {
    const { ctx, dpr, theme } = this;
    const p = e.t;
    const arrive = easeInOut(clamp01(p / 0.2));
    const swap = easeInOut(clamp01((p - 0.2) / 0.22));
    const flash = clamp01((p - 0.45) / 0.08);
    const leave = clamp01((p - (BIRTH - 0.08)) / 0.1);
    const alpha = arrive * (1 - leave);
    if (alpha <= 0.01) return;
    const { x0, rowA, rowB } = this.stripGeometry(e);
    const n = e.first.length;
    const shrink = 0.35 + 0.65 * arrive;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.globalAlpha = alpha;
    for (let k = 0; k < n; k++) {
      const inside = k >= e.from && k < e.to;
      const across = inside ? swap : 0;
      const bow = Math.sin(Math.PI * across) * CELL * 0.45;
      const size = (CELL - 1.5) * shrink;
      // each parent's strip starts at the parent and flies to the middle
      const cellX = x0 + k * CELL;
      for (let strand = 0; strand < 2; strand++) {
        const parent = strand ? e.b : e.a;
        const homeY = strand ? rowB : rowA;
        const otherY = strand ? rowA : rowB;
        const x = parent.x + (cellX - (x0 + (n * CELL) / 2)) * shrink + (x0 + (n * CELL) / 2 - parent.x) * arrive;
        const y = parent.y + (homeY + (otherY - homeY) * across - parent.y) * arrive;
        const endsFirst = strand ? inside : !inside;
        const mutated = (endsFirst ? e.mask1 : e.mask2) & (1 << k);
        const colors = strand ? e.cells.b : e.cells.a;
        const after = endsFirst ? e.cells.first : e.cells.second;
        ctx.fillStyle = mutated && flash > 0.5 ? after[k] : colors[k];
        ctx.fillRect(x + (strand ? -bow : bow), y, size, size);
        if (mutated && flash > 0 && flash < 1) {
          ctx.strokeStyle = theme.primary;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x + (strand ? -bow : bow) - 1.5, y - 1.5, size + 3, size + 3);
        }
      }
    }
    // the cut points
    if (swap > 0 && swap < 1) {
      ctx.globalAlpha = alpha * Math.sin(Math.PI * swap);
      ctx.strokeStyle = theme.ink;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0 + e.from * CELL - 0.75, rowA - 4);
      ctx.lineTo(x0 + e.from * CELL - 0.75, rowB + CELL + 4);
      ctx.moveTo(x0 + e.to * CELL - 0.75, rowA - 4);
      ctx.lineTo(x0 + e.to * CELL - 0.75, rowB + CELL + 4);
      ctx.stroke();
    }
  }

  drawParticles() {
    const { ctx, dpr } = this;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const base = this.play ? 1 : 0.5;
    for (let p = 0; p < PARTICLES; p++) {
      const life = this.plife[p];
      if (life <= 0) continue;
      const t = life / this.pmax[p];
      const x = this.px[p];
      const y = this.py[p];
      ctx.globalAlpha = base * t * (this.play ? 1 : this.shade(x, y));
      ctx.fillStyle = this.pcolor[p];
      ctx.beginPath();
      sparkle(ctx, x, y, this.psize[p] * (0.4 + 0.6 * t));
      ctx.fill();
    }
  }

  /** One creature, fitted into a small canvas of its own (the target, the winner). */
  drawAlone(canvas, ctx, t) {
    const dpr = this.dpr;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return;
    const pw = Math.round(w * dpr);
    const ph = Math.round(h * dpr);
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw;
      canvas.height = ph;
    }
    if (t.painted !== this.theme.version) paint(t, this.theme);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;
    ctx.clearRect(0, 0, pw, ph);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const look = t.look;
    const tall = look.radius * 3.2 + Math.abs(look.bend) * look.length * 0.3 + 8;
    const scale = Math.min(1.3, (w - 16) / (look.length + look.thickness * 2), (h - 8) / tall);
    drawHelix(ctx, look, t.colors, w / 2, h / 2 + (look.bend * look.length * scale) / 6, 0, scale, t.phase, 0, 1, dpr);
  }
}
