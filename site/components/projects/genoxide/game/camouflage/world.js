// The game: moths resting on bark, a bird (you, or genoxide) that catches the ones it can see,
// and the survivors that breed the next generation. Each level adds one idea of genoxide's.

import { EYESPOTS, breed, createRng, front, randomGenome, tournament } from "./ga";
import { HABITATS, paintHabitat, patchAt } from "./habitat";
import { GENE_LABELS, WINGSPAN, decode, drawMoth, drawStrip, giveaway, onMoth, spark, stripWidth, visibility } from "./moth";

const TAU = Math.PI * 2;
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const easeInOut = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

/**
 * The levels. `habitat` is what the bark shows (two for the island level), `population` the
 * moths per habitat, `catches` the moths the bird takes each generation, `goal` the average
 * camouflage that ends the level, and `concept` what of genoxide it shows.
 */
export const LEVELS = [
  {
    id: "selection",
    title: "Natural selection",
    habitats: ["birch"],
    population: 20,
    catches: 10,
    goal: 0.72,
    maxGenerations: 40,
    intro:
      "You're a bird. Catch the moths you can see: the ones you miss live on and have the next generation's moths, which inherit their genes. Watch the moths come to match the bark.",
    concept: "Selection, crossover and mutation: a genetic algorithm.",
    code: "Ga::builder(Real::uniform(5, 0.0..=1.0)?)\n    .select(Tournament::new(3)?)\n    .crossover(PointCrossover::one_point())\n    .mutate(GaussianMutation::per_gene(0.2, 0.12)?)",
  },
  {
    id: "change",
    title: "The world changes",
    habitats: ["birch", "soot"],
    population: 20,
    catches: 10,
    goal: 0.72,
    maxGenerations: 40,
    intro:
      "The moths match their pale birch. Then soot darkens the bark, a little more each generation. What hid them now gives them away: keep hunting, and the population follows the change.",
    concept: "A fitness function that changes during a run: re-evaluate and carry on.",
    code: "Engine::new(ga, fitness)\n    .control(|ga, _| {\n        soot.darken();\n        ga.reevaluate()\n    })",
  },
  {
    id: "islands",
    title: "Two habitats",
    habitats: ["birch", "soot"],
    split: true,
    population: 12,
    catches: 8,
    goal: 0.7,
    maxGenerations: 40,
    migration: 3,
    intro:
      "Two trees, one pale and one sooty, each with its own population. Every 3 generations a moth flies across. Each population adapts to its own tree: isolated, they stay different, and migrants bring new genes.",
    concept: "The island model: populations that evolve apart and exchange migrants.",
    code: "Islands::builder(vec![pale, sooty])\n    .topology(Topology::Ring)\n    .interval(3)\n    .migrants(1)",
  },
  {
    id: "tradeoff",
    title: "Hide or be seen",
    habitats: ["birch"],
    population: 20,
    catches: 10,
    generations: 14,
    maxGenerations: 14,
    mates: true,
    intro:
      "Now mates like bright eyespots: showy moths have more young. But you see them too. No moth can be both the best hidden and the most attractive: the best ones form a front of trade-offs.",
    concept: "Two objectives at once: the Pareto front, as NSGA-II finds it.",
    code: "Nsga2::builder(real, [Maximize, Maximize])\n// hidden from birds, attractive to mates",
  },
  {
    id: "machine",
    title: "genoxide hunts",
    habitats: ["lichen"],
    population: 20,
    catches: 10,
    goal: 0.72,
    maxGenerations: 80,
    auto: true,
    intro:
      "New bark, and this time genoxide is the bird: it takes the most visible of 3 random moths, again and again. A fitness function, a selection and a stop condition: the whole loop, at machine speed.",
    concept: "The engine: a fitness function, tournament selection and a stop condition.",
    code: "Engine::new(ga, camouflage)\n    .stop_when(Stop::target(0.72))\n    .run()?",
  },
];

// seconds for a newborn to fly to its place, between births, and for a catch
const FLIGHT = 0.9;
const BIRTH_GAP = 0.5;
const CATCH = 0.45;

export class CamouflageWorld {
  constructor(canvas, { reduced, onStats, onLevelEnd }) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.reduced = reduced;
    this.onStats = onStats;
    this.onLevelEnd = onLevelEnd;
    this.width = 1;
    this.height = 1;
    this.dpr = 1;
    this.top = 0;
    this.bottom = 0;
    this.theme = { ink: "#222", base: "#fff", primary: "#2563eb", dark: false };
    this.moths = [];
    this.effects = [];
    this.births = [];
    this.history = [];
    this.auto = false;
    this.running = false;
    this.focus = -1;
    this.level = null;
  }

  resize(width, height, dpr) {
    const changed = Math.abs(width - this.width) > 1 || Math.abs(height - this.height) > 1 || dpr !== this.dpr;
    this.width = Math.max(1, width);
    this.height = Math.max(1, height);
    this.dpr = dpr;
    this.canvas.width = Math.round(this.width * dpr);
    this.canvas.height = Math.round(this.height * dpr);
    if (changed && this.level) {
      this.paintHabitats();
      for (const moth of this.moths) {
        moth.x = Math.min(moth.x, this.width - 20);
        moth.y = Math.min(moth.y, this.height - this.bottom - 20);
      }
    }
  }

  setInsets(top, bottom) {
    this.top = top;
    this.bottom = bottom;
  }

  setTheme(theme) {
    this.theme = theme;
  }

  // ---- levels ------------------------------------------------------------------------------

  /** Starts `index`, with the genomes of `carry` (the last level's moths) if it uses them. */
  startLevel(index, seed, carry) {
    this.level = LEVELS[index];
    this.index = index;
    this.seed = seed;
    this.rng = createRng(seed);
    this.generation = 0;
    this.history = [];
    this.births = [];
    this.effects = [];
    this.assisted = false;
    this.auto = Boolean(this.level.auto);
    this.soot = 0;
    this.startTime = performance.now();
    this.paintHabitats();
    const groups = this.level.split ? 2 : 1;
    this.moths = [];
    for (let island = 0; island < groups; island++) {
      for (let i = 0; i < this.level.population; i++) {
        this.moths.push(this.newMoth(this.firstGenome(carry, i), island));
      }
    }
    // paused under the level's intro until `begin`
    this.phase = "intro";
    this.catchesLeft = this.level.catches;
    this.emit();
  }

  /** Starts the hunt of a level shown by `startLevel`. */
  begin() {
    if (this.phase !== "intro") return;
    this.startTime = performance.now();
    this.beginHunt();
  }

  // the genomes a level starts with: the last level's moths for "the world changes", moths
  // already pale for birch otherwise in that level, and random ones elsewhere
  firstGenome(carry, i) {
    const rng = this.rng;
    if (this.level.id === "change") {
      if (carry && carry.length) return Float64Array.from(carry[i % carry.length]);
      return Float64Array.from([0.88 + rng() * 0.1, 0.3 + rng() * 0.2, 0.3 + rng() * 0.3, rng() * 0.1, 0.3 + rng() * 0.4]);
    }
    if (this.level.mates) {
      // camouflaged already, with a little display: the trade-off is what evolves
      return Float64Array.from([0.85 + rng() * 0.12, 0.3 + rng() * 0.25, 0.3 + rng() * 0.4, rng() * 0.3, 0.3 + rng() * 0.4]);
    }
    return randomGenome(rng);
  }

  paintHabitats() {
    const keys = this.level.habitats;
    this.painted = keys.map((key, i) => paintHabitat(key, this.width, this.height, this.seed + i * 101, this.dpr));
  }

  // the habitat a point is in: the left or right tree of the island level
  islandAt(x) {
    return this.level.split && x > this.width / 2 ? 1 : 0;
  }

  /** The patch of bark under a moth: blended while soot settles in "the world changes". */
  patchOf(moth) {
    const radius = moth.look.span * 0.4;
    if (this.level.split) return patchAt(this.painted[moth.island].stats, moth.x, moth.y, radius);
    const birch = patchAt(this.painted[0].stats, moth.x, moth.y, radius);
    if (this.level.id !== "change" || this.soot <= 0) return birch;
    const soot = patchAt(this.painted[1].stats, moth.x, moth.y, radius);
    const t = this.soot;
    return {
      color: birch.color.map((v, k) => v * (1 - t) + soot.color[k] * t),
      roughness: birch.roughness * (1 - t) + soot.roughness * t,
    };
  }

  visibilityOf(moth) {
    return visibility(moth.look, this.patchOf(moth));
  }

  newMoth(genome, island) {
    const moth = {
      genome,
      look: decode(genome),
      seed: Math.floor(this.rng() * 1e9),
      island,
      x: 0,
      y: 0,
      angle: (this.rng() - 0.5) * 0.5,
      state: "rest",
      t: 0,
    };
    this.place(moth);
    return moth;
  }

  // a resting place in the moth's habitat, clear of the others and of the panels
  place(moth) {
    const margin = WINGSPAN;
    const left = this.level.split && moth.island === 1 ? this.width / 2 + margin / 2 : margin;
    const right = this.level.split && moth.island === 0 ? this.width / 2 - margin / 2 : this.width - margin;
    const top = this.top + margin;
    const bottom = Math.max(top + 10, this.height - this.bottom - margin);
    let best = null;
    let bestGap = -1;
    for (let attempt = 0; attempt < 24; attempt++) {
      const x = left + this.rng() * Math.max(1, right - left);
      const y = top + this.rng() * Math.max(1, bottom - top);
      let gap = Infinity;
      for (const other of this.moths) {
        if (other !== moth && other.state !== "caught") gap = Math.min(gap, Math.hypot(other.x - x, other.y - y));
      }
      if (gap > bestGap) {
        bestGap = gap;
        best = [x, y];
      }
      if (gap > WINGSPAN * 1.6) break;
    }
    moth.x = best[0];
    moth.y = best[1];
  }

  living(island) {
    return this.moths.filter((m) => m.state !== "caught" && (island === undefined || m.island === island));
  }

  /** The population's camouflage, 0 to 1: one minus the average visibility, per habitat. */
  camouflage(island) {
    const moths = this.living(island).filter((m) => m.state === "rest");
    if (!moths.length) return 0;
    return moths.reduce((sum, m) => sum + 1 - this.visibilityOf(m), 0) / moths.length;
  }

  beginHunt() {
    this.phase = "hunt";
    this.catchesLeft = this.level.catches;
    this.autoClock = 0;
    this.record();
    this.emit();
  }

  record() {
    const islands = this.level.split ? [this.camouflage(0), this.camouflage(1)] : null;
    this.history.push({ generation: this.generation, camouflage: this.camouflage(), islands });
  }

  // ---- input ----------------------------------------------------------------------------------

  /** A peck at (x, y): catches the top moth there, if it's still resting. */
  clickAt(x, y, slop) {
    if (this.phase !== "hunt" || this.auto) return;
    for (let i = this.moths.length - 1; i >= 0; i--) {
      const moth = this.moths[i];
      if (moth.state === "rest" && onMoth(moth.look, moth.x, moth.y, x, y, slop)) {
        this.catch(moth);
        return;
      }
    }
    this.effects.push({ kind: "miss", x, y, t: 0 });
  }

  catch(moth) {
    moth.state = "caught";
    moth.t = 0;
    moth.seen = this.visibilityOf(moth);
    moth.reason = giveaway(moth.look, this.patchOf(moth));
    this.catchesLeft -= 1;
    this.lastCaught = moth;
    this.effects.push({ kind: "peck", x: moth.x, y: moth.y, t: 0 });
    if (this.catchesLeft <= 0) this.breedSoon = this.auto ? 0.25 : 0.7;
    this.emit();
  }

  setAuto(on) {
    if (this.level?.auto) return;
    this.auto = on;
    if (on) this.assisted = true;
    this.emit();
  }

  /** Moves the keyboard focus to the next (+1) or previous (-1) resting moth. */
  moveFocus(step) {
    const resting = this.moths.filter((m) => m.state === "rest");
    if (!resting.length) return;
    const current = resting.indexOf(this.moths[this.focus]);
    const next = resting[(current + step + resting.length) % resting.length];
    this.focus = this.moths.indexOf(next);
  }

  chooseFocus() {
    const moth = this.moths[this.focus];
    if (moth && moth.state === "rest" && this.phase === "hunt" && !this.auto) this.catch(moth);
  }

  // ---- breeding -----------------------------------------------------------------------------

  // the survivors breed until each habitat has its population again
  breedGeneration() {
    this.phase = "breed";
    this.births = [];
    const groups = this.level.split ? 2 : 1;
    const children = [];
    for (let island = 0; island < groups; island++) {
      const parents = this.living(island);
      const missing = this.level.population - parents.length;
      for (let i = 0; i < missing && parents.length; i++) {
        const mother = this.chooseParent(parents);
        const father = this.chooseParent(parents.length > 1 ? parents.filter((p) => p !== mother) : parents);
        const birth = breed(mother.genome, father.genome, this.rng);
        children.push({ birth, island, mother, father });
      }
    }
    this.moths = this.moths.filter((m) => m.state !== "caught");
    const speed = this.speed();
    let delay = 0.2 / speed;
    const origin = this.panelOrigin();
    for (const { birth, island, mother, father } of children) {
      const moth = this.newMoth(birth.child, island);
      moth.state = "arriving";
      moth.t = -delay;
      moth.from = origin;
      this.moths.push(moth);
      this.births.push({ ...birth, mother: mother.genome, father: father.genome, moth, start: delay });
      delay += BIRTH_GAP / speed;
    }
    this.breedTime = 0;
    this.breedEnd = delay + FLIGHT / speed + 0.2;
    if (this.reduced) this.breedEnd = 0.6;
  }

  // a parent among the survivors: any of them, or in "hide or be seen" mostly the showy ones,
  // which mates prefer
  chooseParent(parents) {
    if (!this.level.mates) return parents[this.rng.int(parents.length)];
    const weights = parents.map((p) => 0.15 + p.genome[EYESPOTS] ** 1.5 * 3);
    let r = this.rng() * weights.reduce((a, b) => a + b, 0);
    for (let i = 0; i < parents.length; i++) {
      r -= weights[i];
      if (r <= 0) return parents[i];
    }
    return parents[parents.length - 1];
  }

  endGeneration() {
    this.generation += 1;
    this.births = [];
    for (const moth of this.moths) {
      moth.state = "rest";
      moth.from = null;
    }
    if (this.level.id === "change") this.soot = clamp01((this.generation - 1) / 4);
    if (this.level.split && this.generation % this.level.migration === 0) this.migrate();
    if (this.levelDone()) {
      this.record();
      this.finish();
      return;
    }
    this.beginHunt();
  }

  // a moth of each tree flies to the other, in place of a random one there
  migrate() {
    for (const from of [0, 1]) {
      const movers = this.living(from).filter((m) => !m.migrated);
      if (!movers.length) continue;
      const mover = movers[this.rng.int(movers.length)];
      const start = { x: mover.x, y: mover.y };
      mover.island = 1 - from;
      mover.migrated = true;
      this.place(mover);
      // it flies across, and rests (and can be caught) once it lands
      mover.state = "arriving";
      mover.from = start;
      mover.t = 0;
      this.effects.push({ kind: "migrate", from: start, to: { x: mover.x, y: mover.y }, t: 0 });
    }
    for (const moth of this.moths) moth.migrated = false;
  }

  levelDone() {
    const level = this.level;
    if (this.generation >= level.maxGenerations) return true;
    if (level.generations) return this.generation >= level.generations;
    if (level.id === "change" && this.soot < 1) return false;
    if (level.split) return this.camouflage(0) >= level.goal && this.camouflage(1) >= level.goal;
    return this.camouflage() >= level.goal;
  }

  finish() {
    this.phase = "done";
    const reached = this.level.generations ? true : this.camouflageMet();
    this.emit();
    this.onLevelEnd?.({
      index: this.index,
      id: this.level.id,
      seed: this.seed,
      generations: this.generation,
      camouflage: this.camouflage(),
      islands: this.level.split ? [this.camouflage(0), this.camouflage(1)] : null,
      history: this.history.slice(),
      front: this.level.mates ? this.frontPoints() : null,
      reached,
      assisted: this.assisted,
      seconds: (performance.now() - this.startTime) / 1000,
      genomes: this.living().map((m) => Array.from(m.genome)),
    });
  }

  camouflageMet() {
    const level = this.level;
    if (level.split) return this.camouflage(0) >= level.goal && this.camouflage(1) >= level.goal;
    return this.camouflage() >= level.goal;
  }

  /** Each resting moth as (hidden, showy): its camouflage and its eyespots, and the front. */
  frontPoints() {
    const points = this.living().map((m) => ({ x: 1 - this.visibilityOf(m), y: m.genome[EYESPOTS] }));
    return { points, front: front(points) };
  }

  speed() {
    if (this.level.auto) return 3.5;
    return this.auto ? 2 : 1;
  }

  // where the births panel is, where newborns fly from
  panelOrigin() {
    return { x: this.width / 2, y: this.height - this.bottom - 70 };
  }

  emit() {
    const level = this.level;
    this.onStats?.({
      index: this.index,
      generation: this.generation,
      camouflage: this.camouflage(),
      islands: level.split ? [this.camouflage(0), this.camouflage(1)] : null,
      goal: level.goal ?? null,
      generations: level.generations ?? null,
      catchesLeft: this.catchesLeft,
      catches: level.catches,
      phase: this.phase,
      auto: this.auto,
      soot: this.soot,
      history: this.history.map((h) => h.camouflage),
    });
  }

  // ---- animation ------------------------------------------------------------------------------

  start() {
    if (this.running) return;
    this.running = true;
    this.last = performance.now();
    const tick = (now) => {
      if (!this.running) return;
      const dt = Math.min(0.05, (now - this.last) / 1000);
      this.last = now;
      this.step(dt);
      this.draw();
      this.frame = requestAnimationFrame(tick);
    };
    this.frame = requestAnimationFrame(tick);
  }

  stop() {
    this.running = false;
    cancelAnimationFrame(this.frame);
  }

  step(dt) {
    if (!this.level) return;
    const speed = this.speed();
    for (const moth of this.moths) {
      moth.t += dt * (moth.state === "caught" ? 1 : speed);
      // a migrant lands during the hunt
      if (moth.state === "arriving" && this.phase === "hunt" && moth.t >= FLIGHT) {
        moth.state = "rest";
        moth.from = null;
      }
    }
    for (const effect of this.effects) effect.t += dt;
    this.effects = this.effects.filter((e) => e.t < (e.kind === "migrate" ? 1.4 : 0.8));
    this.moths = this.moths.filter((m) => m.state !== "caught" || m.t < CATCH || this.phase === "hunt");
    if (this.phase === "hunt") {
      if (this.auto && this.catchesLeft > 0) {
        this.autoClock += dt;
        const every = this.level.auto ? 0.12 : 0.32;
        if (this.autoClock >= every) {
          this.autoClock = 0;
          const resting = this.moths.filter((m) => m.state === "rest");
          if (resting.length) this.catch(tournament(resting, (m) => this.visibilityOf(m), 3, this.rng));
        }
      }
      if (this.breedSoon != null) {
        this.breedSoon -= dt;
        if (this.breedSoon <= 0) {
          this.breedSoon = null;
          this.breedGeneration();
        }
      }
    } else if (this.phase === "breed") {
      this.breedTime += dt;
      if (this.breedTime >= this.breedEnd) this.endGeneration();
    }
  }

  draw() {
    const { ctx, dpr } = this;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (!this.level) return;
    this.drawBark();
    for (const moth of this.moths) this.drawOne(moth);
    for (const effect of this.effects) this.drawEffect(effect);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const focused = this.moths[this.focus];
    if (focused && focused.state === "rest" && document.activeElement === this.canvas) {
      ctx.beginPath();
      ctx.arc(focused.x, focused.y, focused.look.span * 0.62, 0, TAU);
      ctx.strokeStyle = this.theme.primary;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (this.phase === "breed" && this.births.length) this.drawBirths();
    else if (this.lastCaught && this.phase === "hunt" && this.lastCaught.t < 2.2) this.drawCaught(this.lastCaught);
    if (this.level.mates) this.drawFront();
    if (this.level.split) this.drawDivider();
  }

  drawBark() {
    const { ctx, width } = this;
    const [first, second] = this.painted;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (this.level.split) {
      const half = Math.round((width / 2) * this.dpr);
      ctx.drawImage(first.canvas, 0, 0, half, first.canvas.height, 0, 0, half, first.canvas.height);
      ctx.drawImage(second.canvas, half, 0, second.canvas.width - half, second.canvas.height, half, 0, second.canvas.width - half, second.canvas.height);
      return;
    }
    ctx.drawImage(first.canvas, 0, 0);
    if (this.level.id === "change" && this.soot > 0) {
      ctx.globalAlpha = easeInOut(this.soot);
      ctx.drawImage(second.canvas, 0, 0);
      ctx.globalAlpha = 1;
    }
  }

  drawOne(moth) {
    const { ctx, dpr } = this;
    if (moth.state === "caught") {
      const t = clamp01(moth.t / CATCH);
      drawMoth(ctx, moth.look, moth.seed, moth.x, moth.y - t * 18, moth.angle + t * 1.2, 1 - t * 0.6, 1 - t, dpr);
      return;
    }
    if (moth.state === "arriving" && moth.from) {
      if (moth.t < 0) return;
      const t = this.reduced ? 1 : clamp01(moth.t / FLIGHT);
      const e = easeInOut(t);
      const x = moth.from.x + (moth.x - moth.from.x) * e;
      const y = moth.from.y + (moth.y - moth.from.y) * e - Math.sin(Math.PI * t) * 60;
      const flap = t < 1 ? 0.55 + 0.45 * Math.abs(Math.cos(moth.t * 18)) : 1;
      drawMoth(ctx, moth.look, moth.seed, x, y, moth.angle * e, flap, 1, dpr);
      return;
    }
    drawMoth(ctx, moth.look, moth.seed, moth.x, moth.y, moth.angle, 1, 1, dpr);
  }

  drawEffect(effect) {
    const { ctx, dpr } = this;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const t = effect.t / 0.8;
    if (effect.kind === "peck") {
      ctx.beginPath();
      ctx.arc(effect.x, effect.y, 10 + 26 * t, 0, TAU);
      ctx.strokeStyle = `rgb(255 255 255 / ${0.8 * (1 - t)})`;
      ctx.lineWidth = 3 * (1 - t);
      ctx.stroke();
    } else if (effect.kind === "miss") {
      ctx.beginPath();
      ctx.arc(effect.x, effect.y, 4 + 6 * t, 0, TAU);
      ctx.strokeStyle = `rgb(0 0 0 / ${0.35 * (1 - t)})`;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    } else if (effect.kind === "migrate") {
      const p = clamp01(effect.t / 1.2);
      ctx.beginPath();
      ctx.moveTo(effect.from.x, effect.from.y);
      ctx.quadraticCurveTo((effect.from.x + effect.to.x) / 2, Math.min(effect.from.y, effect.to.y) - 80, effect.to.x, effect.to.y);
      ctx.strokeStyle = `rgb(255 255 255 / ${0.7 * (1 - p)})`;
      ctx.setLineDash([5, 6]);
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // a box for the panels: the theme's surface, a little transparent
  box(x, y, w, h) {
    const { ctx } = this;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 12);
    ctx.fillStyle = this.theme.panel;
    ctx.fill();
    ctx.strokeStyle = this.theme.line;
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  text(value, x, y, size, weight = 500, align = "left", alpha = 1) {
    const { ctx } = this;
    ctx.font = `${weight} ${size}px ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = align;
    ctx.textBaseline = "middle";
    ctx.globalAlpha = alpha;
    ctx.fillStyle = this.theme.ink;
    ctx.fillText(value, x, y);
    ctx.globalAlpha = 1;
  }

  // the moth just caught: its DNA, and how visible it was
  drawCaught(moth) {
    const { ctx } = this;
    const w = 34;
    const h = 18;
    const gap = 4;
    const sw = stripWidth(w, gap);
    const bw = Math.max(sw + 24, 200);
    const bh = 82;
    const x = Math.min(this.width - bw - 8, Math.max(8, moth.x - bw / 2));
    const y = Math.max(this.top + 6, moth.y - moth.look.span - bh);
    this.box(x, y, bw, bh);
    this.text(`Caught: ${Math.round(moth.seen * 100)}% visible`, x + 12, y + 14, 11, 600);
    this.text(`given away: ${moth.reason}`, x + 12, y + 29, 11, 400, "left", 0.7);
    drawStrip(ctx, moth.genome, x + 12, y + 42, w, h, gap, this.theme.ink);
    this.labels(x + 12, y + 42 + h + 9, w, gap);
  }

  // the births of this generation, one at a time: two parents' DNA, where crossover cut them,
  // and the child's, with its mutations
  drawBirths() {
    const { ctx } = this;
    const current = this.births.findLast((b) => this.breedTime >= b.start) ?? this.births[0];
    const index = this.births.indexOf(current) + 1;
    const w = 34;
    const h = 18;
    const gap = 4;
    const sw = stripWidth(w, gap);
    const bw = sw * 2 + 110;
    const bh = 124;
    const x = this.width / 2 - bw / 2;
    const y = this.height - this.bottom - bh - 8;
    this.box(x, y, bw, bh);
    this.text(`Birth ${index} of ${this.births.length}`, x + 14, y + 16, 12, 600);
    this.text("one-point crossover, then mutation", x + bw - 14, y + 16, 11, 400, "right", 0.6);
    const mother = this.theme.mother;
    const father = this.theme.father;
    const left = x + 70;
    const top = y + 46;
    this.text("mother", x + 14, top + h / 2, 11, 500, "left", 0.7);
    this.text("father", x + 14, top + h + 14 + h / 2, 11, 500, "left", 0.7);
    this.labels(left, top - 9, w, gap);
    drawStrip(ctx, current.mother, left, top, w, h, gap, this.theme.ink, (k) => (k < current.cut ? mother : null));
    drawStrip(ctx, current.father, left, top + h + 14, w, h, gap, this.theme.ink, (k) => (k >= current.cut ? father : null));
    // the cut, through both parents
    const cx = left + current.cut * (w + gap) - gap / 2;
    ctx.beginPath();
    ctx.moveTo(cx, top - 6);
    ctx.lineTo(cx, top + 2 * h + 20);
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = this.theme.ink;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.setLineDash([]);
    // the child
    const childX = left + sw + 34;
    const childY = top + (h + 14) / 2;
    ctx.beginPath();
    ctx.moveTo(left + sw + 8, childY + h / 2);
    ctx.lineTo(childX - 8, childY + h / 2);
    ctx.strokeStyle = this.theme.ink;
    ctx.globalAlpha = 0.5;
    ctx.stroke();
    ctx.globalAlpha = 1;
    drawStrip(ctx, current.child, childX, childY, w, h, gap, this.theme.ink, (k) => (k < current.cut ? mother : father), current.mutated);
    this.text("child", childX + sw / 2, childY - 10, 11, 600, "center", 0.8);
    const note = current.mutated.length
      ? `${current.mutated.length} mutation${current.mutated.length > 1 ? "s" : ""}`
      : "no mutation";
    spark(ctx, childX + 4, childY + h + 16, 5, this.theme.ink);
    this.text(note, childX + 14, childY + h + 16, 11, 400, "left", 0.7);
  }

  // the genes' names under (or over) a strip
  labels(x, y, w, gap) {
    GENE_LABELS.forEach((label, k) => this.text(label.toLowerCase(), x + k * (w + gap) + w / 2, y, 9, 500, "center", 0.6));
  }

  // "hide or be seen": every moth by how hidden and how showy it is, and the front
  drawFront() {
    const { ctx } = this;
    const { points, front: best } = this.frontPoints();
    const size = Math.min(180, this.width * 0.32);
    const x = this.width - size - 16;
    const y = this.top + 12;
    this.box(x, y, size, size + 22);
    const px = (v) => x + 26 + v * (size - 38);
    const py = (v) => y + size - 12 - v * (size - 38);
    this.text("hidden →", x + size - 12, y + size + 6, 10, 500, "right", 0.6);
    ctx.save();
    ctx.translate(x + 12, y + size / 2);
    ctx.rotate(-Math.PI / 2);
    this.text("showy →", 0, 0, 10, 500, "center", 0.6);
    ctx.restore();
    const sorted = best.slice().sort((a, b) => a.x - b.x);
    ctx.beginPath();
    sorted.forEach((p, i) => (i ? ctx.lineTo(px(p.x), py(p.y)) : ctx.moveTo(px(p.x), py(p.y))));
    ctx.strokeStyle = this.theme.primary;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    for (const p of points) {
      ctx.beginPath();
      ctx.arc(px(p.x), py(p.y), best.includes(p) ? 3.5 : 2.5, 0, TAU);
      ctx.fillStyle = best.includes(p) ? this.theme.primary : this.theme.ink;
      ctx.globalAlpha = best.includes(p) ? 1 : 0.35;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    this.text("the front", x + 30, y + 14, 10, 600, "left", 0.8);
  }

  drawDivider() {
    const { ctx } = this;
    const x = this.width / 2;
    ctx.fillStyle = "rgb(0 0 0 / 0.35)";
    ctx.fillRect(x - 3, 0, 6, this.height);
    for (const [i, name] of [HABITATS.birch.name, HABITATS.soot.name].entries()) {
      const cx = i ? x + this.width / 4 : this.width / 4;
      const y = this.top + 18;
      ctx.font = "600 12px ui-sans-serif, system-ui, sans-serif";
      const w = ctx.measureText(name).width + 20;
      this.box(cx - w / 2, y - 11, w, 22);
      this.text(name, cx, y, 12, 600, "center");
    }
  }
}
