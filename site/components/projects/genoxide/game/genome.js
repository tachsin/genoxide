// The creatures' genomes and the steady-state GA that breeds them: real
// genes in [0, 1], two-point crossover, Gaussian mutation, tournament
// selection, the two worst replaced each generation. The same operators as
// genoxide's PointCrossover::two_point(), GaussianMutation and
// Scheme::SteadyState { replacements: 2 }, in a few lines of JavaScript.

/** mulberry32: a small seeded generator, so a round can be replayed. */
export function createRng(seed) {
  let s = seed >>> 0;
  const next = () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  next.int = (n) => Math.floor(next() * n);
  next.gauss = () => {
    const u = next() || 1e-12;
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
  };
  return next;
}

export function randomSeed() {
  const buffer = new Uint32Array(1);
  crypto.getRandomValues(buffer);
  return buffer[0];
}

// What each gene draws, and how much it counts when comparing two creatures:
// the ones you notice first (colour, length, twist) weigh the most.
export const GENES = [
  { key: "hue", weight: 2.2 },
  { key: "hueShift", weight: 1 },
  { key: "chroma", weight: 0.6 },
  { key: "length", weight: 1.6 },
  { key: "radius", weight: 1.2 },
  { key: "twist", weight: 1.4 },
  { key: "thickness", weight: 1 },
  { key: "rungs", weight: 0.8 },
  { key: "bend", weight: 1 },
  { key: "wobble", weight: 0.5 },
  { key: "pattern", weight: 1 },
  { key: "taper", weight: 0.7 },
];
export const GENE_COUNT = GENES.length;
const HUE = 0;
const PATTERN = 10;
const WEIGHT_SUM = GENES.reduce((sum, g) => sum + g.weight, 0);

export const MUTATION_RATE = 0.3;
export const MUTATION_SIGMA = 0.15;
export const TOURNAMENT_SIZE = 3;
/** A round is won when a creature is this similar to the target. */
export const MATCH = 0.88;

export const patternOf = (value) => Math.min(2, Math.floor(value * 3));

export function randomGenome(rng) {
  const genome = new Float64Array(GENE_COUNT);
  for (let k = 0; k < GENE_COUNT; k++) genome[k] = rng();
  return genome;
}

/** 1 for the same look, about 0.6 for two random creatures. */
export function similarity(a, b) {
  let distance = 0;
  for (let k = 0; k < GENE_COUNT; k++) {
    let d = Math.abs(a[k] - b[k]);
    if (k === HUE) d = Math.min(d, 1 - d) * 2;
    else if (k === PATTERN) d = Math.abs(patternOf(a[k]) - patternOf(b[k])) / 2;
    distance += GENES[k].weight * d;
  }
  return 1 - distance / WEIGHT_SUM;
}

/** Two-point crossover: the children swap the genes in [from, to). */
export function crossover(a, b, rng) {
  let from = rng.int(GENE_COUNT);
  let to = rng.int(GENE_COUNT);
  if (from > to) [from, to] = [to, from];
  to += 1;
  const first = Float64Array.from(a);
  const second = Float64Array.from(b);
  for (let k = from; k < to; k++) {
    first[k] = b[k];
    second[k] = a[k];
  }
  return { first, second, from, to };
}

/** Gaussian mutation, mirrored at the bounds (the hue wraps around). Returns a bit mask of the changed genes. */
export function mutate(genome, rng, rate = MUTATION_RATE, sigma = MUTATION_SIGMA) {
  let mask = 0;
  for (let k = 0; k < GENE_COUNT; k++) {
    if (rng() >= rate) continue;
    let v = genome[k] + rng.gauss() * sigma;
    if (k === HUE) v -= Math.floor(v);
    else {
      if (v < 0) v = -v;
      if (v > 1) v = 2 - v;
      v = Math.min(1, Math.max(0, v));
    }
    genome[k] = v;
    mask |= 1 << k;
  }
  return mask;
}

/**
 * Tournament selection: the fittest of `size` random candidates (ties: the
 * first picked), each with a `fitness`. The picks go into `contenders`.
 */
export function tournament(candidates, rng, contenders, size = TOURNAMENT_SIZE) {
  let best = null;
  contenders.length = 0;
  for (let i = 0; i < size && candidates.length; i++) {
    const pick = candidates[rng.int(candidates.length)];
    contenders.push(pick);
    if (!best || pick.fitness > best.fitness) best = pick;
  }
  return best;
}

/** A creature's look, decoded from its genes. Written into `out`, so nothing is allocated per frame. */
export function decode(genes, out) {
  out.hue = genes[0] * 360;
  out.hue2 = out.hue + (genes[1] - 0.5) * 180;
  out.chroma = 0.07 + genes[2] * 0.15;
  out.length = 34 + genes[3] * 70;
  out.radius = 3.5 + genes[4] * 9;
  out.turns = 0.8 + genes[5] * 2.7;
  out.thickness = 1 + genes[6] * 3;
  out.rungs = Math.max(3, Math.min(36, Math.round(out.turns * (3 + genes[7] * 7))));
  out.bend = (genes[8] - 0.5) * 1.2;
  out.wobble = 0.6 + genes[9] * 2.8;
  out.pattern = patternOf(genes[10]);
  out.taper = genes[11];
  return out;
}

/** Eases `shown` toward `genes` (the hue the short way round); returns whether it moved. */
export function easeGenes(shown, genes, amount) {
  let moved = false;
  for (let k = 0; k < GENE_COUNT; k++) {
    let d = genes[k] - shown[k];
    if (k === HUE) d -= Math.round(d);
    if (Math.abs(d) < 1e-4) {
      shown[k] = genes[k];
      continue;
    }
    let v = shown[k] + d * amount;
    if (k === HUE) v -= Math.floor(v);
    shown[k] = v;
    moved = true;
  }
  return moved;
}
