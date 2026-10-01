// The moths' genes and how they're inherited: the same ideas as genoxide's genetic algorithm, in
// a few lines of JavaScript. A genome is five numbers in [0, 1], one per trait; children come
// from two parents by one-point crossover (PointCrossover::one_point()) and Gaussian mutation
// mirrored at the bounds (GaussianMutation::per_gene(rate, sigma)).

export { createRng, randomSeed } from "../genome";

/**
 * The five genes, in the order of the DNA strip. Each one codes for one thing you can see on the
 * moth, and the strip draws each gene the way the moth shows it.
 */
export const GENES = [
  { key: "shade", label: "Shade", help: "how dark or pale the wings are" },
  { key: "tint", label: "Tint", help: "the colour of the wings, from grey-brown to green" },
  { key: "speckles", label: "Speckles", help: "how many flecks break up the wings" },
  { key: "eyespots", label: "Eyespots", help: "bright spots that attract mates, and birds" },
  { key: "size", label: "Size", help: "how big the moth is" },
];
export const GENE_COUNT = GENES.length;
export const SHADE = 0;
export const TINT = 1;
export const SPECKLES = 2;
export const EYESPOTS = 3;
export const SIZE = 4;

/** The chance that a child's gene mutates, and the size of a mutation. */
export const MUTATION_RATE = 0.2;
export const MUTATION_SIGMA = 0.12;

export function randomGenome(rng) {
  const genome = new Float64Array(GENE_COUNT);
  for (let k = 0; k < GENE_COUNT; k++) genome[k] = rng();
  return genome;
}

// a value mirrored back into [0, 1], as genoxide's Gaussian mutation does at the bounds
function mirror(value) {
  let v = value;
  for (let i = 0; i < 4 && (v < 0 || v > 1); i++) v = v < 0 ? -v : 2 - v;
  return Math.min(1, Math.max(0, v));
}

/**
 * A child of `mother` and `father`: the genes before a random cut from the mother, the rest from
 * the father (one-point crossover), then each gene mutated with probability `rate` by normal
 * noise of standard deviation `sigma`. Returns the child and what happened, for the DNA panel.
 */
export function breed(mother, father, rng, rate = MUTATION_RATE, sigma = MUTATION_SIGMA) {
  // the cut is between two genes: 1 to GENE_COUNT - 1 genes come from the mother
  const cut = 1 + rng.int(GENE_COUNT - 1);
  const child = new Float64Array(GENE_COUNT);
  for (let k = 0; k < GENE_COUNT; k++) child[k] = k < cut ? mother[k] : father[k];
  const inherited = Float64Array.from(child);
  const mutated = [];
  for (let k = 0; k < GENE_COUNT; k++) {
    if (rng() < rate) {
      child[k] = mirror(child[k] + sigma * rng.gauss());
      mutated.push(k);
    }
  }
  return { child, cut, inherited, mutated };
}

/**
 * Tournament selection of the predator: of `size` random moths, the most visible is caught.
 * genoxide's Tournament::new(size), with visibility as the fitness to minimize.
 */
export function tournament(moths, visibility, size, rng) {
  let caught = null;
  for (let i = 0; i < size; i++) {
    const moth = moths[rng.int(moths.length)];
    if (!caught || visibility(moth) > visibility(caught)) caught = moth;
  }
  return caught;
}

/**
 * The moths no other moth beats on both objectives: hidden (camouflage) and attractive (display).
 * The Pareto front, as genoxide's multi-objective algorithms find it.
 */
export function front(points) {
  return points.filter(
    (p) => !points.some((q) => q !== p && q.x >= p.x && q.y >= p.y && (q.x > p.x || q.y > p.y)),
  );
}
