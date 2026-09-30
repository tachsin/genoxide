---
title: Ackley
category: continuous
summary: Minimize a 30-dimensional function with a deep hole in a nearly flat, rippled plain, where a swarm that follows its best particle gets stuck.
reference: "Ackley, D. H. (1987). A Connectionist Machine for Genetic Hillclimbing. Kluwer. Generalized by Bäck, T. (1996). Evolutionary Algorithms in Theory and Practice. Oxford University Press."
reference_url: "https://doi.org/10.1007/978-1-4613-1997-9"
optimum: "0 (at the origin)"
languages: [rust, python]
order: 49
trace_note: "Recorded from another run: PSO with a ring topology in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Ackley

## The problem

Ackley's function is the sum of two terms, one that depends on the distance to the origin and one
that ripples:

```text
f(x) = −20 exp(−0.2 √(Σ xᵢ² / n)) − exp(Σ cos(2πxᵢ) / n) + 20 + e,   each xᵢ in [−32, 32]
```

Ackley (1987) defined it in two dimensions, and Bäck (1996) generalized it to n. The bounds are
those of Yao, Liu and Lin (1999, f10); other papers use [−32.768, 32.768]. Its minimum is 0, at the
origin. Here n = 30.

The first term depends only on r = √(Σ xᵢ² / n), the root mean square of the genes. It is 0 at the
origin, 17.3 at r = 10 and 19.97 at r = 32: a funnel in the middle of a plain of height nearly 20.
The second term depends on the cosines of the genes. It is 0 when every gene is an integer, and at
most e − 1/e ≈ 2.35 when every gene is half an integer. It adds a ripple with a local minimum near
every integer point.

## What makes it hard

Far from the origin, the first term is almost flat: it changes by less than 3 between r = 10 and
the bounds. There, the ripples are what a search sees, and they point nowhere. A search that samples
only a small neighborhood finds a local minimum near an integer point, and learns nothing about the
funnel.

Near the origin, the funnel is steep, but the local minima continue into it. A gene near 1 instead
of 0 costs little: in 30 dimensions, 3 such genes give a value of about 1.16. A search must still
move them to 0, each across a ripple, while the other genes stay put.

## Representation

A `Real` genome of 30 genes, each in [−32, 32]: the point x itself. The fitness is f(x), to
minimize. The function is genoxide's `problems::Ackley`, which brings its bounds and its minimum.
30 is the dimension of Yao, Liu and Lin's comparison, and the one where the example below shows a
difference between the algorithms: in 10 dimensions, PSO solves Ackley too.

## Algorithm

Three runs, each with a budget of 10,000 evaluations per dimension, 300,000 in all, and a target of
1e-8.

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. It uses genoxide's
defaults: a population of 4 + ⌊3 ln 30⌋ = 14, a step size of 0.3 of each gene's range (19.2), and a
random start. With IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776), a run that has
converged starts again from a random point with twice the population. Its first steps are much
wider than a ripple, and it moves its mean by a weighted average of its best samples, not to a
single one: it follows the trend of the funnel more than the ripples.

Particle swarm optimization (Kennedy and Eberhart, 1995, Proceedings of ICNN'95: 1942-1948) moves
40 particles, each pulled towards the best point it has found and towards the best point of its
neighborhood. genoxide uses Clerc and Kennedy's constriction coefficients (2002, IEEE Transactions
on Evolutionary Computation 6(1): 58-73). The example runs it with two topologies, which differ
only in the neighborhood:

- global: every particle follows the best point of the whole swarm. Once a particle finds a good
  point, the whole swarm moves towards it at once;
- ring: the particles sit on a ring, and each follows the best of itself and its two neighbors.
  A good point spreads around the ring one particle at a time, so parts of the swarm search
  different regions for longer.

## Output

One line per run: the best value it found, to 6 decimals, and the evaluations it took. A run stops
as soon as it is within 1e-8 of the minimum, so a value of 0.000000 means that it met the target.
In Python, `run` evaluates the function in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/ackley) plays back another run:
PSO with a ring topology on Ackley in 2 dimensions, so that the population can be drawn on the
function's contour.

## Good results

The minimum is 0. CMA-ES reaches the target after 7,812 evaluations, without a restart. PSO with the
ring topology reaches it after 84,320 evaluations. PSO with the global topology doesn't: it ends at
1.50, where 5 genes are near ±0.91 and the others at 0. The swarm has contracted around its best
point, and no particle is left to look elsewhere.

The difference isn't the seed's. With seeds 1 to 10, CMA-ES takes 7,500 to 8,400 evaluations and
the ring 76,000 to 85,000, while the global topology ends between 1.16 and 2.96 every time.
