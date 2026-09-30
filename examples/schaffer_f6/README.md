---
title: Schaffer F6
category: continuous
summary: Minimize Schaffer's F6, a two-dimensional function of concentric rings whose first ring of local minima surrounds the global minimum behind a ridge, with SHADE, a particle swarm and a genetic algorithm from 30 seeds.
reference: "Schaffer, J. D., Caruana, R. A., Eshelman, L. J. and Das, R. (1989). A study of control parameters affecting online performance of genetic algorithms for function optimization. Proceedings of the Third International Conference on Genetic Algorithms, Morgan Kaufmann: 51-60."
reference_url: ""
optimum: "0 at the origin"
languages: [rust, python]
order: 79
---

# Schaffer F6

## The problem

Schaffer's F6 is a function of two variables to minimize:

```text
f(x₁, x₂) = 0.5 + (sin² √(x₁² + x₂²) − 0.5) / (1 + 0.001 (x₁² + x₂²))²,   x₁, x₂ in [−100, 100]
```

It comes from Schaffer, Caruana, Eshelman and Das (1989), a study of how the settings of a genetic
algorithm affect its performance, whose test suite it is part of. Their paper couldn't be read:
genoxide takes the definition and the bounds from Whitley, Mathias, Rana and Dzubera (1996, table
1, F9), who call it the sine envelope sine wave and credit Schaffer et al., and the CEC 2005 report
(Suganthan et al., 2005) states the same function.

The minimum is 0 at the origin, and it's the global minimum: the numerator is at least −0.5 and the
denominator at least 1, so f is at least 0, and 0 only where the denominator is 1. genoxide's
`problems::SchafferF6` gives it as proven.

## What makes it hard

f depends only on the distance r from the origin, and oscillates with it: sin² r is 0 at r = kπ and
1 in between, and the denominator flattens the oscillation away from the origin. So the landscape is
a set of concentric rings, of local minima near r = kπ and ridges between them:

| r | 0 | 1.5692 | 3.1385 | 4.7078 | 6.2771 | 9.4161 | 12.5555 |
|---|---|---|---|---|---|---|---|
| f | 0 (minimum) | 0.9975 (ridge) | 0.0097 (ring) | 0.9785 (ridge) | 0.0372 (ring) | 0.0782 (ring) | 0.1270 (ring) |

The first ring is only 0.0097 above the minimum, and a ridge at 0.9975, nearly the highest value of
the function, separates it from the central basin. The central basin, r < 1.57, is 0.02% of the
box. Each ring is a whole circle of equally good points, so a search that reaches a ring can move
along it freely, but only a jump across the ridge leads further in.

## Representation

A `Real` genome of 2 genes, each in [−100, 100]: the point (x₁, x₂) itself. The fitness is f, to
minimize. The function, its bounds and its minimum are genoxide's `problems::SchafferF6`.

## Algorithm

Three algorithms, each from seeds 1 to 30, with a budget of 50,000 evaluations per run, and a target
of 1e-6:

- SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78), a differential evolution, with
  genoxide's defaults: a population of 100; each trial point combines a parent with differences
  between other points of the population, and replaces the parent only if it is at least as good;
  the scale factor and crossover rate adapt from the trials that succeeded;
- for contrast, particle swarm optimization (Kennedy and Eberhart, 1995, Proceedings of ICNN'95:
  1942-1948) with 40 particles, in which every particle follows the best point of the whole swarm,
  and Clerc and Kennedy's constriction coefficients (2002, IEEE Transactions on Evolutionary
  Computation 6(1): 58-73);
- and a genetic algorithm of 100 individuals, with tournaments of 3, simulated binary crossover
  (Deb and Agrawal, 1995, Complex Systems 9(2): 115-148) with η = 15, and polynomial mutation of
  each gene with a probability of 0.5 and η = 20; generational, keeping the best.

A run is counted on the ring nearest its best point: ring k where r rounds to kπ, and the minimum
for r < π/2.

## Output

The first line gives the minimum, the seeds and the budget. Then a row per ring that some run ended
on: the lowest value found there, and how many runs of each algorithm ended there. The last row
counts the runs that come within 1e-6 of the minimum. In Python, `run` evaluates the function in
Rust, so both versions print the same table.

The page's plot shows the 30 runs of SHADE, each at its best point so far, over the function's
contour in the square [−25, 25]², where the runs end: the rings are too close together to draw over
the whole box. A curve gives the best and the median run's value, its distance above the minimum,
on a logarithmic axis.

[The project page](https://tachsin.gr/projects/genoxide/examples/schaffer-f6) plays this run back.

## Good results

A good result reaches 0. SHADE reaches it, within 1e-6, in all 30 runs, after a median of about
28,000 evaluations. Each point of its population is replaced only by a better trial point of its
own, rather than pulled towards the best point, as a particle is: the population closes in more
slowly than the swarm, and a ring found early doesn't draw it in.

The particle swarm, for contrast, reaches the minimum in 24 of the 30 runs, and the other 6 end on
the first ring, at 0.0097. The genetic algorithm ends in the central basin in 24 runs and on the
first ring in 6, but comes within 1e-6 of 0 in only 5. Its mutation's steps are relative to a
gene's range, 200: with η = 20, a step smaller than 0.001 has a probability of about 1 in 10,000,
so its points in the central basin approach the origin slowly.

Over seeds 1 to 1,000, SHADE reaches the minimum in 997 runs, after at most 46,900 evaluations,
and the other 3 end in the central basin, 1.2e-6 to 6.3e-5 above the minimum when the budget runs
out; the swarm reaches it in 76% of the runs, and the genetic algorithm in 15%. With the earlier
budget of 20,000 evaluations, SHADE's population of 100 is too slow to close in: it reached the
minimum from 1 of seeds 1 to 30.
