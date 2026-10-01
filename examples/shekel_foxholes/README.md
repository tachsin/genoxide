---
title: Shekel's foxholes
category: continuous
summary: Minimize De Jong's fifth function, a plane at nearly 500 with 25 narrow holes of different depths, with CMA-ES without and with restarts, PSO and a GA from 30 seeds.
reference: "De Jong, K. A. (1975). An Analysis of the Behavior of a Class of Genetic Adaptive Systems. PhD thesis, University of Michigan. Function F5, after Shekel, J. (1971). Test functions for multimodal search techniques. Proceedings of the 5th Annual Princeton Conference on Information Sciences and Systems, Princeton University."
reference_url: "https://hdl.handle.net/2027.42/4507"
optimum: "0.99800 at (−31.97833, −31.97833) (best known)"
languages: [rust, python]
order: 84
---

# Shekel's foxholes

## The problem

Shekel's foxholes, the fifth function of De Jong's test bed, is a function of two variables to
minimize:

```text
1 / f(x) = 1/500 + Σⱼ₌₁²⁵ 1 / (j + (x₁ − a₁ⱼ)⁶ + (x₂ − a₂ⱼ)⁶),   x₁, x₂ in [−65.536, 65.536]
```

where the 25 points aⱼ are the grid (−32, −16, 0, 16, 32)², x₁ varying first: a₁ = (−32, −32), a₂ =
(−16, −32), …, a₂₅ = (32, 32).

It comes from De Jong's thesis (1975, appendix A.6), read in the scan that George Mason University's
EC lab published, where it is test function F5, "synthesized as suggested by Shekel (1971)", with cⱼ
= j, K = 500, this grid and these bounds; De Jong gives the minimum as ≅ 1. Yao, Liu and Lin (1999,
f14) restate it the same way. Shekel's own paper couldn't be read.

The best known minimum is 0.9980038377944502 at (−31.97833483565697, −31.978334837300796), in the
first hole: Newton's method, computed to 40 digits and rounded. The other holes pull it a little
towards the middle of the box, and the value at (−32, −32) is 0.9980038388186489. The other holes'
minima are about their j: 1.99203, 2.98211, 3.96825, 4.95049, … up to 23.8 for the last. It's not
proven global, and genoxide's `problems::ShekelFoxholes` gives it as a best known value.

## What makes it hard

Away from the holes, all 25 terms are tiny and f is nearly 500: half of the box is above 499.95.
Each hole is a flat-bottomed well, since the sixth powers are nearly 0 within about 1 of its center
and very large beyond: a random point falls below 10, in one of the ten deepest holes, with a
probability of 0.36%, and below 2, in the deepest, with 0.026%. Between the holes, the plane gives
no slope that points to a deeper one.

So a search that finds a hole and settles in it is stuck: to find a deeper one, it has to sample the
plane again, 16 away. And the holes are many and of all depths: the deepest is only a little deeper
than the next, 0.998 against 1.992.

## Representation

A `Real` genome of 2 genes, each in [−65.536, 65.536]: the point (x₁, x₂) itself. The fitness is f,
to minimize. The function, its bounds and its best known minimum are genoxide's
`problems::ShekelFoxholes`.

## Algorithm

Four algorithms, each from seeds 1 to 30, each run stopping once its value is within 1e-8 of the
best known minimum, or after 10,000 evaluations:

- CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195), with genoxide's
  defaults: a population of 6 and a step size of 0.3 of each gene's range, 39, from a random start;
- CMA-ES with IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776): a run that has
  converged starts again from a random point with twice the population;
- particle swarm optimization (Kennedy and Eberhart, 1995), 40 particles with Clerc and Kennedy's
  constriction coefficients and a global topology;
- a real-coded genetic algorithm (De Jong's were binary): a population of 50, tournaments of 3,
  simulated binary crossover (Deb and Agrawal, 1995) with η = 15 and polynomial mutation with η = 20
  at a rate of 1/2 per gene.

## Output

The first line gives the best known minimum, the seeds and the budget. Then a row per algorithm: how
many of the 30 runs reach the best known minimum and how many don't, and the median and largest
number of evaluations of the runs that reach it. In Python, `run` evaluates the function in Rust, so
both versions print the same table.

The page's plot shows the 30 runs of CMA-ES without restarts, each at its best point so far, over
the function's contour, with the best known minimum marked: they end in holes all over the grid. A
curve gives the best and the median run's error, on a logarithmic axis.

[The project page](https://tachsin.gr/projects/genoxide/examples/shekel-foxholes) plays this run
back.

## Good results

A good result reaches the best known minimum in every run. The genetic algorithm does, after a
median of 2,049 evaluations and at most 5,081, and so does PSO, after a median of 2,140 and at most
4,120: their populations of 50 and 40 sample the plane widely before they gather.

CMA-ES never does without restarts: the best points of its runs lie in 17 different holes, one of
them the deepest (seed 27, at 0.99894, on its side), 4 in the third, at (0, −32), 4 in the one in
the middle, at (0, 0), and the others elsewhere. Its population of 6 narrows down on a hole that its
first samples find. With IPOP restarts, only one run reaches the minimum, after 5,280 evaluations:
the restarts sample the plane again, and 18 runs have a best point in the deepest hole, down to
0.9980038389, but aren't within 1e-8 of its bottom when the budget ends.
