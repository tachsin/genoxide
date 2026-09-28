---
title: ZDT1
category: multi-objective
summary: Minimize two conflicting objectives over 30 variables, with a convex Pareto front.
reference: "Zitzler, E., Deb, K. and Thiele, L. (2000). Comparison of multiobjective evolutionary algorithms: empirical results. Evolutionary Computation 8(2): 173-195."
reference_url: https://doi.org/10.1162/106365600568202
optimum: "hypervolume 0.8767 (reference point (1.1, 1.1))"
languages: [rust, python]
order: 110
family: ZDT
---

# ZDT1

## The problem

Zitzler, Deb and Thiele (2000) built six test problems with two objectives from one scheme. ZDT1 is
the first. It has 30 variables in [0, 1], and minimizes both objectives:

```text
f₁ = x₁
g  = 1 + 9 (x₂ + … + x₃₀) / 29
f₂ = g (1 − √(f₁ / g))
```

No solution minimizes both. The optimal trade-offs, the Pareto front, are the solutions with g = 1,
that is x₂ = … = x₃₀ = 0. There, f₂ = 1 − √f₁ for f₁ from 0 to 1: a convex curve. With x₁ = 0.25 and
the rest 0, the solution is on the front at (0.25, 0.5). With the rest at 0.5 instead, g = 5.5 and
f₂ = 4.33, far above it.

## What makes it hard

The search has two jobs. It has to converge: drive 29 variables to 0, where g reaches 1. And it has
to spread: cover f₁ from 0 to 1 with solutions, so that the front shows the whole trade-off. A
random solution has g near 5.5, so the initial population sits far above the front.

## Representation

A `Real` genome of 30 genes in [0, 1]: the vector x. The fitness is the pair (f₁, f₂), both
minimized. The Rust version uses genoxide's `Zdt1`; the Python version computes the objectives with
numpy, a generation at a time.

## Algorithm

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation
6(2): 182-197). It ranks solutions by non-dominated sorting: the first front is the solutions that
no other solution beats in both objectives, the second front those beaten only by the first, and so
on. Within a front, it prefers solutions in less crowded regions (crowding distance). Parents and
children compete for the next population, so the population keeps the best solutions found so far
(elitism).

The settings follow the usual ones that genoxide's NSGA-II docs give:

- a population of 100, for 25,000 evaluations, 250 generations, as in the NSGA-II paper;
- simulated binary crossover with η = 15, at genoxide's default rate of 0.9; a larger η makes
  children closer to their parents;
- polynomial mutation with η = 20, at a rate of 1/30 per gene, one gene per child on average.

## Output

One line: how many solutions are on the final non-dominated front, and its hypervolume. The
hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271)
is the area that the front dominates, up to a reference point, here (1.1, 1.1). Larger is better. It
rewards both convergence and spread. For the whole Pareto front, it is 1.1 × 1.1 − 1/3 = 0.8767, the
area of the box minus the area under the curve.

[The project page](https://tachsin.gr/projects/genoxide/examples/zdt1) plays this run back.

## Good results

No set of 100 points reaches 0.8767, the hypervolume of the whole, continuous front. For comparison,
100 points on the front, evenly spaced in f₁, give 0.8714. The run's front has all 100 solutions
non-dominated and a hypervolume of about 0.870: close to the front, and well spread.
