---
title: ZDT2
category: multi-objective
summary: Minimize two conflicting objectives over 30 variables, with a concave Pareto front, with NSGA-II.
reference: "Zitzler, E., Deb, K. and Thiele, L. (2000). Comparison of multiobjective evolutionary algorithms: empirical results. Evolutionary Computation 8(2): 173-195."
reference_url: https://doi.org/10.1162/106365600568202
optimum: "the front f₂ = 1 − f₁² for f₁ in [0, 1]; hypervolume 0.5433 (reference point (1.1, 1.1))"
languages: [rust, python]
order: 131
family: ZDT
---

# ZDT2

## The problem

Zitzler, Deb and Thiele (2000) built six test problems with two objectives from one scheme: f₁
depends on the first variable, a function g on the others, and f₂ on both. ZDT2 is the second. It
has 30 variables in [0, 1], and minimizes both objectives:

```text
f₁ = x₁
g  = 1 + 9 (x₂ + … + x₃₀) / 29
f₂ = g (1 − (f₁ / g)²)
```

It is ZDT1 with the square root replaced by a square. No solution minimizes both objectives. The
optimal trade-offs, the Pareto front, are the solutions with g = 1, that is x₂ = … = x₃₀ = 0.
There, f₂ = 1 − f₁² for f₁ from 0 to 1. The straight line between the two ends, (0, 1) and (1, 0),
passes through (0.5, 0.5), but the front passes through (0.5, 0.75): it bulges away from the
origin, it is concave. Zitzler, Deb and Thiele built ZDT2 as the concave counterpart of ZDT1.

## What makes it hard

A concave front. Minimizing a weighted sum of the objectives, w f₁ + (1 − w) f₂, finds only the
two ends of a concave front, whatever the weight w. On the front, the sum is
w f₁ + (1 − w)(1 − f₁²), a curve that bends down, so it is smallest at f₁ = 0 or f₁ = 1, never in
between. An algorithm that ranks solutions by Pareto dominance, not by a sum, can find the middle.

A front that shrinks to a point far from the optimum. A random solution has g near 5.5. For a large
g, f₂ = g − f₁² / g falls only slowly as f₁ grows: a solution with f₁ > 0 is beaten by a solution
with f₁ = 0, unless its g is almost as small. So the non-dominated solutions gather at f₁ = 0, and
the others fall behind, whatever their x₁. The search has to drive g down without losing the
spread in x₁ that it needs later.

## Representation

A `Real` genome of 30 genes in [0, 1]: the vector x. The problem is genoxide's `Zdt2`, whose
fitness is the pair (f₁, f₂). In Python, `run` evaluates it in Rust, so both versions print the
same.

## Algorithm

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation
6(2): 182-197), as the ZDT1 example runs it. It ranks solutions by non-dominated sorting: the first
front is the solutions that no other solution beats in both objectives, the second front those
beaten only by the first, and so on. Within a front, it prefers solutions in less crowded regions
(crowding distance). Parents and children compete for the next population, so it keeps the best
solutions found so far.

- a population of 100, for 250 generations, as in the NSGA-II paper;
- simulated binary crossover with η = 15, at genoxide's default rate of 0.9;
- polynomial mutation with η = 20, at a rate of 1/30 per gene, one gene per child on average.

## Output

The first line gives the size of the final front.

The second gives its IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to 500 points of
the optimal front, evenly spaced in f₁. IGD+ averages, over those 500 points, the distance to the
nearest point of the found front, counting only the objectives in which the found point is worse.
0 means that the found front covers the optimal one. Smaller is better.

The third gives the front's hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on
Evolutionary Computation 3(4): 257-271): the area it dominates, up to the reference point
(1.1, 1.1). Larger is better. For the whole front, it is 1.1 × 1.1 − 2/3 = 0.5433: the box minus
the area under the curve.

[The project page](https://tachsin.gr/projects/genoxide/examples/zdt2) plays this run back.

## Good results

A good front has 100 solutions spread from (0, 1) to (1, 0), an IGD+ near 0 and a hypervolume near
0.5433. No set of 100 points reaches that hypervolume: 100 points of the optimal front, evenly
spaced in f₁, give 0.5383 and an IGD+ of 0.0023.

The run shows the shrinking front. From generation 15 to 72, the front's largest f₁ is below 0.0001
in all but 7 generations, and below 0.04 in all, while g falls from about 2.4 to 1.1. Only when g
is near 1 does the front spread again: its largest f₁ reaches 0.5 at generation 133, and 1 at
about 175; it has 100 solutions from generation 120. It ends with 100 solutions, g below 1.008, an
IGD+ of 0.0032 and a hypervolume of 0.5366, 98.8% of the whole front's.

On seeds 1 to 5, NSGA-II ends between 0.5359 and 0.5366, with an IGD+ of 0.0032 to 0.0035. With
the same settings, SPEA2 ends between 0.5361 and 0.5370, and SMS-EMOA, which keeps the solutions
that add the most hypervolume, between 0.5376 and 0.5382, with an IGD+ of about 0.0025: about as
close as the 100 evenly spaced points.
