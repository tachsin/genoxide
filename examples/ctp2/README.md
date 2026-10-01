---
title: CTP2
category: multi-objective
summary: Minimize two objectives subject to a constraint whose wavy boundary cuts the optimal front into 13 disconnected pieces, with NSGA-II.
reference: "Deb, K., Pratap, A. and Meyarivan, T. (2001). Constrained test problems for multi-objective evolutionary optimization. Evolutionary Multi-Criterion Optimization (EMO 2001), LNCS 1993: 284-298."
reference_url: https://doi.org/10.1007/3-540-44719-9_20
optimum: "13 disconnected pieces of the constraint's boundary, from (0, 1) to (0.9845, 0.2872); hypervolume 0.6901 in objectives scaled by the ideal and nadir points (reference point (1.1, 1.1))"
languages: [rust, python]
order: 166
family: CTP
---

# CTP2

## The problem

Deb, Pratap and Meyarivan (2001) built the CTP problems to test how multi-objective algorithms
handle constraints. CTP2 to CTP7 share one form, a generator whose six parameters θ, a, b, c, d
and e shape a single constraint:

```text
minimize   f₁ = x₁
           f₂ = g (1 − √(f₁/g)),  g = 1 + x₂
subject to cos θ (f₂ − e) − sin θ f₁ ≥ a |sin(bπ (sin θ (f₂ − e) + cos θ f₁)^c)|^d
x₁, x₂ in [0, 1]
```

CTP2 has θ = −0.2π, a = 0.2, b = 10, c = 1, d = 6 and e = 1. The constraint turns the objective
space by θ: its left side, u, measures the distance above the line (f₂ − e) cos θ = f₁ sin θ,
which is f₂ = 1 − 0.7265 f₁ here, and the right side waves along that line, with period 1/b in
the coordinate v along it. A solution is feasible where u is above the wave.

The unconstrained front, f₂ = 1 − √f₁ at g = 1, lies below the line and is infeasible. The wave
touches the line where its sine is 0, at v = 0, 0.1, …, 1.2, and rises between. The optimal front
is 13 pieces of the wave, each starting on the line and ending where the next piece dominates it,
from (0, 1) to about (0.9845, 0.2872). genoxide's `optimal_front` samples the boundaries of the
feasible region densely and keeps the feasible non-dominated points.

The definitions come from the authors' KanGAL report 200005 (October 2000), the paper's preprint:
eq. 5 and the parameters that follow it on p. 7. The report leaves g, the number of variables and
their bounds open, and prints f₂ as g (1 − f₁/g); its figures draw the unconstrained front as the
curve 1 − √f₁, and the authors' NSGA-II code (version 1.1.6, KanGAL) computes g (1 − √(f₁/g)) with
g = 1 + x₂ and two variables in [0, 1], which genoxide follows. The report's own experiments used
five variables and a Rastrigin function for g, without giving its formula.

## What makes it hard

The front is disconnected: 13 pieces, each about 0.025 wide in f₁, with gaps of about 0.055
between them. An algorithm has to find every piece and keep solutions on all of them, and it
can't slide from one piece to the next, since the feasible region between them rises into waves.
The larger b, the more pieces; the report calls finding them all the task.

About 45% of random genomes are feasible: the waves cut the region near the front, not far from
it.

## Representation

A `Real` genome of 2 genes in [0, 1]: x₁ and x₂. The problem is genoxide's `Ctp2`, whose fitness
is the two objectives and the constraint violation.

Solutions compare by constrained dominance, the rule of the NSGA-II paper. A feasible solution
beats an infeasible one; of two infeasible ones, the smaller violation wins; of two feasible ones,
Pareto dominance decides.

## Algorithm

NSGA-II with the settings of the report's experiments:

- a population of 100, for 500 generations;
- simulated binary crossover with η = 20, at a rate of 0.9;
- polynomial mutation with η = 20, at a rate of 1/n per gene for n genes: 0.5.

## Output

The first line gives the size of the final front and how many of its solutions are feasible.

The second counts the pieces of the optimal front that the run reaches: that have a solution
within 0.02 of one of their points, in scaled objectives.

The third gives the front's IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to 2,000
points of the optimal front, and its hypervolume, the area it dominates up to the reference point
(1.1, 1.1), as a share of the whole optimal front's. Both use objectives scaled to [0, 1] on the
front, by its ideal point (0, 0.2872) and nadir point (0.9845, 1). IGD+ averages, over the points
of the optimal front, the distance to the nearest solution, counting only the objectives in
which the solution is worse: 0 means that the front covers the optimal one. The whole front's
hypervolume, from 100,000 of its points, is 0.6901. In Python, `run` evaluates the problem in
Rust, so both versions print the same.

The run's `trace.json` also has the problem's feasible region, which the page shades.
[The project page](https://tachsin.gr/projects/genoxide/examples/ctp2) plays this run back.

## Good results

A good front is all feasible, with solutions on all 13 pieces, an IGD+ well under 0.01 and a
hypervolume close to the whole front's. 100 points of the optimal front, spread over the pieces by
their lengths, give 99.90% of its hypervolume and an IGD+ of 0.0010.

The run's front has 100 solutions, all feasible, on all 13 pieces, with an IGD+ of 0.0016 and
99.78% of the whole front's hypervolume. Over seeds 1 to 20, every run reaches all 13 pieces, with
an IGD+ from 0.0015 to 0.0019 and 99.75% to 99.82% of the hypervolume. The report found the same
with its five variables: NSGA-II found all the disconnected pieces.
