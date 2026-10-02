---
title: CTP7
category: multi-objective
summary: Minimize two objectives with infeasible bands running across the unconstrained front, which leave six disconnected pieces of it and a lone point, with NSGA-II.
reference: "Deb, K., Pratap, A. and Meyarivan, T. (2001). Constrained test problems for multi-objective evolutionary optimization. Evolutionary Multi-Criterion Optimization (EMO 2001), LNCS 1993: 284-298."
reference_url: https://doi.org/10.1007/3-540-44719-9_20
optimum: "six pieces of the curve f₂ = 1 − √f₁, the last ending at (1, 0), and the point (0, 1.0446); hypervolume 0.8443 in objectives scaled by the ideal and nadir points (reference point (1.1, 1.1))"
languages: [rust, python]
order: 191
family: CTP
---

# CTP7

## The problem

Deb, Pratap and Meyarivan (2001) built the CTP problems to test how multi-objective algorithms
handle constraints. CTP2 to CTP7 share one form, a generator whose six parameters θ, a, b, c, d
and e shape a single constraint:

```text
minimize   f₁ = x₁
           f₂ = g (1 − √(f₁/g)),  g = 1 + x₂
subject to cos θ (f₂ − e) − sin θ f₁ ≥ a |sin(bπ (sin θ (f₂ − e) + cos θ f₁)^c)|^d
x₁ in [0, 1], x₂ in [0, 10]
```

CTP7 has θ = −0.05π, a = 40, b = 5, c = 1, d = 6 and e = 0. Turned by only −0.05π, the coordinate
v = cos θ f₁ − sin(0.05π) f₂ runs almost along f₁, and the wave 40 sin⁶(5πv) is 0 where v is a
multiple of 0.2. The left side, u, is close to f₂. With a = 40, the constraint holds only in bands
around v = 0, 0.2, 0.4, …, nearly upright, and the high power d = 6 makes them wide, with
infeasible bands between: they cross the whole objective space, and the unconstrained front.

The bands leave parts of the unconstrained front, f₂ = 1 − √f₁ at g = 1, feasible, and those are
optimal: six pieces of it, from f₁ = 0.0792 to 0.1343, 0.2490 to 0.3056, 0.4285 to 0.4840, 0.6122
to 0.6659, 0.7986 to 0.8500 and 0.9871 to 1, where the last ends at (1, 0). At f₁ = 0, the least
feasible f₂ is 1.0446, on the edge of a band, and that point is optimal too, since nothing has a
smaller f₁. The front was found by sampling the
boundaries of the feasible region.

The definitions come from the authors' KanGAL report 200005 (October 2000), the paper's preprint:
eq. 5 on p. 7 and CTP7's parameters on pp. 10-11. The report leaves g, the number of variables and
their bounds open, and prints f₂ as g (1 − f₁/g); its figures draw the unconstrained front as the
curve 1 − √f₁, and the authors' NSGA-II code (version 1.1.6, KanGAL) computes g (1 − √(f₁/g))
with g = 1 + x₂, x₁ in [0, 1] and x₂ in [0, 10], which genoxide follows.

## What makes it hard

The report: "In order to find all such disconnected regions, an algorithm has to maintain an
adequate diversity right from the beginning of a simulation run. Moreover, the algorithm also has
to maintain its solutions feasible as it proceeds towards the Pareto-optimal region." A group of
solutions that converges in one band can't cross into the next: the bands run from far above the
front down to it. 47% of random genomes are feasible.

In the report's experiments, with five variables and a Rastrigin g, CTP7 was the hardest problem:
neither algorithm got close to the front. With the two variables and g = 1 + x₂ of the authors'
code, converging is easy, since x₂ = 0 is optimal everywhere, and the difficulty is the diversity
alone.

## Representation

A `Real` genome of 2 genes: x₁ in [0, 1] and x₂ in [0, 10]. The problem is genoxide's `Ctp7`,
whose fitness is the two objectives and the constraint violation.

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

The second counts the pieces of the optimal front that the run reaches, the six pieces and the
point: that have a solution within 0.02 of one of their points, in scaled objectives.

The third gives the front's IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to 2,000
points of the optimal front, and its hypervolume, the area it dominates up to the reference point
(1.1, 1.1), as a share of the whole optimal front's. Both use objectives scaled to [0, 1] on the
front, by its ideal point (0, 0) and nadir point (1, 1.0446). IGD+ averages, over the points of
the optimal front, the distance to the nearest solution, counting only the objectives in which
the solution is worse. The whole front's hypervolume, from 100,000 of its points, is 0.8443. In
Python, `run` evaluates the problem in Rust, so both versions print the same.

The run's `trace.json` also has the problem's feasible region, which the page shades.
[The project page](https://tachsin.gr/projects/genoxide/examples/ctp7) plays this run back.

## Good results

A good front is all feasible, with solutions on all six pieces and at the lone point, an IGD+ well
under 0.01 and a hypervolume close to the whole front's. 100 points of the optimal front, spread
over the pieces by their lengths, give 99.96% of its hypervolume and an IGD+ of 0.0007.

The run's front has 100 solutions, all feasible, reaching all seven parts, with an IGD+ of 0.0010
and 99.94% of the whole front's hypervolume. Over seeds 1 to 20, every run reaches all seven, with
an IGD+ from 0.0009 to 0.0011 and 99.93% to 99.94% of the hypervolume.
