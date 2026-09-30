---
title: CTP6
category: multi-objective
summary: Minimize two objectives behind infeasible bands that cross the whole objective space parallel to the optimal front, with NSGA-II.
reference: "Deb, K., Pratap, A. and Meyarivan, T. (2001). Constrained test problems for multi-objective evolutionary optimization. Evolutionary Multi-Criterion Optimization (EMO 2001), LNCS 1993: 284-298."
reference_url: https://doi.org/10.1007/3-540-44719-9_20
optimum: "one piece of a constraint boundary, from (0, 3.6958) to (1, 0.8813); hypervolume 0.7124 in objectives scaled by the ideal and nadir points (reference point (1.1, 1.1))"
languages: [rust, python]
order: 170
family: CTP
---

# CTP6

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

The report's section on difficulty in the whole search space gives CTP6 very different
parameters: θ = 0.1π, a = 40, b = 0.5, c = 1, d = 2 and e = −2. Turned by θ = 0.1π, the
coordinate v = sin θ (f₂ + 2) + cos θ f₁ runs across the objective space, and the wave
40 sin²(πv/2) is 0 only where v is an even number. The left side, u, stays between 1.6 and 12.4
over the space, far below the wave's height of 40: the constraint holds only in bands around
v = 2, 4, 6, …, parallel to the front and to each other, with infeasible bands between.

The unconstrained front, f₂ = 1 − √f₁ at g = 1, lies in the infeasible band below v = 2. The
optimal front is the lower edge of the first feasible band, one continuous piece from (0, 3.6958)
to (1, 0.8813), found by sampling the boundaries of the feasible region. On it, v runs from 1.76
to 1.84; the report puts the front where v is between 1 and 2.

The definitions come from the authors' KanGAL report 200002 (October 2000), the paper's preprint:
eq. 5 on p. 7 and CTP6's parameters on p. 10. The report leaves g, the number of variables and
their bounds open, and prints f₂ as g (1 − f₁/g); its figures draw the unconstrained front as the
curve 1 − √f₁, and the authors' NSGA-II code (version 1.1.6, KanGAL) computes g (1 − √(f₁/g))
with g = 1 + x₂, x₁ in [0, 1] and x₂ in [0, 10], which genoxide follows.

## What makes it hard

Solutions in the upper feasible bands have to cross the infeasible bands between them,
"infeasible holes of differing widths", in the report's words, to reach the band that holds the
front. 22% of random genomes are feasible. Once
the population is in the right band, the front is its lower edge: every optimal solution meets
the constraint exactly, and just below it lies infeasible space.

## Representation

A `Real` genome of 2 genes: x₁ in [0, 1] and x₂ in [0, 10]. The problem is genoxide's `Ctp6`,
whose fitness is the two objectives and the constraint violation.

Solutions compare by constrained dominance, the rule of the NSGA-II paper. A feasible solution
beats an infeasible one; of two infeasible ones, the smaller violation wins; of two feasible ones,
Pareto dominance decides. An infeasible solution closer to a band's edge has the smaller
violation, which leads the search across the infeasible bands.

## Algorithm

NSGA-II with the settings of the report's experiments:

- a population of 100, for 500 generations;
- simulated binary crossover with η = 20, at a rate of 0.9;
- polynomial mutation with η = 20, at a rate of 1/n per gene for n genes: 0.5.

## Output

The first line gives the size of the final front and how many of its solutions are feasible.

The second counts the pieces of the optimal front that the run reaches: that have a solution
within 0.02 of one of their points, in scaled objectives. CTP6's front is one piece.

The third gives the front's IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to 2,000
points of the optimal front, and its hypervolume, the area it dominates up to the reference point
(1.1, 1.1), as a share of the whole optimal front's. Both use objectives scaled to [0, 1] on the
front, by its ideal point (0, 0.8813) and nadir point (1, 3.6958). IGD+ averages, over the points
of the optimal front, the distance to the nearest solution, counting only the objectives in which
the solution is worse. The whole front's hypervolume, from 100,000 of its points, is 0.7124. In
Python, `run` evaluates the problem in Rust, so both versions print the same.

The run's `trace.json` also has the problem's feasible region, which the page shades.
[The project page](https://tachsin.gr/projects/genoxide/examples/ctp6) plays this run back.

## Good results

A good front is all feasible, spread over the whole edge, with an IGD+ well under 0.01. 100 points
of the optimal front, spread evenly along it, give 99.31% of its hypervolume and an IGD+ of
0.0025.

The run's front has 100 solutions, all feasible, an IGD+ of 0.0052 and 98.61% of the whole
front's hypervolume. Over seeds 1 to 20, every run reaches the front, with an IGD+ from 0.0046 to
0.0053 and 98.56% to 98.75% of the hypervolume. The report found NSGA-II "very near to the true
Pareto-optimal front" with its five variables too.
