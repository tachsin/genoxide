---
title: Constrained two-objective fronts
category: multi-objective
summary: Minimize two objectives under constraints on four classic problems, SRN, TNK, OSY and CONSTR, with NSGA-II and constrained dominance.
reference: "Deb, K., Pratap, A., Agarwal, S. and Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation 6(2): 182-197."
reference_url: https://doi.org/10.1109/4235.996017
optimum: "the known optimal fronts; normalized hypervolume of 500 of their points: SRN 0.7496, TNK 0.5184, OSY 0.9680, CONSTR 0.9907 (reference point (1.1, 1.1))"
languages: [rust, python]
order: 84
---

# Constrained two-objective fronts

## The problem

Four classic test problems, each with two objectives to minimize and constraints to satisfy. The
NSGA-II paper (Deb, Pratap, Agarwal and Meyarivan, 2002) tests its constraint handling on them, and
its table V restates CONSTR, SRN and TNK. Each has a known optimal front, derived from its
definition.

**SRN**, from Srinivas and Deb (1994, Evolutionary Computation 2(3): 221-248), over x₁ and x₂ in
[−20, 20]:

```text
minimize   f₁ = (x₁ − 2)² + (x₂ − 1)² + 2
           f₂ = 9x₁ − (x₂ − 1)²
subject to x₁² + x₂² ≤ 225
           x₁ − 3x₂ ≤ −10
```

The first constraint keeps x inside a circle of radius 15. The second keeps it above a line. The
front has three pieces: along the line, from x₂ = 3.7 down to 2.5; along x₁ = −2.5, up to the
circle; and along the circle, to x ≈ (−4.841, 14.197). It runs from (10.1, 2.61) to (222.97,
−217.74). The solutions on x₁ = −2.5 are the ones usually quoted, but they are only part of the
front.

**TNK**, from Tanaka, Watanabe, Furukawa and Tanino (1995, Proceedings of the IEEE International
Conference on Systems, Man and Cybernetics 2: 1556-1561), over x₁ and x₂ in [0, π]:

```text
minimize   f₁ = x₁
           f₂ = x₂
subject to x₁² + x₂² − 1 − 0.1 cos(16 arctan(x₁/x₂)) ≥ 0
           (x₁ − 0.5)² + (x₂ − 0.5)² ≤ 0.5
```

The objectives are the variables themselves: the problem is all in its constraints. The first keeps
x outside a wavy curve, whose distance from the origin, √(1 + 0.1 cos 16φ) at the angle φ from the
x₂ axis, swings between 0.95 and 1.05 four times over the quarter circle in the box. The second
keeps x inside a circle of radius √0.5 around (0.5, 0.5). The front lies on the parts of the wavy
curve that no other feasible point dominates. It runs from about (0.042, 1.038) to (1.038, 0.042)
in five pieces, with wide gaps between f₁ ≈ 0.20 and 0.45 and between f₂ ≈ 0.20 and 0.45.

**OSY**, from Osyczka and Kundu (1995, Structural Optimization 10(2): 94-99), in six variables:

```text
minimize   f₁ = −[25 (x₁ − 2)² + (x₂ − 2)² + (x₃ − 1)² + (x₄ − 4)² + (x₅ − 1)²]
           f₂ = x₁² + x₂² + x₃² + x₄² + x₅² + x₆²
subject to x₁ + x₂ ≥ 2           x₁ + x₂ ≤ 6
           x₂ − x₁ ≤ 2           x₁ − 3x₂ ≤ 2
           (x₃ − 3)² + x₄ ≤ 4    (x₅ − 3)² + x₆ ≥ 4
```

with x₁, x₂ and x₆ in [0, 10], x₃ and x₅ in [1, 5], and x₄ in [0, 6]. On the front, x₄ = x₆ = 0.
It has five pieces, each on a different set of active constraints, from (−274, 76) to (−42, 4).

**CONSTR**, defined in the NSGA-II paper itself, over x₁ in [0.1, 1] and x₂ in [0, 5]:

```text
minimize   f₁ = x₁
           f₂ = (1 + x₂)/x₁
subject to x₂ + 9x₁ ≥ 6
           −x₂ + 9x₁ ≥ 1
```

Without the constraints, x₂ = 0 would be best for any x₁, and the front would be f₂ = 1/f₁. The
first constraint cuts off its left part: for x₁ below 2/3, the front follows the constraint's
boundary x₂ = 6 − 9x₁, where f₂ = 7/f₁ − 9, down to x₁ = 7/18, where the second constraint closes
the feasible region. The front is convex and runs from (7/18, 9) to (1, 1).

genoxide restates SRN and TNK as the NSGA-II paper does, and OSY as Deb, Pratap and Meyarivan
(2001, EMO 2001, LNCS 1993: 284-298) do; it hasn't checked them against the original papers yet.

## What makes it hard

On all four, at least part of the optimal front lies on the edge of the feasible region, so the
search has to approach the constraints without crossing them. Many random solutions are
infeasible: in the run's first population, only 15 of 100 are feasible on SRN, 6 on TNK and OSY,
and 56 on CONSTR.

Each problem adds its own difficulty. SRN's front has three pieces, on the line, inside the region
and on the circle, and its objectives span about 213 and 220. TNK's front is disconnected, and its pieces follow the bumps
of a wavy curve. OSY has six constraints in six variables, and each of its five pieces lies on a
different combination of them: a population that finds one piece doesn't find the others by small
steps. CONSTR's left part lies exactly on a constraint's boundary.

## Representation

A `Real` genome with the problem's bounds: 2 genes for SRN, TNK and CONSTR, 6 for OSY. The problems
are genoxide's `Srn`, `Tnk`, `Osy` and `Constr`, whose fitness is the two objectives and the total
constraint violation: the sum of how far x breaks each constraint, 0 when it is feasible.

Solutions compare by constrained dominance, the rule of the NSGA-II paper. A feasible solution
beats an infeasible one. Of two infeasible ones, the smaller violation wins. Of two feasible ones,
the usual Pareto dominance decides. The infeasible solutions thus lead the search towards the
feasible region, and the front is feasible as soon as any solution is.

## Algorithm

NSGA-II, which ranks solutions into non-dominated fronts and prefers the less crowded ones within a
front, with parents and children competing for the next population. The settings are those of the
NSGA-II paper, with 200 generations instead of its 250:

- a population of 100, for 200 generations, on each problem;
- simulated binary crossover with η = 20, at genoxide's default rate of 0.9;
- polynomial mutation with η = 20, at a rate of 1/n per gene for n genes: 0.5 for the problems in
  2 variables, 1/6 for OSY, one gene per child on average.

## Output

A table with a row per problem. `front` is the size of the final front, and `feasible` how many of
its solutions are feasible.

The other columns measure the feasible front with its objectives normalized: each mapped to [0, 1]
by the problem's ideal point (the best value of each objective on the optimal front) and nadir
point (the worst). The four problems then share one scale.

`hypervolume` is the area that the front dominates, up to the reference point (1.1, 1.1). Larger is
better. `(optimal)` is the same for 500 points of the optimal front, for comparison. The whole,
continuous fronts reach about 0.7507, 0.5189, 0.9688 and 0.9917.

`IGD+` (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) averages, over the 500 points of the
optimal front, the distance to the nearest point of the found front, counting only the objectives
in which the found point is worse. 0 means that the found front covers the optimal one. Smaller is
better. In Python, `run` evaluates the problems in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/constrained-fronts) plays this run back.

## Good results

A good front is all feasible, with a hypervolume close to the `(optimal)` column and an IGD+ near 0.
A front of 100 solutions stays a little below it: 100 evenly spread points of the optimal fronts
give 0.7448, 0.5164, 0.9648 and 0.9865.

The run's fronts have 100 solutions each, all feasible, with IGD+ below 0.005 on every problem: the
normalized hypervolumes are 0.7427, 0.5147, 0.9612 and 0.9853. Every population is all feasible by
generation 8. OSY's hypervolume rises slowest, as the population finds its five pieces one after
another: the middle ones by generation 32, the one at the end near (−274, 76) only by generation
64. On TNK, part of the population stays dominated, off the front, until about generation 90.
