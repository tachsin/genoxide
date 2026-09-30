---
title: TNK (Tanaka)
category: multi-objective
summary: Minimize the two variables themselves, subject to staying outside a wavy curve and inside a circle, with NSGA-II; the optimal front is in five pieces along the curve.
reference: "Tanaka, M., Watanabe, H., Furukawa, Y. and Tanino, T. (1995). GA-based decision support system for multicriteria optimization. Proceedings of the IEEE International Conference on Systems, Man and Cybernetics 2: 1556-1561."
reference_url: https://doi.org/10.1109/ICSMC.1995.537993
optimum: "a front in five pieces on the first constraint's boundary, from (0.0417, 1.0384) to (1.0384, 0.0417); hypervolume 0.6551 (reference point (1.2, 1.2))"
languages: [rust, python]
order: 143
---

# TNK (Tanaka)

## The problem

Tanaka, Watanabe, Furukawa and Tanino (1995) built this problem to test a genetic algorithm for
decision support. It has two objectives and two constraints, over x₁ and x₂ in [0, π]:

```text
minimize   f₁ = x₁
           f₂ = x₂
subject to x₁² + x₂² − 1 − 0.1 cos(16 arctan(x₁/x₂)) ≥ 0
           (x₁ − 0.5)² + (x₂ − 0.5)² ≤ 0.5
```

The objectives are the variables themselves: the problem is all in its constraints, and a plot of
the solutions is also a plot of their objectives.

The first constraint keeps x outside a wavy curve. At the angle φ from the x₂ axis, the curve lies
at the distance √(1 + 0.1 cos 16φ) from the origin, which swings between 0.95 and 1.05 four times
over the quarter circle in the box: it bulges out at 0°, 22.5°, 45°, 67.5° and 90°, and dips in
between. genoxide takes the angle arctan(x₁/x₂) as `atan2(x₁, x₂)`, π/2 at x₂ = 0.

The second constraint keeps x inside the circle of radius √0.5 ≈ 0.71 around (0.5, 0.5), which
passes through (0, 0), (1, 0), (0, 1) and (1, 1). Together they leave a thin crescent, about 5% of
the box: inside the circle, outside the wavy curve.

To minimize both variables, a solution moves towards the origin until the wavy curve stops it. The
optimal front thus lies on the curve, where it runs inside the circle: from (0.0417, 1.0384), where
the curve enters the circle, to (1.0384, 0.0417). But not all of it. Where the curve bulges out at
22.5°, it climbs from f₂ = 0.929 to 0.976 and back, and every point of the bulge is dominated by
the point where it begins, (0.1996, 0.9290), lower in both objectives. The same holds at 67.5°
with x₁ and x₂ swapped. The front breaks into five pieces, the same on both sides of the diagonal:

| Piece | f₁ from | to | f₂ from | to |
|---|---|---|---|---|
| 1 | 0.0417 | 0.1996 | 1.0384 | 0.9290 |
| 2 | 0.4469 | 0.6147 | 0.9290 | 0.7731 |
| 3 | 0.6202 | 0.7731 | 0.7731 | 0.6202 |
| 4 | 0.7731 | 0.9290 | 0.6147 | 0.4469 |
| 5 | 0.9290 | 1.0384 | 0.1996 | 0.0417 |

Between the first and the second, and between the fourth and the fifth, lie wide gaps. The middle
piece follows the bulge at 45°, and two narrow gaps set it apart from its neighbours.

genoxide restates the problem as the NSGA-II paper (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE
Transactions on Evolutionary Computation 6(2): 182-197, table V) and Deb, Pratap and Meyarivan
(2001, EMO 2001, eq. 2) do; it hasn't checked it against the original paper yet.

## What makes it hard

Three things. The feasible region is small: 6 of the run's first 100 random solutions are
feasible. The whole front lies on a constraint's boundary, so the best solutions sit right next to
infeasible ones. And the front is disconnected: a solution between two pieces is dominated, so the
population has to keep a group of solutions on each of the five pieces, and small steps along the
curve don't lead from one to the next.

The objectives, x₁ and x₂, are as simple as objectives get. The difficulty comes from the
constraints alone, which is why TNK is a common test of constraint handling.

## Representation

A `Real` genome of 2 genes in [0, π]: the point x. The problem is genoxide's `Tnk`, whose fitness
is the two objectives and the total constraint violation: the sum of how far x breaks each
constraint, 0 when it is feasible.

Solutions compare by constrained dominance, the rule of the NSGA-II paper. A feasible solution
beats an infeasible one. Of two infeasible ones, the smaller violation wins. Of two feasible ones,
the usual Pareto dominance decides. Infeasible solutions thus lead the search towards the feasible
region.

## Algorithm

NSGA-II, which ranks solutions into non-dominated fronts and prefers the less crowded ones within a
front, with parents and children competing for the next population. The settings are those of the
NSGA-II paper:

- a population of 100, for 250 generations;
- simulated binary crossover with η = 20, at genoxide's default rate of 0.9;
- polynomial mutation with η = 20, at a rate of 1/n per gene for n genes: 0.5, one of the two
  genes per child on average.

## Output

The first line gives the size of the final front, and how many of its solutions are feasible.

The second counts the solutions on each of the five pieces of the optimal front, in order of f₁,
told apart by the gaps between them.

The third gives the front's IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to 500
points of the optimal front. IGD+ averages, over those 500 points, the distance to the nearest
point of the found front, counting only the objectives in which the found point is worse. 0 means
that the found front covers the optimal one. Smaller is better.

The fourth gives the front's hypervolume: the area it dominates, up to the reference point
(1.2, 1.2), a little beyond the worst value of each objective on the front. Larger is better. For
the whole optimal front, computed from 2,000,000 of its points, it is 0.6551. In Python, `run`
evaluates the problem in Rust, so both versions print the same.

The run's `trace.json` has one field more than a front's trace: `feasible`, the share of the
population that is feasible in each recorded generation.

[The project page](https://tachsin.gr/projects/genoxide/examples/tnk) plays this run back.

## Good results

A good front is all feasible, has solutions on all five pieces, and has an IGD+ near 0 and a
hypervolume near 0.6551. 100 points of the optimal front, spread evenly along it, give a
hypervolume of 0.6526 and an IGD+ of 0.0015, and put 17, 22, 22, 22 and 17 points on the five
pieces.

The run's population is all feasible by generation 4. Its front grows more slowly than on problems
with a connected front: 14 solutions at generation 4, 75 at generation 40, and all 100 only at
generation 90. Until then, part of the population is feasible but dominated, off the front. The
final front has 100 solutions, 18, 23, 19, 22 and 18 on the five pieces, an IGD+ of 0.0020 and a
hypervolume of 0.6511, 99.4% of the whole front's. The hypervolume keeps rising slowly to the end,
as the solutions settle onto the curve.
