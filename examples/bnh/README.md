---
title: BNH, a constrained two-objective problem
category: multi-objective
summary: Minimize two objectives subject to two constraints with NSGA-II, and measure the front against the optimal one.
reference: "Binh, T. T. and Korn, U. (1997). MOBES: a multiobjective evolution strategy for constrained optimization problems. Proceedings of the Third International Conference on Genetic Algorithms (Mendel 97), Brno: 176-182."
reference_url: ""
optimum: "the front f = (8t², 2(t − 5)²) for t in [0, 5]; hypervolume 9883.33 (reference point (210, 55))"
languages: [rust, python]
order: 111
---

# BNH, a constrained two-objective problem

## The problem

Binh and Korn (1997) test their evolution strategy on this problem with two objectives and two
constraints, over x₁ and x₂ in [−15, 30]:

```text
minimize   f₁ = 4x₁² + 4x₂²
           f₂ = (x₁ − 5)² + (x₂ − 5)²
subject to (x₁ − 5)² + x₂² ≤ 25
           (x₁ − 8)² + (x₂ + 3)² ≥ 7.7
```

f₁ is four times the squared distance to (0, 0), and f₂ the squared distance to (5, 5). Getting
closer to one point moves away from the other. The optimal trade-offs lie on the segment between the
two points: x₁ = x₂ = t for t from 0 to 5. Their objectives form the Pareto front f = (8t², 2(t −
5)²), from (0, 50) to (200, 0). At t = 2.5, the solution (2.5, 2.5) scores (50, 12.5).

## What makes it hard

The constraints. The first keeps x inside a circle of radius 5 around (5, 0). The second keeps it
outside a circle of radius √7.7 ≈ 2.77 around (8, −3). Together they leave about 3% of the box
feasible, so almost every random solution is infeasible, and the search has to find the feasible
region first.

The constraints don't cut the front itself: both ends of the segment lie on the first circle, and
the second circle doesn't reach it. The objectives also have different scales, 0 to 200 and 0 to 50.

## Representation

A `Real` genome of 2 genes in [−15, 30]: the point x. The problem is genoxide's `Bnh`, whose fitness
is the two objectives and the total constraint violation.

Solutions compare by constrained dominance, the rule of the NSGA-II paper (Deb, Pratap, Agarwal and
Meyarivan, 2002, IEEE Transactions on Evolutionary Computation 6(2): 182-197). A feasible solution
beats an infeasible one. Of two infeasible ones, the smaller violation wins. Of two feasible ones,
the usual Pareto dominance decides. Infeasible solutions thus lead the search towards the feasible
region.

## Algorithm

NSGA-II, which ranks solutions into non-dominated fronts and prefers the less crowded ones within a
front, with parents and children competing for the next population:

- a population of 100, for 250 generations;
- simulated binary crossover with η = 20, at genoxide's default rate of 0.9;
- polynomial mutation with η = 20, at a rate of 0.5 per gene, one of the two genes per child on
  average.

## Output

The first line gives the size of the final front, and how many of its solutions are feasible.

The second gives its IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to 500 points of
the optimal front. IGD+ averages, over those 500 points, the distance to the nearest point of the
found front, counting only the objectives in which the found point is worse. 0 means that the found
front covers the optimal one. Smaller is better.

The third gives the front's hypervolume: the area it dominates, up to the reference point (210, 55).
Larger is better. For the whole optimal front, it is 210 × 55 − 5000/3 = 9883.33. In Python, `run`
evaluates the problem in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/bnh) plays this run back. Its `trace.json` has one field more than a front's trace:
`feasible`, the share of the population that is feasible in each recorded generation.

## Good results

A good front is all feasible, with an IGD+ near 0 and a hypervolume near 9883.33. No set of 100
points reaches that hypervolume: 100 points of the optimal front, evenly spaced in t, give 9849.3.
The run's front has 100 solutions, all feasible, an IGD+ of about 0.24 on objectives that span 200
and 50, and a hypervolume of about 9830.
