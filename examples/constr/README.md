---
title: CONSTR
category: multi-objective
summary: Minimize two objectives subject to two linear constraints with NSGA-II; a constraint cuts off part of the unconstrained front, and the optimal front follows the constraint's boundary instead.
reference: "Deb, K., Pratap, A., Agarwal, S. and Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation 6(2): 182-197."
reference_url: https://doi.org/10.1109/4235.996017
optimum: "the front f₂ = 7/f₁ − 9 for f₁ in [7/18, 2/3], then f₂ = 1/f₁ for f₁ in [2/3, 1]; hypervolume 5.3327 (reference point (1.1, 10))"
languages: [rust, python]
order: 100
---

# CONSTR

## The problem

The NSGA-II paper (Deb, Pratap, Agarwal and Meyarivan, 2002) defines this problem in its table V,
to test its constraint handling. It has two objectives and two constraints, over x₁ in [0.1, 1] and
x₂ in [0, 5]:

```text
minimize   f₁ = x₁
           f₂ = (1 + x₂)/x₁
subject to x₂ + 9x₁ ≥ 6
           −x₂ + 9x₁ ≥ 1
```

Without the constraints, the problem is simple. For any x₁, x₂ = 0 gives the smallest f₂, and the
optimal front is f₂ = 1/f₁, from (0.1, 10) to (1, 1): a smaller f₁ costs a larger f₂.

The constraints are two lines. The first keeps x above x₂ = 6 − 9x₁, the second below
x₂ = 9x₁ − 1. They cross at x = (7/18, 2.5), and the feasible region is the wedge between them, to
the right of that corner: about half of the box. No solution with x₁ below 7/18 ≈ 0.389 is
feasible.

x₂ = 0 is feasible only for x₁ ≥ 2/3, where the first line meets x₂'s lower bound. For a smaller
x₁, the smallest feasible x₂ is on the first line, x₂ = 6 − 9x₁, which gives
f₂ = (7 − 9x₁)/x₁ = 7/f₁ − 9. The optimal front, derived from the definition, has two pieces:

| Piece | x | f₁ from | to | f₂ | Shaped by |
|---|---|---|---|---|---|
| 1 | x₂ = 6 − 9x₁ | 7/18 | 2/3 | 7/f₁ − 9, from 9 to 1.5 | the first constraint |
| 2 | x₂ = 0 | 2/3 | 1 | 1/f₁, from 1.5 to 1 | the bound on x₂ |

The front is convex and runs from (7/18, 9) to (1, 1), with a kink at (2/3, 1.5): its slope jumps
there from −15.75 to −2.25. The second constraint only ends it, at the corner (7/18, 2.5). The first
constraint has cut off the unconstrained front's left part, f₂ = 1/f₁ for f₁ below 2/3, and
replaced it with a steeper curve along its own boundary.

## What makes it hard

Little, compared with the other constrained test problems: it is the gentle one. Half the box is
feasible, 56 of the run's first 100 random solutions are, and the objectives are smooth.

What it tests is whether the search can hold solutions exactly on a constraint's boundary. The
first piece, most of the front in f₂, lies on the first constraint's boundary, with infeasible
solutions just across it; its left end lies in the corner where both constraints meet. The
objectives also have different scales: f₁ spans 0.61, f₂ spans 8.

## Representation

A `Real` genome of 2 genes, x₁ in [0.1, 1] and x₂ in [0, 5]: the point x. The problem is genoxide's
`Constr`, whose fitness is the two objectives and the total constraint violation: the sum of how far
x breaks each constraint, 0 when it is feasible.

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

The second counts the solutions along each of the two pieces of the optimal front, told apart by
f₁ = x₁: below 2/3, and from 2/3 on.

The third gives the front's IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to 500
points of the optimal front. IGD+ averages, over those 500 points, the distance to the nearest
point of the found front, counting only the objectives in which the found point is worse. 0 means
that the found front covers the optimal one. Smaller is better.

The fourth gives the front's hypervolume: the area it dominates, up to the reference point
(1.1, 10), a little beyond the worst value of each objective on the front. Larger is better. For
the whole optimal front, it is 19 · 5/18 − 7 ln(12/7) + 10/3 − ln(3/2) + 0.9 = 5.3327: the areas
above the two pieces, and the strip from f₁ = 1 to 1.1. In Python, `run` evaluates the problem in
Rust, so both versions print the same.

The run's `trace.json` has one field more than a front's trace: `feasible`, the share of the
population that is feasible in each recorded generation.

[The project page](https://tachsin.gr/projects/genoxide/examples/constr) plays this run back.

## Good results

A good front is all feasible, and has an IGD+ near 0 and a hypervolume near 5.3327. 100 points of
the optimal front, spread evenly along it, give a hypervolume of 5.3072 and an IGD+ of 0.0025.

The run's population is all feasible from generation 1, and its front has 100 solutions by
generation 12. The hypervolume reaches 5.2921 at generation 32 and ends at 5.3024, 99.4% of the
whole front's, with an IGD+ of 0.0035. The final front puts 73 solutions on the first piece and 27
on the second. Points spread evenly along the front in its own units would put 93 on the first
piece, where f₂ falls by 7.5. NSGA-II measures crowding with each objective scaled to its range
instead, and there the second piece, which spans more than half of f₁'s range, weighs more.
