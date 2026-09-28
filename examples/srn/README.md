---
title: SRN (Srinivas and Deb)
category: multi-objective
summary: Minimize two objectives inside a circle and above a line with NSGA-II, and find all three pieces of the optimal front, not only the one usually quoted.
reference: "Srinivas, N. and Deb, K. (1994). Multiobjective optimization using nondominated sorting in genetic algorithms. Evolutionary Computation 2(3): 221-248."
reference_url: https://doi.org/10.1162/evco.1994.2.3.221
optimum: "a front in three pieces, from (10.1, 2.61) to (222.97, −217.74); hypervolume 35478.6 (reference point (245, 25))"
languages: [rust, python]
order: 122
---

# SRN (Srinivas and Deb)

## The problem

Srinivas and Deb (1994) test NSGA, the first non-dominated sorting genetic algorithm, on this
problem with two objectives and two constraints, over x₁ and x₂ in [−20, 20]:

```text
minimize   f₁ = (x₁ − 2)² + (x₂ − 1)² + 2
           f₂ = 9x₁ − (x₂ − 1)²
subject to x₁² + x₂² ≤ 225
           x₁ − 3x₂ ≤ −10
```

f₁ is the squared distance to the point (2, 1), plus 2. f₂ falls as x₁ falls and as x₂ moves away
from 1. The first constraint keeps x inside the circle of radius 15 around the origin. The second
keeps it above the line x₂ = (x₁ + 10)/3, which runs from (−10, 0) to (2, 4). Together they leave
about 16% of the box feasible: the part of the disc above the line.

genoxide restates the problem as the NSGA-II paper (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE
Transactions on Evolutionary Computation 6(2): 182-197, table V) and Binh and Korn (1997, section
5.1) do; it hasn't checked it against the original paper yet.

The optimal front follows from one identity: f₁ + f₂ = (x₁ − 2)² + 2 + 9x₁ = (x₁ + 2.5)² − 0.25.
x₂ cancels out. No solution has f₁ + f₂ below −0.25, and every solution on the line x₁ = −2.5
reaches it. Moving x₂ along that line trades f₁ against f₂ one for one. So the feasible part of
x₁ = −2.5 is optimal, and its objectives lie on the straight line f₁ + f₂ = −0.25.

The constraints cut that line at both ends, and the front continues along them. It has three
pieces:

| Piece | x | f₁ from | to | Shaped by |
|---|---|---|---|---|
| 1 | x₁ = 3x₂ − 10, x₂ from 3.7 down to 2.5 | 10.1 | 24.5 | the line constraint |
| 2 | x₁ = −2.5, x₂ from 2.5 up to √218.75 ≈ 14.79 | 24.5 | 212.42 | the identity |
| 3 | on the circle, to x ≈ (−4.841, 14.197) | 212.42 | 222.97 | the circle constraint |

On the first piece, the line constraint keeps x₂ from moving closer to 1, so the front slides along
the constraint's boundary to (1.1, 3.7), the feasible point nearest to (2, 1). On the third, x₂
can't grow past the circle, and f₂ still falls a little further round it, until it stops at
x ≈ (−4.841, 14.197). The front runs from (10.1, 2.61) to (222.97, −217.74). The solutions on
x₁ = −2.5 are the ones usually quoted, but they are only part of the front.

## What makes it hard

Not much, by the standards of constrained problems, which is why it is a common first test.
A sixth of the box is feasible: 15 of the run's first 100 random solutions are.

Its difficulties are in the details. The front has three pieces, two of them on constraint
boundaries, and a search that only finds the middle one misses both ends. The objectives span about
213 and 220. And the middle piece lies along a flat valley: a solution at x₁ = −2.5 + d is worse
than the optimal line by only d² in f₁ + f₂. A solution 0.5 away from x₁ = −2.5 costs 0.25, on
objectives that run over hundreds, so the selection pressure towards the exact line is weak.

## Representation

A `Real` genome of 2 genes in [−20, 20]: the point x. The problem is genoxide's `Srn`, whose fitness
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

The second counts the solutions along each piece of the optimal front, told apart by f₁: below
24.5, between 24.5 and 212.42, and above.

The third gives the front's IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to 500
points of the optimal front. IGD+ averages, over those 500 points, the distance to the nearest
point of the found front, counting only the objectives in which the found point is worse. 0 means
that the found front covers the optimal one. Smaller is better.

The fourth gives the front's hypervolume: the area it dominates, up to the reference point
(245, 25), a little beyond the worst value of each objective on the front. Larger is better. For
the whole optimal front, computed from 2,000,000 of its points, it is 35478.6. In Python, `run`
evaluates the problem in Rust, so both versions print the same.

The run's `trace.json` has one field more than a front's trace: `feasible`, the share of the
population that is feasible in each recorded generation.

[The project page](https://tachsin.gr/projects/genoxide/examples/srn) plays this run back.

## Good results

A good front is all feasible, covers the three pieces, and has an IGD+ near 0 and a hypervolume
near 35478.6. No set of 100 points reaches that hypervolume or an IGD+ of 0: 100 points of the
optimal front, spread evenly along it, give a hypervolume of 35198.5 and an IGD+ of 0.54, since the
500 reference points fall between them. They put 10, 86 and 4 points on the three pieces.

The run's population is all feasible by generation 3, and its front has 100 solutions from
generation 5 on, spread over all three pieces: 9, 86 and 5 at the end. Its IGD+ is 0.79 and its
hypervolume 35121.5, 99.0% of the whole front's. The gap to the evenly spread points comes from the
flat valley: on the middle piece, the solutions' x₁ lies between about −3.2 and −1.2 instead of at
−2.5. After about generation 10, the hypervolume stays between 35040 and 35140, as the population
trades one near-optimal solution for another.
