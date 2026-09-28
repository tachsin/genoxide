---
title: Schaffer 2
category: multi-objective
summary: Minimize a piecewise linear objective and a parabola over one variable with NSGA-II, and cover both pieces of a disconnected front.
reference: "Schaffer, J. D. (1985). Multiple objective optimization with vector evaluated genetic algorithms. Proceedings of the First International Conference on Genetic Algorithms: 93-100."
reference_url: ""
optimum: "the front f₂ = (f₁ − 3)² for f₁ in [−1, 0) and f₂ = (f₁ − 1)² for f₁ in [0, 1]; hypervolume 26.053 (reference point (1.2, 17.6))"
languages: [rust, python]
order: 107
---

# Schaffer 2

## The problem

Schaffer's (1985) second problem has one variable, x in [−5, 10], and two objectives, both
minimized. f₁ is piecewise linear, a zigzag with minima at x = 1 and x = 4:

```text
f₁ = −x      for x ≤ 1
     x − 2   for 1 < x ≤ 3
     4 − x   for 3 < x ≤ 4
     x − 4   for x > 4
f₂ = (x − 5)²
```

f₁ is −1 at x = 1, its minimum, and 0 at x = 4, a second, worse minimum. f₂ is smallest at x = 5.
The best trade-offs, the Pareto set, are x in [1, 2) and x in [4, 5]. From x = 1, moving right
makes f₂ better and f₁ worse, until x = 2, where f₁ = 0. There, the solution is (0, 9), and x = 4
beats it: (0, 1). From x = 4 to 5, f₁ grows to 1 and f₂ falls to 0. Every other x is beaten in
both objectives: x in [2, 3] by x + 2, x in (3, 4) by 8 − x, x < 1 by x = 1, and x > 5 by x = 5.

The Pareto front, the objective values of the Pareto set, is in two pieces:

- f₂ = (f₁ − 3)² for f₁ from −1 to 0, from (−1, 16) down to (0, 9), without its end;
- f₂ = (f₁ − 1)² for f₁ from 0 to 1, from (0, 1) to (1, 0).

Between them, f₂ drops from 9 to 1 at f₁ = 0. The definition and bounds are as Van Veldhuizen
(1999, PhD thesis, Air Force Institute of Technology, table B.1) restates them, after Srinivas and
Deb (1994). genoxide hasn't yet checked them against Schaffer's original.

## What makes it hard

The front is disconnected. Its two pieces come from two separate intervals of x, and the solutions
between them, x in [2, 4), are all dominated. The population has to keep two separate groups, and
a child of one parent from each group can land between them, where it's lost.

The pieces also differ in size. In objective space, the first spans 1 in f₁ and 7 in f₂; the second
1 in f₁ and 1 in f₂. An algorithm that spreads its solutions evenly by some measure puts different
numbers of them on each piece, depending on that measure.

f₁ has kinks at x = 1, 3 and 4, but the search doesn't use gradients, so they don't matter. With
one variable, and a Pareto set that fills 2 of the 15 units of the interval, the search finds both
pieces fast: the difficulty is keeping them and spreading solutions over them.

## Representation

A `Real` genome of 1 gene in [−5, 10]: the variable x. The problem is genoxide's `Schaffer2`, whose
fitness is the pair (f₁, f₂). In Python, `run` evaluates it in Rust, so both versions print the
same.

## Algorithm

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation
6(2): 182-197). It ranks solutions by non-dominated sorting: the first front is the solutions that
no other solution beats in both objectives, the second front those beaten only by the first, and so
on. Within a front, it prefers solutions in less crowded regions (crowding distance). Parents and
children compete for the next population, so it keeps the best solutions found so far. Both pieces
of the front are in the first front, so non-dominated sorting keeps both groups.

The settings are the usual ones of the NSGA-II paper:

- a population of 100, for 250 generations;
- simulated binary crossover with η = 15, at genoxide's default rate of 0.9;
- polynomial mutation with η = 20, at a rate of 1/n per gene for n variables: here 1, so every child
  is mutated.

## Output

The first line gives the size of the final front, and how many of its solutions are on each piece:
on the first, f₁ < 0; on the second, f₁ ≥ 0.

The second gives its IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to 500 points of
the optimal front, on both pieces. IGD+ averages, over those 500 points, the distance to the
nearest point of the found front, counting only the objectives in which the found point is worse.
0 means that the found front covers the optimal one. Smaller is better.

The third gives the front's hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on
Evolutionary Computation 3(4): 257-271): the area it dominates, up to a reference point. Larger is
better. The reference point is (1.2, 17.6), 10% of the front's range beyond its worst point (1, 16).
For the whole front, the hypervolume is 2.2 × 17.6 − 37/3 − 1/3 = 26.053: the box minus the areas
under the two pieces.

[The project page](https://tachsin.gr/projects/genoxide/examples/schaffer2) plays this run back.

## Good results

A good front has solutions on both pieces, spread over each, an IGD+ near 0 and a hypervolume near
26.053. No set of 100 points reaches that hypervolume. genoxide's `optimal_front(100)` shares 100
points between the pieces by their lengths in objective space, 83 on the first and 17 on the
second, and they give 25.979 and an IGD+ of 0.0068.

The run's front has 100 solutions after 4 generations, on both pieces. It ends with 61 on the
first piece and 39 on the second, an IGD+ of 0.0087 and a hypervolume of 25.964, 99.7% of the whole
front's. The split follows from crowding distance, which adds up each solution's gaps to its
neighbors, with each objective divided by its range on the front, 2 for f₁ and 16 for f₂. Along the
first piece these gaps add up to 1/2 + 7/16, and along the second to 1/2 + 1/16: 62.5% and 37.5% of
the total.

From generation 12 on, the hypervolume moves between 25.95 and 25.97. NSGA-II doesn't keep it from
falling: when more than 100 solutions are non-dominated, crowding distance drops some of them, not
always those that add the least area. On seeds 1 to 5, NSGA-II ends between 25.964 and 25.971,
SPEA2 between 25.972 and 25.977, and SMS-EMOA, which keeps the solutions that add the most
hypervolume, between 25.983 and 25.986, more than the 100 points of `optimal_front(100)`.
