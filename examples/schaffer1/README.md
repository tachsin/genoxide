---
title: Schaffer 1
category: multi-objective
summary: Minimize x² and (x − 2)² over one variable in [−1000, 1000] with NSGA-II, and fill the convex front between the two minima.
reference: "Schaffer, J. D. (1985). Multiple objective optimization with vector evaluated genetic algorithms. Proceedings of the First International Conference on Genetic Algorithms: 93-100."
reference_url: ""
optimum: "the front f₂ = (√f₁ − 2)² for f₁ in [0, 4]; hypervolume 16.693 (reference point (4.4, 4.4))"
languages: [rust, python]
order: 106
family: Schaffer
---

# Schaffer 1

## The problem

Schaffer (1985) tested his vector evaluated genetic algorithm (VEGA), one of the first genetic
algorithms for several objectives, on a problem with one variable and two objectives, both
minimized. In the form that Deb, Pratap, Agarwal and Meyarivan (2002, IEEE Transactions on
Evolutionary Computation 6(2): 182-197, table I) restate for NSGA-II, x is in [−1000, 1000]:

```text
f₁ = x²
f₂ = (x − 2)²
```

f₁ is smallest at x = 0, and f₂ at x = 2. Between them, moving x towards one minimum moves it away
from the other, so every x in [0, 2] is a best trade-off: no other x is better in both objectives.
These solutions are the Pareto set. Outside it, x < 0 is worse in both objectives than x = 0, and
x > 2 than x = 2.

The objective values of the Pareto set are the Pareto front: f₂ = (√f₁ − 2)² for f₁ from 0 to 4, a
convex curve from (0, 4) to (4, 0) that bows towards the origin. At x = 1, its middle, both
objectives are 1.

The definition and bounds are the NSGA-II paper's; other papers use other bounds. genoxide hasn't
yet checked them against Schaffer's original.

## What makes it hard

The objectives are two parabolas, and the front is convex, so even a weighted sum of the objectives
finds every point of it. The difficulty is the bounds. The Pareto set, [0, 2], is 0.1% of
[−1000, 1000]. A random population of 100 has one solution in it with a probability of about 10%,
and far from it both objectives are huge: in this run, the best initial solution is x ≈ 30.5, at
(927.5, 809.7), and it dominates all the others.

The width of the bounds also sets the size of the mutation's steps. Polynomial mutation moves a
gene by a fraction of its range, and with η = 20 the median step is about 3% of it: 65 here, 30
times the width of the Pareto set. Only about 1 mutated child in 100 of a parent in [0, 2] lands
back in [0, 2]. The population reaches the Pareto set in a few generations, but then fills it
slowly, about one new solution per generation.

## Representation

A `Real` genome of 1 gene in [−1000, 1000]: the variable x. The problem is genoxide's `Schaffer1`,
whose fitness is the pair (f₁, f₂). In Python, `run` evaluates it in Rust, so both versions print
the same.

## Algorithm

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002), with the settings of the paper that restates
this problem. It ranks solutions by non-dominated sorting: the first front is the solutions that no
other solution beats in both objectives, the second front those beaten only by the first, and so on.
Within a front, it prefers solutions in less crowded regions (crowding distance). Parents and
children compete for the next population, so it keeps the best solutions found so far.

- a population of 100, for 250 generations;
- simulated binary crossover with η = 15, at genoxide's default rate of 0.9;
- polynomial mutation with η = 20, at a rate of 1/n per gene for n variables: here 1, so every child
  is mutated.

A lower mutation rate fills the front sooner: at a rate of 0.2 per gene, the front has 100
solutions after 10 generations instead of about 90, because crossover of two parents in [0, 2]
gives children in [0, 2]. But on seeds 1 to 5 it ends with a slightly smaller hypervolume, 16.626
on average against 16.630, so the example keeps the paper's rate.

## Output

The first line gives the size of the final front.

The second gives its IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to 500 points of
the optimal front, evenly spaced in x. IGD+ averages, over those 500 points, the distance to the
nearest point of the found front, counting only the objectives in which the found point is worse.
0 means that the found front covers the optimal one. Smaller is better.

The third gives the front's hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on
Evolutionary Computation 3(4): 257-271): the area it dominates, up to a reference point. Larger is
better. The reference point is (4.4, 4.4), 10% of the front's range beyond its worst point (4, 4).
For the whole front, the hypervolume is 4.4² − 8/3 = 16.693: the box minus the area under the
curve.

[The project page](https://tachsin.gr/projects/genoxide/examples/schaffer1) plays this run back.

## Good results

A good front has 100 solutions spread over the whole curve, from (0, 4) to (4, 0), with an IGD+
near 0 and a hypervolume near 16.693. No set of 100 points reaches that hypervolume: 100 points of
the optimal front, evenly spaced in x, give 16.639 and an IGD+ of 0.0068.

The run's front has 100 solutions, an IGD+ of 0.0082 and a hypervolume of 16.628, 99.6% of the
whole front's. It has 5 solutions after 4 generations, 100 after 93, and its hypervolume grows
until about generation 200, while the solutions spread more evenly. Its last solutions include a
few just past x = 2, with f₁ up to 4.008, which x = 2 would dominate.

On seeds 1 to 5, NSGA-II ends between 16.628 and 16.633. SPEA2 and SMS-EMOA, with the same
settings, end between 16.625 and 16.638: on this problem, the three are as good.
