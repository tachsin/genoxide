---
title: Fonseca-Fleming
category: multi-objective
summary: Minimize two objectives with a concave front over 3 variables in [−4, 4] with NSGA-II, as in the NSGA-II paper.
reference: "Fonseca, C. M. and Fleming, P. J. (1995). An overview of evolutionary algorithms in multiobjective optimization. Evolutionary Computation 3(1): 1-16."
reference_url: https://doi.org/10.1162/evco.1995.3.1.1
optimum: "the front (1 − exp(−(s − 1)²), 1 − exp(−(s + 1)²)) for s in [−1, 1]; hypervolume 0.5521 (reference point (1.1, 1.1))"
languages: [rust, python]
order: 98
---

# Fonseca-Fleming

## The problem

Fonseca and Fleming (1995) posed a problem with two objectives, both minimized, in n variables,
each in [−4, 4]:

```text
f₁ = 1 − exp(−Σ (xᵢ − 1/√n)²)
f₂ = 1 − exp(−Σ (xᵢ + 1/√n)²)
```

The sums are the squared distances from x to two points: a = (1/√n, …, 1/√n) and its opposite −a.
Both points are at distance 1 from the origin. Each objective is 0 at its point, and rises towards 1
with the distance from it. The definition and bounds are as Deb, Thiele, Laumanns and Zitzler
(2001, TIK-Report 112) restate them, and, for 3 variables, as the NSGA-II paper (Deb, Pratap,
Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation 6(2): 182-197, table I)
does. This example uses 3 variables, as that paper does. genoxide hasn't yet checked the
restatements against the original.

The best trade-offs, the Pareto set, are the points on the segment from −a to a: all variables
equal, to t from −1/√n to 1/√n. Any other point is beaten by the nearest point of the segment,
which is closer to both a and −a. With
s = t√n from −1 to 1, the objectives on the segment are:

```text
f₁ = 1 − exp(−(s − 1)²)
f₂ = 1 − exp(−(s + 1)²)
```

That is the Pareto front, the same for every n: a curve from (0, 0.9817) at s = 1 to (0.9817, 0) at
s = −1, where 0.9817 = 1 − e⁻⁴. At the origin, s = 0, both objectives are 1 − e⁻¹ = 0.632. The
straight line between the two ends passes through (0.491, 0.491), so the front bulges away from the
origin: it is concave.

## What makes it hard

A concave front. Minimizing a weighted sum of the objectives, w f₁ + (1 − w) f₂, finds only the
two ends of a concave front, whatever the weight w: every point in between has a larger sum than
one of the ends. An algorithm that ranks solutions by Pareto dominance, not by a sum, can find the
middle.

A plateau. Far from the segment, both exponentials vanish, and both objectives are close to 1. In
87% of the box [−4, 4]³, both are above 0.99, so most random solutions look alike, and the search
has little to follow. The more variables, the larger the plateau's share of the box.

## Representation

A `Real` genome of 3 genes in [−4, 4]: the vector x. The problem is genoxide's `FonsecaFleming`
with 3 variables, whose fitness is the pair (f₁, f₂). In Python, `run` evaluates it in Rust, so
both versions print the same.

## Algorithm

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002), with the settings of the paper that restates
this problem. It ranks solutions by non-dominated sorting: the first front is the solutions that no
other solution beats in both objectives, the second front those beaten only by the first, and so on.
Within a front, it prefers solutions in less crowded regions (crowding distance). Parents and
children compete for the next population, so it keeps the best solutions found so far.

- a population of 100, for 250 generations;
- simulated binary crossover with η = 15, at genoxide's default rate of 0.9;
- polynomial mutation with η = 20, at a rate of 1/3 per gene, one gene per child on average.

## Output

The first line gives the size of the final front.

The second gives its IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to 500 points of
the optimal front, evenly spaced in s. IGD+ averages, over those 500 points, the distance to the
nearest point of the found front, counting only the objectives in which the found point is worse.
0 means that the found front covers the optimal one. Smaller is better.

The third gives the front's hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on
Evolutionary Computation 3(4): 257-271): the area it dominates, up to a reference point. Larger is
better. The reference point is (1.1, 1.1), beyond the front's worst point (0.9817, 0.9817) and
every objective value, which is below 1. For the whole front, the hypervolume is 0.5521: the box
minus the area under the curve, found numerically.

[The project page](https://tachsin.gr/projects/genoxide/examples/fonseca-fleming) plays this run back.

## Good results

A good front has 100 solutions spread from (0, 0.9817) to (0.9817, 0), an IGD+ near 0 and a
hypervolume near 0.5521. No set of 100 points reaches that hypervolume: 100 points of the optimal
front, evenly spaced in s, give 0.5469 and an IGD+ of 0.0022.

The run starts with 4 solutions on its front, and a hypervolume of 0.167. It has 50 after 8
generations and 100 after 12, and its hypervolume reaches 0.544 after about 50 generations. It
ends with 100 solutions, an IGD+ of 0.0038 and a hypervolume of 0.5446, 98.6% of the whole front's.

On seeds 1 to 5, NSGA-II ends between 0.5437 and 0.5448, with an IGD+ of 0.0036 to 0.0043. Other
algorithms with the same settings do better here. SPEA2, which removes crowded solutions one at a
time, ends between 0.5457 and 0.5460, with an IGD+ of about 0.0031. SMS-EMOA, which keeps the
solutions that add the most hypervolume, ends between 0.5472 and 0.5474, with an IGD+ of about
0.0024: more than the 100 evenly spaced points, because it places its points where they add the
most area, not evenly. Both cost more per generation than NSGA-II.
