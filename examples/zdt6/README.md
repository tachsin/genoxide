---
title: ZDT6
category: multi-objective
summary: Minimize two conflicting objectives over 10 variables, with a concave Pareto front and a search space that crowds solutions at one end of it, with NSGA-II.
reference: "Zitzler, E., Deb, K. and Thiele, L. (2000). Comparison of multiobjective evolutionary algorithms: empirical results. Evolutionary Computation 8(2): 173-195."
reference_url: https://doi.org/10.1162/106365600568202
optimum: "the front f₂ = 1 − f₁² for f₁ from 0.2808 to 1; hypervolume 0.5079 (reference point (1.1, 1.1))"
languages: [rust, python]
order: 105
family: ZDT
---

# ZDT6

## The problem

Zitzler, Deb and Thiele (2000) built six test problems with two objectives from one scheme: f₁
depends on the first variable, a function g on the others, and f₂ on both. ZDT6 is the sixth (the
fifth has binary variables). It has 10 variables in [0, 1], and minimizes both objectives:

```text
f₁ = 1 − exp(−4 x₁) sin⁶(6π x₁)
g  = 1 + 9 ((x₂ + … + x₁₀) / 9)^0.25
f₂ = g (1 − (f₁ / g)²)
```

f₂ is that of ZDT2, but f₁ and g are new. The best solutions have g = 1, that is
x₂ = … = x₁₀ = 0, where f₂ = 1 − f₁². f₁ can't go below 0.2808, its value at
x₁ = atan(9π) / (6π) ≈ 0.0815. So the Pareto front is the concave curve f₂ = 1 − f₁² for f₁ from
0.2808 to 1: from (0.2808, 0.9212) to (1, 0).

## What makes it hard

Zitzler, Deb and Thiele built ZDT6 to test a search space that is not uniform, in two ways.

The solutions are uneven along the front. f₁ is 1 wherever sin(6π x₁) is 0, at x₁ = 0, 1/6, 2/6,
…, 1, and dips below 1 between them, less and less as exp(−4 x₁) shrinks. So most values of x₁
give an f₁ near 1. Of 1,001 evenly spaced values of x₁, only 60 give an f₁ in the lower half of the
front's range, below 0.6404; 804 give more than 0.9. A random population crowds at the lower right
end of the front, and the search has to find the few x₁ that reach its upper left end.

The solutions thin out towards the front. g − 1 is 9 times the fourth root of the mean m of
x₂ to x₁₀. A random solution has m near 0.5, and g near 8.6. To bring g within 0.01 of 1, m has to
fall below 1.5 × 10⁻¹². Halving g − 1 takes dividing m by 16, so each step closer to the front
takes a much smaller m than the one before, and a random change to x₂ to x₁₀ rarely helps.

## Representation

A `Real` genome of 10 genes in [0, 1]: the vector x. The problem is genoxide's `Zdt6`, whose
fitness is the pair (f₁, f₂). In Python, `run` evaluates it in Rust, so both versions print the
same.

## Algorithm

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation
6(2): 182-197), as the ZDT1 example runs it. It ranks solutions by non-dominated sorting: the first
front is the solutions that no other solution beats in both objectives, the second front those
beaten only by the first, and so on. Within a front, it prefers solutions in less crowded regions
(crowding distance). Parents and children compete for the next population, so it keeps the best
solutions found so far.

Crowding distance works in the objectives, not in x. So NSGA-II spreads its front evenly along the
curve, however unevenly x₁ maps onto it.

- a population of 100, for 250 generations, as in the NSGA-II paper;
- simulated binary crossover with η = 15, at genoxide's default rate of 0.9;
- polynomial mutation with η = 20, at a rate of 1/10 per gene, one gene per child on average.

## Output

The first line shows the uneven map from x₁ to f₁: of 1,001 evenly spaced values of x₁, with the
other variables at 0, how many give an f₁ below 0.6404, the middle of the front's range.

The second gives the size of the final front, how many of its solutions have an f₁ below 0.6404,
and its smallest f₁.

The third gives its IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to 500 points of
the optimal front, evenly spaced in f₁. IGD+ averages, over those 500 points, the distance to the
nearest point of the found front, counting only the objectives in which the found point is worse.
0 means that the found front covers the optimal one. Smaller is better.

The fourth gives the front's hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on
Evolutionary Computation 3(4): 257-271): the area it dominates, up to the reference point
(1.1, 1.1). Larger is better. For the whole front, it is 0.5079: the integral of 1.1 − f₂ over f₁
from 0.2808 to 1, plus the strip from 1 to 1.1.

[The project page](https://tachsin.gr/projects/genoxide/examples/zdt6) plays this run back.

## Good results

A good front has 100 solutions spread from (0.2808, 0.9212) to (1, 0), an IGD+ near 0 and a
hypervolume near 0.5079. No set of 100 points reaches that hypervolume: 100 points of the optimal
front, evenly spaced in f₁, give 0.5045 and an IGD+ of 0.0020.

The first front has 6 solutions, only one of them with an f₁ below 0.6404. NSGA-II finds the
front's upper left end quickly: its front has the smallest f₁, 0.2808, from generation 8. It ends
with 100 solutions, 45 of them in the lower half of the range: the front is spread evenly, against
the 6% that the map from x₁ gives.

It is the convergence that is slow. The front's smallest g falls below 2 at about generation 49,
and below 1.1 at about 112, but after 250 generations g is still 1.005 to 1.010 on the front. The
run ends with an IGD+ of 0.0071 and a hypervolume of 0.4959, 97.6% of the whole front's: just above
the front all along it.

On seeds 1 to 5, NSGA-II ends between 0.4953 and 0.4960, with an IGD+ of about 0.0071 to 0.0076.
More generations help: after 500, it ends between 0.5029 and 0.5035, with an IGD+ of about 0.0028,
and after 1,000 between 0.5038 and 0.5039. With the same settings and 250 generations, SPEA2 ends
between 0.4931 and 0.4952, SMS-EMOA between 0.4905 and 0.4951, and MOEA/D lower, between 0.4865
and 0.4883.
