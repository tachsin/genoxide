---
title: MW8
category: multi-objective
summary: Minimize three objectives over 15 variables subject to one constraint, whose Pareto front is the unit sphere in four bands, with NSGA-III.
reference: "Ma, Z. and Wang, Y. (2019). Evolutionary constrained multiobjective optimization: test suite construction and performance comparisons. IEEE Transactions on Evolutionary Computation 23(6): 972-986."
reference_url: https://doi.org/10.1109/TEVC.2019.2896967
optimum: "the unit sphere with non-negative coordinates where arcsin f₃ is in [0, π/24], [π/8, 5π/24], [7π/24, 3π/8] or [11π/24, π/2]; hypervolume 0.7677 (reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 191
family: MW
---

# MW8

## The problem

Ma and Wang (2019) built fourteen constrained test problems, MW1 to MW14; three take any number of objectives. MW8 (eq. 22), with 3 objectives and 15 variables in [0, 1], is DTLZ2's shape with the multimodal distance function g₂:

```text
f₁ = g₂ cos(0.5πx₁) cos(0.5πx₂)
f₂ = g₂ cos(0.5πx₁) sin(0.5πx₂)
f₃ = g₂ sin(0.5πx₁)
subject to (1.25 − 0.5 sin(6l)²)² − f₁² − f₂² − f₃² ≥ 0,  l = arcsin(f₃ / √(f₁² + f₂² + f₃²))
g₂ = 1 + Σᵢ₌₃¹⁵ (1.5 + 0.1 zᵢ²/15 − 1.5 cos 2πzᵢ),  zᵢ = 1 − exp(−10 (xᵢ − (i − 1)/15)²)
```

All three are minimized. At g₂ = 1 the solutions lie on the unit sphere. The constraint keeps them inside the radius 1.25 − 0.5 sin(6l)², which falls below 1 where sin(6l)² > 1/2: MW8 is of type II, and its optimal front, derived here, is the sphere in four bands of the angle l, [0, π/24], [π/8, 5π/24], [7π/24, 3π/8] and [11π/24, π/2]. Between them, no direction has a feasible solution. The ideal point is the origin and the nadir point (1, 1, 1).

## What makes it hard

g₂: each of its thirteen terms has a second, local minimum where zᵢ reaches 1, far from its best value, only 0.0067 higher, behind a barrier of 3 at zᵢ = 0.5; a solution whose distance variables stopped there sits just outside the sphere. And the bands: a solution slightly above g₂ = 1 is still feasible in a band's middle, but not near its edges.

## Representation

A `Real` genome of 15 genes in [0, 1]: the vector x, the paper's size. The problem is genoxide's `Mw8::<3>::default()`, whose fitness is the three objectives and the constraint violation; solutions compare by constrained dominance, feasible ones first. In Python, `gx.problems.Mw8()`, which `run` evaluates in Rust, so both versions print the same.

## Algorithm

NSGA-III (Deb and Jain, 2014) ranks solutions into non-dominated fronts, like NSGA-II, and chooses among the last front that fits by reference directions: it normalizes the objectives by the ideal point and the front's extreme points, attaches each solution to its nearest direction, and prefers the directions with the fewest members. The directions here are Das and Dennis's 91 points with 12 divisions (1998, SIAM Journal on Optimization 8(3): 631-657), all (a/12, b/12, c/12) with a + b + c = 12, with a population of 92, the multiple of four above 91, as in Deb and Jain's experiments. Ma and Wang use NSGA-III for their many-objective tests of MW8. Twice:

- with the paper's settings: simulated binary crossover and polynomial mutation both with η = 20, the mutation at a rate of 1/15 per gene, 600 generations;
- with Deb and Jain's crossover, η = 30, and polynomial mutation with η = 2 at the same rate, for 2,000 generations.

With η = 20, the median mutation step is about 3% of the range, and fewer than 1 in 1,000 steps cover more than 30% of it; with η = 2, the median step is 8% to 17% of the range, and about 1 in 5 steps cover more than 30%, enough to cross g₂'s barriers.

## Output

A line per run: the size of its final front, its hypervolume, as a share of the sample's, and the median and largest g₂ of its solutions. Then the hypervolumes of the whole optimal front, from 3,000 or more of its points, and of the sample.

The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the volume that the front dominates up to a reference point. Larger is better. Here the objectives are divided by the front's nadir point, (1, 1, 1), so that the front spans [0, 1] in each, and the reference point is (1.1, 1.1, 1.1). No finite set of solutions reaches the whole front's hypervolume: the benchmark is the sample.

The sample is genoxide's `optimal_front(91)`: Das and Dennis's points projected on the sphere and kept in the bands, with more divisions until there are at least 91, here 99: what a front of 91 solutions can be. g₂ comes from f₁² + f₂² + f₃² = g₂²: 1 on the front. [The project page](https://tachsin.gr/projects/genoxide/examples/mw8) plays both runs back, each front in a panel of its own over points of the optimal front, and each hypervolume over the generations.

## Good results

The target: a hypervolume at least 99% of the sample's, with g₂ near 1.

With the paper's settings, the front has 96.5% of the sample's hypervolume and median g₂ 1.014: most of the population stopped in g₂'s local minima. With η = 2, it has 100.1%, with median g₂ 1.00000, on all four bands.

Over seeds 1 to 20, the paper's settings reach 99% on one seed (89.4% to 100.1%, median 94.3%), and η = 2 on every seed, with 100.0% to 100.4%. Ma and Wang report a mean IGD of 0.106 for the same NSGA-III on 3 objectives (supplement, table S-R-IX).
