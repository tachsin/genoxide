---
title: MW4
category: multi-objective
summary: Minimize three objectives over 15 variables subject to one constraint, whose Pareto front is the triangle f₁ + f₂ + f₃ = 1, with NSGA-III and NSGA-II.
reference: "Ma, Z. and Wang, Y. (2019). Evolutionary constrained multiobjective optimization: test suite construction and performance comparisons. IEEE Transactions on Evolutionary Computation 23(6): 972-986."
reference_url: https://doi.org/10.1109/TEVC.2019.2896967
optimum: "the triangle f₁ + f₂ + f₃ = 1 with every fᵢ ≥ 0; hypervolume 1.1577 (reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 207
family: MW
---

# MW4

## The problem

Ma and Wang (2019) built fourteen constrained test problems, MW1 to MW14, each objective a distance function g of some variables times a shape of the others, and constraints near the front shaped by a periodic term; three of them take any number of objectives. MW4 (eq. 18), with 3 objectives and 15 variables in [0, 1], is DTLZ1's shape with 1 − x for x:

```text
f₁ = g₁ (1 − x₁)(1 − x₂)
f₂ = g₁ (1 − x₁) x₂
f₃ = g₁ x₁
subject to 1 + 0.4 sin(2.5πl)⁸ − f₁ − f₂ − f₃ ≥ 0,  l = f₃ − f₁ − f₂
g₁ = 1 + Σᵢ₌₃¹⁵ (1 − exp(−10 (zᵢ − 0.5 − (i − 1)/30)²)),  zᵢ = xᵢ¹²
```

All three are minimized. At g₁ = 1 the objectives sum to 1, and the constraint, whose sine term is never negative, holds: MW4 is of type I, and its optimal front is the triangle f₁ + f₂ + f₃ = 1. A larger g₁ moves a solution away from it, where the constraint leaves only a thin, wavy layer. The ideal point is the origin and the nadir point (1, 1, 1).

## What makes it hard

The biased distance function g₁: each zᵢ = xᵢ¹² must be 0.5 + (i − 1)/30, between 0.57 and 0.97, which needs every xᵢ between 0.954 and 0.997. And the feasible region, under 0.1‰ of the search space (table II): a random genome is far from it.

## Representation

A `Real` genome of 15 genes in [0, 1]: the vector x, the paper's size. The problem is genoxide's `Mw4::<3>::default()`, whose fitness is the three objectives and the constraint violation; solutions compare by constrained dominance, feasible ones first. In Python, `gx.problems.Mw4()`, which `run` evaluates in Rust, so both versions print the same.

## Algorithm

NSGA-III (Deb and Jain, 2014) ranks solutions into non-dominated fronts, like NSGA-II, and chooses among the last front that fits by reference directions: it normalizes the objectives by the ideal point and the front's extreme points, attaches each solution to its nearest direction, and prefers the directions with the fewest members. The directions here are Das and Dennis's 91 points with 12 divisions (1998, SIAM Journal on Optimization 8(3): 631-657), all (a/12, b/12, c/12) with a + b + c = 12, with a population of 92, the multiple of four above 91, as in Deb and Jain's experiments. Ma and Wang use NSGA-III for their many-objective tests of MW4 (supplement, table S-R-VIII). Simulated binary crossover and polynomial mutation both with η = 20, the mutation at a rate of 1/15 per gene, as in the paper, for 1,000 generations.

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation 6(2): 182-197), with the same population and operators, which spreads the front by crowding distance instead of directions.

## Output

A line per run: the size of its final front, its hypervolume, as a share of the sample's, and the median and largest g₁ of its solutions. Then the hypervolumes of the whole optimal front, from 3,003 of its points, and of the sample.

The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the volume that the front dominates up to a reference point. Larger is better. Here the objectives are divided by the front's nadir point, (1, 1, 1), so that the front spans [0, 1] in each, and the reference point is (1.1, 1.1, 1.1). No finite set of solutions reaches the whole front's hypervolume: the benchmark is the sample.

The sample is genoxide's `optimal_front(91)`: Das and Dennis's 91 points, where NSGA-III's directions meet the front: what a front of 91 solutions can be. g₁ comes from f₁ + f₂ + f₃ = g₁: 1 on the front. [The project page](https://tachsin.gr/projects/genoxide/examples/mw4) plays both runs back, each front in a panel of its own over points of the optimal front, and each hypervolume over the generations.

## Good results

The target: a hypervolume at least 99% of the sample's, with g₁ near 1.

NSGA-III's front has 92 solutions and 100.0% of the sample's hypervolume, with median g₁ 1.00008. NSGA-II's has 97.4%: crowding distance spreads three objectives less evenly, and a few of its solutions are still off the front (g₁ up to 1.07).

Over seeds 1 to 20, NSGA-III reaches 99.9% to 100.0% of the sample's hypervolume, with median g₁ at most 1.0002, and NSGA-II 96.1% to 97.9%.
