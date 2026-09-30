---
title: Scaled DTLZ1 with 3 objectives
category: multi-objective
summary: Minimize three objectives whose ranges differ a hundredfold, with a linear front behind 11⁵ − 1 local fronts, with NSGA-III and MOEA/D.
reference: "Deb, K. and Jain, H. (2014). An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, part I: solving problems with box constraints. IEEE Transactions on Evolutionary Computation 18(4): 577-601."
reference_url: https://doi.org/10.1109/TEVC.2013.2281535
optimum: "the plane f₁ + f₂/10 + f₃/100 = 0.5 with every fᵢ ≥ 0; hypervolume 1.1577 (objectives divided by the nadir point (0.5, 5, 50), reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 201
family: DTLZ
tab: Scaled DTLZ1
---

# Scaled DTLZ1 with 3 objectives

## The problem

Deb and Jain (2014) built scaled DTLZ problems to test algorithms on objectives with different ranges, as real problems have (their section V-C). Scaled DTLZ1 multiplies DTLZ1's objective i by 10^(i − 1) for 3 objectives: with 7 variables in [0, 1],

```text
g  = 100 (5 + Σᵢ₌₃⁷ ((xᵢ − 0.5)² − cos(20π (xᵢ − 0.5))))
f₁ = 0.5 x₁ x₂ (1 + g)
f₂ = 10 × 0.5 x₁ (1 − x₂) (1 + g)
f₃ = 100 × 0.5 (1 − x₁) (1 + g)
```

All three are minimized. The optimal solutions are DTLZ1's, x₃ = … = x₇ = 0.5 and g = 0, and the front is the triangle f₁ + f₂/10 + f₃/100 = 0.5, from the origin's corners (0.5, 0, 0), (0, 5, 0) and (0, 0, 50): its ideal point is the origin and its nadir point (0.5, 5, 50). The paper's text says a factor 10ⁱ, and its example multiplies f₁, f₂ and f₃ by 10⁰, 10¹ and 10², as its figures show, while the captions of its tables VII and VIII read 10ⁱ for i from 1: genoxide follows the example. Table VIII sets smaller factors for more objectives (3 for 8, 2 for 10, 1.2 for 15).

## What makes it hard

Two things. DTLZ1's distance function g has 11⁵ − 1 local optima, so the population descends through local fronts, planes parallel to the optimal one. And the scaling: an algorithm that spreads its solutions by the raw objectives crowds them where f₃, a hundred times larger than f₁, dominates the geometry. Deb and Jain show NSGA-III, which normalizes, finding an even spread, and MOEA/D, which doesn't, missing most of the front.

## Representation

A `Real` genome of 7 genes in [0, 1]: the vector x. The problem is genoxide's `ScaledDtlz1::<3>::default()`, with the paper's factor 10 for 3 objectives, whose fitness is the three objectives; in Python, `gx.problems.ScaledDtlz1()`, which `run` evaluates in Rust, so both versions print the same.

## Algorithm

NSGA-III (Deb and Jain, 2014) ranks solutions into non-dominated fronts, like NSGA-II, and chooses among the last front that fits by reference directions: it normalizes the objectives by the ideal point and the front's extreme points, attaches each solution to its nearest direction, and prefers the directions with the fewest members. The directions here are Das and Dennis's 91 points with 12 divisions (1998, SIAM Journal on Optimization 8(3): 631-657), all (a/12, b/12, c/12) with a + b + c = 12, with a population of 92, the multiple of four above 91, as in Deb and Jain's experiments. Simulated binary crossover with η = 30 and polynomial mutation with η = 20 at a rate of 1/7 per gene, as they use; 1,000 generations, more than the 400 of their table VIII, which leave g near 0.01.

MOEA/D (Zhang and Li, 2007, IEEE Transactions on Evolutionary Computation 11(6): 712-731) splits the problem into one subproblem per weight vector, here the same 91 points, each minimizing the Tchebycheff distance to the ideal point, and neighboring subproblems share their solutions. genoxide's MOEA/D doesn't normalize the objectives: the weights apply to their raw values. Simulated binary crossover with η = 20, the same mutation, 1,000 generations.

## Output

A line per run: the size of its final front, its hypervolume, as a share of the sample's, and the median and largest distance g of its solutions from the front. Then the hypervolumes of the whole optimal front, from 3,003 of its points, and of the sample.

The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the volume that the front dominates up to a reference point. Larger is better. Here the objectives are divided by the front's nadir point, (0.5, 5, 50), so that the front spans [0, 1] in each, and the reference point is (1.1, 1.1, 1.1). No finite set of solutions reaches the whole front's hypervolume: the benchmark is the sample.

The sample is genoxide's `optimal_front(91)`: Das and Dennis's 91 points, halved and scaled, the points where NSGA-III's directions meet the front once the objectives are normalized: what a front of 91 solutions can be. g comes from 2 (f₁ + f₂/10 + f₃/100) = 1 + g: 0 on the front, and 1 or more on the nearest local front. [The project page](https://tachsin.gr/projects/genoxide/examples/scaled-dtlz1) plays both runs back, each front in a panel of its own over points of the optimal front, and each hypervolume over the generations.

## Good results

The target: a hypervolume at least 99% of the sample's, with g near 0.

NSGA-III's front has 92 solutions and 99.9% of the sample's hypervolume, with g 0.0015. MOEA/D keeps 65 solutions and 64.6% of the sample's hypervolume: with raw objectives, most of its solutions crowd toward the corner (0.5, 0, 0), where f₂ and f₃, the objectives with the large ranges, are near 0 (its median normalized f₁ is 0.88), and the rest of the triangle stays nearly empty.

Over seeds 1 to 20, NSGA-III reaches 99.9% to 100.0% of the sample's hypervolume, with median g at most 0.0017. MOEA/D reaches 64.3% to 68.4%.
