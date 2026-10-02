---
title: Scaled DTLZ2 with 3 objectives
category: multi-objective
summary: Minimize three objectives whose ranges differ a hundredfold, with a front that is an eighth of an ellipsoid, with NSGA-III and MOEA/D.
reference: "Deb, K. and Jain, H. (2014). An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, part I: solving problems with box constraints. IEEE Transactions on Evolutionary Computation 18(4): 577-601."
reference_url: https://doi.org/10.1109/TEVC.2013.2281535
optimum: "the ellipsoid f₁² + (f₂/10)² + (f₃/100)² = 1 with every fᵢ ≥ 0; hypervolume 0.7971 (objectives divided by the nadir point (1, 10, 100), reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 218
family: DTLZ
tab: Scaled DTLZ2
---

# Scaled DTLZ2 with 3 objectives

## The problem

Deb and Jain (2014) built scaled DTLZ problems to test algorithms on objectives with different ranges (their section V-C). Scaled DTLZ2 multiplies DTLZ2's objective i by 10^(i − 1) for 3 objectives: with 12 variables in [0, 1],

```text
g  = (x₃ − 0.5)² + … + (x₁₂ − 0.5)²
f₁ = (1 + g) cos(x₁π/2) cos(x₂π/2)
f₂ = 10 (1 + g) cos(x₁π/2) sin(x₂π/2)
f₃ = 100 (1 + g) sin(x₁π/2)
```

All three are minimized. The optimal solutions are DTLZ2's, x₃ = … = x₁₂ = 0.5 and g = 0, and the front is the eighth of the ellipsoid f₁² + (f₂/10)² + (f₃/100)² = 1 with non-negative coordinates: its ideal point is the origin and its nadir point (1, 10, 100). As for scaled DTLZ1, genoxide follows the paper's example and figures, 10^(i − 1), where its table captions read 10ⁱ; for 10 and 15 objectives, table VIII sets factors (3 and 2) that differ from scaled DTLZ1's.

## What makes it hard

The scaling: f₃ ranges over 100 and f₁ over 1, so a spread that is even in the raw objectives crowds the front where f₃ is small. Deb and Jain show NSGA-III, which normalizes, finding an even spread, and MOEA/D, which doesn't, missing most of the front. The ten distance variables also have to converge to 0.5.

## Representation

A `Real` genome of 12 genes in [0, 1]: the vector x. The problem is genoxide's `ScaledDtlz2::<3>::default()`, with the paper's factor 10 for 3 objectives, whose fitness is the three objectives; in Python, `gx.problems.ScaledDtlz2()`, which `run` evaluates in Rust, so both versions print the same.

## Algorithm

NSGA-III (Deb and Jain, 2014) ranks solutions into non-dominated fronts, like NSGA-II, and chooses among the last front that fits by reference directions: it normalizes the objectives by the ideal point and the front's extreme points, attaches each solution to its nearest direction, and prefers the directions with the fewest members. The directions here are Das and Dennis's 91 points with 12 divisions (1998, SIAM Journal on Optimization 8(3): 631-657), all (a/12, b/12, c/12) with a + b + c = 12, with a population of 92, the multiple of four above 91, as in Deb and Jain's experiments. Simulated binary crossover with η = 30 and polynomial mutation with η = 20 at a rate of 1/12 per gene, as they use; 500 generations, twice the 250 of their table VIII.

MOEA/D (Zhang and Li, 2007, IEEE Transactions on Evolutionary Computation 11(6): 712-731) splits the problem into one subproblem per weight vector, here the same 91 points, each minimizing the Tchebycheff distance to the ideal point, and neighboring subproblems share their solutions. genoxide's MOEA/D doesn't normalize the objectives: the weights apply to their raw values. Simulated binary crossover with η = 20, the same mutation, 500 generations.

## Output

A line per run: the size of its final front, its hypervolume, as a share of the sample's, and the median and largest distance g of its solutions from the front. Then the hypervolumes of the whole optimal front, from 3,003 of its points, and of the sample.

The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the volume that the front dominates up to a reference point. Larger is better. Here the objectives are divided by the front's nadir point, (1, 10, 100), so that the front spans [0, 1] in each, and the reference point is (1.1, 1.1, 1.1). No finite set of solutions reaches the whole front's hypervolume: the benchmark is the sample.

The sample is genoxide's `optimal_front(91)`: Das and Dennis's 91 points projected on the sphere and scaled, the points where NSGA-III's directions meet the front once the objectives are normalized: what a front of 91 solutions can be. g comes from f₁² + (f₂/10)² + (f₃/100)² = (1 + g)²: 0 on the front. [The project page](https://tachsin.gr/projects/genoxide/examples/scaled-dtlz2) plays both runs back, each front in a panel of its own over points of the optimal front, and each hypervolume over the generations.

## Good results

The target: a hypervolume at least 99% of the sample's, with g near 0.

NSGA-III's front has 92 solutions and 100.0% of the sample's hypervolume, with median g 0.00002. MOEA/D keeps 73 solutions and 63.6% of the sample's hypervolume, crowded toward (1, 0, 0), where f₂ and f₃ are near 0 (its median normalized f₁ is 0.99).

Over seeds 1 to 20, NSGA-III reaches 99.9% to 100.0% of the sample's hypervolume, with median g at most 0.00004. MOEA/D reaches 63.6% on every seed.
