---
title: Inverted DTLZ1 with 3 objectives
category: multi-objective
summary: Minimize three objectives whose Pareto front is DTLZ1's triangle turned upside down, with NSGA-III and its usual reference directions, and with directions turned the same way.
reference: "Jain, H. and Deb, K. (2014). An evolutionary many-objective optimization algorithm using reference-point based nondominated sorting approach, part II: handling constraints and extending to an adaptive approach. IEEE Transactions on Evolutionary Computation 18(4): 602-622."
reference_url: https://doi.org/10.1109/TEVC.2013.2281534
optimum: "the triangle f₁ + f₂ + f₃ = 1 with every fᵢ in [0, 0.5]; hypervolume 0.3392 (objectives divided by the nadir point (0.5, 0.5, 0.5), reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 219
family: DTLZ
tab: Inverted DTLZ1
---

# Inverted DTLZ1 with 3 objectives

## The problem

Jain and Deb (2014), in the second part of their NSGA-III paper, invert DTLZ1 to test an adaptive version of it (their section VIII-A, eq. 9): DTLZ1's objectives are computed, and each is replaced by fᵢ ← 0.5 (1 + g) − fᵢ. With 3 objectives and 7 variables in [0, 1],

```text
g  = 100 (5 + Σᵢ₌₃⁷ ((xᵢ − 0.5)² − cos(20π (xᵢ − 0.5))))
f₁ = 0.5 (1 + g) (1 − x₁ x₂)
f₂ = 0.5 (1 + g) (1 − x₁ (1 − x₂))
f₃ = 0.5 (1 + g) x₁
```

All three are minimized. The optimal solutions are DTLZ1's, x₃ = … = x₇ = 0.5 and g = 0, and the front, derived here, is the triangle f₁ + f₂ + f₃ = 1 with each fᵢ at most 0.5: DTLZ1's triangle turned upside down, with corners (0, 0.5, 0.5), (0.5, 0, 0.5) and (0.5, 0.5, 0). Its ideal point is the origin and its nadir point (0.5, 0.5, 0.5).

## What makes it hard

NSGA-III's reference directions are spread over the whole triangle of the normalized objectives, point up; the front points down. Many directions have no part of the front near them, and a front of 91 solutions gets fewer useful directions: Jain and Deb report that NSGA-III finds only 28 well-spread points for 91 directions, and propose an adaptive NSGA-III (A-NSGA-III) that moves the directions to where the solutions are. DTLZ1's distance function also has 11⁵ − 1 local optima.

## Representation

A `Real` genome of 7 genes in [0, 1]: the vector x. The problem is genoxide's `InvertedDtlz1::<3>::default()`, whose fitness is the three objectives; in Python, `gx.problems.InvertedDtlz1()`, which `run` evaluates in Rust, so both versions print the same.

## Algorithm

NSGA-III (Deb and Jain, 2014) ranks solutions into non-dominated fronts, like NSGA-II, and chooses among the last front that fits by reference directions: it normalizes the objectives by the ideal point and the front's extreme points, attaches each solution to its nearest direction, and prefers the directions with the fewest members. The directions here are Das and Dennis's 91 points with 12 divisions (1998, SIAM Journal on Optimization 8(3): 631-657), all (a/12, b/12, c/12) with a + b + c = 12, with a population of 92, the multiple of four above 91, as in Deb and Jain's experiments. Simulated binary crossover with η = 30 and polynomial mutation with η = 20 at a rate of 1/7 per gene, for 2,000 generations, twice:

- with the usual directions, Das and Dennis's points;
- with the same points turned upside down, (1 − d)/2 for each direction d: on the triangle of the normalized objectives, the directions fill the inverted triangle that the front makes. genoxide has no adaptive NSGA-III, but its NSGA-III takes any directions, and these are what an adaptive version aims for on this front.

## Output

A line per run: the size of its final front, its hypervolume, as a share of the sample's, and the median and largest distance g of its solutions from the front. Then the hypervolumes of the whole optimal front, from 3,003 of its points, and of the sample.

The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the volume that the front dominates up to a reference point. Larger is better. Here the objectives are divided by the front's nadir point, (0.5, 0.5, 0.5), so that the front spans [0, 1] in each, and the reference point is (1.1, 1.1, 1.1). No finite set of solutions reaches the whole front's hypervolume: the benchmark is the sample.

The sample is genoxide's `optimal_front(91)`: Das and Dennis's 91 points on the inverted triangle, 0.5 (1 − p) for each point p, where the inverted directions meet the front: what a front of 91 solutions can be. g comes from f₁ + f₂ + f₃ = 1 + g: 0 on the front. [The project page](https://tachsin.gr/projects/genoxide/examples/inverted-dtlz1) plays both runs back, each front in a panel of its own over points of the optimal front, and each hypervolume over the generations.

## Good results

The target: a hypervolume at least 99% of the sample's, with g near 0.

With the usual directions, the front has 92 solutions and 93.7% of the sample's hypervolume: the solutions gather on fewer directions. With the inverted directions, it has 100.0%: one solution per direction, each at the sample's point. Both have converged, with g below 0.001.

Over seeds 1 to 20, the usual directions reach 91.1% to 93.7% of the sample's hypervolume, and the inverted directions 99.7% to 100.0%, with median g at most 0.001.
