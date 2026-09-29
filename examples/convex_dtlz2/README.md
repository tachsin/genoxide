---
title: Convex DTLZ2 with 3 objectives
category: multi-objective
summary: Minimize three objectives whose Pareto front is convex, flat near its edges and steep in between, with NSGA-III and MOEA/D.
reference: "Deb, K. and Jain, H. (2014). An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, part I: solving problems with box constraints. IEEE Transactions on Evolutionary Computation 18(4): 577-601."
reference_url: https://doi.org/10.1109/TEVC.2013.2281535
optimum: "the surface f₃ + √f₁ + √f₂ = 1 with fᵢ in [0, 1]; hypervolume 1.2943 (reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 180
family: DTLZ
tab: Convex DTLZ2
---

# Convex DTLZ2 with 3 objectives

## The problem

Deb and Jain (2014) introduced NSGA-III with tests on DTLZ problems, whose fronts are a plane (DTLZ1) or a sphere (DTLZ2 to DTLZ4), and on variants they built to test it further. The convex DTLZ2 (their section V-D) takes DTLZ2's objectives and raises them to powers: with M = 3 objectives and 12 variables in [0, 1],

```text
g  = (x₃ − 0.5)² + … + (x₁₂ − 0.5)²
f₁ = [(1 + g) cos(x₁π/2) cos(x₂π/2)]⁴
f₂ = [(1 + g) cos(x₁π/2) sin(x₂π/2)]⁴
f₃ = [(1 + g) sin(x₁π/2)]²
```

All three are minimized. The optimal solutions are DTLZ2's, x₃ = … = x₁₂ = 0.5 and g = 0, and the front is the convex surface f₃ + √f₁ + √f₂ = 1 (their eq. 8), from the corners (1, 0, 0), (0, 1, 0) and (0, 0, 1). Its ideal point is the origin and its nadir point (1, 1, 1).

## What makes it hard

The shape. The front is almost flat near its edges and changes sharply between them, so evenly spread reference directions meet it unevenly: Deb and Jain use it to check that NSGA-III still finds a spread of points, and they report that MOEA/D with Tchebycheff decomposition finds a non-uniform spread and MOEA/D with the PBI decomposition misses the front's boundary. The ten distance variables also have to converge to 0.5.

## Representation

A `Real` genome of 12 genes in [0, 1]: the vector x. The problem is genoxide's `ConvexDtlz2::<3>::default()`, whose fitness is the three objectives; in Python, `gx.problems.ConvexDtlz2()`, which `run` evaluates in Rust, so both versions print the same.

## Algorithm

NSGA-III (Deb and Jain, 2014) ranks solutions into non-dominated fronts, like NSGA-II, and chooses among the last front that fits by reference directions: it normalizes the objectives by the ideal point and the front's extreme points, attaches each solution to its nearest direction, and prefers the directions with the fewest members. The directions here are Das and Dennis's 91 points with 12 divisions (1998, SIAM Journal on Optimization 8(3): 631-657), all (a/12, b/12, c/12) with a + b + c = 12, with a population of 92, the multiple of four above 91, as in Deb and Jain's experiments. Simulated binary crossover with η = 30 and polynomial mutation with η = 20 at a rate of 1/12 per gene, as they use; 500 generations, twice the 250 of their table IX.

MOEA/D (Zhang and Li, 2007, IEEE Transactions on Evolutionary Computation 11(6): 712-731) splits the problem into one subproblem per weight vector, here the same 91 points, each minimizing the Tchebycheff distance to the ideal point, and neighboring subproblems share their solutions. genoxide's MOEA/D doesn't normalize the objectives: the weights apply to their raw values. Simulated binary crossover with η = 20, the same mutation, 500 generations.

## Output

A line per run: the size of its final front, its hypervolume, as a share of the sample's, and the median and largest distance g of its solutions from the front. Then the hypervolumes of the whole optimal front, from 3,003 of its points, and of the sample.

The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the volume that the front dominates up to a reference point. Larger is better. Here the objectives are divided by the front's nadir point, (1, 1, 1), so that the front spans [0, 1] in each, and the reference point is (1.1, 1.1, 1.1). No finite set of solutions reaches the whole front's hypervolume: the benchmark is the sample.

The sample is genoxide's `optimal_front(91)`: DTLZ2's front at Das and Dennis's 91 directions, raised to the same powers: what a front of 91 solutions can be. g comes from f₃ + √f₁ + √f₂ = (1 + g)²: 0 on the front. [The project page](https://tachsin.gr/projects/genoxide/examples/convex-dtlz2) plays both runs back, each front in a panel of its own over points of the optimal front, and each hypervolume over the generations.

## Good results

The target: a hypervolume at least 99% of the sample's, with g near 0.

NSGA-III's front has 92 solutions and a hypervolume of 1.2755, 100.2% of the sample's: the directions don't meet the front at the sample's points, and its spread happens to dominate a little more. Its median g is 0.00002. MOEA/D keeps 71 different solutions, 94.9% of the sample's hypervolume: several subproblems share their best solutions near the corners.

Over seeds 1 to 20, NSGA-III reaches 100.1% to 100.2% of the sample's hypervolume, with median g at most 0.00005 (the largest g of a front is 0.0005 to 0.026: a stray solution that no other dominates yet). MOEA/D reaches 94.9% to 96.1%.
