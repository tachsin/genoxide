---
title: MW14
category: multi-objective
summary: Minimize three objectives over 15 variables subject to one constraint, whose Pareto front is four disconnected patches, with NSGA-III and SMS-EMOA; neither reaches 99% of the hypervolume of 100 points of the front.
reference: "Ma, Z. and Wang, Y. (2019). Evolutionary constrained multiobjective optimization: test suite construction and performance comparisons. IEEE Transactions on Evolutionary Computation 23(6): 972-986."
reference_url: https://doi.org/10.1109/TEVC.2019.2896967
optimum: "f₃ = (φ(f₁) + φ(f₂))/2, φ(t) = 6 − eᵗ − 1.5 sin(1.1πt²), with f₁ and f₂ each in [0, 0.7314] or (1.3296, 1.5]; hypervolume 0.6742 (objectives normalized by the ideal and nadir points, reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 233
family: MW
---

# MW14

## The problem

Ma and Wang (2019) built fourteen constrained test problems, MW1 to MW14; three take any number of objectives. MW14 (eq. 28), with 3 objectives and 15 variables in [0, 1.5], has the distance function with linked variables g₃:

```text
f₁ = x₁,  f₂ = x₂
f₃ = g₃ (φ(f₁) + φ(f₂))/2,  φ(t) = 6 − eᵗ − 1.5 sin(1.1πt²)
subject to ((6.1 − α(f₁)) + (6.1 − α(f₂)))/2 − f₃ ≥ 0,  α(t) = 1 + t + 0.5t² + 1.5 sin(1.1πt²)
g₃ = 1 + Σᵢ₌₃¹⁵ 2 (xᵢ + (xᵢ₋₁ − 0.5)² − 1)²
```

All three are minimized. At g₃ = 1 the constraint holds everywhere (it compares eᵗ with the first terms of its series, which are smaller), so MW14 is of type I: its optimal front is the non-dominated part of the unconstrained one. As in DTLZ7, f₁ and f₂ each count on their own, and a value is optimal where φ falls below its values at every smaller t: in [0, 0.7314] or (1.3296, 1.5] (derived here, the first local minimum of φ and where φ falls back to it). The front is four patches, f₃ = (φ(f₁) + φ(f₂))/2 over the four combinations. Its ideal point is (0, 0, 0.0229) and its nadir point (1.5, 1.5, 5).

## What makes it hard

The patches: a solution with f₁ or f₂ in the gap between 0.7314 and 1.3296 is dominated by one with the gap's lower edge, but only if the population holds that solution, and a finite population keeps such solutions. NSGA-III's directions that point into a gap attach solutions there. And g₃: every solution on the front has its own chain of distance variables, which starts from x₂.

## Representation

A `Real` genome of 15 genes in [0, 1.5]: the vector x, the paper's size. The problem is genoxide's `Mw14::<3>::default()`, whose fitness is the three objectives and the constraint violation; solutions compare by constrained dominance, feasible ones first. In Python, `gx.problems.Mw14()`, which `run` evaluates in Rust, so both versions print the same.

## Algorithm

NSGA-III (Deb and Jain, 2014) ranks solutions into non-dominated fronts, like NSGA-II, and chooses among the last front that fits by reference directions: it normalizes the objectives by the ideal point and the front's extreme points, attaches each solution to its nearest direction, and prefers the directions with the fewest members. The directions here are Das and Dennis's 91 points with 12 divisions (1998, SIAM Journal on Optimization 8(3): 631-657), all (a/12, b/12, c/12) with a + b + c = 12, with a population of 92, the multiple of four above 91, as in Deb and Jain's experiments. Ma and Wang use NSGA-III for their many-objective tests of MW14. Simulated binary crossover and polynomial mutation both with η = 20, the mutation at a rate of 1/15 per gene, for 3,000 generations.

SMS-EMOA (Beume, Naujoks and Emmerich, 2007, European Journal of Operational Research 181(3): 1653-1669), in genoxide's generational form, with a population of 92 and the same operators, for 2,000 generations: from the last front that fits only in part, it removes the solutions that add the least hypervolume, which drops solutions in the gaps.

## Output

A line per run: the size of its final front, its hypervolume, as a share of the sample's, and how many of its solutions lie in the gaps between the patches. Then the hypervolumes of the whole optimal front, from 3,025 of its points, and of the sample.

The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the volume that the front dominates up to a reference point. Larger is better. Here the objectives are divided by the front's nadir point, (1.5, 1.5, 5) after subtracting the ideal point (0, 0, 0.0229), so that the front spans [0, 1] in each, and the reference point is (1.1, 1.1, 1.1). No finite set of solutions reaches the whole front's hypervolume: the benchmark is the sample.

The sample is genoxide's `optimal_front(91)`, a grid of 10 × 10 values of f₁ and f₂ over the patches, 100 points: what a front of 91 solutions can be. [The project page](https://tachsin.gr/projects/genoxide/examples/mw14) plays both runs back, each front in a panel of its own over points of the optimal front, and each hypervolume over the generations.

## Good results

The target: a hypervolume at least 99% of the sample's. Neither algorithm reaches it.

NSGA-III's front has 92 solutions and 98.0% of the sample's hypervolume, 7 of them in the gaps. SMS-EMOA's has 97.8%, with 1 in the gaps.

Over seeds 1 to 20, NSGA-III reaches 97.3% to 98.2% of the sample's hypervolume, with 4 to 16 solutions in the gaps, and SMS-EMOA, over seeds 1 to 10, 97.2% to 97.8%, with at most 1. Other settings do no better within a few seconds: NSGA-III for 10,000 generations 98.2%, NSGA-III with mutation η = 2 97.7% to 97.9% (seeds 1 and 2), SPEA2 with a population of 100 97.7%, NSGA-II 93% to 95%, and MOEA/D 79%. A front of 92 solutions can do better: 92 points of the front chosen greedily, each adding the most hypervolume, reach 100.9% of the sample's. So these algorithms stop about 3% short, with some solutions still off the front or in the gaps. Ma and Wang report a mean IGD of 0.114 for NSGA-III on MW14 with 3 objectives (supplement, table S-R-IX), the worst of their three many-objective problems.
