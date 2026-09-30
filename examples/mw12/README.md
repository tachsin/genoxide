---
title: MW12
category: multi-objective
summary: Minimize two objectives over 15 variables subject to two constraints, whose Pareto front lies on a wavy constraint boundary, with NSGA-II.
reference: "Ma, Z. and Wang, Y. (2019). Evolutionary constrained multiobjective optimization: test suite construction and performance comparisons. IEEE Transactions on Evolutionary Computation 23(6): 972-986."
reference_url: https://doi.org/10.1109/TEVC.2019.2896967
optimum: "the boundary T₁ = 0 from (0, 1) to (1.3164, 0.0039); hypervolume 0.7397 (normalized objectives, reference point (1.1, 1.1))"
languages: [rust, python]
order: 215
family: MW
---

# MW12

## The problem

Ma and Wang (2019) built fourteen constrained test problems, MW1 to MW14. Each objective is a distance function g of some variables times a shape of the others, so that g = 1, its least value, puts a solution on the unconstrained front, and the constraints are curves near that front whose shapes a periodic term adjusts. The paper sorts the problems by what the constraints do to the front: type I leaves it whole, type II cuts parts of it out, type III replaces parts of it with pieces of a constraint's boundary, and type IV moves all of it onto boundaries. In every problem the feasible region is small: under 0.1‰ of the search space for most (table II).

MW12 (eq. 26) has two objectives and two constraints, over 15 variables in [0, 1], with the biased distance function g₁ (eq. 12):

```text
f₁ = g₁ x₁
f₂ = g₁ (0.85 − 0.8 f₁/g₁ − 0.08 |sin(3.2π f₁/g₁)|)
subject to T₁ T₄ ≤ 0 and T₂ T₃ ≥ 0
T₁ = 1 − 0.8f₁ − f₂ + 0.08 sin(2π(f₂ − f₁/1.5))
T₂ = 1 − 0.625f₁ − f₂ + 0.08 sin(2π(f₂ − f₁/1.6))
T₃ = 1.4 − 0.875f₁ − f₂ + 0.08 sin(2π(f₂/1.4 − f₁/1.6))
T₄ = 1.8 − 1.125f₁ − f₂ + 0.08 sin(2π(f₂/1.8 − f₁/1.6))
g₁ = 1 + Σᵢ₌₂¹⁵ (1 − exp(−10 (zᵢ − 0.5 − (i − 1)/30)²)),  zᵢ = xᵢ¹³
```

Both objectives are minimized. The unconstrained front, a wavy line from (0, 0.85), is infeasible: the constraints keep solutions between wavy lines above it. MW12 is of type IV: its optimal front, derived here, is the lower boundary T₁ = 0 of the feasible band, one curve from (0, 1) to (1.3164, 0.0039), the first feasible point in the direction of x₁ = 1. At f₁ = 0, T₁ and T₂ vanish together at f₂ = 1, which makes (0, 1) feasible in exact arithmetic, the curve's limit, while rounding leaves it infeasible by about 10⁻¹⁷. The ideal point is (0, 0.0039) and the nadir point (1.3164, 1).

## What makes it hard

The whole front is on a boundary, with g₁ between 1.13 and 1.39 along it: the distance variables must stop at the right distance from their best values, different along the front. g₁ is 1 where each zᵢ = xᵢ¹³ is 0.5 + (i − 1)/30, between 0.53 and 0.97: every xᵢ must be between 0.952 and 0.997, where the thirteenth power is steep. Ma and Wang call the power a bias: most values of xᵢ put zᵢ near 0.

## Representation

A `Real` genome of 15 genes in [0, 1]: the vector x, the paper's size. The problem is genoxide's `Mw12`, whose fitness is the two objectives and the total constraint violation: the sum of how far x breaks each constraint, 0 when it is feasible. In Python, `gx.problems.Mw12()`, which `run` evaluates in Rust, so both versions print the same.

Solutions compare by constrained dominance, the rule of the NSGA-II paper and the constraint handling that Ma and Wang call CDP. A feasible solution beats an infeasible one. Of two infeasible ones, the smaller violation wins. Of two feasible ones, Pareto dominance decides.

## Algorithm

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation 6(2): 182-197), twice, with a population of 100 and simulated binary crossover with η = 20 at genoxide's default rate of 0.9:

- with the paper's settings: polynomial mutation with η = 20 at a rate of 1/15 per gene, for 600 generations, 60,000 evaluations, as in Ma and Wang's comparison (section IV-C);
- with polynomial mutation with η = 2, for 5,000 generations, 500,000 evaluations.

The distribution index η sets how far a mutation moves a gene. With η = 20, the median step is about 3% of the range, and fewer than 1 in 1,000 steps cover more than 30% of it; with η = 2, the median step is 8% to 17% of the range, depending on where the gene is, and about 1 in 5 steps cover more than 30% (measured with genoxide's polynomial mutation). On MW12 both settings usually reach the front.

## Output

A line per run: the size of its final front, how many of its solutions are feasible, and the front's IGD+ and hypervolume. Then the hypervolume of the whole optimal front.

Both indicators use the objectives normalized by the front's ideal and nadir points, (0, 0.0039) and (1.3164, 1), so that the front spans [0, 1] in each. IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) averages, over 500 points of the optimal front from genoxide's `optimal_front`, the distance to the nearest point of the found front, counting only the objectives in which the found point is worse. Smaller is better, and 0 means the found front covers the optimal one. The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the area the front dominates up to the reference point (1.1, 1.1). Larger is better; the whole front's is computed from 20,000 of its points. Only feasible solutions count.

The page plays both runs back over the grey feasible region, which the trace samples from genomes on the unconstrained front's rays; hollow points are infeasible solutions of the populations, and the line is the optimal front, in its pieces.

[The project page](https://tachsin.gr/projects/genoxide/examples/mw12) plays this run back.

## Good results

The target: an IGD+ of at most 0.01 in normalized objectives.

With the paper's settings, the run with seed 1 already reaches it: IGD+ 0.0031, hypervolume 0.7334. Over seeds 1 to 20, 16 runs do; the median IGD+ is 0.0034, and the worst 0.6474. Ma and Wang report a mean IGD of 0.0534 for the same NSGA-II (supplement, table S-R-I).

With η = 2 and 5,000 generations, the run's front has the same IGD+, 0.0031, and hypervolume, 0.7334, 99.1% of the whole front's. Over seeds 1 to 20, every run reaches the target, with IGD+ from 0.0029 to 0.0035.
