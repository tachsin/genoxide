---
title: MW10
category: multi-objective
summary: Minimize two objectives over 15 variables subject to three constraints, whose Pareto front is two pieces on feasible islands, with f₁ biased toward 0, with NSGA-II.
reference: "Ma, Z. and Wang, Y. (2019). Evolutionary constrained multiobjective optimization: test suite construction and performance comparisons. IEEE Transactions on Evolutionary Computation 23(6): 972-986."
reference_url: https://doi.org/10.1109/TEVC.2019.2896967
optimum: "two pieces, each part constraint boundary and part the parabola f₂ = 1 − f₁², from (0.2325, 1.1350) to (1, 0); hypervolume 0.6883 (normalized objectives, reference point (1.1, 1.1))"
languages: [rust, python]
order: 229
family: MW
---

# MW10

## The problem

Ma and Wang (2019) built fourteen constrained test problems, MW1 to MW14. Each objective is a distance function g of some variables times a shape of the others, so that g = 1, its least value, puts a solution on the unconstrained front, and the constraints are curves near that front whose shapes a periodic term adjusts. The paper sorts the problems by what the constraints do to the front: type I leaves it whole, type II cuts parts of it out, type III replaces parts of it with pieces of a constraint's boundary, and type IV moves all of it onto boundaries. In every problem the feasible region is small: under 0.1‰ of the search space for most (table II).

MW10 (eq. 24) has two objectives and three constraints, over 15 variables in [0, 1], with the multimodal distance function g₂ (eq. 13):

```text
f₁ = g₂ x₁¹⁵
f₂ = g₂ (1 − (f₁/g₂)²)
subject to (2 − 4f₁² − f₂)(2 − 8f₁² − f₂) ≥ 0
           (2 − 2f₁² − f₂)(2 − 16f₁² − f₂) ≤ 0
           (1 − f₁² − f₂)(1.2 − 1.2f₁² − f₂) ≤ 0
g₂ = 1 + Σᵢ₌₂¹⁵ (1.5 + 0.1 zᵢ²/15 − 1.5 cos 2πzᵢ),  zᵢ = 1 − exp(−10 (xᵢ − (i − 1)/15)²)
```

Both objectives are minimized. At g₂ = 1 the solutions lie on the parabola f₂ = 1 − f₁². The constraints are products of pairs of parabolas, and together they leave two islands between them, above the unconstrained front. MW10 is of type III, disconnected: its optimal front, derived here, has two pieces, each starting on a constraint's boundary and ending on the parabola: from (0.2325, 1.1350) along the second constraint's boundary to (0.2582, 0.9333) and the parabola to (0.3779, 0.8572); and from (0.5345, 0.8571) along the first constraint's boundary to (0.5774, 0.6667) and the parabola to (1, 0). The ideal point is (0.2325, 0) and the nadir point (1, 1.1350).

## What makes it hard

The fifteenth power of x₁: f₁ = x₁¹⁵ at g₂ = 1, so f₁ = 0.25 needs x₁ = 0.912 and most of x₁'s range maps to f₁ near 0, left of both islands; Ma and Wang call it a polynomial bias that keeps an algorithm from finding the islands. g₂ is 1 where each xᵢ is (i − 1)/15. Each term has a second, local minimum where zᵢ reaches 1, far from that value, only 0.0067 higher; between the two lies a barrier of 3 at zᵢ = 0.5. A variable that settles on the wrong side stays there unless a single step carries it across.

## Representation

A `Real` genome of 15 genes in [0, 1]: the vector x, the paper's size. The problem is genoxide's `Mw10`, whose fitness is the two objectives and the total constraint violation: the sum of how far x breaks each constraint, 0 when it is feasible. In Python, `gx.problems.Mw10()`, which `run` evaluates in Rust, so both versions print the same.

Solutions compare by constrained dominance, the rule of the NSGA-II paper and the constraint handling that Ma and Wang call CDP. A feasible solution beats an infeasible one. Of two infeasible ones, the smaller violation wins. Of two feasible ones, Pareto dominance decides.

## Algorithm

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation 6(2): 182-197), twice, with a population of 100 and simulated binary crossover with η = 20 at genoxide's default rate of 0.9:

- with the paper's settings: polynomial mutation with η = 20 at a rate of 1/15 per gene, for 600 generations, 60,000 evaluations, as in Ma and Wang's comparison (section IV-C);
- with polynomial mutation with η = 2, for 5,000 generations, 500,000 evaluations.

The distribution index η sets how far a mutation moves a gene. With η = 20, the median step is about 3% of the range, and fewer than 1 in 1,000 steps cover more than 30% of it; with η = 2, the median step is 8% to 17% of the range, depending on where the gene is, and about 1 in 5 steps cover more than 30% (measured with genoxide's polynomial mutation). The larger steps move x₁ into the narrow range where the islands are.

## Output

A line per run: the size of its final front, how many of its solutions are feasible, and the front's IGD+ and hypervolume. Then the hypervolume of the whole optimal front.

Both indicators use the objectives normalized by the front's ideal and nadir points, (0.2325, 0) and (1, 1.1350), so that the front spans [0, 1] in each. IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) averages, over 500 points of the optimal front from genoxide's `optimal_front`, the distance to the nearest point of the found front, counting only the objectives in which the found point is worse. Smaller is better, and 0 means the found front covers the optimal one. The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the area the front dominates up to the reference point (1.1, 1.1). Larger is better; the whole front's is computed from 20,000 of its points. Only feasible solutions count.

The page plays both runs back over the grey feasible region, which the trace samples from genomes on the unconstrained front's rays; hollow points are infeasible solutions of the populations, and the line is the optimal front, in its pieces.

[The project page](https://tachsin.gr/projects/genoxide/examples/mw10) plays this run back.

## Good results

The target: an IGD+ of at most 0.01 in normalized objectives.

With the paper's settings, the run with seed 1 covers only part of the pieces: IGD+ 0.0922, hypervolume 0.5486. Over seeds 1 to 20, no run reaches the target; the median IGD+ is 0.0812. Ma and Wang report a mean IGD of 0.130 for the same NSGA-II (supplement, table S-R-I).

With η = 2 and 5,000 generations, the run's front covers both pieces: IGD+ 0.0027, hypervolume 0.6833, 99.3% of the whole front's. Over seeds 1 to 20, every run reaches the target, with IGD+ from 0.0025 to 0.0029.
