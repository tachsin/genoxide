---
title: MW2
category: multi-objective
summary: Minimize two objectives over 15 variables subject to one constraint, whose Pareto front is a line behind many local fronts, with NSGA-II.
reference: "Ma, Z. and Wang, Y. (2019). Evolutionary constrained multiobjective optimization: test suite construction and performance comparisons. IEEE Transactions on Evolutionary Computation 23(6): 972-986."
reference_url: https://doi.org/10.1109/TEVC.2019.2896967
optimum: "the line f₂ = 1 − f₁ for f₁ in [0, 1]; hypervolume 0.7100 (normalized objectives, reference point (1.1, 1.1))"
languages: [rust, python]
order: 205
family: MW
---

# MW2

## The problem

Ma and Wang (2019) built fourteen constrained test problems, MW1 to MW14. Each objective is a distance function g of some variables times a shape of the others, so that g = 1, its least value, puts a solution on the unconstrained front, and the constraints are curves near that front whose shapes a periodic term adjusts. The paper sorts the problems by what the constraints do to the front: type I leaves it whole, type II cuts parts of it out, type III replaces parts of it with pieces of a constraint's boundary, and type IV moves all of it onto boundaries. In every problem the feasible region is small: under 0.1‰ of the search space for most (table II).

MW2 (eq. 16) has two objectives and one constraint, over 15 variables in [0, 1], with the multimodal distance function g₂ (eq. 13):

```text
f₁ = x₁
f₂ = g₂ (1 − f₁/g₂)
subject to 1 − f₁ − f₂ + 0.5 sin(3πl)⁸ ≥ 0,  l = √2 f₂ − √2 f₁
g₂ = 1 + Σᵢ₌₂¹⁵ (1.5 + 0.1 zᵢ²/15 − 1.5 cos 2πzᵢ),  zᵢ = 1 − exp(−10 (xᵢ − (i − 1)/15)²)
```

Both objectives are minimized. At g₂ = 1 the solutions lie on the line f₂ = 1 − f₁, where f₁ + f₂ = 1 and the constraint holds whatever the sine term: MW2 is of type I, and its optimal front is the whole line, from (0, 1) to (1, 0). A larger g₂ moves a solution straight up, where the constraint leaves only narrow teeth. The ideal point is (0, 0) and the nadir point (1, 1).

## What makes it hard

The constraint doesn't change the front, but it keeps the population in a thin region above it. The difficulty is g₂: g₂ is 1 where each xᵢ is (i − 1)/15. Each term has a second, local minimum where zᵢ reaches 1, far from that value, only 0.0067 higher; between the two lies a barrier of 3 at zᵢ = 0.5. A variable that settles on the wrong side stays there unless a single step carries it across. Solutions whose distance variables stopped in the local minima form local fronts just above the line, feasible, and only a jump of more than about a third of the range in one variable leads down.

## Representation

A `Real` genome of 15 genes in [0, 1]: the vector x, the paper's size. The problem is genoxide's `Mw2`, whose fitness is the two objectives and the total constraint violation: the sum of how far x breaks each constraint, 0 when it is feasible. In Python, `gx.problems.Mw2()`, which `run` evaluates in Rust, so both versions print the same.

Solutions compare by constrained dominance, the rule of the NSGA-II paper and the constraint handling that Ma and Wang call CDP. A feasible solution beats an infeasible one. Of two infeasible ones, the smaller violation wins. Of two feasible ones, Pareto dominance decides.

## Algorithm

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation 6(2): 182-197), twice, with a population of 100 and simulated binary crossover with η = 20 at genoxide's default rate of 0.9:

- with the paper's settings: polynomial mutation with η = 20 at a rate of 1/15 per gene, for 600 generations, 60,000 evaluations, as in Ma and Wang's comparison (section IV-C);
- with polynomial mutation with η = 2, for 5,000 generations, 500,000 evaluations.

The distribution index η sets how far a mutation moves a gene. With η = 20, the median step is about 3% of the range, and fewer than 1 in 1,000 steps cover more than 30% of it; with η = 2, the median step is 8% to 17% of the range, depending on where the gene is, and about 1 in 5 steps cover more than 30% (measured with genoxide's polynomial mutation). The larger steps carry distance variables across g₂'s barriers, which the small ones rarely cross.

## Output

A line per run: the size of its final front, how many of its solutions are feasible, and the front's IGD+ and hypervolume. Then the hypervolume of the whole optimal front.

Both indicators use the objectives normalized by the front's ideal and nadir points, (0, 0) and (1, 1), so that the front spans [0, 1] in each. IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) averages, over 500 points of the optimal front from genoxide's `optimal_front`, the distance to the nearest point of the found front, counting only the objectives in which the found point is worse. Smaller is better, and 0 means the found front covers the optimal one. The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the area the front dominates up to the reference point (1.1, 1.1). Larger is better; the whole front's is computed from 20,000 of its points. Only feasible solutions count.

The page plays both runs back over the grey feasible region, which the trace samples from genomes on the unconstrained front's rays; hollow points are infeasible solutions of the populations, and the line is the optimal front, in its pieces.

[The project page](https://tachsin.gr/projects/genoxide/examples/mw2) plays this run back.

## Good results

The target: an IGD+ of at most 0.01 in normalized objectives. 100 points of the line, evenly spread, have an IGD+ of 0.0025 and a hypervolume of 0.7049, 99.3% of the whole front's.

With the paper's settings, the run with seed 1 ends on a local front: IGD+ 0.0253, hypervolume 0.6595. Over seeds 1 to 20, 2 runs reach the target; the median IGD+ is 0.0241. Ma and Wang report a mean IGD of 0.0240 for the same NSGA-II (supplement, table S-R-I).

With η = 2 and 5,000 generations, the run's front has an IGD+ of 0.0035 and a hypervolume of 0.7030, 99.0% of the whole front's: all 100 solutions on the line. Over seeds 1 to 20, every run reaches the target, with IGD+ from 0.0032 to 0.0037.
