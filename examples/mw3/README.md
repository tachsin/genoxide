---
title: MW3
category: multi-objective
summary: Minimize two objectives over 15 variables subject to two constraints, whose Pareto front runs along a line and along a constraint's boundary, with NSGA-II.
reference: "Ma, Z. and Wang, Y. (2019). Evolutionary constrained multiobjective optimization: test suite construction and performance comparisons. IEEE Transactions on Evolutionary Computation 23(6): 972-986."
reference_url: https://doi.org/10.1109/TEVC.2019.2896967
optimum: "the line f₂ = 1 − f₁ and two stretches of the second constraint's boundary, from (0, 1) to (1, 0); hypervolume 0.6650 (normalized objectives, reference point (1.1, 1.1))"
languages: [rust, python]
order: 186
family: MW
---

# MW3

## The problem

Ma and Wang (2019) built fourteen constrained test problems, MW1 to MW14. Each objective is a distance function g of some variables times a shape of the others, so that g = 1, its least value, puts a solution on the unconstrained front, and the constraints are curves near that front whose shapes a periodic term adjusts. The paper sorts the problems by what the constraints do to the front: type I leaves it whole, type II cuts parts of it out, type III replaces parts of it with pieces of a constraint's boundary, and type IV moves all of it onto boundaries. In every problem the feasible region is small: under 0.1‰ of the search space for most (table II).

MW3 (eq. 17) has two objectives and two constraints, over 15 variables in [0, 1], with the distance function with linked variables g₃ (eq. 14):

```text
f₁ = x₁
f₂ = g₃ (1 − f₁/g₃)
subject to 1.05 − f₁ − f₂ + 0.45 sin(0.75πl)⁶ ≥ 0
           0.85 − f₁ − f₂ + 0.3 sin(0.75πl)² ≤ 0,  l = √2 f₂ − √2 f₁
g₃ = 1 + Σᵢ₌₂¹⁵ 2 (xᵢ + (xᵢ₋₁ − 0.5)² − 1)²
```

Both objectives are minimized. The unconstrained front is the line f₂ = 1 − f₁. The two constraints leave a band around it: f₁ + f₂ at most 1.05 plus a wave, and at least 0.85 plus another. The second wave rises above the line where sin(0.75πl)² > 1/2, around f₁ = 0.26 and 0.74, and there the band starts above the line. MW3 is of type III: its optimal front, derived here, is the line for f₁ in [0, 0.1464], [0.3821, 0.6179] and [0.8536, 1], and the second constraint's boundary in between, up to 0.15 above the line; it drops back to the line by 0.019 at f₁ = 0.3821. The ideal point is (0, 0) and the nadir point (1, 1).

## What makes it hard

The feasible band is narrow, and parts of the front are on its lower edge, where the best solutions sit next to infeasible ones. And g₃ links the variables: every solution on the front has its own distance variables, and a child of two parents far apart along the front inherits a chain that fits neither.

## Representation

A `Real` genome of 15 genes in [0, 1]: the vector x, the paper's size. The problem is genoxide's `Mw3`, whose fitness is the two objectives and the total constraint violation: the sum of how far x breaks each constraint, 0 when it is feasible. In Python, `gx.problems.Mw3()`, which `run` evaluates in Rust, so both versions print the same.

Solutions compare by constrained dominance, the rule of the NSGA-II paper and the constraint handling that Ma and Wang call CDP. A feasible solution beats an infeasible one. Of two infeasible ones, the smaller violation wins. Of two feasible ones, Pareto dominance decides.

## Algorithm

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation 6(2): 182-197), twice, with a population of 100 and simulated binary crossover with η = 20 at genoxide's default rate of 0.9:

- with the paper's settings: polynomial mutation with η = 20 at a rate of 1/15 per gene, for 600 generations, 60,000 evaluations, as in Ma and Wang's comparison (section IV-C);
- with polynomial mutation with η = 2, for 5,000 generations, 500,000 evaluations.

The distribution index η sets how far a mutation moves a gene. With η = 20, the median step is about 3% of the range, and fewer than 1 in 1,000 steps cover more than 30% of it; with η = 2, the median step is 8% to 17% of the range, depending on where the gene is, and about 1 in 5 steps cover more than 30% (measured with genoxide's polynomial mutation). On MW3 both settings usually reach the front; the longer run does so on every seed tried.

## Output

A line per run: the size of its final front, how many of its solutions are feasible, and the front's IGD+ and hypervolume. Then the hypervolume of the whole optimal front.

Both indicators use the objectives normalized by the front's ideal and nadir points, (0, 0) and (1, 1), so that the front spans [0, 1] in each. IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) averages, over 500 points of the optimal front from genoxide's `optimal_front`, the distance to the nearest point of the found front, counting only the objectives in which the found point is worse. Smaller is better, and 0 means the found front covers the optimal one. The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the area the front dominates up to the reference point (1.1, 1.1). Larger is better; the whole front's is computed from 20,000 of its points. Only feasible solutions count.

The page plays both runs back over the grey feasible region, which the trace samples from genomes on the unconstrained front's rays; hollow points are infeasible solutions of the populations, and the line is the optimal front, in its pieces.

[The project page](https://tachsin.gr/projects/genoxide/examples/mw3) plays this run back.

## Good results

The target: an IGD+ of at most 0.01 in normalized objectives. 100 points of the front, spread along it, have an IGD+ of 0.0019 and 99.4% of the whole front's hypervolume.

With the paper's settings, the run with seed 1 misses part of the front: IGD+ 0.0620, hypervolume 0.5734. Over seeds 1 to 20, 18 runs reach the target; one ends with an IGD+ of 0.7476, its front far from the optimal one. Ma and Wang report a mean IGD of 0.0376 for the same NSGA-II (supplement, table S-R-I).

With η = 2 and 5,000 generations, the run's front has an IGD+ of 0.0031 and a hypervolume of 0.6581, 99.0% of the whole front's, with solutions on the line and on both stretches of the boundary. Over seeds 1 to 20, every run reaches the target, with IGD+ from 0.0027 to 0.0038.
