---
title: MW9
category: multi-objective
summary: Minimize two objectives over 15 variables subject to one constraint, whose Pareto front lies entirely on the boundary of the feasible region, with NSGA-II.
reference: "Ma, Z. and Wang, Y. (2019). Evolutionary constrained multiobjective optimization: test suite construction and performance comparisons. IEEE Transactions on Evolutionary Computation 23(6): 972-986."
reference_url: https://doi.org/10.1109/TEVC.2019.2896967
optimum: "f₂ = 1 − 0.64 f₁² for f₁ in [0, 0.5868], then f₂ = 1.15² − (f₁ + 0.15)² to (1, 0); hypervolume 0.4934 (normalized objectives, reference point (1.1, 1.1))"
languages: [rust, python]
order: 192
family: MW
---

# MW9

## The problem

Ma and Wang (2019) built fourteen constrained test problems, MW1 to MW14. Each objective is a distance function g of some variables times a shape of the others, so that g = 1, its least value, puts a solution on the unconstrained front, and the constraints are curves near that front whose shapes a periodic term adjusts. The paper sorts the problems by what the constraints do to the front: type I leaves it whole, type II cuts parts of it out, type III replaces parts of it with pieces of a constraint's boundary, and type IV moves all of it onto boundaries. In every problem the feasible region is small: under 0.1‰ of the search space for most (table II).

MW9 (eq. 23) has two objectives and one constraint, over 15 variables in [0, 1], with the biased distance function g₁ (eq. 12):

```text
f₁ = g₁ x₁
f₂ = g₁ (1 − (f₁/g₁)^0.6)
subject to min(T₁, T₂ T₃) ≤ 0
T₁ = (1 − 0.64 f₁² − f₂)(1 − 0.36 f₁² − f₂)
T₂ = 1.35² − (f₁ + 0.35)² − f₂,  T₃ = 1.15² − (f₁ + 0.15)² − f₂
g₁ = 1 + Σᵢ₌₂¹⁵ (1 − exp(−10 (zᵢ − 0.5 − (i − 1)/30)²)),  zᵢ = xᵢ¹³
```

Both objectives are minimized. At g₁ = 1 the solutions lie on the convex curve f₂ = 1 − f₁^0.6; a larger g₁ moves a solution outward from the origin. The constraint holds between the parabolas 1 − 0.64 f₁² and 1 − 0.36 f₁², or between the curves 1.15² − (f₁ + 0.15)² and 1.35² − (f₁ + 0.35)²; the unconstrained front lies below both bands, infeasible but for its ends. MW9 is of type IV: its optimal front is on the bands' lower edges, derived here in closed form: f₂ = 1 − 0.64 f₁² for f₁ in [0, 0.5868] and f₂ = 1.15² − (f₁ + 0.15)² for f₁ in [0.5868, 1], the two meeting at f₁ = (√0.522 − 0.3)/0.72. It is concave, from (0, 1) to (1, 0); the ideal point is (0, 0) and the nadir point (1, 1).

## What makes it hard

Every optimal solution is on a constraint's boundary, with infeasible solutions just below, and it has g₁ above 1, up to 1.67 near f₁ = 0: the distance variables must stop short of their best values, at a g₁ that differs along the front. g₁ is 1 where each zᵢ = xᵢ¹³ is 0.5 + (i − 1)/30, between 0.53 and 0.97: every xᵢ must be between 0.952 and 0.997, where the thirteenth power is steep. Ma and Wang call the power a bias: most values of xᵢ put zᵢ near 0.

## Representation

A `Real` genome of 15 genes in [0, 1]: the vector x, the paper's size. The problem is genoxide's `Mw9`, whose fitness is the two objectives and the total constraint violation: the sum of how far x breaks each constraint, 0 when it is feasible. In Python, `gx.problems.Mw9()`, which `run` evaluates in Rust, so both versions print the same.

Solutions compare by constrained dominance, the rule of the NSGA-II paper and the constraint handling that Ma and Wang call CDP. A feasible solution beats an infeasible one. Of two infeasible ones, the smaller violation wins. Of two feasible ones, Pareto dominance decides.

## Algorithm

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation 6(2): 182-197), twice, with a population of 100 and simulated binary crossover with η = 20 at genoxide's default rate of 0.9:

- with the paper's settings: polynomial mutation with η = 20 at a rate of 1/15 per gene, for 600 generations, 60,000 evaluations, as in Ma and Wang's comparison (section IV-C);
- with polynomial mutation with η = 2, for 5,000 generations, 500,000 evaluations.

The distribution index η sets how far a mutation moves a gene. With η = 20, the median step is about 3% of the range, and fewer than 1 in 1,000 steps cover more than 30% of it; with η = 2, the median step is 8% to 17% of the range, depending on where the gene is, and about 1 in 5 steps cover more than 30% (measured with genoxide's polynomial mutation). The larger steps let the population explore the thin band above the unconstrained front.

## Output

A line per run: the size of its final front, how many of its solutions are feasible, and the front's IGD+ and hypervolume. Then the hypervolume of the whole optimal front.

Both indicators use the objectives normalized by the front's ideal and nadir points, (0, 0) and (1, 1), so that the front spans [0, 1] in each. IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) averages, over 500 points of the optimal front from genoxide's `optimal_front`, the distance to the nearest point of the found front, counting only the objectives in which the found point is worse. Smaller is better, and 0 means the found front covers the optimal one. The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the area the front dominates up to the reference point (1.1, 1.1). Larger is better; the whole front's is computed from 20,000 of its points. Only feasible solutions count.

The page plays both runs back over the grey feasible region, which the trace samples from genomes on the unconstrained front's rays; hollow points are infeasible solutions of the populations, and the line is the optimal front, in its pieces.

[The project page](https://tachsin.gr/projects/genoxide/examples/mw9) plays this run back.

## Good results

The target: an IGD+ of at most 0.01 in normalized objectives. 100 points of the front, spread along it, have an IGD+ of 0.0021 and 99.1% of the whole front's hypervolume.

With the paper's settings, the run with seed 1 is short of the boundary: IGD+ 0.0316, hypervolume 0.4334. Over seeds 1 to 20, 4 runs reach the target; the median IGD+ is 0.0160. Ma and Wang report a mean IGD of 0.0211 for the same NSGA-II (supplement, table S-R-I).

With η = 2 and 5,000 generations, the run's front has an IGD+ of 0.0036 and a hypervolume of 0.4807, 97.4% of the whole front's: its solutions are on the boundary, but crowding distance spreads them less evenly than those 100 points. Over seeds 1 to 20, every run reaches the target, with IGD+ from 0.0031 to 0.0068 and hypervolumes of 95.6% to 98.1% of the whole front's.
