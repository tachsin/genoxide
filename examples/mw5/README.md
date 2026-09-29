---
title: MW5
category: multi-objective
summary: Minimize two objectives over 15 variables subject to three constraints, whose Pareto front is sixteen points at the ends of narrow feasible tunnels, with NSGA-II.
reference: "Ma, Z. and Wang, Y. (2019). Evolutionary constrained multiobjective optimization: test suite construction and performance comparisons. IEEE Transactions on Evolutionary Computation 23(6): 972-986."
reference_url: https://doi.org/10.1109/TEVC.2019.2896967
optimum: "sixteen points of the unit circle and two short curves near the axes; hypervolume 0.3930 (normalized objectives, reference point (1.1, 1.1))"
languages: [rust, python]
order: 188
family: MW
---

# MW5

## The problem

Ma and Wang (2019) built fourteen constrained test problems, MW1 to MW14. Each objective is a distance function g of some variables times a shape of the others, so that g = 1, its least value, puts a solution on the unconstrained front, and the constraints are curves near that front whose shapes a periodic term adjusts. The paper sorts the problems by what the constraints do to the front: type I leaves it whole, type II cuts parts of it out, type III replaces parts of it with pieces of a constraint's boundary, and type IV moves all of it onto boundaries. In every problem the feasible region is small: under 0.1‰ of the search space for most (table II).

MW5 (eq. 19) has two objectives and three constraints, over 15 variables in [0, 1], with the biased distance function g₁ (eq. 12):

```text
f₁ = g₁ x₁
f₂ = g₁ √(1 − (f₁/g₁)²)
subject to (1.7 − 0.2 sin 2l₁)² − f₁² − f₂² ≥ 0
           (1 + 0.5 sin 6l₂³)² − f₁² − f₂² ≤ 0
           (1 − 0.45 sin 6l₂³)² − f₁² − f₂² ≤ 0
l₁ = arctan(f₂/f₁),  l₂ = 0.5π − 2 |l₁ − 0.25π|
g₁ = 1 + Σᵢ₌₂¹⁵ (1 − exp(−10 (zᵢ − 0.5 − (i − 1)/30)²)),  zᵢ = xᵢ¹³
```

Both objectives are minimized. At g₁ = 1 the solutions lie on the quarter of the unit circle; a larger g₁ moves a solution outward along its direction. The second and third constraints both hold on the circle only where sin 6l₂³ = 0: at l₂ = (kπ/6)^(1/3) for k from 0 to 7, sixteen points, symmetric about the diagonal (derived here). Elsewhere the first feasible radius is 1 + 0.5 |sin 6l₂³| or 1 + 0.45 |sin 6l₂³|, and the feasible region is a fan of narrow tunnels that end at the sixteen points. MW5 is of type II, with a discrete front. Near the axes, where l₂ is below about 0.028, the first feasible points are so close to the circle that they aren't dominated either: the front also has two short curves, from (1, 0) to about (0.99997, 0.0139) and the mirror image. The paper's figure and the authors' sampled front show points there; genoxide's front has the curves. Its ideal point is (0, 0) and its nadir point (1, 1).

## What makes it hard

Each optimal point sits at the tip of a tunnel whose width goes to 0: a solution in a tunnel improves only by moving toward the tip, and the neighbouring tunnels are separated by infeasible wedges. g₁ is 1 where each zᵢ = xᵢ¹³ is 0.5 + (i − 1)/30, between 0.53 and 0.97: every xᵢ must be between 0.952 and 0.997, where the thirteenth power is steep. Ma and Wang call the power a bias: most values of xᵢ put zᵢ near 0.

## Representation

A `Real` genome of 15 genes in [0, 1]: the vector x, the paper's size. The problem is genoxide's `Mw5`, whose fitness is the two objectives and the total constraint violation: the sum of how far x breaks each constraint, 0 when it is feasible. In Python, `gx.problems.Mw5()`, which `run` evaluates in Rust, so both versions print the same.

Solutions compare by constrained dominance, the rule of the NSGA-II paper and the constraint handling that Ma and Wang call CDP. A feasible solution beats an infeasible one. Of two infeasible ones, the smaller violation wins. Of two feasible ones, Pareto dominance decides.

## Algorithm

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation 6(2): 182-197), twice, with a population of 100 and simulated binary crossover with η = 20 at genoxide's default rate of 0.9:

- with the paper's settings: polynomial mutation with η = 20 at a rate of 1/15 per gene, for 600 generations, 60,000 evaluations, as in Ma and Wang's comparison (section IV-C);
- with polynomial mutation with η = 2, for 5,000 generations, 500,000 evaluations.

The distribution index η sets how far a mutation moves a gene. With η = 20, the median step is about 3% of the range, and fewer than 1 in 1,000 steps cover more than 30% of it; with η = 2, the median step is 8% to 17% of the range, depending on where the gene is, and about 1 in 5 steps cover more than 30% (measured with genoxide's polynomial mutation). With η = 20 the population often ends inside a few tunnels, short of their tips, or misses tunnels.

## Output

A line per run: the size of its final front, how many of its solutions are feasible, and the front's IGD+ and hypervolume. Then the hypervolume of the whole optimal front.

Both indicators use the objectives normalized by the front's ideal and nadir points, (0, 0) and (1, 1), so that the front spans [0, 1] in each. IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) averages, over 500 points of the optimal front from genoxide's `optimal_front`, the distance to the nearest point of the found front, counting only the objectives in which the found point is worse. Smaller is better, and 0 means the found front covers the optimal one. The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the area the front dominates up to the reference point (1.1, 1.1). Larger is better; the whole front's is computed from 20,000 of its points. Only feasible solutions count.

The page plays both runs back over the grey feasible region, which the trace samples from genomes on the unconstrained front's rays; hollow points are infeasible solutions of the populations, and the line is the optimal front, in its pieces.

[The project page](https://tachsin.gr/projects/genoxide/examples/mw5) plays this run back.

## Good results

The target: an IGD+ of at most 0.01 in normalized objectives. genoxide's 500 points of the front share them by length along the front, a gap counting to the points beside it, so each of the fourteen inner points stands for its stretch of the circle, and a front that misses one pays for it.

With the paper's settings, the run with seed 1 has 75 feasible solutions, an IGD+ of 0.0161 and a hypervolume of 0.3800. Over seeds 1 to 20, 5 runs reach the target; the median IGD+ is 0.1828. Ma and Wang report a mean IGD of 0.175 for the same NSGA-II (supplement, table S-R-I).

With η = 2 and 5,000 generations, the run's front reaches every tip: IGD+ 0.0002, hypervolume 0.3928, 99.9% of the whole front's. Over seeds 1 to 20, every run reaches the target, with IGD+ from 0.0001 to 0.0025.
