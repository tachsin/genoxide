---
title: MW11
category: multi-objective
summary: Minimize two objectives over 15 variables subject to four constraints, whose Pareto front is two constraint boundaries and an isolated point that no search reaches in floating point, with NSGA-II.
reference: "Ma, Z. and Wang, Y. (2019). Evolutionary constrained multiobjective optimization: test suite construction and performance comparisons. IEEE Transactions on Evolutionary Computation 23(6): 972-986."
reference_url: https://doi.org/10.1109/TEVC.2019.2896967
optimum: "two pieces on constraint boundaries and the isolated point (1, 1), which needs x₁ = 1 and g₃ = 1 exactly; hypervolume 0.8099 (normalized objectives, reference point (1.1, 1.1))"
languages: [rust, python]
order: 194
family: MW
---

# MW11

## The problem

Ma and Wang (2019) built fourteen constrained test problems, MW1 to MW14. Each objective is a distance function g of some variables times a shape of the others, so that g = 1, its least value, puts a solution on the unconstrained front, and the constraints are curves near that front whose shapes a periodic term adjusts. The paper sorts the problems by what the constraints do to the front: type I leaves it whole, type II cuts parts of it out, type III replaces parts of it with pieces of a constraint's boundary, and type IV moves all of it onto boundaries. In every problem the feasible region is small: under 0.1‰ of the search space for most (table II).

MW11 (eq. 25) has two objectives and four constraints, over 15 variables in [0, √2], with the distance function with linked variables g₃ (eq. 14):

```text
f₁ = g₃ x₁
f₂ = g₃ √(2 − (f₁/g₃)²)
subject to (3 − f₁² − f₂)(3 − 2f₁² − f₂) ≥ 0
           (3 − 0.625f₁² − f₂)(3 − 7f₁² − f₂) ≤ 0
           (1.62 − 0.18f₁² − f₂)(1.125 − 0.125f₁² − f₂) ≥ 0
           (2.07 − 0.23f₁² − f₂)(0.63 − 0.07f₁² − f₂) ≤ 0
g₃ = 1 + Σᵢ₌₂¹⁵ 2 (xᵢ + (xᵢ₋₁ − 0.5)² − 1)²
```

Both objectives are minimized. At g₃ = 1 the solutions lie on the circle of radius √2; the constraints leave three feasible islands outside it, and touch it at one point, (1, 1), where the first and third constraints are both 0. MW11 is of type IV: its optimal front, derived here, is two pieces on the islands' boundaries, from (0.3707, 2.0383) to (0.8707, 1.4835) and from (1.4639, 0.8570) to (2.0662, 0.3311), and the point (1, 1), which dominates the third island. The paper names the point, and the authors' sampled front has it. The ideal point is (0.3707, 0.3311) and the nadir point (2.0662, 2.0383).

## What makes it hard

The point (1, 1) is feasible alone: beside it on the circle one of the two constraints fails, and outside the circle both do. A solution reaches it only with x₁ = 1 exactly and g₃ = 1, every distance variable on its chain (within about 10⁻⁹, where g₃'s squares vanish in double precision). Simulated binary crossover and polynomial mutation make continuous steps and don't land on a given value. g₃ is 1 where each xᵢ is 1 − (xᵢ₋₁ − 0.5)², a chain that starts from x₁: every solution on the front has its own distance variables, and a child of two parents far apart along the front inherits a chain that fits neither.

## Representation

A `Real` genome of 15 genes in [0, √2]: the vector x, the paper's size. The problem is genoxide's `Mw11`, whose fitness is the two objectives and the total constraint violation: the sum of how far x breaks each constraint, 0 when it is feasible. In Python, `gx.problems.Mw11()`, which `run` evaluates in Rust, so both versions print the same.

Solutions compare by constrained dominance, the rule of the NSGA-II paper and the constraint handling that Ma and Wang call CDP. A feasible solution beats an infeasible one. Of two infeasible ones, the smaller violation wins. Of two feasible ones, Pareto dominance decides.

## Algorithm

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation 6(2): 182-197), twice, with a population of 100 and simulated binary crossover with η = 20 at genoxide's default rate of 0.9:

- with the paper's settings: polynomial mutation with η = 20 at a rate of 1/15 per gene, for 600 generations, 60,000 evaluations, as in Ma and Wang's comparison (section IV-C);
- with polynomial mutation with η = 2, for 5,000 generations, 500,000 evaluations.

The distribution index η sets how far a mutation moves a gene. With η = 20, the median step is about 3% of the range, and fewer than 1 in 1,000 steps cover more than 30% of it; with η = 2, the median step is 8% to 17% of the range, depending on where the gene is, and about 1 in 5 steps cover more than 30% (measured with genoxide's polynomial mutation). The larger steps help the population reach both islands' boundaries.

## Output

A line per run: the size of its final front, how many of its solutions are feasible, and the front's IGD+ and hypervolume. Then the hypervolume of the whole optimal front, and a last line for the second run: its IGD+ to the optimal front without the isolated point (1, 1), and the distance from that point to the front's nearest solution.

Both indicators use the objectives normalized by the front's ideal and nadir points, (0.3707, 0.3311) and (2.0662, 2.0383), so that the front spans [0, 1] in each. IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) averages, over 500 points of the optimal front from genoxide's `optimal_front`, the distance to the nearest point of the found front, counting only the objectives in which the found point is worse. Smaller is better, and 0 means the found front covers the optimal one. The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the area the front dominates up to the reference point (1.1, 1.1). Larger is better; the whole front's is computed from 20,000 of its points. Only feasible solutions count.

The page plays both runs back over the grey feasible region, which the trace samples from genomes on the unconstrained front's rays; hollow points are infeasible solutions of the populations, and the line is the optimal front, in its pieces.

[The project page](https://tachsin.gr/projects/genoxide/examples/mw11) plays this run back.

## Good results

The target, an IGD+ of at most 0.01 in normalized objectives, can't be reached without the point (1, 1), and nothing reaches it: in the 500 points of the optimal front, spread by length, the point stands for the stretch of front around it, and a front without it has an IGD+ above 0.046. The example also prints the IGD+ to the optimal front without the point, which shows how well the rest is found, and how far the found front is from the point.

With the paper's settings, the run with seed 1 has an IGD+ of 0.1873 and a hypervolume of 0.5631. Over seeds 1 to 20, the IGD+ is 0.1861 to 0.2856. Ma and Wang report a mean IGD of 0.613 for the same NSGA-II (supplement, table S-R-I).

With η = 2 and 5,000 generations, the run's front covers both pieces and the third island's boundary, which (1, 1) dominates: IGD+ 0.0472, hypervolume 0.7433, 91.8% of the whole front's. Without the point, its IGD+ is 0.0015, and its nearest solution to (1, 1) is 0.4390 away. Over seeds 1 to 20, every run ends the same way: IGD+ 0.0467 to 0.0473, 0.0015 to 0.0016 without the point, and no solution within 0.43 of it. The pieces are found; the isolated point, which needs an exact x₁ = 1, isn't, by either run.
