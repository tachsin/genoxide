---
title: MW7
category: multi-objective
summary: Minimize two objectives over 15 variables subject to two constraints, whose Pareto front runs along a circle and a constraint's wavy boundary, with NSGA-II.
reference: "Ma, Z. and Wang, Y. (2019). Evolutionary constrained multiobjective optimization: test suite construction and performance comparisons. IEEE Transactions on Evolutionary Computation 23(6): 972-986."
reference_url: https://doi.org/10.1109/TEVC.2019.2896967
optimum: "three pieces, of the unit circle and of the second constraint's boundary, from (0, 1.15) to (1.15, 0); hypervolume 0.5025 (normalized objectives, reference point (1.1, 1.1))"
languages: [rust, python]
order: 226
family: MW
---

# MW7

## The problem

Ma and Wang (2019) built fourteen constrained test problems, MW1 to MW14. Each objective is a distance function g of some variables times a shape of the others, so that g = 1, its least value, puts a solution on the unconstrained front, and the constraints are curves near that front whose shapes a periodic term adjusts. The paper sorts the problems by what the constraints do to the front: type I leaves it whole, type II cuts parts of it out, type III replaces parts of it with pieces of a constraint's boundary, and type IV moves all of it onto boundaries. In every problem the feasible region is small: under 0.1‰ of the search space for most (table II).

MW7 (eq. 21) has two objectives and two constraints, over 15 variables in [0, 1], with the distance function with linked variables g₃ (eq. 14):

```text
f₁ = g₃ x₁
f₂ = g₃ √(1 − (f₁/g₃)²)
subject to (1.2 + 0.4 sin(4l)¹⁶)² − f₁² − f₂² ≥ 0
           (1.15 − 0.2 sin(4l)⁸)² − f₁² − f₂² ≤ 0,  l = arctan(f₂/f₁)
g₃ = 1 + Σᵢ₌₂¹⁵ 2 (xᵢ + (xᵢ₋₁ − 0.5)² − 1)²
```

Both objectives are minimized. At g₃ = 1 the solutions lie on the quarter of the unit circle. The second constraint keeps them outside the radius 1.15 − 0.2 sin(4l)⁸, which is 1.15 near the axes and the diagonal and dips below 1 only around l = π/8 and 3π/8; the first keeps them inside a wavy outer boundary. MW7 is of type III: its optimal front, derived here, runs along the second constraint's boundary from (0, 1.15) to f₁ = 0.3203, along the circle to f₁ = 0.4434, along the boundary again from (0.7202, 0.8963) to (0.8963, 0.7202), along the circle to f₁ = 0.9473 and along the boundary to (1.15, 0): three pieces, symmetric about the diagonal. The ideal point is (0, 0) and the nadir point (1.15, 1.15).

## What makes it hard

The parts of the front on the boundary lie where the feasible band is narrow, and Ma and Wang note that the optimal solutions there are hard to find. g₃ is 1 where each xᵢ is 1 − (xᵢ₋₁ − 0.5)², a chain that starts from x₁: every solution on the front has its own distance variables, and a child of two parents far apart along the front inherits a chain that fits neither.

## Representation

A `Real` genome of 15 genes in [0, 1]: the vector x, the paper's size. The problem is genoxide's `Mw7`, whose fitness is the two objectives and the total constraint violation: the sum of how far x breaks each constraint, 0 when it is feasible. In Python, `gx.problems.Mw7()`, which `run` evaluates in Rust, so both versions print the same.

Solutions compare by constrained dominance, the rule of the NSGA-II paper and the constraint handling that Ma and Wang call CDP. A feasible solution beats an infeasible one. Of two infeasible ones, the smaller violation wins. Of two feasible ones, Pareto dominance decides.

## Algorithm

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation 6(2): 182-197), twice, with a population of 100 and simulated binary crossover with η = 20 at genoxide's default rate of 0.9:

- with the paper's settings: polynomial mutation with η = 20 at a rate of 1/15 per gene, for 600 generations, 60,000 evaluations, as in Ma and Wang's comparison (section IV-C);
- with polynomial mutation with η = 2, for 5,000 generations, 500,000 evaluations.

The distribution index η sets how far a mutation moves a gene. With η = 20, the median step is about 3% of the range, and fewer than 1 in 1,000 steps cover more than 30% of it; with η = 2, the median step is 8% to 17% of the range, depending on where the gene is, and about 1 in 5 steps cover more than 30% (measured with genoxide's polynomial mutation). On MW7 both settings usually reach the front.

## Output

A line per run: the size of its final front, how many of its solutions are feasible, and the front's IGD+ and hypervolume. Then the hypervolume of the whole optimal front.

Both indicators use the objectives normalized by the front's ideal and nadir points, (0, 0) and (1.15, 1.15), so that the front spans [0, 1] in each. IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) averages, over 500 points of the optimal front from genoxide's `optimal_front`, the distance to the nearest point of the found front, counting only the objectives in which the found point is worse. Smaller is better, and 0 means the found front covers the optimal one. The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the area the front dominates up to the reference point (1.1, 1.1). Larger is better; the whole front's is computed from 20,000 of its points. Only feasible solutions count.

The page plays both runs back over the grey feasible region, which the trace samples from genomes on the unconstrained front's rays; hollow points are infeasible solutions of the populations, and the line is the optimal front, in its pieces.

[The project page](https://tachsin.gr/projects/genoxide/examples/mw7) plays this run back.

## Good results

The target: an IGD+ of at most 0.01 in normalized objectives.

With the paper's settings, the run with seed 1 already reaches it: IGD+ 0.0026, hypervolume 0.4973. Over seeds 1 to 20, 18 runs do; the other two end with IGD+ up to 0.2305. Ma and Wang report a mean IGD of 0.0265 for the same NSGA-II (supplement, table S-R-I).

With η = 2 and 5,000 generations, the run's front has an IGD+ of 0.0025 and a hypervolume of 0.4976, 99.0% of the whole front's. Over seeds 1 to 20, every run reaches the target, with IGD+ from 0.0020 to 0.0026.
