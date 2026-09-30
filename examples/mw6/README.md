---
title: MW6
category: multi-objective
summary: Minimize two objectives over 15 variables subject to one constraint, whose Pareto front is twelve pieces of a circle, with NSGA-II.
reference: "Ma, Z. and Wang, Y. (2019). Evolutionary constrained multiobjective optimization: test suite construction and performance comparisons. IEEE Transactions on Evolutionary Computation 23(6): 972-986."
reference_url: https://doi.org/10.1109/TEVC.2019.2896967
optimum: "twelve pieces of the circle f₁² + f₂² = 1.21, from f₁ = 0.0163 to (1.1, 0); hypervolume 0.4043 (normalized objectives, reference point (1.1, 1.1))"
languages: [rust, python]
order: 209
family: MW
---

# MW6

## The problem

Ma and Wang (2019) built fourteen constrained test problems, MW1 to MW14. Each objective is a distance function g of some variables times a shape of the others, so that g = 1, its least value, puts a solution on the unconstrained front, and the constraints are curves near that front whose shapes a periodic term adjusts. The paper sorts the problems by what the constraints do to the front: type I leaves it whole, type II cuts parts of it out, type III replaces parts of it with pieces of a constraint's boundary, and type IV moves all of it onto boundaries. In every problem the feasible region is small: under 0.1‰ of the search space for most (table II).

MW6 (eq. 20) has two objectives and one constraint, over 15 variables in [0, 1.1], with the multimodal distance function g₂ (eq. 13):

```text
f₁ = g₂ x₁
f₂ = g₂ √(1.1² − (f₁/g₂)²)
subject to 1 − (f₁/(1 + 0.15l))² − (f₂/(1 + 0.75l))² ≥ 0,  l = cos(6 arctan(f₂/f₁)⁴)¹⁰
g₂ = 1 + Σᵢ₌₂¹⁵ (1.5 + 0.1 zᵢ²/15 − 1.5 cos 2πzᵢ),  zᵢ = 1 − exp(−10 (xᵢ − (i − 1)/15)²)
```

Both objectives are minimized. At g₂ = 1 the solutions lie on the circle of radius 1.1; a larger g₂ moves a solution outward. The constraint keeps solutions inside an ellipse whose axes 1 + 0.15l and 1 + 0.75l grow with l, which is near 1 in narrow spokes of direction and near 0 between them. The circle is feasible only where the ellipse reaches it, and along a direction the constraint only gets worse outward: MW6 is of type II, and its optimal front, derived here, is twelve pieces of the circle, eleven short ones near f₂ = 1.1 that narrow toward f₁ = 0, from f₁ = 0.0163, and a long one from (0.9549, 0.5461) to (1.1, 0). l is the tenth power of the cosine of 6 arctan(f₂/f₁)⁴, as the paper's local adjustment A sin(B l^C)^D (eq. 9) and the authors' code have it. The ideal point is (0.0163, 0) and the nadir point (1.1, 1.0999).

## What makes it hard

Nothing outside the circle is feasible in the gaps, so the population has to reach g₂ = 1 exactly in eleven narrow windows. g₂ is 1 where each xᵢ is (i − 1)/15. Each term has a second, local minimum where zᵢ reaches 1, far from that value, only 0.0067 higher; between the two lies a barrier of 3 at zᵢ = 0.5. A variable that settles on the wrong side stays there unless a single step carries it across.

## Representation

A `Real` genome of 15 genes in [0, 1.1]: the vector x, the paper's size. The problem is genoxide's `Mw6`, whose fitness is the two objectives and the total constraint violation: the sum of how far x breaks each constraint, 0 when it is feasible. In Python, `gx.problems.Mw6()`, which `run` evaluates in Rust, so both versions print the same.

Solutions compare by constrained dominance, the rule of the NSGA-II paper and the constraint handling that Ma and Wang call CDP. A feasible solution beats an infeasible one. Of two infeasible ones, the smaller violation wins. Of two feasible ones, Pareto dominance decides.

## Algorithm

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation 6(2): 182-197), twice, with a population of 100 and simulated binary crossover with η = 20 at genoxide's default rate of 0.9:

- with the paper's settings: polynomial mutation with η = 20 at a rate of 1/15 per gene, for 600 generations, 60,000 evaluations, as in Ma and Wang's comparison (section IV-C);
- with polynomial mutation with η = 2, for 5,000 generations, 500,000 evaluations.

The distribution index η sets how far a mutation moves a gene. With η = 20, the median step is about 3% of the range, and fewer than 1 in 1,000 steps cover more than 30% of it; with η = 2, the median step is 8% to 17% of the range, depending on where the gene is, and about 1 in 5 steps cover more than 30% (measured with genoxide's polynomial mutation). The larger steps carry distance variables across g₂'s barriers; on MW6 a solution even slightly above g₂ = 1 falls outside the narrow pieces.

## Output

A line per run: the size of its final front, how many of its solutions are feasible, and the front's IGD+ and hypervolume. Then the hypervolume of the whole optimal front.

Both indicators use the objectives normalized by the front's ideal and nadir points, (0.0163, 0) and (1.1, 1.0999), so that the front spans [0, 1] in each. IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) averages, over 500 points of the optimal front from genoxide's `optimal_front`, the distance to the nearest point of the found front, counting only the objectives in which the found point is worse. Smaller is better, and 0 means the found front covers the optimal one. The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the area the front dominates up to the reference point (1.1, 1.1). Larger is better; the whole front's is computed from 20,000 of its points. Only feasible solutions count.

The page plays both runs back over the grey feasible region, which the trace samples from genomes on the unconstrained front's rays; hollow points are infeasible solutions of the populations, and the line is the optimal front, in its pieces.

[The project page](https://tachsin.gr/projects/genoxide/examples/mw6) plays this run back.

## Good results

The target: an IGD+ of at most 0.01 in normalized objectives.

With the paper's settings, the run with seed 1 ends on part of the long piece and little else: IGD+ 0.2932, hypervolume 0.1635. Over seeds 1 to 20, 3 runs reach the target; the median IGD+ is 0.0258. Ma and Wang report a mean IGD of 0.102 for the same NSGA-II (supplement, table S-R-I).

With η = 2 and 5,000 generations, the run's front covers all twelve pieces: IGD+ 0.0017, hypervolume 0.4026, 99.6% of the whole front's. Over seeds 1 to 20, every run reaches the target, with IGD+ from 0.0015 to 0.0020.
