---
title: DAS-CMOP7
category: multi-objective
summary: Minimize three objectives over 30 variables subject to 7 constraints of adjustable difficulty, whose Pareto front is patches of the plane f₁ + f₂ + f₃ = 2.5, with NSGA-II and NSGA-III.
reference: "Fan, Z., Li, W., Cai, X., Li, H., Wei, C., Zhang, Q., Deb, K. and Goodman, E. (2020). Difficulty adjustable and scalable constrained multiobjective test problem toolkit. Evolutionary Computation 28(3): 339-378."
reference_url: https://doi.org/10.1162/evco_a_00259
optimum: "for the difficulty triplet (0.5, 0.5, 0.5), patches of the plane f₁ + f₂ + f₃ = 2.5; ideal point (0.5, 0.5, 0.5), nadir point (1.4479, 1.5, 1.5); hypervolume 1.1300 (normalized objectives, reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 193
family: DAS-CMOP
---

# DAS-CMOP7

## The problem

Fan et al. (2020) built a toolkit of constrained test problems whose difficulty is set by a triplet (η, ζ, γ) in [0, 1]³: η for diversity (type I constraints, which cut the front into pieces), ζ for feasibility (a type II constraint, a band of distances from the unconstrained front) and γ for convergence (type III constraints, infeasible regions in the objective space). DAS-CMOP7 to DAS-CMOP9 have three objectives; see [DAS-CMOP1](../das_cmop1/) for the two-objective problems.

DAS-CMOP7 (table 2) has three objectives over 30 variables in [0, 1]:

```text
f₁ = x₁x₂ + g
f₂ = x₂(1 − x₁) + g
f₃ = 1 − x₂ + g
g = 28 + Σⱼ₌₃³⁰ ((xⱼ − 0.5)² − cos(20π(xⱼ − 0.5)))
subject to
  sin(20πx₁) − b ≥ 0,  cos(20πx₂) − b ≥ 0,     b = 2η − 1       (type I)
  (e − g)(g − 0.5) ≥ 0,                        e = 0.5 − ln ζ   (type II)
  Σⱼ fⱼ² − fₖ² + (fₖ − 1)² − r² ≥ 0,  k = 1, 2, 3               (type III)
  Σⱼ (fⱼ − 1/√3)² − r² ≥ 0,                    r = γ/2          (type III)
```

All three are minimized. At g = 0 the unconstrained front is the simplex f₁ + f₂ + f₃ = 1. The distance function g is 0 with every xⱼ at 0.5, and has 11²⁸ − 1 local minima, the cosine's, as DTLZ1's. The example uses the triplet with which the paper's figure 6 plots DAS-CMOP7, (0.5, 0.5, 0.5): b = 0, which keeps x₁ in [0, 0.05], [0.1, 0.15], … and x₂ in [0, 0.025], [0.075, 0.125], …; g between 0.5 and 0.5 + ln 2 ≈ 1.193; and spheres of radius 0.25.

The front depends on the triplet, and the paper samples it; genoxide samples it in the same way, as the first feasible point of each ray α(x₁, x₂) + g (1, 1, 1) from the 41,905 points of Das and Dennis's method with 288 divisions, non-dominated. With this triplet it is patches of the plane f₁ + f₂ + f₃ = 2.5, the simplex moved out by 0.5 along the diagonal, where x₁ and x₂ are both in the type I intervals. Its ideal point is (0.5, 0.5, 0.5) and its nadir point (1.4479, 1.5, 1.5) (1.4479, short of 1.45, where the sampled rays end).

## What makes it hard

The grid of patches first: the type I constraints keep x₁ and x₂ each in ten intervals, and the front is the 100 or so patches where both are, which a population must spread over. Then the multimodal g, whose local minima put local fronts parallel to the true one, with the band of feasible g, between 0.5 and 0.5 + ln 2, among them. The four type III spheres, of radius 0.25, are centered at the unit vectors and at (1, 1, 1)/√3, which with this triplet lie between the front and the origin, where the band of g lets no solution be: they don't block the way to it.

## Representation

A `Real` genome of 30 genes in [0, 1]. The problem is genoxide's `DasCmop7`, whose fitness is the three objectives and the total constraint violation, 0 when it is feasible; in Python, `gx.problems.DasCmop7()`, with `difficulty=(η, ζ, γ)` or the number of one of the paper's sixteen triplets. Solutions compare by constrained dominance: a feasible solution beats an infeasible one, of two infeasible ones the smaller violation wins, and of two feasible ones Pareto dominance decides.

## Algorithm

Two runs, each with a population of 300 for 1,000 generations, 300,000 evaluations (the paper's budget, section 7.1), simulated binary crossover with η = 20 and polynomial mutation at a rate of 1/30 per gene:

- NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation 6(2): 182-197) with the paper's settings: crossover at a rate of 0.9, mutation with η = 20;
- NSGA-III (Deb and Jain, 2014, IEEE Transactions on Evolutionary Computation 18(4): 577-601) with the 276 reference directions of Das and Dennis's method with 22 divisions, crossover at a rate of 1, and mutation with η = 5, whose steps are larger (a median of about 11% of the range, against 3% with η = 20).

## Output

A line per run: the size of its final front, how many of its solutions the problem finds feasible, their IGD+ and hypervolume, and the hypervolume as a share of that of a sample of the optimal front with at least as many points as the population. Then the hypervolumes of the whole front and of the sample.

The indicators use the objectives normalized by the front's ideal and nadir points, so that the front spans [0, 1] in each. IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) averages, over 2,000 points of the optimal front from genoxide's `optimal_front`, the distance to the nearest feasible solution of the found front, counting only the objectives in which the solution is worse. Smaller is better; a front of as many points as the population can't cover the 2,000 exactly, and the sample shows what it can. The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the volume the feasible solutions dominate up to the reference point (1.1, 1.1, 1.1). Larger is better; the whole front's is computed from 3,000 of its points, and the sample's is what a front of that many points reaches. In Python, `run` evaluates the problem in Rust, so both versions print the same.

The page plays the runs back side by side, each population as the problem scores it: its feasible non-dominated solutions, its infeasible ones (hollow, at most 100 a frame), and the optimal front, sampled.

[The project page](https://tachsin.gr/projects/genoxide/examples/das-cmop7) plays these runs back.

## Good results

The target: every solution feasible, and 99% of the hypervolume of the sample of 649 points of the front, about what a front of 300 solutions spread like it has.

With the paper's settings, NSGA-II's run with seed 1 ends with 300 solutions, 300 feasible, IGD+ 0.0190, hypervolume 1.0978, 98.2% of the sample's. NSGA-III's ends with 300 solutions, 300 feasible, IGD+ 0.0136, hypervolume 1.1115, 99.4% of the sample's. Over seeds 1 to 20, NSGA-II ends with IGD+ from 0.0174 to 0.0193 and 98.0% to 98.6% of the sample's hypervolume; with NSGA-III, every run reaches the target, with IGD+ from 0.0124 to 0.0135 and 99.1% to 99.7% of the sample's hypervolume.

The paper's NSGA-II-CDP ends with a mean IGD of 0.0225 on DAS-CMOP7 with this triplet (its table 5, number 8), and MOEA/D-CDP with 0.102. NSGA-III, whose reference directions spread the population evenly, covers the patches better than NSGA-II's crowding distance does.
