---
title: DAS-CMOP9
category: multi-objective
summary: Minimize three objectives over 30 variables subject to 7 constraints of adjustable difficulty, whose Pareto front is patches of a sphere, with linked variables, with MOEA/D-DE, which reaches the front and spreads on it as well as its 300 subproblems allow, and NSGA-II, which covers only part of it.
reference: "Fan, Z., Li, W., Cai, X., Li, H., Wei, C., Zhang, Q., Deb, K. and Goodman, E. (2020). Difficulty adjustable and scalable constrained multiobjective test problem toolkit. Evolutionary Computation 28(3): 339-378."
reference_url: https://doi.org/10.1162/evco_a_00259
optimum: "for the difficulty triplet (0.5, 0.5, 0.5), DAS-CMOP8's, patches of the unit sphere's octant moved to (0.5, 0.5, 0.5); ideal point (0.5, 0.5, 0.5), nadir point (1.5, 1.5, 1.4969); hypervolume 0.7853 (normalized objectives, reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 215
family: DAS-CMOP
---

# DAS-CMOP9

## The problem

Fan et al. (2020) built a toolkit of constrained test problems whose difficulty is set by a triplet (η, ζ, γ) in [0, 1]³: η for diversity (type I constraints, which cut the front into pieces), ζ for feasibility (a type II constraint, a band of distances from the unconstrained front) and γ for convergence (type III constraints, infeasible regions in the objective space). DAS-CMOP7 to DAS-CMOP9 have three objectives; see [DAS-CMOP1](../das_cmop1/) for the two-objective problems.

DAS-CMOP9 (table 2) has three objectives over 30 variables in [0, 1]:

```text
f₁ = cos(0.5πx₁) cos(0.5πx₂) + g
f₂ = cos(0.5πx₁) sin(0.5πx₂) + g
f₃ = sin(0.5πx₁) + g
g = Σⱼ₌₃³⁰ (xⱼ − cos(0.25jπ(x₁ + x₂)/30))²
subject to
  sin(20πx₁) − b ≥ 0,  cos(20πx₂) − b ≥ 0,     b = 2η − 1       (type I)
  (e − g)(g − 0.5) ≥ 0,                        e = 0.5 − ln ζ   (type II)
  Σⱼ fⱼ² − fₖ² + (fₖ − 1)² − r² ≥ 0,  k = 1, 2, 3               (type III)
  Σⱼ (fⱼ − 1/√3)² − r² ≥ 0,                    r = γ/2          (type III)
```

All three are minimized. At g = 0 the unconstrained front is the unit sphere's octant. The distance function g is 0 where each xⱼ is its own cosine of x₁ + x₂, different for each j: the variables are linked to the position variables. The example uses the triplet with which the paper's figure 6 plots DAS-CMOP9, (0.5, 0.5, 0.5): b = 0, which keeps x₁ in [0, 0.05], [0.1, 0.15], … and x₂ in [0, 0.025], [0.075, 0.125], …; g between 0.5 and 0.5 + ln 2 ≈ 1.193; and spheres of radius 0.25.

The front depends on the triplet, and the paper samples it; genoxide samples it in the same way, as the first feasible point of each ray α(x₁, x₂) + g (1, 1, 1) from the 41,905 points of Das and Dennis's method with 288 divisions, non-dominated. With this triplet it is DAS-CMOP8's: patches of the sphere of radius 1 centered at (0.5, 0.5, 0.5) where x₁ and x₂ are both in the type I intervals. Its ideal point is (0.5, 0.5, 0.5) and its nadir point (1.5, 1.5, 1.4969) (1.4969 for f₃, where x₁ is at most 0.95).

## What makes it hard

The linked variables. A solution on the front at (x₁, x₂) has every other variable at its own cosine of x₁ + x₂: to move along the front, all 28 must move together, each by its own amount. Simulated binary crossover and polynomial mutation change each variable on its own, so a population that has settled on some values of x₁ + x₂ keeps them, and covers only part of the front's patches. Differential evolution moves them together, as its steps are differences between solutions of the population: the paper's MOEA/D-CDP uses it, and so does MOEA/D-DE here. Then, as for DAS-CMOP8, the band of feasible g and the grid of patches.

## Representation

A `Real` genome of 30 genes in [0, 1]. The problem is genoxide's `DasCmop9`, whose fitness is the three objectives and the total constraint violation, 0 when it is feasible; in Python, `gx.problems.DasCmop9()`, with `difficulty=(η, ζ, γ)` or the number of one of the paper's sixteen triplets. Solutions compare by constrained dominance: a feasible solution beats an infeasible one, of two infeasible ones the smaller violation wins, and of two feasible ones Pareto dominance decides.

## Algorithm

Two runs, each with the paper's population of 300 for 1,000 generations, 300,000 evaluations (section 7.1), and polynomial mutation with η = 20 at a rate of 1/30 per gene:

- MOEA/D with differential evolution, MOEA/D-DE (Li and Zhang, 2009, IEEE Transactions on Evolutionary Computation 13(2): 284-302): a subproblem for each of the 300 weight vectors of Das and Dennis's method with 23 divisions, each the weighted Tchebycheff distance to the ideal point, with the paper's 30 neighbours (0.1 N) and at most 2 replacements per child; a child is its subproblem's solution moved by F = 0.5 times the difference of two parents, every gene (CR = 1), the parents from the neighbourhood with probability δ = 0.2, where the paper and Li and Zhang use 0.9 (DAS-CMOP1's page says why);
- NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation 6(2): 182-197) with the paper's settings, simulated binary crossover with η = 20 at a rate of 0.9, as the contrast.

Both compare solutions by constrained dominance.

## Output

A line per run: the size of its final front, how many of its solutions the problem finds feasible, their IGD+ and hypervolume, the hypervolume as a share of that of a sample of the optimal front with at least as many points as the population, and the median and largest distance from its solutions to the nearest point of a dense sample of the front (44,643 points). Then a line for what MOEA/D's decomposition can reach: for each of its 300 weight vectors, the point of the dense sample with the least Tchebycheff value, the best that subproblem can have, and the same indicators for these points; and the same with 666 weight vectors (35 divisions). Then the hypervolumes of the whole front and of the sample.

The indicators use the objectives normalized by the front's ideal and nadir points, so that the front spans [0, 1] in each. IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) averages, over 2,000 points of the optimal front from genoxide's `optimal_front`, the distance to the nearest feasible solution of the found front, counting only the objectives in which the solution is worse. Smaller is better; a front of as many points as the population can't cover the 2,000 exactly, and the sample shows what it can. The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the volume the feasible solutions dominate up to the reference point (1.1, 1.1, 1.1). Larger is better; the whole front's is computed from 3,000 of its points, and the sample's is what a front of that many points reaches. In Python, `run` evaluates the problem in Rust, so both versions print the same.

The page plays the runs back side by side, each population as the problem scores it: its feasible non-dominated solutions, its infeasible ones (hollow, at most 100 a frame), and the optimal front, sampled.

[The project page](https://tachsin.gr/projects/genoxide/examples/das-cmop9) plays these runs back.

## Good results

The target: every solution feasible, and 99% of the hypervolume of the sample of 346 points of the front, about what a front of 300 solutions spread like it has.

MOEA/D-DE's run with seed 1 ends with 196 solutions, 196 feasible, IGD+ 0.0158, hypervolume 0.7484, 97.9% of the sample's; half of its solutions are within 0.0018 of the front, and the farthest 0.0200 away. NSGA-II's ends with 300 solutions, 300 feasible, IGD+ 0.2006, hypervolume 0.4880, 63.8% of the sample's: as close to the front (a median of 0.0017), but on only part of it. Over seeds 1 to 20, MOEA/D-DE ends with IGD+ from 0.0151 to 0.0167 and 97.3% to 98.0% of the sample's hypervolume; half of each run's solutions within 0.0018 to 0.0023 of the front, 89% to 97% of them within 0.01, and the farthest 0.014 to 0.047 away. NSGA-II ends with IGD+ from 0.0158 to 0.3408 and 45% to 97% of the sample's hypervolume. With δ = 0.9, MOEA/D-DE ends with IGD+ from 0.0152 to 0.5144 and 20% to 98%.

MOEA/D-DE reaches the front, and spreads on it as well as its decomposition allows, but no run reaches the 99% target, because the target doesn't fit 300 Tchebycheff subproblems. Each subproblem can at best hold the front's point with its least Tchebycheff value. Those best points, one per weight vector, are only 223 distinct points, as several weight vectors share one on the patched front. Together they have IGD+ 0.0184 and 96.7% of the sample's hypervolume: every run of MOEA/D-DE does better on both. With 666 weight vectors, the best points reach 99.4%: the target asks for a front spread evenly by length, as the sample is, which the 300 subproblems can't give. The few solutions farther from the front than 0.01 haven't converged yet: with seed 1, the farthest is dominated by 81 points of the dense sample.

The paper's NSGA-II-CDP ends with a mean IGD of 0.227 on DAS-CMOP9 with this triplet (its table 5, number 8), and MOEA/D-CDP, with differential evolution, with 0.0767.
