---
title: DC1-DTLZ1 with 3 objectives
category: multi-objective
summary: Minimize three objectives of DTLZ1 subject to a constraint on the first variable, whose Pareto front is two bands of DTLZ1's, with NSGA-III and SMS-EMOA.
reference: "Li, K., Chen, R., Fu, G. and Yao, X. (2019). Two-archive evolutionary algorithm for constrained multiobjective optimization. IEEE Transactions on Evolutionary Computation 23(2): 303-315."
reference_url: https://doi.org/10.1109/TEVC.2018.2855411
optimum: "two bands of the triangle f₁ + f₂ + f₃ = 1/2, f₃ in [1/9, 2/9] or [4/9, 1/2]; ideal point (0, 0, 1/9), nadir point (7/18, 7/18, 1/2); hypervolume 1.0709 (normalized objectives, reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 181
family: DC-DTLZ
tab: DC1-DTLZ1
---

# DC1-DTLZ1 with 3 objectives

## The problem

Li, Chen, Fu and Yao (2019) proposed C-TAEA, a two-archive algorithm for constrained problems, and, to test it, the DC-DTLZ problems: DTLZ1 and DTLZ3 with constraints on the decision variables, where Jain and Deb's C-DTLZ problems constrain the objectives. Type 1 (DC1) constrains the first variable and cuts the front into cones from the origin; type 2 (DC2) constrains the distance function g and leaves almost nothing but the front feasible, with local optima of the violation on the way; type 3 (DC3) does both, on every position variable. The paper's supplement defines them (section 1.2) with parameters a and b, and gives a = 3 and b = 0.5 for DC1 only; genoxide takes the others from the code of the authors' laboratory, EMOC, and the docs of `multi::problems::Dc1Dtlz1` say what else the supplement leaves open and how genoxide settles it.

DC1-DTLZ1 with 3 objectives and 7 variables in [0, 1]:

```text
minimize   f₁ = ½ x₁ x₂ (1 + g)
           f₂ = ½ x₁ (1 − x₂) (1 + g)
           f₃ = ½ (1 − x₁) (1 + g)
           g = 100 (5 + Σᵢ₌₃⁷ ((xᵢ − 0.5)² − cos(20π (xᵢ − 0.5))))
subject to cos(aπx₁) ≥ b,  a = 3, b = 0.5
```

DTLZ1's front is the triangle where the objectives sum to 1/2, reached with the distance variables at 0.5, where g = 0. The constraint keeps x₁ in [0, 1/9] or [5/9, 7/9], a third of its range, and the front is the parts of the triangle with f₃ = (1 − x₁)/2 in [1/9, 2/9] or [4/9, 1/2]: two bands, each a cone from the origin through the triangle. Its ideal point is (0, 0, 1/9) and its nadir point (7/18, 7/18, 1/2). The supplement gives no front; genoxide derives it (g = 0 is feasible under every constraint, and the position constraints don't involve g, so DTLZ1's front points are optimal wherever their position variables are feasible, and no other point is), and it agrees with the sampled front that EMOC ships.

## What makes it hard

DTLZ1's g, with 11⁵ − 1 local minima, puts local fronts parallel to the true one, at g = 1, 2, …, which the population climbs down. The constraint cuts every one of them into the same cones, all feasible: the difficulty is to keep the population spread over both bands, when a third of the reference directions of NSGA-III point into the infeasible gaps between them.

## Representation

A `Real` genome of 7 genes in [0, 1]. The problem is genoxide's `Dc1Dtlz1`, whose fitness is the three objectives and the total constraint violation, 0 when it is feasible; in Python, `gx.problems.Dc1Dtlz1()`. Solutions compare by constrained dominance: a feasible solution beats an infeasible one, of two infeasible ones the smaller violation wins, and of two feasible ones Pareto dominance decides.

## Algorithm

Two runs, with a population of 92 for 1,000 generations each:

- NSGA-III (Deb and Jain, 2014; Jain and Deb, 2014, IEEE Transactions on Evolutionary Computation 18(4): 577-601 and 602-622) with the settings that the C-TAEA paper gives its C-NSGA-III (supplement, tables 3 and 4): the 91 reference directions of Das and Dennis's method with 12 divisions and a population of 92, simulated binary crossover with η = 30 at a rate of 1, and polynomial mutation with η = 20 at a rate of 1/n per gene, for 1,000 generations (the supplement gives no budget for the DC-DTLZ problems; 2,000 on DTLZ3, whose local fronts take longer);
- SMS-EMOA (Beume, Naujoks and Emmerich, 2007, European Journal of Operational Research 181(3): 1653-1669), which keeps the solutions that add the most hypervolume, with the same operators and as many children a generation as the population.

## Output

A line per run: the size of its final front, how many of its solutions the problem finds feasible, their IGD+ and hypervolume, and the hypervolume as a share of that of a sample of the optimal front with at least as many points as the population. Then the hypervolumes of the whole front and of the sample.

The indicators use the objectives normalized by the front's ideal and nadir points, so that the front spans [0, 1] in each. IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) averages, over 2,000 points of the optimal front from genoxide's `optimal_front`, the distance to the nearest feasible solution of the found front, counting only the objectives in which the solution is worse. Smaller is better; a front of as many points as the population can't cover the 2,000 exactly, and the sample shows what it can. The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the volume the feasible solutions dominate up to the reference point (1.1, 1.1, 1.1). Larger is better; the whole front's is computed from 3,000 of its points, and the sample's is what a front of that many points reaches. In Python, `run` evaluates the problem in Rust, so both versions print the same.

The page plays the runs back side by side, each population as the problem scores it: its feasible non-dominated solutions, its infeasible ones (hollow, at most 100 a frame), and the optimal front, sampled.

[The project page](https://tachsin.gr/projects/genoxide/examples/dc1-dtlz1-3obj) plays these runs back.

## Good results

The target: every solution feasible, and 99% of the hypervolume of the sample of 93 points of the front, about what a front of 92 solutions spread like it has.

NSGA-III's run with seed 1 ends with 92 solutions, 92 feasible, IGD+ 0.0267, hypervolume 1.0434, 103.1% of the sample's; SMS-EMOA's with 92 solutions, 92 feasible, IGD+ 0.0247, hypervolume 1.0543, 104.2% of the sample's. Over seeds 1 to 20, NSGA-III ends with IGD+ from 0.0248 to 0.0271 and 102.8% to 103.4% of the sample's hypervolume, every run reaching the target, and SMS-EMOA with IGD+ from 0.0240 to 0.0253 and 104.1% to 104.3% of the sample's hypervolume, every run reaching the target. A front of 92 points spread over two bands can't come close to the 2,000 points that IGD+ measures with, and both exceed the sample's hypervolume: the solutions crowd the bands' edges, which add more volume. The C-TAEA paper's C-NSGA-III ends with a median IGD of 0.0599 (its table 3), and C-TAEA with 0.0564.
