---
title: DC3-DTLZ1 with 3 objectives
category: multi-objective
summary: Minimize three objectives of DTLZ1 subject to constraints on its position variables and its distance function, whose Pareto front is four patches, with NSGA-III; constrained dominance stalls in a band of the distance function, and dropping that constraint reaches the front.
reference: "Li, K., Chen, R., Fu, G. and Yao, X. (2019). Two-archive evolutionary algorithm for constrained multiobjective optimization. IEEE Transactions on Evolutionary Computation 23(2): 303-315."
reference_url: https://doi.org/10.1109/TEVC.2018.2855411
optimum: "four patches of DTLZ1's front; ideal point (0, 0, 1/9), nadir point (49/162, 7/18, 1/2) = (0.3025, 0.3889, 0.5); hypervolume 0.9010 (normalized objectives, reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 205
family: DC-DTLZ
tab: DC3-DTLZ1
---

# DC3-DTLZ1 with 3 objectives

## The problem

Li, Chen, Fu and Yao (2019) proposed C-TAEA, a two-archive algorithm for constrained problems, and, to test it, the DC-DTLZ problems: DTLZ1 and DTLZ3 with constraints on the decision variables, where Jain and Deb's C-DTLZ problems constrain the objectives. Type 1 (DC1) constrains the first variable and cuts the front into cones from the origin; type 2 (DC2) constrains the distance function g and leaves almost nothing but the front feasible, with local optima of the violation on the way; type 3 (DC3) does both, on every position variable. The paper's supplement defines them (section 1.2) with parameters a and b, and gives a = 3 and b = 0.5 for DC1 only; genoxide takes the others from the code of the authors' laboratory, EMOC, and the docs of `multi::problems::Dc1Dtlz1` say what else the supplement leaves open and how genoxide settles it.

DC3-DTLZ1 with 3 objectives and 7 variables in [0, 1]:

```text
minimize   f₁ = ½ x₁ x₂ (1 + g)
           f₂ = ½ x₁ (1 − x₂) (1 + g)
           f₃ = ½ (1 − x₁) (1 + g)
           g = 100 (5 + Σᵢ₌₃⁷ ((xᵢ − 0.5)² − cos(20π (xᵢ − 0.5))))
subject to cos(aπx₁) ≥ b
           cos(aπx₂) ≥ b
           cos(aπg) ≥ b,  a = 3, b = 0.5
```

DTLZ1's front is the triangle where the objectives sum to 1/2, reached with the distance variables at 0.5, where g = 0. Each position variable must be in [0, 1/9] or [5/9, 7/9], and g in [0, 1/9], [5/9, 7/9], [11/9, 13/9], …: the front is the parts of DTLZ1's where both position variables are, four patches. (The supplement writes the position constraints for j = 1, …, m, which would include the first distance variable, 0.5 on the front, and make it all infeasible; genoxide constrains the m − 1 position variables, as EMOC does and the supplement's table 2 confirms.) Its ideal point is (0, 0, 1/9) and its nadir point (49/162, 7/18, 1/2) = (0.3025, 0.3889, 0.5). The supplement gives no front; genoxide derives it (g = 0 is feasible under every constraint, and the position constraints don't involve g, so DTLZ1's front points are optimal wherever their position variables are feasible, and no other point is), and it agrees with the sampled front that EMOC ships.

## What makes it hard

The constraint on g: it's feasible in bands, g in [0, 1/9], [5/9, 7/9], [11/9, 13/9], …, and with DTLZ's g, 100 times a sum, a population finds a band far from the front, feasible, and stays in it, as constrained dominance accepts no step through the infeasible gap below. Then the patches, a ninth of the front, which the population must spread over.

## Representation

A `Real` genome of 7 genes in [0, 1]. The problem is genoxide's `Dc3Dtlz1`, whose fitness is the three objectives and the total constraint violation, 0 when it is feasible; in Python, `gx.problems.Dc3Dtlz1()`. Solutions compare by constrained dominance: a feasible solution beats an infeasible one, of two infeasible ones the smaller violation wins, and of two feasible ones Pareto dominance decides.

## Algorithm

Two runs of NSGA-III, with a population of 92 for 1,000 generations each:

- NSGA-III (Deb and Jain, 2014; Jain and Deb, 2014, IEEE Transactions on Evolutionary Computation 18(4): 577-601 and 602-622) with the settings that the C-TAEA paper gives its C-NSGA-III (supplement, tables 3 and 4): the 91 reference directions of Das and Dennis's method with 12 divisions and a population of 92, simulated binary crossover with η = 30 at a rate of 1, and polynomial mutation with η = 20 at a rate of 1/n per gene, for 1,000 generations (the supplement gives no budget for the DC-DTLZ problems; 2,000 on DTLZ3, whose local fronts take longer), with constrained dominance;
- the same without the constraint on g, the position constraints only, its final front then scored by DC3-DTLZ1: the constraint on g holds at g = 0, so a population that converges to the front is feasible.

## Output

A line per run: the size of its final front, how many of its solutions the problem finds feasible, their IGD+ and hypervolume, and the hypervolume as a share of that of a sample of the optimal front with at least as many points as the population. Then the hypervolumes of the whole front and of the sample.

The indicators use the objectives normalized by the front's ideal and nadir points, so that the front spans [0, 1] in each. IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) averages, over 2,000 points of the optimal front from genoxide's `optimal_front`, the distance to the nearest feasible solution of the found front, counting only the objectives in which the solution is worse. Smaller is better; a front of as many points as the population can't cover the 2,000 exactly, and the sample shows what it can. The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the volume the feasible solutions dominate up to the reference point (1.1, 1.1, 1.1). Larger is better; the whole front's is computed from 3,000 of its points, and the sample's is what a front of that many points reaches. In Python, `run` evaluates the problem in Rust, so both versions print the same.

The page plays the runs back side by side, each population as the problem scores it: its feasible non-dominated solutions, its infeasible ones (hollow, at most 100 a frame), and the optimal front, sampled.

[The project page](https://tachsin.gr/projects/genoxide/examples/dc3-dtlz1-3obj) plays these runs back.

## Good results

The target: every solution feasible, and 99% of the hypervolume of the sample of 93 points of the front, about what a front of 92 solutions spread like it has.

The run with constrained dominance ends with 92 solutions, 92 feasible, IGD+ 0.4606, hypervolume 0.1355, 15.6% of the sample's: in a band of g away from the front. Without the constraint on g, it ends with 92 solutions, 92 feasible, IGD+ 0.0212, hypervolume 0.8726, 100.6% of the sample's. Over seeds 1 to 20, constrained dominance ends with IGD+ from 0.0202 to 1.0190, a median of 0.4597, and without the constraint on g with IGD+ from 0.0197 to 0.2537 and 76.3% to 101.9% of the sample's hypervolume, 18 runs reaching the target. The C-TAEA paper's C-NSGA-III ends with a median IGD of 9.75 (its table 3), the same failure, and C-TAEA with 0.0503.
