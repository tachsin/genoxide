---
title: Convex C2-DTLZ2 with 3 objectives
category: multi-objective
summary: Minimize three objectives of the convex DTLZ2 with an infeasible cylinder around the diagonal that cuts a hole in the middle of the front, with NSGA-III.
reference: "Jain, H. and Deb, K. (2014). An evolutionary many-objective optimization algorithm using reference-point based nondominated sorting approach, part II: handling constraints and extending to an adaptive approach. IEEE Transactions on Evolutionary Computation 18(4): 602-622."
reference_url: https://doi.org/10.1109/TEVC.2013.2281534
optimum: "the convex front f₃ + √f₁ + √f₂ = 1 outside the cylinder of radius 0.225 around the diagonal; the 47 feasible target points' hypervolume is 1.2546 (reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 156
family: C-DTLZ
tab: convex C2-DTLZ2
---

# Convex C2-DTLZ2 with 3 objectives

## The problem

Jain and Deb (2014) extended NSGA-III to constrained problems and built, for its tests, the
constrained DTLZ problems: DTLZ problems of any number of objectives with constraints of three
types. Type 2 makes parts of the front infeasible. Convex C2-DTLZ2 starts from the convex DTLZ2 of
the paper's part I (Deb and Jain, 2014), DTLZ2 with its first objectives raised to the power 4 and
its last squared, and adds one constraint:

```text
minimize   f₁ = [(1 + g) cos(x₁π/2) cos(x₂π/2)]⁴
           f₂ = [(1 + g) cos(x₁π/2) sin(x₂π/2)]⁴
           f₃ = [(1 + g) sin(x₁π/2)]²
           g = Σᵢ₌₃¹² (xᵢ − 0.5)²
subject to Σᵢ (fᵢ − λ)² − r² ≥ 0,  λ = (f₁ + f₂ + f₃)/3,  r = 0.225
x in [0, 1]¹²
```

The front of convex DTLZ2, at g = 0, is the convex surface f₃ + √f₁ + √f₂ = 1 (part I, eq. 8),
from the corners (1, 0, 0), (0, 1, 0) and (0, 0, 1). Σ (fᵢ − λ)² is the squared distance from the
diagonal (1, 1, 1): the constraint makes a cylinder of radius r around it infeasible, which cuts a
round hole in the middle of the front. r is 0.225 for 3 and 5 objectives, 0.26 for 8 and 10, and
0.27 for 15. The front of convex C2-DTLZ2 is the rest of the surface.

The problem is genoxide's `ConvexC2Dtlz2`, checked against the paper's eq. 6 and its radii
(section V-C), and the convex DTLZ2 against part I (section VII-C), both in their accepted
manuscripts (the journal's final texts weren't compared), with 12 variables (k = 10), as the paper
uses. The paper's table V counts 47 of the 91 reference directions with a Pareto-optimal
solution for 3 objectives, and 47 of them meet the front outside the cylinder here (and 97 of
210 for 5 objectives, as the table says too).

## What makes it hard

The front is convex, and the hole in its middle takes 44 of the 91 reference directions, whose
niches have no optimal solution. The solutions that would fill the hole are infeasible, and those
next to it lie on the cylinder's edge. Behind the hole, solutions with larger g that stay outside
the cylinder never beat the front: genoxide's tests check that no feasible solution does, against
random ones.

## Representation

A `Real` genome of 12 genes in [0, 1]. The problem's fitness is the three objectives and the
constraint violation, 0 when it's feasible.

## Algorithm

NSGA-III (Deb and Jain, 2014) with Jain and Deb's constraint handling: parents are paired at
random, except that of two infeasible ones the smaller violation wins; the next population takes
the feasible solutions first, sorted into non-dominated fronts and spread along the reference
directions, and fills the rest with the least infeasible ones. The settings are the paper's:

- the 91 reference directions of Das and Dennis's method with 12 divisions, and a population of
  92;
- simulated binary crossover with η = 30, at a rate of 1;
- polynomial mutation with η = 20, at a rate of 1/n per gene for n genes: 1/12;
- 250 generations.

## Output

The first line gives the size of the final front and how many of its solutions are feasible.

NSGA-III aims at one solution per reference direction: its targets are the points where the
directions w meet the front, t w with f₃ + √f₁ + √f₂ = 1, the 47 of the 91 that are feasible. The
second line counts the targets that a solution comes within 0.02 of. The front's nadir point is
(1, 1, 1) and its ideal point the origin, so the objectives need no scaling.

The third gives the front's IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to the
47 targets: the mean, over the targets, of the distance to the nearest solution, counting only the
objectives in which the solution is worse. It's the measure of the paper, which uses IGD to the
same targets. Then the front's hypervolume, the volume it dominates up to the reference point
(1.1, 1.1, 1.1), as a share of the 47 targets' hypervolume, 1.2546. The population has 92
solutions for 47 targets, and the others fill in between: the front's hypervolume can pass the
targets'. In Python, `run` evaluates the problem in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/convex-c2-dtlz2-3obj) plays this
run back, with the population's infeasible solutions; the true front is drawn as the surface
around the hole.

## Good results

A good front is all feasible, around the whole hole, with a solution near each of the 47 targets
and an IGD+ well under 0.01.

The run's front has 92 solutions, all feasible, within 0.02 of 46 of the 47 targets, with an IGD+
of 0.0033 and 100.94% of the targets' hypervolume. Over seeds 1 to 20, the runs come within 0.02
of 40 to 47 targets (46 or 47 in 15 of them), with an IGD+ from 0.0016 to 0.0056 and 100.9% to
101.1% of the targets' hypervolume. The target this run misses lies on the front's edge, where
f₁ = 0: the nearest solution is 0.028 away, but worse by only 0.0015, the distance that IGD+
counts. The paper reports a median IGD of 0.0059 over its 20 runs.
