---
title: C1-DTLZ1 with 3 objectives
category: multi-objective
summary: Minimize three objectives of DTLZ1 subject to a constraint that leaves only a thin wedge next to the front feasible, across DTLZ1's many local fronts, with NSGA-III.
reference: "Jain, H. and Deb, K. (2014). An evolutionary many-objective optimization algorithm using reference-point based nondominated sorting approach, part II: handling constraints and extending to an adaptive approach. IEEE Transactions on Evolutionary Computation 18(4): 602-622."
reference_url: https://doi.org/10.1109/TEVC.2013.2281534
optimum: "DTLZ1's front, the plane f₁ + f₂ + f₃ = 1/2, all feasible; the 91 target points' hypervolume is 1.1204 in objectives scaled by the nadir point (reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 153
family: C-DTLZ
tab: C1-DTLZ1
---

# C1-DTLZ1 with 3 objectives

## The problem

Jain and Deb (2014) extended NSGA-III to constrained problems and built, for its tests, the
constrained DTLZ problems: DTLZ problems of any number of objectives with constraints of three
types. Type 1 keeps the front and puts an infeasible barrier before it. C1-DTLZ1 is DTLZ1 with one
constraint:

```text
minimize   f₁ = ½ x₁ x₂ (1 + g)
           f₂ = ½ x₁ (1 − x₂) (1 + g)
           f₃ = ½ (1 − x₁) (1 + g)
           g = 100 (5 + Σᵢ₌₃⁷ ((xᵢ − 0.5)² − cos(20π (xᵢ − 0.5))))
subject to 1 − f₃/0.6 − f₁/0.5 − f₂/0.5 ≥ 0
x in [0, 1]⁷
```

DTLZ1's front is the triangle where the objectives sum to 1/2, reached with the last five
variables at 0.5, where g = 0. The constraint is a plane through (0.5, 0, 0), (0, 0.5, 0) and
(0, 0, 0.6): below it is feasible. It touches the front's two corners on the f₁ and f₂ axes and
rises above its third, so the whole front is feasible (on it, the constraint is f₃/3 ≥ 0), with
only a thin wedge above it: a solution with g more than 0.2 is infeasible everywhere.

The problem is genoxide's `C1Dtlz1`, checked against the paper's eq. 4 (section V-B) in its
accepted manuscript (the journal's final text wasn't compared), with 7 variables (k = 5), as the
paper uses.

## What makes it hard

DTLZ1's g has 11⁵ − 1 local optima, which put local fronts parallel to the true one at g = 1, 2,
…: an algorithm usually climbs down them one by one. Here all of them are infeasible, and only the
last steps, g below 0.2, are feasible: the population has to cross the infeasible space above the
wedge, following the constraint violation, and then converge inside the wedge.

## Representation

A `Real` genome of 7 genes in [0, 1]. The problem's fitness is the three objectives and the
constraint violation, 0 when it's feasible.

## Algorithm

NSGA-III (Deb and Jain, 2014) with Jain and Deb's constraint handling: parents are paired at
random, except that of two infeasible ones the smaller violation wins; the next population takes
the feasible solutions first, sorted into non-dominated fronts and spread along the reference
directions, and fills the rest with the least infeasible ones. The settings are the paper's:

- the 91 reference directions of Das and Dennis's method with 12 divisions, and a population of
  92;
- simulated binary crossover with η = 30, at a rate of 1;
- polynomial mutation with η = 20, at a rate of 1/n per gene for n genes: 1/7;
- 2,000 generations, where the paper runs 500 (see Good results).

## Output

The first line gives the size of the final front and how many of its solutions are feasible.

NSGA-III aims at one solution per reference direction: its targets are the 91 points where the
directions meet the front. The second line counts the targets that a solution comes within 0.02
of, in objectives scaled to [0, 1] by the front's nadir point (0.5, 0.5, 0.5); its ideal point is
the origin.

The third gives the front's IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to the
91 targets, in scaled objectives: the mean, over the targets, of the distance to the nearest
solution, counting only the objectives in which the solution is worse. It's the measure of the
paper, which uses IGD to the same targets. Then the front's hypervolume, the volume it dominates
up to the reference point (1.1, 1.1, 1.1), as a share of the 91 targets' hypervolume, 1.1204. In
Python, `run` evaluates the problem in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/c1-dtlz1-3obj) plays this run
back, with the population's infeasible solutions.

## Good results

A good front is all feasible, with a solution at each of the 91 targets, an IGD+ well under 0.01
and a hypervolume close to the targets'.

The run's front has 92 solutions, all feasible, reaching all 91 targets, with an IGD+ of 0.0020
and 99.86% of the targets' hypervolume. Over seeds 1 to 20, every run reaches all 91 targets,
with an IGD+ from 0.0001 to 0.0039 and 99.68% to 100.01% of the hypervolume.

With the paper's 500 generations, the population is often still on its way down: over the same
20 seeds, only 6 runs end with an IGD+ under 0.01, from 0.0014 to 0.035 in all, and 3 reach all
91 targets. The paper reports a median IGD of 0.0049 over its 20 runs of 500 generations.
