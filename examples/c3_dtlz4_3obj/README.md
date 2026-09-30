---
title: C3-DTLZ4 with 3 objectives
category: multi-objective
summary: Minimize three objectives of DTLZ4 subject to three quadratic constraints that make its front infeasible, so that the optimal front lies on three ellipsoids, against DTLZ4's bias, with NSGA-III.
reference: "Jain, H. and Deb, K. (2014). An evolutionary many-objective optimization algorithm using reference-point based nondominated sorting approach, part II: handling constraints and extending to an adaptive approach. IEEE Transactions on Evolutionary Computation 18(4): 602-622."
reference_url: https://doi.org/10.1109/TEVC.2013.2281534
optimum: "the front minⱼ [fⱼ²/4 + Σ_{i≠j} fᵢ²] = 1, three ellipsoids from 2 × the unit vectors to (2/3, 2/3, 2/3); the 91 target points' hypervolume is 1.0598 in objectives scaled by the nadir point (reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 178
family: C-DTLZ
tab: C3-DTLZ4
---

# C3-DTLZ4 with 3 objectives

## The problem

Jain and Deb (2014) extended NSGA-III to constrained problems and built, for its tests, the
constrained DTLZ problems: DTLZ problems of any number of objectives with constraints of three
types. Type 3 makes the whole front infeasible, and the new front lies on the constraints'
boundaries. C3-DTLZ4 is DTLZ4 with one quadratic constraint per objective:

```text
minimize   f₁ = (1 + g) cos(x₁¹⁰⁰π/2) cos(x₂¹⁰⁰π/2)
           f₂ = (1 + g) cos(x₁¹⁰⁰π/2) sin(x₂¹⁰⁰π/2)
           f₃ = (1 + g) sin(x₁¹⁰⁰π/2)
           g = Σᵢ₌₃⁷ (xᵢ − 0.5)²
subject to fⱼ²/4 + Σ_{i≠j} fᵢ² − 1 ≥ 0,  j = 1, 2, 3
x in [0, 1]⁷
```

The objectives lie on a sphere of radius 1 + g, and DTLZ4's front is the unit sphere, which the
constraints make infeasible: constraint j asks f to lie outside an ellipsoid stretched to 2 along
fⱼ. Each constraint grows with every objective, so, derived from the definition, the front of
C3-DTLZ4 is where the least of them is 0: minⱼ [fⱼ²/4 + Σ_{i≠j} fᵢ²] = 1, three pieces of
ellipsoids that reach 2 × the unit vectors and meet at (2/3, 2/3, 2/3). Every point of it is
reached by DTLZ4 with g = |f| − 1, between 0.15 and 1.

The paper uses 7 variables for this problem (k = 5, n = M + 4), where DTLZ4 has 12, and genoxide's
`C3Dtlz4` does the same. The definition was checked in the paper's accepted manuscript (eq. 8,
section V-D; the journal's final text wasn't compared).

## What makes it hard

DTLZ4's angles are x₁ and x₂ raised to the power 100: most of [0, 1] maps to angles near 0, which
put a solution near the f₁ axis, and a solution in the middle of the front is much harder to find.
The population has to spread against that bias, which can lose whole regions of the front for
good, and to sit on the constraints' boundaries, with infeasible space just below.

## Representation

A `Real` genome of 7 genes in [0, 1]. The problem's fitness is the three objectives and the total
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
- 750 generations.

## Output

The first line gives the size of the final front and how many of its solutions are feasible.

NSGA-III aims at one solution per reference direction: its targets are the 91 points where the
directions meet the front. The second line counts the targets that a solution comes within 0.02
of, in objectives scaled to [0, 1] by the front's nadir point (2, 2, 2); its ideal point is the
origin.

The third gives the front's IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to the
91 targets, in scaled objectives: the mean, over the targets, of the distance to the nearest
solution, counting only the objectives in which the solution is worse. It's the measure of the
paper, which uses IGD to the same targets. Then the front's hypervolume, the volume it dominates
up to the reference point (1.1, 1.1, 1.1), as a share of the 91 targets' hypervolume, 1.0598. In
Python, `run` evaluates the problem in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/c3-dtlz4-3obj) plays this run
back, with the population's infeasible solutions.

## Good results

A good front is all feasible, spread over the three ellipsoids, with an IGD+ well under 0.01 and a
hypervolume close to the targets'.

The run's front has 92 solutions, all feasible, within 0.02 of 84 of the 91 targets, with an
IGD+ of 0.0045 and 99.70% of the targets' hypervolume. Over seeds 1 to 50, 47 runs end with an
IGD+ under 0.01 (from 0.0029), and three lose part of the front to the bias, with an IGD+ of 0.20
to 0.64. The paper found the same: over its 20 runs, a median IGD of 0.025 and a worst of 0.56.
