---
title: C3-DTLZ1 with 3 objectives
category: multi-objective
summary: Minimize three objectives of DTLZ1 subject to three constraints that make its front infeasible, so that the optimal front lies on the constraints' boundaries, with NSGA-III.
reference: "Jain, H. and Deb, K. (2014). An evolutionary many-objective optimization algorithm using reference-point based nondominated sorting approach, part II: handling constraints and extending to an adaptive approach. IEEE Transactions on Evolutionary Computation 18(4): 602-622."
reference_url: https://doi.org/10.1109/TEVC.2013.2281534
optimum: "the front f₁ + f₂ + f₃ + min fⱼ = 1, three planes from the unit vectors to (1/4, 1/4, 1/4); the 91 target points' hypervolume is 1.1624 (reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 177
family: C-DTLZ
tab: C3-DTLZ1
---

# C3-DTLZ1 with 3 objectives

## The problem

Jain and Deb (2014) extended NSGA-III to constrained problems and built, for its tests, the
constrained DTLZ problems: DTLZ problems of any number of objectives with constraints of three
types. Type 3 makes the whole front infeasible, and the new front lies on the constraints'
boundaries. C3-DTLZ1 is DTLZ1 with one linear constraint per objective:

```text
minimize   f₁ = ½ x₁ x₂ (1 + g)
           f₂ = ½ x₁ (1 − x₂) (1 + g)
           f₃ = ½ (1 − x₁) (1 + g)
           g = 100 (5 + Σᵢ₌₃⁷ ((xᵢ − 0.5)² − cos(20π (xᵢ − 0.5))))
subject to Σ_{i≠j} fᵢ + fⱼ/0.5 − 1 ≥ 0,  j = 1, 2, 3
x in [0, 1]⁷
```

With S = f₁ + f₂ + f₃, constraint j reads S + fⱼ ≥ 1, and all three together S + min fⱼ ≥ 1.
DTLZ1's front, where S = 1/2, breaks them all. Derived from the definition, the front of C3-DTLZ1
is where S + min fⱼ = 1: three planes, one per constraint, that meet at (1/4, 1/4, 1/4) and reach
the unit vectors (1, 0, 0), (0, 1, 0) and (0, 0, 1). Each point below one of them breaks a
constraint, and each point of them is reached by DTLZ1 with g = 2S − 1.

The paper prints the constraint as Σ_{i≠j} fⱼ + fᵢ/0.5 − 1 ≥ 0, i and j swapped. Read that way,
each constraint is the same, 2S − 1 ≥ 0, which DTLZ1's front meets, against the paper's text
("the unconstrained Pareto-optimal front is now infeasible") and its figure 13, which draws the
two lines of the two-objective version: genoxide's `C3Dtlz1` uses Σ_{i≠j} fᵢ + fⱼ/0.5 − 1 ≥ 0.
The definition was checked in the paper's accepted manuscript (eq. 7, section V-D; the journal's
final text wasn't compared), with 7 variables (k = 5), as the paper uses.

## What makes it hard

Three things. DTLZ1's g has 11⁵ − 1 local optima, local fronts at whole values of g, which the
population has to climb down. The optimal front lies at g between 1/2 and 1, not at g = 0, so the
population must stop short of DTLZ1's own front and sit on the constraints' boundaries, with
infeasible space just below. And the front bends where the planes meet: each optimal solution
meets one, two or all three constraints exactly.

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
- 2,000 generations, where the paper runs 750 (see Good results).

## Output

The first line gives the size of the final front and how many of its solutions are feasible.

NSGA-III aims at one solution per reference direction: its targets are the 91 points where the
directions w meet the front, w / (1 + min w) for w on the simplex. The second line counts the
targets that a solution comes within 0.02 of. The front's nadir point is (1, 1, 1) and its ideal
point the origin, so the objectives need no scaling.

The third gives the front's IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to the
91 targets: the mean, over the targets, of the distance to the nearest solution, counting only the
objectives in which the solution is worse. It's the measure of the paper, which uses IGD to the
same targets. Then the front's hypervolume, the volume it dominates up to the reference point
(1.1, 1.1, 1.1), as a share of the 91 targets' hypervolume, 1.1624. In Python, `run` evaluates the
problem in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/c3-dtlz1-3obj) plays this run
back, with the population's infeasible solutions.

## Good results

A good front is all feasible, with a solution at each of the 91 targets, an IGD+ well under 0.01
and a hypervolume close to the targets'.

The run's front has 92 solutions, all feasible, reaching all 91 targets, with an IGD+ of 0.0002
and 99.99% of the targets' hypervolume. Over seeds 1 to 20, every run ends with an IGD+ under
0.01, from 0.0002 to 0.0040, 99.79% to 100.00% of the hypervolume, and within 0.02 of 86 to 91
targets (all 91 in 10 runs). With the paper's 750 generations, 15 of the 20 runs end with an IGD+
under 0.01, and the others end with an IGD+ up to 0.049.
The paper reports a median IGD of 0.0091 over its 20 runs of 750 generations.
