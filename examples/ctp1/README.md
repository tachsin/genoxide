---
title: CTP1
category: multi-objective
summary: Minimize two objectives subject to two constraints that cut off two thirds of the unconstrained front, so that most of the optimal front lies on constraint boundaries, with NSGA-II.
reference: "Deb, K., Pratap, A. and Meyarivan, T. (2001). Constrained test problems for multi-objective evolutionary optimization. Evolutionary Multi-Criterion Optimization (EMO 2001), LNCS 1993: 284-298."
reference_url: https://doi.org/10.1007/3-540-44719-9_20
optimum: "the front f₂ = max(e^−f₁, 0.858 e^−0.541f₁, 0.728 e^−0.295f₁) for f₁ in [0, 1]; hypervolume 0.8829 in objectives scaled by the ideal and nadir points (reference point (1.1, 1.1))"
languages: [rust, python]
order: 185
family: CTP
---

# CTP1

## The problem

Deb, Pratap and Meyarivan (2001) built the CTP problems to test how multi-objective algorithms
handle constraints, with difficulty that a few parameters tune. CTP1 has two objectives, two
variables and two constraints:

```text
minimize   f₁ = x₁
           f₂ = g exp(−f₁/g),  g = 1 + x₂
subject to f₂ − aⱼ exp(−bⱼ f₁) ≥ 0,  j = 1, 2
           a = (0.858, 0.728), b = (0.541, 0.295)
x₁, x₂ in [0, 1]
```

Without the constraints, the best solutions have g = 1 (x₂ = 0), and the front is the curve
f₂ = exp(−f₁). Each constraint asks f₂ to stay above another exponential curve,
aⱼ exp(−bⱼ f₁), flatter than the front. The paper builds a and b so that the first curve meets
the front at f₁ = 1/3 and the second meets the first at f₁ = 2/3, and prints them to three
digits, the values used here. With those, the curves cross at f₁ = 0.33367 and 0.66789.

The optimal front, derived from the definition, is the highest of the three curves at g = 1: the
unconstrained front up to f₁ = 0.33367, the first constraint's boundary up to 0.66789, and the
second's up to f₁ = 1, where f₂ = 0.728 e^−0.295 = 0.5420. Two thirds of it lies on constraint
boundaries, as the paper says; the rest of the unconstrained front, below them, is infeasible.

The definitions are the paper's: CTP1 is its eq. 4, with the table of a and b on p. 289. Its
preprint, the authors' KanGAL report 200005 (October 2000, p. 6), and Deb's 2001 book
(*Multi-Objective Optimization Using Evolutionary Algorithms*, Wiley, eq. 8.45 on p. 353) give the
same. All three leave g, the number of variables and their bounds open; genoxide takes them from the
authors' NSGA-II code (version 1.1.6, KanGAL), which has g = 1 + x₂, the g the book names on p. 360,
and two variables in [0, 1]. The paper's own experiments used five variables and a Rastrigin
function for g, without giving its formula.

## What makes it hard

Two thirds of the front lies on constraint boundaries. A solution there has x₂ = 0 and meets a
constraint exactly; a little lower in f₂ and it's infeasible, a little higher and it's dominated.
The algorithm has to hold a population on two curved boundaries and on the part of the
unconstrained front that stays feasible, with the corners between them.

The feasible region itself is large: every solution above the constraints' curves is feasible,
93% of random genomes. The difficulty lies near the front, not in finding feasible solutions.

## Representation

A `Real` genome of 2 genes in [0, 1]: x₁ and x₂. The problem is genoxide's `Ctp1`, whose fitness
is the two objectives and the total constraint violation.

Solutions compare by constrained dominance, the rule of the NSGA-II paper. A feasible solution
beats an infeasible one; of two infeasible ones, the smaller violation wins; of two feasible ones,
Pareto dominance decides.

## Algorithm

NSGA-II with the settings of the paper's experiments:

- a population of 100, for 500 generations;
- simulated binary crossover with η = 20, at a rate of 0.9;
- polynomial mutation with η = 20, at a rate of 1/n per gene for n genes: 0.5.

## Output

The first line gives the size of the final front and how many of its solutions are feasible.

The second counts the pieces of the optimal front that the run reaches: that have a solution
within 0.02 of one of their points, in scaled objectives. CTP1's front is one connected piece.

The third gives the front's IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to 2,000
points of the optimal front, and its hypervolume, the area it dominates up to the reference point
(1.1, 1.1), as a share of the whole optimal front's. Both use objectives scaled to [0, 1] on the
front, by its ideal point (0, 0.5420) and nadir point (1, 1). IGD+ averages, over the points of
the optimal front, the distance to the nearest solution, counting only the objectives in which
the solution is worse: 0 means that the front covers the optimal one. The whole front's
hypervolume, from 100,000 of its points, is 0.8829. In Python, `run` evaluates the problem in
Rust, so both versions print the same.

The run's `trace.json` also has the problem's feasible region, which the page shades.
[The project page](https://tachsin.gr/projects/genoxide/examples/ctp1) plays this run back.

## Good results

A good front is all feasible, spread over the whole curve, with an IGD+ well under 0.01 and a
hypervolume close to the whole front's. 100 points of the optimal front, spread evenly along it,
give 99.46% of its hypervolume and an IGD+ of 0.0023.

The run's front has 100 solutions, all feasible, an IGD+ of 0.0037 and 99.18% of the whole front's
hypervolume. Over seeds 1 to 20, every run ends the same way: IGD+ from 0.0034 to 0.0040, and 99.15%
to 99.22% of the hypervolume.
