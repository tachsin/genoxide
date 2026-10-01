---
title: CTP8
category: multi-objective
summary: Minimize two objectives subject to two constraints, bands parallel to the front and bands across it, which leave feasible patches and a front of three disconnected pieces, with NSGA-II.
reference: "Deb, K. (2001). Multi-Objective Optimization Using Evolutionary Algorithms. Wiley, Chichester."
reference_url: ""
optimum: "three pieces of CTP6's front, from (0, 3.6958) to (0.1345, 3.3128), (0.3263, 2.7686) to (0.4790, 2.3372) and (0.6823, 1.7654) to (0.8229, 1.3727); hypervolume 0.6540 in objectives scaled by the ideal and nadir points (reference point (1.1, 1.1))"
languages: [rust, python]
order: 172
family: CTP
---

# CTP8

## The problem

Deb, Pratap and Meyarivan (2001) built the CTP problems from a generator, a constraint whose six
parameters θ, a, b, c, d and e set its difficulty, and noted that "a combination of two or more
effects can be achieved together in a problem by considering more than one such constraints".
CTP8 does that with two:

```text
minimize   f₁ = x₁
           f₂ = g (1 − √(f₁/g)),  g = 1 + x₂
subject to cos θⱼ (f₂ − eⱼ) − sin θⱼ f₁ ≥ aⱼ |sin(bⱼπ (sin θⱼ (f₂ − eⱼ) + cos θⱼ f₁)^cⱼ)|^dⱼ
           for j = 1, 2, with
           θ₁ = 0.1π,   a₁ = 40, b₁ = 0.5, c₁ = 1, d₁ = 2, e₁ = −2
           θ₂ = −0.05π, a₂ = 40, b₂ = 2,   c₂ = 1, d₂ = 6, e₂ = 0
x₁ in [0, 1], x₂ in [0, 10]
```

The first constraint is CTP6's: feasible bands parallel to the front, with the front on the lower
edge of the first one. The second is CTP7's with b = 2 instead of 5: wide bands nearly upright,
across the objective space. Together they leave feasible patches where the bands cross, and the
front is the parts of CTP6's front that the second constraint allows: three pieces, from
(0, 3.6958) to (0.1345, 3.3128), from (0.3263, 2.7686) to (0.4790, 2.3372) and from
(0.6823, 1.7654) to (0.8229, 1.3727), found by sampling the boundaries of the feasible region.
Only 10% of random genomes are feasible.

CTP8 isn't in the EMO 2001 paper, which has CTP1 to CTP7. Later papers credit it to Deb's book
(2001, *Multi-Objective Optimization Using Evolutionary Algorithms*, Wiley), which I couldn't
read. The definition here is the one in the NSGA-II code of Deb's group (version 1.1.6, KanGAL),
with the same g, variables and bounds as its CTP6 and CTP7; the parameters of the two constraints
come from there. The constraint's form is the one of the EMO 2001 paper, checked in the authors'
KanGAL report 200005 (eq. 5).

## What makes it hard

Both difficulties at once. The feasible patches are islands: a population has to cross
infeasible bands in two directions to reach the ones on the front, and it has to keep a group of
solutions on each of the three pieces, since no path inside the feasible region joins them. Every
optimal solution lies on the first constraint's boundary.

## Representation

A `Real` genome of 2 genes: x₁ in [0, 1] and x₂ in [0, 10]. The problem is genoxide's `Ctp8`,
whose fitness is the two objectives and the total constraint violation.

Solutions compare by constrained dominance, the rule of the NSGA-II paper. A feasible solution
beats an infeasible one; of two infeasible ones, the smaller violation wins; of two feasible ones,
Pareto dominance decides.

## Algorithm

NSGA-II with the settings of the EMO 2001 paper's experiments:

- a population of 100, for 500 generations;
- simulated binary crossover with η = 20, at a rate of 0.9;
- polynomial mutation with η = 20, at a rate of 1/n per gene for n genes: 0.5.

## Output

The first line gives the size of the final front and how many of its solutions are feasible.

The second counts the pieces of the optimal front that the run reaches: that have a solution
within 0.02 of one of their points, in scaled objectives.

The third gives the front's IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to 2,000
points of the optimal front, and its hypervolume, the area it dominates up to the reference point
(1.1, 1.1), as a share of the whole optimal front's. Both use objectives scaled to [0, 1] on the
front, by its ideal point (0, 1.3727) and nadir point (0.8229, 3.6958). IGD+ averages, over the
points of the optimal front, the distance to the nearest solution, counting only the objectives
in which the solution is worse. The whole front's hypervolume, from 100,000 of its points, is
0.6540. In Python, `run` evaluates the problem in Rust, so both versions print the same.

The run's `trace.json` also has the problem's feasible region, which the page shades.
[The project page](https://tachsin.gr/projects/genoxide/examples/ctp8) plays this run back.

## Good results

A good front is all feasible, with solutions on all three pieces, an IGD+ well under 0.01 and a
hypervolume close to the whole front's. 100 points of the optimal front, spread over the pieces by
their lengths, give 99.80% of its hypervolume and an IGD+ of 0.0013.

The run's front has 100 solutions, all feasible, on all three pieces, with an IGD+ of 0.0024 and
99.55% of the whole front's hypervolume. Over seeds 1 to 20, every run reaches the three pieces,
with an IGD+ from 0.0024 to 0.0028 and 99.38% to 99.60% of the hypervolume.
