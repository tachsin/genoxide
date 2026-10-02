---
title: CTP3
category: multi-objective
summary: Minimize two objectives subject to a constraint that shrinks the optimal front to 13 separate points, with NSGA-II for 500 and for 3,000 generations.
reference: "Deb, K., Pratap, A. and Meyarivan, T. (2001). Constrained test problems for multi-objective evolutionary optimization. Evolutionary Multi-Criterion Optimization (EMO 2001), LNCS 1993: 284-298."
reference_url: https://doi.org/10.1007/3-540-44719-9_20
optimum: "13 points on the line f₂ = 1 − tan(0.2π) f₁, from (0, 1) to (0.9708, 0.2947); hypervolume 0.6683 in objectives scaled by the ideal and nadir points (reference point (1.1, 1.1))"
languages: [rust, python]
order: 187
family: CTP
---

# CTP3

## The problem

Deb, Pratap and Meyarivan (2001) built the CTP problems to test how multi-objective algorithms
handle constraints. CTP2 to CTP7 share one form, a generator whose six parameters θ, a, b, c, d
and e shape a single constraint:

```text
minimize   f₁ = x₁
           f₂ = g (1 − √(f₁/g)),  g = 1 + x₂
subject to cos θ (f₂ − e) − sin θ f₁ ≥ a |sin(bπ (sin θ (f₂ − e) + cos θ f₁)^c)|^d
x₁, x₂ in [0, 1]
```

CTP3 is CTP2 with a smaller wave and a sharper power: θ = −0.2π, a = 0.1, b = 10, c = 1, d = 0.5
and e = 1. The left side, u, measures the distance above the line f₂ = 1 − tan(0.2π) f₁ =
1 − 0.7265 f₁, and the right side waves along it, touching it where the sine is 0: at v = 0, 0.1,
…, 1.2 along the line. With d = 0.5, the wave rises steeply on both sides of those points, like a
square root, and every point of it near them is dominated by the point itself.

The optimal front is 13 points, one per touch, from (0, 1) to (0.9708, 0.2947), derived from the
definition: (cos θ v, 1 + sin θ v) for v = k/10. The report calls them "a singular feasible
Pareto-optimal solution" in each region. At the points themselves the constraint is exactly 0;
in floating point, sin(kπ) is about 1e-15 and its square root 3e-8, so the points are barely
infeasible to the computer, and the best solutions sit next to them.

The definitions come from the authors' KanGAL report 200005 (October 2000), the paper's preprint:
eq. 5 on p. 7 and CTP3's parameters on p. 8. The report leaves g, the number of variables and their
bounds open, and prints f₂ as g (1 − f₁/g); its figures draw the unconstrained front as the curve
1 − √f₁, and the authors' NSGA-II code (version 1.1.6, KanGAL) computes g (1 − √(f₁/g)) with
g = 1 + x₂ and two variables in [0, 1], which genoxide follows.

## What makes it hard

Each optimal point is the tip of a narrow feasible region. Near the tip, at a height u above the
line, the region reaches only about (u/a)²/(bπ) to each side along it: at u = 0.01, 3e-4. A solution
gets closer to the tip only by an offspring that lands inside this shrinking wedge. The population
also has to keep a solution at each of the 13 tips at once. Far from the front, the feasible region
is connected, and 42% of random genomes are feasible.

## Representation

A `Real` genome of 2 genes in [0, 1]: x₁ and x₂. The problem is genoxide's `Ctp3`, whose fitness
is the two objectives and the constraint violation.

Solutions compare by constrained dominance, the rule of the NSGA-II paper. A feasible solution
beats an infeasible one; of two infeasible ones, the smaller violation wins; of two feasible ones,
Pareto dominance decides.

## Algorithm

NSGA-II with the settings of the report's experiments, run twice:

- a population of 100, for 500 generations as in the report, then for 3,000;
- simulated binary crossover with η = 20, at a rate of 0.9;
- polynomial mutation with η = 20, at a rate of 1/n per gene for n genes: 0.5.

## Output

For each run, the first line gives the size of the final front and how many of its solutions are
feasible.

The second counts the optimal points that the run reaches: that have a solution within 0.02, in
scaled objectives.

The third gives the front's IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to 2,000
points of the optimal front (the 13 points, each repeated), and its hypervolume, the area it
dominates up to the reference point (1.1, 1.1), as a share of the optimal front's. Both use
objectives scaled to [0, 1] on the front, by its ideal point (0, 0.2947) and nadir point
(0.9708, 1). IGD+ averages, over the points of the optimal front, the distance to the nearest
solution, counting only the objectives in which the solution is worse. The 13 points' hypervolume
is 0.6683. In Python, `run` evaluates the problem in Rust, so both versions print the same.

The run's `trace.json`, of the longer run, also has the problem's feasible region, which the page
shades. [The project page](https://tachsin.gr/projects/genoxide/examples/ctp3) plays this run
back.

## Good results

A good front has a solution next to each of the 13 points, and an IGD+ under 0.01. No finite set
of feasible solutions reaches the points' hypervolume, since the points themselves are the limit.

After 500 generations, the report's budget, the front has 45 solutions near 11 of the 13 points,
an IGD+ of 0.0155 and 96.68% of the hypervolume: the solutions are in the right wedges, not yet at
their tips. After 3,000, it has 100 solutions near all 13 points, an IGD+ of 0.0060 and 98.70% of
the hypervolume. Over seeds 1 to 20, the 500-generation runs end with an IGD+ from 0.010 to 0.016,
none under 0.01, and 13 of them near all 13 points; the 3,000-generation runs all reach every
point, with an IGD+ from 0.0046 to 0.0063. The report saw NSGA-II find a solution very close to
each point with its five variables.
