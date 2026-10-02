---
title: CTP5
category: multi-objective
summary: Minimize two objectives whose optimal front is one continuous piece and 15 separate points crowding towards its end, with NSGA-II for 500 and for 10,000 generations.
reference: "Deb, K., Pratap, A. and Meyarivan, T. (2001). Constrained test problems for multi-objective evolutionary optimization. Evolutionary Multi-Criterion Optimization (EMO 2001), LNCS 1993: 284-298."
reference_url: https://doi.org/10.1007/3-540-44719-9_20
optimum: "a piece of the constraint's boundary from (0, 1) to f₁ = 0.2558, and 15 points on the line f₂ = 1 − tan(0.2π) f₁ at √(k/10) along it, the last at (0.9908, 0.2801); hypervolume 0.6613 in objectives scaled by the ideal and nadir points (reference point (1.1, 1.1))"
languages: [rust, python]
order: 189
family: CTP
---

# CTP5

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

CTP5 is CTP3 with c = 2: θ = −0.2π, a = 0.1, b = 10, c = 2, d = 0.5 and e = 1. The wave now
touches the line f₂ = 1 − 0.7265 f₁ where v² is a multiple of 1/10, at v = √(k/10) along it, so
its touches crowd together as f₁ grows: 16 of them in the objective space, from (0, 1) to
(0.9908, 0.2801).

The report describes CTP5's optimal solutions as separate points, like CTP3's. Derived from the
definition, the front is one continuous piece and 15 points. Near v = 0, the wave |sin(bπv²)|^0.5
grows like √(bπ) v, only linearly, and the boundary leaves the line at a slope shallow enough to
keep going down in f₂: the whole first stretch of the boundary, from (0, 1) to f₁ = 0.2558, is
optimal. Near the other touches, the wave rises like a square root, and only the touches themselves
are optimal, as in CTP3. The report's own figure 16 shows NSGA-II's solutions spread along that
first stretch. genoxide's `optimal_front` samples the boundaries of the feasible region densely and
keeps the feasible non-dominated points.

The definitions come from the authors' KanGAL report 200005 (October 2000), the paper's preprint:
eq. 5 on p. 7 and CTP5's parameters on p. 9. The report leaves g, the number of variables and their
bounds open, and prints f₂ as g (1 − f₁/g); its figures draw the unconstrained front as the curve
1 − √f₁, and the authors' NSGA-II code (version 1.1.6, KanGAL) computes g (1 − √(f₁/g)) with
g = 1 + x₂ and two variables in [0, 1], which genoxide follows.

## What makes it hard

The optimal points crowd together: 0.034 apart in f₁ at the right end, against 0.106 between the
first two of them. Each is the tip of a narrow feasible wedge, as in CTP3, and the population has to
keep a solution at each of 15 tips and along the continuous piece, which takes most of the front's
length and pulls the crowding distance of NSGA-II towards it. The report found this non-uniform
spacing "not a great difficulty to NSGA-II". About 43% of random genomes are feasible.

## Representation

A `Real` genome of 2 genes in [0, 1]: x₁ and x₂. The problem is genoxide's `Ctp5`, whose fitness
is the two objectives and the constraint violation.

Solutions compare by constrained dominance, the rule of the NSGA-II paper. A feasible solution
beats an infeasible one; of two infeasible ones, the smaller violation wins; of two feasible ones,
Pareto dominance decides.

## Algorithm

NSGA-II with the settings of the report's experiments, run twice:

- a population of 100, for 500 generations as in the report, then for 10,000;
- simulated binary crossover with η = 20, at a rate of 0.9;
- polynomial mutation with η = 20, at a rate of 1/n per gene for n genes: 0.5.

## Output

For each run, the first line gives the size of the final front and how many of its solutions are
feasible.

The second counts the pieces of the optimal front that the run reaches, the continuous piece and
the 15 points: that have a solution within 0.02 of one of their points, in scaled objectives.

The third gives the front's IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to 2,000
points of the optimal front, and its hypervolume, the area it dominates up to the reference point
(1.1, 1.1), as a share of the whole optimal front's. Both use objectives scaled to [0, 1] on the
front, by its ideal point (0, 0.2801) and nadir point (0.9908, 1). IGD+ averages, over the points
of the optimal front, the distance to the nearest solution, counting only the objectives in which
the solution is worse. The 2,000 points spread over the pieces by their lengths, so all but 15 lie
on the continuous piece: IGD+ says little about the 15 points, which the second line counts. The
whole front's hypervolume, from 100,000 of its points, is 0.6613. In Python, `run` evaluates the
problem in Rust, so both versions print the same.

The run's `trace.json`, of the longer run, also has the problem's feasible region, which the page
shades. [The project page](https://tachsin.gr/projects/genoxide/examples/ctp5) plays this run
back.

## Good results

A good front reaches the continuous piece and all 15 points, with a hypervolume close to the whole
front's. 100 points of the optimal front, one on each point and the rest on the continuous piece,
give 99.98% of it.

After 500 generations, the report's budget, the front reaches the continuous piece and 10 of the
15 points: an IGD+ of 0.0012, but 96.73% of the hypervolume. After 10,000, it reaches all 16
pieces, with an IGD+ of 0.0007 and 99.02% of the hypervolume. Over seeds 1 to 20, no
500-generation run reaches all 16 pieces (from 10 to 15 of them), while every 10,000-generation
run does, with 98.99% to 99.25% of the hypervolume.
