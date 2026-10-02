---
title: CTP4
category: multi-objective
summary: Minimize two objectives whose 13 optimal points each lie at the end of a long, narrow feasible tunnel; NSGA-II with the paper's budget stops short of them, and MOEA/D with 7 million evaluations reaches them all.
reference: "Deb, K., Pratap, A. and Meyarivan, T. (2001). Constrained test problems for multi-objective evolutionary optimization. Evolutionary Multi-Criterion Optimization (EMO 2001), LNCS 1993: 284-298."
reference_url: https://doi.org/10.1007/3-540-44719-9_20
optimum: "13 points on the line f₂ = 1 − tan(0.2π) f₁, from (0, 1) to (0.9708, 0.2947); hypervolume 0.6683 in objectives scaled by the ideal and nadir points (reference point (1.1, 1.1))"
languages: [rust, python]
order: 188
family: CTP
---

# CTP4

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

CTP4 is CTP3 with a wave 7.5 times higher: θ = −0.2π, a = 0.75, b = 10, c = 1, d = 0.5 and e = 1.
The left side, u, measures the distance above the line f₂ = 1 − 0.7265 f₁, and the right side,
a wave of height up to 0.75, touches the line where its sine is 0: at v = 0, 0.1, …, 1.2 along it.
Where the wave is higher than the whole objective space above the line, only thin feasible
tunnels are left, one down to each touch.

The optimal front is the same as CTP3's: 13 points, (cos θ v, 1 + sin θ v) for v = k/10, from
(0, 1) to (0.9708, 0.2947), derived from the definition. At the points themselves the constraint
is exactly 0; in floating point, sin(kπ) is about 1e-15 and its square root 3e-8, so the best
feasible solutions sit next to them.

The definitions come from the authors' KanGAL report 200005 (October 2000), the paper's preprint:
eq. 5 on p. 7 and CTP4's parameters on p. 8. The report leaves g, the number of variables and their
bounds open, and prints f₂ as g (1 − f₁/g); its figures draw the unconstrained front as the curve
1 − √f₁, and the authors' NSGA-II code (version 1.1.6, KanGAL) computes g (1 − √(f₁/g)) with
g = 1 + x₂ and two variables in [0, 1], which genoxide follows.

## What makes it hard

The report puts it this way: "an algorithm now has to travel through a long narrow feasible
tunnel in search of the lone Pareto-optimal solution at the end of tunnel". At a height u above the
line, a tunnel reaches only about (u/a)²/(bπ) to each side: 6e-6 at u = 0.01, 60 times less than
CTP3's. Only 3% of random genomes are feasible, and the tunnels near f₁ = 1 don't connect to the
open region above: the population has to land in each of them. Every step closer to a tip needs
an offspring inside a narrower sliver than the last, and the steps of crossover and mutation,
which change x₁ and x₂ each on its own, rarely follow a tunnel's slant.

The report found that neither NSGA-II nor Ray et al.'s algorithm got near the 13 points.

## Representation

A `Real` genome of 2 genes in [0, 1]: x₁ and x₂. The problem is genoxide's `Ctp4`, whose fitness
is the two objectives and the constraint violation.

Solutions compare by constrained dominance: a feasible solution beats an infeasible one, of two
infeasible ones the smaller violation wins, and of two feasible ones Pareto dominance, or for
MOEA/D the subproblem's value, decides.

## Algorithm

Two runs with the same operators: simulated binary crossover with η = 20, at a rate of 0.9, and
polynomial mutation with η = 20, at a rate of 1/n per gene for n genes, 0.5.

- NSGA-II with the settings of the report's experiments, a population of 100 for 500
  generations.
- MOEA/D (Zhang and Li, 2007) with 100 subproblems, weight vectors evenly spread on the simplex,
  Tchebycheff decomposition and genoxide's default neighborhoods of 20, for 70,000 generations:
  7 million evaluations, 3.5 seconds on a desktop. Each subproblem keeps its best solution for its
  direction and lets a child replace a neighbor's only when better, so the solutions in a tunnel
  creep down it, one small improvement at a time. NSGA-II gets there too with enough generations
  (an IGD+ of 0.0077 to 0.0089 over seeds 1 to 4 after 100,000), but MOEA/D is faster.

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
solution, counting only the objectives in which the solution is worse. The 13 points'
hypervolume is 0.6683. In Python, `run` evaluates the problem in Rust, so both versions print the
same.

The run's `trace.json`, of the MOEA/D run, also has the problem's feasible region, which the page
shades; the tunnels near their tips are narrower than its cells.
[The project page](https://tachsin.gr/projects/genoxide/examples/ctp4) plays this run back.

## Good results

A good front has a solution next to each of the 13 points, and an IGD+ under 0.01. No finite set
of feasible solutions reaches the points' hypervolume, since the points themselves are the limit.

NSGA-II with the report's budget ends in the tunnels, far from their tips: 22 solutions, none
within 0.02 of a point, an IGD+ of 0.097 and 79.18% of the hypervolume. Over seeds 1 to 20 its
IGD+ is 0.070 to 0.139, and only one run comes within 0.02 of a single point: the report's
finding again.

MOEA/D reaches all 13: 16 solutions, an IGD+ of 0.0066 and 98.57% of the hypervolume. Over seeds
1 to 20, every run reaches all 13 points, with an IGD+ from 0.0065 to 0.0088.
