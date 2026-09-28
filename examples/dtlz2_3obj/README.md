---
title: DTLZ2 with 3 objectives
category: multi-objective
summary: Minimize three conflicting objectives whose Pareto front is an eighth of the unit sphere, with NSGA-III.
reference: "Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2002). Scalable multi-objective optimization test problems. Proceedings of the 2002 Congress on Evolutionary Computation, pp. 825-830."
reference_url: https://doi.org/10.1109/CEC.2002.1007032
optimum: "hypervolume 0.8074 (reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 116
---

# DTLZ2 with 3 objectives

## The problem

Deb, Thiele, Laumanns and Zitzler (2002) built test problems that scale to any number of objectives
M. DTLZ2 has M + k − 1 variables in [0, 1]; with M = 3 objectives and the suggested k = 10, that's
12 variables. All three objectives are minimized:

```text
g  = (x₃ − 0.5)² + … + (x₁₂ − 0.5)²
f₁ = (1 + g) cos(x₁ π/2) cos(x₂ π/2)
f₂ = (1 + g) cos(x₁ π/2) sin(x₂ π/2)
f₃ = (1 + g) sin(x₁ π/2)
```

f₁² + f₂² + f₃² = (1 + g)². The Pareto front is where g = 0, that is x₃ = … = x₁₂ = 0.5: the part of
the unit sphere with non-negative coordinates. x₁ and x₂ are angles that choose the point on it.
With x₁ = x₂ = 0.5 and g = 0, the objectives are (0.5, 0.5, 0.71).

## What makes it hard

The front is a surface, not a curve. Covering it evenly takes many more solutions than covering a
curve. genoxide's docs note that crowding distance, NSGA-II's measure, spreads a front poorly beyond
2 or 3 objectives. At the same time, ten variables must converge to 0.5 to reach the front.

## Representation

A `Real` genome of 12 genes in [0, 1]: the vector x. The fitness is the three objectives. The Rust
version uses genoxide's `Dtlz2`; the Python version computes the objectives with numpy, a generation
at a time.

## Algorithm

NSGA-III (Deb and Jain, 2014, IEEE Transactions on Evolutionary Computation 18(4): 577-601). Like
NSGA-II, it ranks solutions into non-dominated fronts, and parents and children compete. Instead of
crowding distance, it spreads the front along reference directions. Each solution joins the
direction nearest to it, and directions with few members get more.

The settings are Deb and Jain's for DTLZ2 with 3 objectives:

- 91 reference directions from Das and Dennis's method (1998, SIAM Journal on Optimization 8(3):
  631-657) with 12 divisions: all points (a/12, b/12, c/12) with a + b + c = 12;
- a population of 92, the multiple of four just above 91;
- simulated binary crossover with η = 30, and polynomial mutation with η = 20 at a rate of 1/12 per
  gene;
- 250 generations.

## Output

One line: how many solutions are on the final front, and its hypervolume. The hypervolume is the
volume that the front dominates, up to the reference point (1.1, 1.1, 1.1). Larger is better. For
the whole front it is 1.1³ − π/6 = 0.8074: the cube minus the eighth of the unit ball.

[The project page](https://tachsin.gr/projects/genoxide/examples/dtlz2-3obj) plays this run back.

## Good results

No finite set reaches 0.8074. NSGA-III aims at one solution per reference direction. The 91 points
where the directions meet the sphere have a hypervolume of 0.7449. The run's front has 92 solutions
and a hypervolume of about 0.7443, within 0.1% of those 91 points.
