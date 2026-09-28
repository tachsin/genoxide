---
title: DTLZ4 with 3 objectives
category: multi-objective
summary: Minimize three conflicting objectives whose Pareto front is an eighth of the unit sphere, where a bias crowds solutions towards its edges, with NSGA-III.
reference: "Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2002). Scalable multi-objective optimization test problems. Proceedings of the 2002 Congress on Evolutionary Computation, pp. 825-830."
reference_url: https://doi.org/10.1109/CEC.2002.1007032
optimum: "the unit sphere's eighth with f ≥ 0; hypervolume 0.8074 (reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 128
---

# DTLZ4 with 3 objectives

## The problem

Deb, Thiele, Laumanns and Zitzler (2002) built test problems that scale to any number of objectives
M. genoxide uses the numbering of their technical report (TIK-Report 112, ETH Zürich, 2001), which
is the same as the paper's for DTLZ1 to DTLZ4. DTLZ4 has M + k − 1 variables in [0, 1]; with M = 3
objectives and the suggested k = 10, that's 12 variables. All three objectives are minimized:

```text
g  = (x₃ − 0.5)² + … + (x₁₂ − 0.5)²
f₁ = (1 + g) cos(x₁¹⁰⁰ π/2) cos(x₂¹⁰⁰ π/2)
f₂ = (1 + g) cos(x₁¹⁰⁰ π/2) sin(x₂¹⁰⁰ π/2)
f₃ = (1 + g) sin(x₁¹⁰⁰ π/2)
```

This is DTLZ2 (see the [DTLZ2 example](../dtlz2_3obj/)) with each angle variable raised to the
power α = 100. f₁² + f₂² + f₃² = (1 + g)², and g is 0 where x₃ = … = x₁₂ = 0.5. So the Pareto front
is DTLZ2's: the part of the unit sphere with non-negative coordinates. How x₁ and x₂ place a
solution on it is different. In DTLZ2, x₁ = x₂ = 0.5 with g = 0 gives (0.5, 0.5, 0.71), the
middle of the front. Here, 0.5¹⁰⁰ is about 8 × 10⁻³¹, so both angles are nearly 0, and the
objectives are (1, 0, 0) to double precision: a corner of the front.

## What makes it hard

Converging is as easy as on DTLZ2: g has one minimum and no local fronts. What DTLZ4 tests, the
paper says, is whether an algorithm keeps a good spread of solutions.

x¹⁰⁰ is below 0.01 for every x below 0.954. So for 95.4% of the range of x₂, the angle x₂¹⁰⁰ π/2
is below 0.016, and f₂ is nearly 0; the same holds for x₁ and f₃. The middle of the front, where
all three objectives are well above 0, takes both x₁ and x₂ between about 0.95 and 1. Random
solutions, and the children of parents in the usual range, crowd along the edges of the front and
at the corner (1, 0, 0).

In this run's initial population, every non-dominated solution has f₂ below 10⁻¹³: all of them
lie in the plane f₂ = 0, which meets the front along its edge, the quarter circle from (1, 0, 0) to
(0, 0, 1). An algorithm that spreads its solutions over what it has found spreads them along that
edge, and has no reason to leave it until a child happens to get x₂ near 1.

## Representation

A `Real` genome of 12 genes in [0, 1]: the vector x. The problem is genoxide's `Dtlz4`, whose
fitness is the three objectives. In Python, `run` evaluates it in Rust.

## Algorithm

NSGA-III (Deb and Jain, 2014, IEEE Transactions on Evolutionary Computation 18(4): 577-601). Like
NSGA-II, it ranks solutions into non-dominated fronts, and parents and children compete for the
next population. Instead of crowding distance, it spreads the front along reference directions:
each solution joins the direction nearest to it, and directions with few members get more. A
non-dominated solution in an empty direction is taken first, so once one appears in the middle of
the front, it survives and has children. genoxide's docs recommend NSGA-III for three or more
objectives, and Deb and Jain test it on DTLZ4.

The settings are Deb and Jain's for DTLZ4 with 3 objectives:

- 91 reference directions from Das and Dennis's method (1998, SIAM Journal on Optimization 8(3):
  631-657) with 12 divisions: all points (a/12, b/12, c/12) with a + b + c = 12;
- a population of 92, the multiple of four just above 91;
- simulated binary crossover with η = 30, and polynomial mutation with η = 20 at a rate of 1/12 per
  gene, one gene per child on average;
- 600 generations, 55,292 evaluations.

The run shows the bias. By generation 24, the population covers the edge where f₂ = 0: its 13
reference directions, those with b = 0, and no other. The solutions converge (the median g falls
below 0.001 by generation 120), but the front stays on the edge, with a hypervolume of about 0.45,
for about 500 generations. Meanwhile x₂, which barely matters there, drifts. The largest f₂ on the
front creeps up from 10⁻¹³ at generation 496 to 0.002 at 536 and 0.12 at 544. Then, within 30
generations, the front covers the whole eighth of the sphere: 50 directions at generation 552, all
91 at 576. The hypervolume rises from 0.45 to 0.74.

## Output

Three lines. The first gives how many solutions are on the final front, and its hypervolume. The
hypervolume is the volume that the front dominates, up to a reference point. Larger is better. The
reference point here is (1.1, 1.1, 1.1), 1.1 times the nadir point (1, 1, 1), the worst value of
each objective on the front. For the whole front, the hypervolume is 1.1³ − π/6 = 0.8074: the cube
minus the eighth of the unit ball. A front on one edge only has about 0.45.

The second gives the front's IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to 1,035
points of the optimal front, from genoxide's `optimal_front`: Das and Dennis's points with 44
divisions, projected onto the sphere. IGD+ averages, over those points, the distance to the
nearest point of the found front, counting only the objectives in which the found point is worse.
0 means that the found front covers the optimal one. Smaller is better. It measures spread here: a
front on one edge has an IGD+ of about 0.23, ten times the final one.

The third gives the largest g among the front's solutions, computed as √(f₁² + f₂² + f₃²) − 1: 0
is on the true front.

[The project page](https://tachsin.gr/projects/genoxide/examples/dtlz4-3obj) plays this run back.

## Good results

No finite set of solutions reaches 0.8074. NSGA-III aims at one solution per reference direction.
The 91 points where the directions meet the sphere have a hypervolume of 0.7449 and an IGD+ of
0.0221.

The run's front has 92 solutions, a hypervolume of about 0.7414 and an IGD+ of about 0.0233: close
to those 91 points. Every solution has g below 0.012. The front reached the middle of the sphere
late, so it's less refined than on DTLZ2, where the same settings give 0.7443 after 250
generations.

The escape is a matter of chance. Over seeds 1 to 20, 9 runs cover the whole front within 50
generations, and 9 more between generations 100 and 600, all ending with hypervolumes from 0.7317
to 0.7451. Two don't: seed 17 stays on another edge, where f₃ = 0, with a hypervolume of 0.452,
and in seed 2 the front shrinks to the corner (1, 0, 0), with a hypervolume of 0.121.
