---
title: DTLZ3 with 3 objectives
category: multi-objective
summary: Minimize three conflicting objectives whose Pareto front is an eighth of the unit sphere, behind 3¹⁰ − 1 local fronts, with NSGA-III.
reference: "Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2002). Scalable multi-objective optimization test problems. Proceedings of the 2002 Congress on Evolutionary Computation, pp. 825-830."
reference_url: https://doi.org/10.1109/CEC.2002.1007032
optimum: "the unit sphere's eighth with f ≥ 0; hypervolume 0.8074 (reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 107
---

# DTLZ3 with 3 objectives

## The problem

Deb, Thiele, Laumanns and Zitzler (2002) built test problems that scale to any number of objectives
M. genoxide uses the numbering of their technical report (TIK-Report 112, ETH Zürich, 2001), which
is the same as the paper's for DTLZ1 to DTLZ4. DTLZ3 has M + k − 1 variables in [0, 1]; with M = 3
objectives and the suggested k = 10, that's 12 variables. All three objectives are minimized:

```text
g  = 100 (10 + Σᵢ₌₃¹² ((xᵢ − 0.5)² − cos(20π (xᵢ − 0.5))))
f₁ = (1 + g) cos(x₁ π/2) cos(x₂ π/2)
f₂ = (1 + g) cos(x₁ π/2) sin(x₂ π/2)
f₃ = (1 + g) sin(x₁ π/2)
```

f₁² + f₂² + f₃² = (1 + g)²: the objectives are a point at distance 1 + g from the origin. x₁ and
x₂ are angles that choose the direction; the other ten variables, the distance variables, set g.
g is 0 where x₃ = … = x₁₂ = 0.5, and positive everywhere else. So the Pareto front is where g = 0:
the part of the unit sphere with non-negative coordinates. With x₁ = x₂ = 0.5 and g = 0, the
objectives are (0.5, 0.5, 0.71).

DTLZ3 combines the two problems before it: the front of DTLZ2 (see the
[DTLZ2 example](../dtlz2_3obj/)), and the g of DTLZ1 (see the [DTLZ1 example](../dtlz1_3obj/)),
over ten distance variables instead of five.

## What makes it hard

g is a Rastrigin function of the distance variables, scaled by 100. It has a local minimum wherever
each distance variable is near one of 0, 0.1, 0.2, …, 1. At each local minimum, the solutions form
a local front: a sphere of radius 1 + g, where no small change of the distance variables lowers g.
The paper counts 3ᵏ − 1 of them, 59,048 for k = 10, and one global front; for DTLZ1, with the same
g, it counts 11ᵏ − 1. Either way, far too many to search one by one.

The nearest local fronts have one distance variable at 0.4 or 0.6 and the others at 0.5. There,
g is about 1, and the sphere has radius about 2. To reach the true front, that variable has to move
by 0.1. Halfway, g is about 200 higher, so a child that moves only part of the way is far worse
than its parent and doesn't survive. The search needs a child that jumps across the ridge in one
step.

With ten distance variables, a random solution is further away than in DTLZ1: the non-dominated
solutions of this run's initial population have g from 423 to 1,110. And the whole population can
settle on a local front, since every solution there is non-dominated among the others.

## Representation

A `Real` genome of 12 genes in [0, 1]: the vector x. The problem is genoxide's `Dtlz3`, whose
fitness is the three objectives. In Python, `run` evaluates it in Rust.

## Algorithm

NSGA-III (Deb and Jain, 2014, IEEE Transactions on Evolutionary Computation 18(4): 577-601). Like
NSGA-II, it ranks solutions into non-dominated fronts, and parents and children compete for the
next population. Instead of crowding distance, it spreads the front along reference directions:
each solution joins the direction nearest to it, and directions with few members get more.
genoxide's docs recommend it for three or more objectives, and Deb and Jain test it on DTLZ3.

The settings are Deb and Jain's for DTLZ3 with 3 objectives:

- 91 reference directions from Das and Dennis's method (1998, SIAM Journal on Optimization 8(3):
  631-657) with 12 divisions: all points (a/12, b/12, c/12) with a + b + c = 12;
- a population of 92, the multiple of four just above 91;
- simulated binary crossover with η = 30, and polynomial mutation with η = 20 at a rate of 1/12 per
  gene, one gene per child on average;
- 1,000 generations, 92,092 evaluations: 2.5 times DTLZ1's 400, for the harder convergence.

The run shows the trap. By generation 370, the front has reached a local front with g ≈ 1.07; at
generation 450, every solution on it has x₄ near 0.4 and the other distance variables near 0.5.
The population stays there for about 100 generations. The hypervolume is 0 all this time: on a
sphere of radius 2, no point has all three objectives below the reference point's 1.1. Between
generations 464 and 480, a child lands across the ridge, the best g falls from 1.03 to 0.05, and
the population follows. By generation 512, the median g is 0.025 and the hypervolume 0.61. By
generation 625, the front covers the whole eighth of the sphere, with a hypervolume of 0.72. The
last 375 generations refine it: the largest g on the front falls to 0.006, and the hypervolume
reaches 0.737.

## Output

Three lines. The first gives how many solutions are on the final front, and its hypervolume. The
hypervolume is the volume that the front dominates, up to a reference point. Larger is better. The
reference point here is (1.1, 1.1, 1.1), 1.1 times the nadir point (1, 1, 1), the worst value of
each objective on the front. For the whole front, the hypervolume is 1.1³ − π/6 = 0.8074: the cube
minus the eighth of the unit ball.

The second gives the front's IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to 1,035
points of the optimal front, from genoxide's `optimal_front`: Das and Dennis's points with 44
divisions, projected onto the sphere. IGD+ averages, over those points, the distance to the
nearest point of the found front, counting only the objectives in which the found point is worse.
0 means that the found front covers the optimal one. Smaller is better.

The third gives the largest g among the front's solutions, computed as √(f₁² + f₂² + f₃²) − 1. It
shows convergence alone: 0 is on the true front, and about 1 is the nearest local front.

[The project page](https://tachsin.gr/projects/genoxide/examples/dtlz3-3obj) plays this run back.

## Good results

No finite set of solutions reaches 0.8074. NSGA-III aims at one solution per reference direction.
The 91 points where the directions meet the sphere have a hypervolume of 0.7449 and an IGD+ of
0.0221.

The run's front has 92 solutions, a hypervolume of about 0.7373 and an IGD+ of about 0.0256. Every
solution has g below 0.0062: on the true front, not a local one, but not yet as close to it as on
DTLZ1. Over seeds 1 to 20, every run escapes the local fronts, between generations 250 and 550, and
ends with a hypervolume from 0.7284 to 0.7432, 0.7363 in the median.
