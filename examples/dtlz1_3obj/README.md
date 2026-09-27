---
title: DTLZ1 with 3 objectives
category: multi-objective
summary: Minimize three conflicting objectives whose Pareto front is a triangle in a plane, behind 11⁵ − 1 local fronts, with NSGA-III.
reference: "Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2002). Scalable multi-objective optimization test problems. Proceedings of the 2002 Congress on Evolutionary Computation, pp. 825-830."
reference_url: https://doi.org/10.1109/CEC.2002.1007032
optimum: "the plane f₁ + f₂ + f₃ = 0.5; hypervolume 0.1455 (reference point (0.55, 0.55, 0.55))"
languages: [rust, python]
order: 101
---

# DTLZ1 with 3 objectives

## The problem

Deb, Thiele, Laumanns and Zitzler (2002) built test problems that scale to any number of objectives
M. genoxide uses the numbering of their technical report (TIK-Report 112, ETH Zürich, 2001), which
is the same as the paper's for DTLZ1 to DTLZ4. DTLZ1 has M + k − 1 variables in [0, 1]; with M = 3
objectives and the suggested k = 5, that's 7 variables. All three objectives are minimized:

```text
g  = 100 (5 + Σᵢ₌₃⁷ ((xᵢ − 0.5)² − cos(20π (xᵢ − 0.5))))
f₁ = ½ x₁ x₂ (1 + g)
f₂ = ½ x₁ (1 − x₂) (1 + g)
f₃ = ½ (1 − x₁) (1 + g)
```

The three objectives add up to ½ (1 + g). x₁ and x₂ choose how that sum is shared; the other five
variables, the distance variables, set g. g is 0 where x₃ = … = x₇ = 0.5, and positive everywhere
else. So the Pareto front is where g = 0: the triangle of points with f₁ + f₂ + f₃ = 0.5 and no
negative objective, with corners (0.5, 0, 0), (0, 0.5, 0) and (0, 0, 0.5). With x₁ = x₂ = 0.5 and
g = 0, the objectives are (0.125, 0.125, 0.25).

The technical report says that the optimal solutions have x₃ = … = x₇ = 0; the paper's equation
gives 0.5, which is correct. genoxide's `Dtlz1` uses 0.5.

## What makes it hard

g is a Rastrigin function of the distance variables, scaled by 100. Each term of the sum, with
y = xᵢ − 0.5 in [−0.5, 0.5], has a local minimum near each of y = 0, ±0.1, ±0.2, …, ±0.5: 11 per
variable, and 11⁵ combinations for the five. One is the true front, g = 0. Each of the other
11⁵ − 1 = 161,050 is a local front: a plane parallel to the true one, where no small change of the
distance variables lowers g. The paper points out that each of them can attract an algorithm.

The nearest local fronts have one distance variable at 0.4 or 0.6 and the others at 0.5. There,
g is about 1, and the objectives sum to about 1: twice the true front's sum. To reach the true
front, that variable has to move by 0.1. Halfway, g is about 200 higher, so a child that moves
only part of the way is far worse than its parent and doesn't survive. The search needs a child
that jumps across the ridge in one step.

A random solution lies far from all of this: the non-dominated solutions of this run's initial
population have g from 98 to 683, objectives that sum to between 50 and 342. Once the population
reaches the true front, it still has to spread over the whole triangle.

## Representation

A `Real` genome of 7 genes in [0, 1]: the vector x. The problem is genoxide's `Dtlz1`, whose
fitness is the three objectives. In Python, `run` evaluates it in Rust.

## Algorithm

NSGA-III (Deb and Jain, 2014, IEEE Transactions on Evolutionary Computation 18(4): 577-601). Like
NSGA-II, it ranks solutions into non-dominated fronts, and parents and children compete for the
next population. Instead of crowding distance, it spreads the front along reference directions:
each solution joins the direction nearest to it, and directions with few members get more.
genoxide's docs recommend it for three or more objectives, and Deb and Jain test it on DTLZ1.

The settings are Deb and Jain's for DTLZ1 with 3 objectives:

- 91 reference directions from Das and Dennis's method (1998, SIAM Journal on Optimization 8(3):
  631-657) with 12 divisions: all points (a/12, b/12, c/12) with a + b + c = 12;
- a population of 92, the multiple of four just above 91;
- simulated binary crossover with η = 30, and polynomial mutation with η = 20 at a rate of 1/7 per
  gene, one gene per child on average;
- 400 generations, 36,892 evaluations.

The run goes through three phases. Until generation 80, the population descends through the local
fronts: the best g falls from 98 to about 1. The hypervolume stays 0, because no solution is yet
inside the box of the reference point. Between generations 88 and 120, the population crosses the
last ridges, and the median g falls from 1.5 to 0.02. By generation 170, the front covers the whole
triangle, with a hypervolume of 0.1392. The last 230 generations refine it: the largest g on the
front falls from 0.06 to 0.001, and the hypervolume reaches 0.1400.

## Output

Three lines. The first gives how many solutions are on the final front, and its hypervolume. The
hypervolume is the volume that the front dominates, up to a reference point. Larger is better. The
reference point here is (0.55, 0.55, 0.55), 1.1 times the nadir point (0.5, 0.5, 0.5), the worst
value of each objective on the front, as genoxide's benchmarks use for DTLZ1. For the whole front,
the hypervolume is 0.55³ − 0.5³/6 = 0.1455: the cube minus the corner that the plane cuts off.

The second gives the front's IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to 1,035
points of the optimal front, from genoxide's `optimal_front`: Das and Dennis's points with 44
divisions, halved. IGD+ averages, over those points, the distance to the nearest point of the
found front, counting only the objectives in which the found point is worse. 0 means that the found
front covers the optimal one. Smaller is better.

The third gives the largest g among the front's solutions, computed as 2 (f₁ + f₂ + f₃) − 1. It
shows convergence alone: 0 is on the true front, and about 1 is the nearest local front.

[The project page](https://tachsin.gr/projects/genoxide/examples/dtlz1-3obj) plays this run back.

## Good results

No finite set of solutions reaches 0.1455. NSGA-III aims at one solution per reference direction.
The 91 points where the directions meet the plane have a hypervolume of 0.1400 and an IGD+ of
0.0143.

The run's front has 92 solutions, a hypervolume of about 0.1400 and an IGD+ of about 0.0145: as good
as those 91 points. Every solution has g below 0.0012, far below the nearest local front's 1. Over
seeds 1 to 20, every run converges, with hypervolumes from 0.1387 to 0.1400 and IGD+ from 0.0144
to 0.0180.
