---
title: DTLZ1 with 3 objectives
category: multi-objective
summary: Minimize three conflicting objectives whose Pareto front is a triangle in a plane, behind 11⁵ − 1 local fronts, with NSGA-III and 91 or 861 reference directions.
reference: "Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2002). Scalable multi-objective optimization test problems. Proceedings of the 2002 Congress on Evolutionary Computation, pp. 825-830."
reference_url: https://doi.org/10.1109/CEC.2002.1007032
optimum: "the plane f₁ + f₂ + f₃ = 0.5; hypervolume 0.1455 (reference point (0.55, 0.55, 0.55))"
languages: [rust, python]
order: 155
family: DTLZ
tab: DTLZ1
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

A random solution lies far from all of this: the non-dominated solutions of the first run's initial
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

The example runs it twice, with Deb and Jain's operators for DTLZ1 with 3 objectives: simulated
binary crossover with η = 30, and polynomial mutation with η = 20 at a rate of 1/7 per gene, one
gene per child on average. The reference directions come from Das and Dennis's method (1998, SIAM
Journal on Optimization 8(3): 631-657): all points (a/H, b/H, c/H) with a + b + c = H, for H
divisions.

- Deb and Jain's settings: 12 divisions, 91 directions, a population of 92, the multiple of four
  just above 91, and 400 generations, 36,892 evaluations.
- 40 divisions: 861 directions, a population of 861, one solution per direction, and 300
  generations, 259,161 evaluations.

The second run is there for the target: a front within 1% of the optimal one, measured by IGD+
(see Good results). NSGA-III aims at one solution per direction, and 91 points can't cover a
triangle that closely, however well they converge. 861 can.

With 861 directions, the run goes through three phases. Until generation 38, the population descends
through the local fronts: the best g falls from 95 to 0.13. From generation 41, some solutions are
inside the box of the reference point, and the hypervolume grows. Between generations 80 and 89, the
front grows from 292 solutions to all 861, and by generation 100 it covers the whole triangle, with
a hypervolume of 0.1423. The last 200 generations refine it: the median g on the front falls from
0.019 to 0.0002.

## Output

A line per run: how many solutions are on its final front, the front's hypervolume, its IGD+ and
the largest g among its solutions. The last line gives the whole front's hypervolume.

The hypervolume is the volume that the front dominates, up to a reference point. Larger is better.
The reference point here is (0.55, 0.55, 0.55), 1.1 times the nadir point (0.5, 0.5, 0.5), the
worst value of each objective on the front, as genoxide's benchmarks use for DTLZ1. For the whole
front, the hypervolume is 0.55³ − 0.5³/6 = 0.1455: the cube minus the corner that the plane cuts
off.

IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) is measured to 1,035 points of the
optimal front, from genoxide's `optimal_front`: Das and Dennis's points with 44 divisions, halved.
It averages, over those points, the distance to the nearest point of the found front, counting
only the objectives in which the found point is worse. 0 means that the found front covers the
optimal one. Smaller is better. Scaled, it is divided by 0.5, the range of each objective on the
front: the IGD+ of the front with every objective scaled to [0, 1].

The largest g is computed as 2 (f₁ + f₂ + f₃) − 1. It shows convergence alone: 0 is on the true
front, and about 1 is the nearest local front.

[The project page](https://tachsin.gr/projects/genoxide/examples/dtlz1-3obj) plays the second
run back.

## Good results

The target is a front close to the whole optimal one: a scaled IGD+ of at most 0.01, the front
within about 1% of the objectives' range, or a hypervolume of at least 99% of the whole front's,
0.1440. No finite set reaches 0.1455, and neither target can be met with 91 points: the 91 points
where the directions meet the plane have a hypervolume of 0.1400 and a scaled IGD+ of 0.0286. For a
scaled IGD+ of 0.01 it takes about 700 points evenly spread over the triangle, and for 99% of the
hypervolume about 1,100.

With Deb and Jain's settings, the run's front has 92 solutions, a hypervolume of 0.1400 and a
scaled IGD+ of 0.0289: as good as those 91 points. Every solution has g below 0.0012, far below the
nearest local front's 1. Over seeds 1 to 20, every run converges, with hypervolumes from 0.1387 to
0.1400 and IGD+ from 0.0144 to 0.0180, scaled 0.0288 to 0.0360.

With 861 directions, the front has 861 solutions, a hypervolume of 0.1439 (98.9% of the whole
front's) and a scaled IGD+ of 0.0085: within the target. Its largest g is 0.0039. Over seeds 1 to
20, every run reaches the target, with scaled IGD+ from 0.0084 to 0.0089 and hypervolumes from
0.1438 to 0.1439. After 200 generations, 19 of the 20 already have.
