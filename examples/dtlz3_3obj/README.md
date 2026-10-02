---
title: DTLZ3 with 3 objectives
category: multi-objective
summary: Minimize three conflicting objectives whose Pareto front is an eighth of the unit sphere, behind 3¹⁰ − 1 local fronts, with NSGA-III and 91 or 703 reference directions.
reference: "Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2002). Scalable multi-objective optimization test problems. Proceedings of the 2002 Congress on Evolutionary Computation, pp. 825-830."
reference_url: https://doi.org/10.1109/CEC.2002.1007032
optimum: "the unit sphere's eighth with f ≥ 0; hypervolume 0.8074 (reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 177
family: DTLZ
tab: DTLZ3
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
solutions of the first run's initial population have g from 423 to 1,110. And the whole population
can settle on a local front, since every solution there is non-dominated among the others.

## Representation

A `Real` genome of 12 genes in [0, 1]: the vector x. The problem is genoxide's `Dtlz3`, whose
fitness is the three objectives. In Python, `run` evaluates it in Rust.

## Algorithm

NSGA-III (Deb and Jain, 2014, IEEE Transactions on Evolutionary Computation 18(4): 577-601). Like
NSGA-II, it ranks solutions into non-dominated fronts, and parents and children compete for the next
population. Instead of crowding distance, it spreads the front along reference directions: each
solution joins the direction nearest to it, and directions with few members get more. genoxide's
docs recommend it for three or more objectives, and Deb and Jain test it on DTLZ3.

The example runs it twice, with Deb and Jain's operators for DTLZ3 with 3 objectives: simulated
binary crossover with η = 30, and polynomial mutation with η = 20 at a rate of 1/12 per gene, one
gene per child on average. The reference directions come from Das and Dennis's method (1998, SIAM
Journal on Optimization 8(3): 631-657): all points (a/H, b/H, c/H) with a + b + c = H, for H
divisions.

- Deb and Jain's settings: 12 divisions, 91 directions, a population of 92, the multiple of four
  just above 91, and 1,000 generations, 92,092 evaluations.
- 36 divisions: 703 directions, a population of 703, one solution per direction, and 600
  generations, 422,503 evaluations.

The second run is there for the target: a front within 1% of the optimal one, measured by IGD+ (see
Good results). NSGA-III aims at one solution per direction, and 91 points can't cover an eighth of a
sphere that closely, however well they converge. 703 can.

The first run shows the trap. By generation 350, the front has reached a local front with g ≈ 1.07;
at generation 400, every solution on it has x₄ near 0.4 and the other distance variables near 0.5.
The population stays there for over 60 generations. The hypervolume is 0 all this time: on a sphere
of radius 2, no point has all three objectives below the reference point's 1.1. At generation 415, a
child lands across the ridge, the best g falls from 1.01 to 0.03, and the population follows. By
generation 500, the front covers the whole eighth of the sphere, with a hypervolume of 0.72.

The second run escapes sooner, with more children a generation to try. By generation 180, its best
solution is on a local front with g ≈ 1; by generation 200, one is across the ridge, with g = 0.10.
From generation 260 on, all 703 solutions are non-dominated and the hypervolume is 0.72. The last
340 generations refine the front: the largest g on it falls from 0.066 to 0.0008, and the IGD+ from
0.030 to 0.0078.

## Output

A line per run: how many solutions are on its final front, the front's hypervolume, its IGD+ and the
largest g among its solutions. The last line gives the whole front's hypervolume.

The hypervolume is the volume that the front dominates, up to a reference point. Larger is better.
The reference point here is (1.1, 1.1, 1.1), 1.1 times the nadir point (1, 1, 1), the worst value of
each objective on the front. For the whole front, the hypervolume is 1.1³ − π/6 = 0.8074: the cube
minus the eighth of the unit ball.

IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) is measured to 1,035 points of the
optimal front, from genoxide's `optimal_front`: Das and Dennis's points with 44 divisions, projected
onto the sphere. It averages, over those points, the distance to the nearest point of the found
front, counting only the objectives in which the found point is worse. 0 means that the found front
covers the optimal one. Smaller is better. Each objective spans 1 on the front, so IGD+ is already
on the scale of the front's range.

The largest g is computed as √(f₁² + f₂² + f₃²) − 1. It shows convergence alone: 0 is on the true
front, and about 1 is the nearest local front.

[The project page](https://tachsin.gr/projects/genoxide/examples/dtlz3-3obj) plays the second
run back.

## Good results

The target is a front close to the whole optimal one: an IGD+ of at most 0.01, the front within
about 1% of the objectives' range, or a hypervolume of at least 99% of the whole front's, 0.7993. No
finite set reaches 0.8074, and neither target can be met with 91 points: the 91 points where the
directions meet the sphere have a hypervolume of 0.7449 and an IGD+ of 0.0221. For an IGD+ of 0.01
it takes about 450 points evenly spread over the sphere; the hypervolume needs far more, since 1,225
points still give only 98.0% of it.

With Deb and Jain's settings, the run's front has 92 solutions, a hypervolume of 0.7435 and an IGD+
of 0.0227. Every solution has g below 0.00094: on the true front, not a local one. Over seeds 1 to
20, every run escapes the local fronts, between generations 230 and 580, and ends with a hypervolume
from 0.7171 to 0.7435 and an IGD+ from 0.0227 to 0.0350.

With 703 directions, the front has 703 solutions, a hypervolume of 0.7856 (97.3% of the whole
front's), an IGD+ of 0.0078 and a largest g of 0.00083: within the target. Over seeds 1 to 20, every
run reaches it, with IGD+ from 0.0078 to 0.0085 and hypervolumes from 0.7840 to 0.7857. After 500
generations, all 20 are already within the target, the worst at 0.0098.
