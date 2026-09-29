---
title: DTLZ2 with 3 objectives
category: multi-objective
summary: Minimize three conflicting objectives whose Pareto front is an eighth of the unit sphere, with NSGA-III and 91 or 703 reference directions.
reference: "Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2002). Scalable multi-objective optimization test problems. Proceedings of the 2002 Congress on Evolutionary Computation, pp. 825-830."
reference_url: https://doi.org/10.1109/CEC.2002.1007032
optimum: "hypervolume 0.8074 (reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 136
family: DTLZ
tab: DTLZ2
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
NSGA-II, it ranks solutions into non-dominated fronts, and parents and children compete for the next
population. Instead of crowding distance, it spreads the front along reference directions: each
solution joins the direction nearest to it, and directions with few members get more.

The example runs it twice, with Deb and Jain's operators for DTLZ2 with 3 objectives: simulated
binary crossover with η = 30, and polynomial mutation with η = 20 at a rate of 1/12 per gene, one
gene per child on average. The reference directions come from Das and Dennis's method (1998, SIAM
Journal on Optimization 8(3): 631-657): all points (a/H, b/H, c/H) with a + b + c = H, for H
divisions.

- Deb and Jain's settings: 12 divisions, 91 directions, a population of 92, the multiple of four
  just above 91, and 250 generations, 23,092 evaluations.
- 36 divisions: 703 directions, a population of 703, one solution per direction, and 250
  generations, 176,453 evaluations.

The second run is there for the target: a front within 1% of the optimal one, measured by IGD+ (see
Good results). NSGA-III aims at one solution per direction, and 91 points can't cover an eighth of a
sphere that closely, however well they converge. 703 can.

## Output

A line per run: how many solutions are on its final front, the front's hypervolume and its IGD+. The
last line gives the whole front's hypervolume.

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

[The project page](https://tachsin.gr/projects/genoxide/examples/dtlz2-3obj) plays the second
run back.

## Good results

The target is a front close to the whole optimal one: an IGD+ of at most 0.01, the front within
about 1% of the objectives' range, or a hypervolume of at least 99% of the whole front's, 0.7993. No
finite set reaches 0.8074, and neither target can be met with 91 points: the 91 points where the
directions meet the sphere have a hypervolume of 0.7449 and an IGD+ of 0.0221. For an IGD+ of 0.01
it takes about 450 points evenly spread over the sphere; the hypervolume needs far more, since 1,225
points still give only 98.0% of it.

With Deb and Jain's settings, the run's front has 92 solutions, a hypervolume of 0.7443 and an IGD+
of 0.0224: as good as those 91 points. Over seeds 1 to 20, the hypervolume is 0.7433 to 0.7443 and
the IGD+ 0.0224 to 0.0230.

With 703 directions, the front has 703 solutions, a hypervolume of 0.7854 (97.3% of the whole
front's) and an IGD+ of 0.0079: within the target. Over seeds 1 to 20, every run reaches it, with
IGD+ from 0.0079 to 0.0080 and hypervolumes from 0.7852 to 0.7854.
