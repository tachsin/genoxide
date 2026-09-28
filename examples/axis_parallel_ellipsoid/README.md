---
title: Axis-parallel ellipsoid
category: continuous
summary: Minimize a sphere stretched along each axis in 30 dimensions, where each gene has its own scale.
reference: "Molga, M. and Smutnicki, C. (2005). Test functions for optimization needs."
reference_url: ""
optimum: "0 (at the origin)"
languages: [rust, python]
order: 42
trace_note: "Recorded from another run: CMA-ES in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Axis-parallel ellipsoid

## The problem

The axis-parallel hyper-ellipsoid weighs each gene's square by its position:

```text
f(x) = Σ i xᵢ²  (i from 1 to n),   each xᵢ in [−5.12, 5.12]
```

Its minimum is 0, at the origin. Here n = 30. The level sets are ellipsoids whose axes lie along
the genes: the first gene has weight 1, the last 30. It is the sphere stretched along each axis,
the more for the later genes.

Its origin is unknown. The earliest source genoxide's docs found is Pohlheim's GEATbx
documentation (function 1a, "the weighted sphere model"), which Molga and Smutnicki (2005, section
2.2) restate with the same bounds. It isn't among Schwefel's (1977) problems. It is not the rotated
hyper-ellipsoid, and not [Schwefel's problem 1.2](../schwefel_1_2/).

## What makes it hard

The genes have different scales. Where the curvature is largest, along the last gene, a step
changes f 30 times as much as the same step along the first gene. The ratio of the largest to the
smallest curvature, the condition number, is 30, and the ellipsoid's axes differ by a factor √30 ≈
5.5. An algorithm that takes steps of the same size in every gene has to make them small enough for
the steepest gene, and then crawls along the flattest one. What helps is a step size per gene.

The scaling is mild. The ellipsoid often used to test CMA-ES, with coefficients from 1 to 10⁶, has a
condition number of a million; this one has 30, in 30 dimensions. It still shows what scaling costs
each algorithm, compared with the [sphere](../sphere/), which has none.

The axes are aligned with the genes, so each gene can be treated on its own: the function is
separable. [Schwefel's problem 1.2](../schwefel_1_2/) is an ellipsoid whose axes are not.

## Representation

A `Real` genome of 30 genes, each in [−5.12, 5.12]: the point x itself. The fitness is f(x), to
minimize. The function is genoxide's `problems::AxisParallelEllipsoid`, which brings its bounds and
its minimum.

## Algorithm

The same algorithms as on the [sphere](../sphere/), CMA-ES with a full and with a diagonal
covariance matrix, PSO and a GA, each with a budget of 10,000 evaluations per dimension, 300,000 in
all, and a target of 1e-8.

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
of 14 from a normal distribution, and adapts its mean, its step size and its covariance matrix,
from a step size of 0.3 of each gene's range and a random start. The covariance matrix learns a
scale per gene, and would learn correlations between genes if there were any. There are no
restarts, since there is only one minimum.

sep-CMA-ES (Ros and Hansen, 2008, PPSN X: 296-305) is the same CMA-ES with a diagonal covariance
matrix, `.covariance(cmaes::Covariance::Diagonal)` in Rust and `covariance="diagonal"` in Python:
it learns only a scale per gene, which is all this function has, with learning rates (n + 2) / 3,
about 11, times as large.

Particle swarm optimization (Kennedy and Eberhart, 1995, Proceedings of ICNN'95: 1942-1948) moves
40 particles, each pulled towards its own best position and the swarm's, with Clerc and Kennedy's
constriction coefficients (2002, IEEE Transactions on Evolutionary Computation 6(1): 58-73). Each
gene of a velocity is updated on its own, with its own random weights, from that gene's distances
to the best positions: the steps in each gene shrink at their own pace.

A real-coded genetic algorithm: a population of 100, tournaments of 3, simulated binary crossover
(Deb and Agrawal, 1995, Complex Systems 9(2): 115-148) with η = 15, and polynomial mutation with η =
20 at a rate of 1/30 per gene. Both operators work gene by gene. The mutation's steps are a fraction
of each gene's range, and they don't adapt.

## Output

The first two lines give the dimension and the budget. Then a row per algorithm: the evaluations it
had used when its best error first reached 1, 1e-2, 1e-4, 1e-6 and 1e-8, and the best error it
found, to two significant digits. The error is the best value, since the minimum is 0. A dash is an
error not reached. The function has no `sin`, `cos` or `exp`, so the runs, and their counts, are
the same on every platform. In Python, `run` evaluates the function in Rust, so both versions print
the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/axis-parallel-ellipsoid) plays
back another run: CMA-ES on the ellipsoid in 2 dimensions, with weights 1 and 2, so that the
population can be drawn on the function's contour.

## Good results

The minimum is 0. CMA-ES reaches 1e-8 after 6,622 evaluations, against 4,900 on the sphere. From an
error of 1 on, it needs about 590 evaluations per decade, against 370 on the sphere: its covariance
matrix takes time to learn 30 scales, and the error falls faster once it has.

PSO reaches 1e-8 after 21,800 evaluations, about 2,000 per decade, as on the sphere: the scaling
doesn't slow it. The genetic algorithm reaches 1e-4 after about 186,000 evaluations and ends at
1.6e-5, short of the target, as its steps don't shrink.

Here a diagonal covariance matrix is all the problem needs, and it learns faster. sep-CMA-ES
reaches 1e-8 after 4,018 evaluations, 40% fewer than the full matrix: from an error of 1 on, about
340 evaluations per decade, as on the sphere, so the scaling costs it nothing once it has learned
the scales. On [Schwefel's problem 1.2](../schwefel_1_2/), where the axes are rotated, the diagonal
matrix is far slower.
