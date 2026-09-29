---
title: Sphere
category: continuous
summary: Minimize the sum of the squares of 30 genes, and compare how fast CMA-ES, with a full and a diagonal covariance matrix, PSO and a GA close in on the minimum.
reference: "De Jong, K. A. (1975). An Analysis of the Behavior of a Class of Genetic Adaptive Systems. PhD thesis, University of Michigan."
reference_url: "https://hdl.handle.net/2027.42/4507"
optimum: "0 (at the origin)"
languages: [rust, python]
order: 41
trace_note: "Recorded from another run: CMA-ES in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Sphere

## The problem

The sphere is the sum of the squares of the genes:

```text
f(x) = Σ xᵢ²,   each xᵢ in [−100, 100]
```

Its minimum is 0, at the origin. Here n = 30. The value is the squared distance to the origin, so
the level sets are spheres around it: every direction is alike, and every gene counts the same.

It is De Jong's F1 (1975). De Jong used it in 3 dimensions on [−5.12, 5.12], as Pohlheim (GEATbx)
and Laguna and Martí (2005) restate it, and Schwefel (1977, problem 1.1) states it in any
dimension, unbounded. The bounds and the 30 dimensions here are those of Yao, Liu and Lin (1999,
IEEE Transactions on Evolutionary Computation 3(2): 82-102, f1), which genoxide follows. genoxide's
docs note that the definition is not yet checked against De Jong's thesis.

## What makes it hard

Nothing, in a sense: the sphere has one minimum, no plateaus, and no gene depends on another. That
is what it is for. It is the baseline of the unimodal functions: it measures how fast an algorithm
converges, when nothing else gets in the way. The axis-parallel ellipsoid, Schwefel's problem 1.2
and Zakharov's function each add one difficulty to it, and their examples compare against this one.

Convergence is measured in evaluations per decade: how many it takes to divide the error by 10.
An algorithm that shrinks its steps as it closes in needs about the same number for every decade.
One whose steps don't shrink gets slower and slower, and stops improving at some precision.

The 30 dimensions make the difference visible. In 2 dimensions almost any method finds the origin
quickly.

## Representation

A `Real` genome of 30 genes, each in [−100, 100]: the point x itself. The fitness is f(x), to
minimize. The function is genoxide's `problems::Sphere`, which brings its bounds and its minimum.

## Algorithm

Three algorithms, one of them in two variants, each with a budget of 10,000 evaluations per
dimension, 300,000 in all, and a target of 1e-8.

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, its step size and its covariance matrix. It uses
genoxide's defaults: a population of 4 + ⌊3 ln 30⌋ = 14, a step size of 0.3 of each gene's range,
and a random start. There are no restarts, since there is only one minimum. As the samples close
in, the step size shrinks with them.

sep-CMA-ES (Ros and Hansen, 2008, PPSN X: 296-305) is the same CMA-ES with a diagonal covariance
matrix, `.covariance(cmaes::Covariance::Diagonal)` in Rust and `covariance="diagonal"` in Python.
It learns a variance per gene but no correlations between genes, and each sample costs O(n)
instead of O(n²). With fewer entries to learn, its learning rates are (n + 2) / 3 times as large,
about 11 times here. It has the same population, initial step size and seed.

Particle swarm optimization (Kennedy and Eberhart, 1995, Proceedings of ICNN'95: 1942-1948) moves
40 particles, each pulled towards its own best position and the swarm's. It uses Clerc and
Kennedy's constriction coefficients (2002, IEEE Transactions on Evolutionary Computation 6(1):
58-73), genoxide's defaults. The velocities shrink as the particles gather.

A real-coded genetic algorithm: a population of 100, tournaments of 3, simulated binary crossover
(Deb and Agrawal, 1995, Complex Systems 9(2): 115-148) with η = 15, and polynomial mutation with η =
20 at a rate of 1/30 per gene, one gene per child on average. These are the usual settings that
genoxide's docs give for real genes. The mutation's steps are a fraction of each gene's range, and
they don't adapt.

## Output

The first two lines give the dimension and the budget. Then a row per algorithm: the evaluations it
had used when its best error first reached 1, 1e-2, 1e-4, 1e-6 and 1e-8, and the best error it
found, to two significant digits. The error is the best value, since the minimum is 0. A dash is an
error not reached. The counts are taken after each generation, so they are multiples of the
population size for CMA-ES, sep-CMA-ES and PSO. The function has no `sin`, `cos` or `exp`, so the runs, and
their counts, are the same on every platform. In Python, `run` evaluates the function in Rust, so
both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/sphere) plays back another run:
CMA-ES on the sphere in 2 dimensions, so that the population can be drawn on the function's
contour.

## Good results

The minimum is 0. CMA-ES reaches 1e-8 after 4,900 evaluations. From an error of 1 on, it needs
about 370 evaluations per decade, 27 generations, and the same for each decade: its step size
shrinks at the rate the error does. PSO also converges at a steady rate, about 2,000 evaluations
per decade, and reaches 1e-8 after 24,720: five times as many.

The genetic algorithm reaches 1e-2 after about 90,000 evaluations, and then stalls: after all
300,000 its best error is 1.8e-4. Its polynomial mutation makes steps of a fixed fraction of the
range, 200 wide here, and selection alone narrows the population only slowly. On the
[axis-parallel ellipsoid](../axis_parallel_ellipsoid/), whose genes span 10.24, the same GA gets to
1.6e-5: about the same precision relative to the range.

A diagonal covariance matrix is enough here: the sphere has no correlations to learn. sep-CMA-ES
reaches 1e-8 after 4,564 evaluations, 7% fewer than the full matrix: it gets to an error of 1
sooner, and then needs about 360 evaluations per decade, against 370. The other unimodal examples show where the full matrix matters.
