---
title: Zakharov
category: continuous
summary: Minimize a sum of squares plus the square and fourth power of a weighted sum, in 30 dimensions, where one oblique direction is far steeper than the rest.
reference: "Laguna, M. and Martí, R. (2005). Experimental testing of advanced scatter search designs for global optimization of multimodal functions. Journal of Global Optimization 33(2): 235-255."
reference_url: "https://doi.org/10.1007/s10898-004-1936-z"
optimum: "0 (at the origin)"
languages: [rust, python]
order: 46
trace_note: "Recorded from another run: CMA-ES in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Zakharov

## The problem

Zakharov's function adds to a sphere the square and the fourth power of a weighted sum of the
genes:

```text
f(x) = Σ xᵢ² + S² + S⁴,   S = Σ 0.5 i xᵢ  (i from 1 to n),   each xᵢ in [−5, 10]
```

Its minimum is 0, at the origin. Here n = 30. The weights of S grow with the position of the gene,
from 0.5 for the first to 15 for the last.

Its origin is unknown. genoxide takes the definition and the bounds from Laguna and Martí (2005,
function 12), and its docs note that they are not yet checked against an original. The bounds are
not symmetric: the origin is a third of the way from the lower bound, not in the middle of the box.

## What makes it hard

Each term is convex, so f is too: it has one minimum and no plateaus. What makes it hard is its
scaling, in two ways.

Far from the minimum, the fourth power dominates. At a random point of the box, S is about 580, and
f about 10¹¹. The values span eleven decades between the start and an error of 1, and a step that
changes S a little changes f a lot.

Near the minimum, the fourth power vanishes and f is a quadratic. Along the direction of the
weights, w = (0.5, 1, …, 15), it curves 1 + |w|² = 2,365 times as much as in any direction at right
angles to it: a narrow valley around the hyperplane S = 0. That direction mixes all 30 genes, so,
as on [Schwefel's problem 1.2](../schwefel_1_2/), the genes interact, and an algorithm with a step
size per gene can't line its steps up with the valley. In 2 dimensions, the ratio is only 2.25.

## Representation

A `Real` genome of 30 genes, each in [−5, 10]: the point x itself. The fitness is f(x), to minimize.
The function is genoxide's `problems::Zakharov`, which brings its bounds and its minimum.

## Algorithm

The same algorithms as on the [sphere](../sphere/), CMA-ES with a full and with a diagonal
covariance matrix, PSO and a GA, each with a budget of 10,000 evaluations per dimension, 300,000 in
all, and a target of 1e-8.

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
of 14 from a normal distribution, and adapts its mean, its step size and its covariance matrix,
from a step size of 0.3 of each gene's range and a random start. It uses only the ranking of the
samples, not their values, so the eleven decades of the quartic don't matter to it, only the shape
of the level sets. Its full covariance matrix can learn the oblique valley. There are no restarts,
since there is only one minimum.

sep-CMA-ES (Ros and Hansen, 2008, PPSN X: 296-305) is the same CMA-ES with a diagonal covariance
matrix, `.covariance(cmaes::Covariance::Diagonal)` in Rust and `covariance="diagonal"` in Python.
It learns a scale per gene but no correlations, so it can't line its steps up with the valley.

Particle swarm optimization (Kennedy and Eberhart, 1995, Proceedings of ICNN'95: 1942-1948) moves
40 particles, each pulled towards its own best position and the swarm's, with Clerc and Kennedy's
constriction coefficients (2002, IEEE Transactions on Evolutionary Computation 6(1): 58-73). Each
gene of a velocity is updated on its own, with its own random weights.

A real-coded genetic algorithm: a population of 100, tournaments of 3, simulated binary crossover
(Deb and Agrawal, 1995, Complex Systems 9(2): 115-148) with η = 15, and polynomial mutation with η =
20 at a rate of 1/30 per gene. Both operators work gene by gene, and the mutation's steps don't
adapt.

## Output

The first two lines give the dimension and the budget. Then a row per algorithm: the evaluations it
had used when its best error first reached 1, 1e-2, 1e-4, 1e-6 and 1e-8, and the best error it
found, to two significant digits. The error is the best value, since the minimum is 0. A dash is an
error not reached. The function has no `sin`, `cos` or `exp`, so the runs, and their counts, are
the same on every platform. In Python, `run` evaluates the function in Rust, so both versions print
the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/zakharov) plays back another run:
CMA-ES on Zakharov's function in 2 dimensions, so that the population can be drawn on the
function's contour.

## Good results

The minimum is 0. CMA-ES reaches 1e-8 after 13,692 evaluations: 8,484 to get down to an error of 1,
then about 650 per decade, 1.7 times as many as on the [sphere](../sphere/) and a little fewer than
on Schwefel's problem 1.2.

PSO reaches the target after 117,760 evaluations, about 9,400 per decade, almost five times as many
as on the sphere. The genetic algorithm reaches an error of 1 after about 85,000 evaluations and
ends at 0.025.

sep-CMA-ES, whose diagonal matrix can't line up with the valley either, takes 103,306 evaluations,
about seven and a half times as many as the full matrix: 36,022 to reach an error of 1, and then
about 8,400 per decade, 13 times as many. That is closer to PSO than to the full matrix.
