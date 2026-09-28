---
title: Schwefel 1.2
category: continuous
summary: Minimize a sum of squared partial sums in 30 dimensions, a rotated ellipsoid whose genes interact.
reference: "Schwefel, H.-P. (1981). Numerical Optimization of Computer Models. Wiley. Problem 1.2."
reference_url: ""
optimum: "0 (at the origin)"
languages: [rust, python]
order: 53
family: Schwefel
trace_note: "Recorded from another run: CMA-ES in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Schwefel 1.2

## The problem

Schwefel's problem 1.2 sums the squares of the partial sums of the genes:

```text
f(x) = Σᵢ (Σⱼ≤ᵢ xⱼ)²  (i from 1 to n),   each xᵢ in [−100, 100]
```

Its minimum is 0, at the origin. Here n = 30. The first term is x₁², the second (x₁ + x₂)², and the
last the square of the sum of all 30 genes.

It is problem 1.2 of Schwefel's *Numerical Optimization of Computer Models* (1981, Wiley), the
translation of *Numerische Optimierung von Computer-Modellen* (1977, Birkhäuser, p. 319), which
states it in any dimension and unbounded. The bounds and the 30 dimensions here are those of Yao,
Liu and Lin (1999, IEEE Transactions on Evolutionary Computation 3(2): 82-102, f3), which genoxide
follows.

## What makes it hard

The genes interact: x₁ appears in every term, x₂ in all but the first, and so on. The best value
of one gene depends on all the others. f is a quadratic, so its level sets are ellipsoids, as those
of the [axis-parallel ellipsoid](../axis_parallel_ellipsoid/) are, but their axes don't lie along
the genes: it is a rotated ellipsoid. In 30 dimensions its condition number, the ratio of the
largest to the smallest curvature, is about 1,500. In 2 dimensions, f = x₁² + (x₁ + x₂)², it is
6.9, and the contour is a tilted ellipse: its long axis points along (1, −1.62), about 58° from
the x₁ axis.

Rotation is what separates the algorithms. One that adapts a step size per gene can stretch its
steps along the axes, but not along a diagonal: on a rotated ellipsoid it has to take steps small
enough for the steepest direction, in every gene. Only a full covariance matrix, which learns the
correlations between genes, can line its steps up with the ellipsoid.

The start is also far away. At a random point of the box, the partial sums grow with i, and f is
about 465 × 100²/3 ≈ 1.5 × 10⁶, fifteen times what the [sphere](../sphere/) starts from.

## Representation

A `Real` genome of 30 genes, each in [−100, 100]: the point x itself. The fitness is f(x), to
minimize. The function is genoxide's `problems::Schwefel1_2`, which brings its bounds and its
minimum.

## Algorithm

The same algorithms as on the [sphere](../sphere/), CMA-ES with a full and with a diagonal
covariance matrix, PSO and a GA, each with a budget of 10,000 evaluations per dimension, 300,000 in
all, and a target of 1e-8.

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
of 14 from a normal distribution, and adapts its mean, its step size and its covariance matrix,
from a step size of 0.3 of each gene's range and a random start. Its full covariance matrix learns
the correlations between the genes. CMA-ES behaves the same on a function and on any rotation of
it, once the matrix has adapted, so the rotation costs it only the time to learn it. There are no
restarts, since there is only one minimum.

sep-CMA-ES (Ros and Hansen, 2008, PPSN X: 296-305) is the same CMA-ES with a diagonal covariance
matrix, `.covariance(cmaes::Covariance::Diagonal)` in Rust and `covariance="diagonal"` in Python.
It learns a scale per gene but no correlations, so it can't follow the rotation: like PSO, it
steps along the axes.

Particle swarm optimization (Kennedy and Eberhart, 1995, Proceedings of ICNN'95: 1942-1948) moves
40 particles, each pulled towards its own best position and the swarm's, with Clerc and Kennedy's
constriction coefficients (2002, IEEE Transactions on Evolutionary Computation 6(1): 58-73). Each
gene of a velocity is updated on its own, with its own random weights, so the swarm's moves depend
on the axes.

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

[The project page](https://tachsin.gr/projects/genoxide/examples/schwefel-1-2) plays back another
run: CMA-ES on Schwefel 1.2 in 2 dimensions, so that the population can be drawn on the function's
contour.

## Good results

The minimum is 0. CMA-ES reaches 1e-8 after 13,062 evaluations. Most of them go to learning the
covariance matrix: it takes 7,098 to reach an error of 1, and then about 750 per decade, twice as
many as on the [sphere](../sphere/).

PSO reaches the target too, after 171,560 evaluations: about 14,000 per decade, seven times as many
as on the sphere and on the axis-parallel ellipsoid. The genetic algorithm never reaches an error
of 1, and ends at 14.

sep-CMA-ES, whose diagonal matrix can't follow the rotation, takes 61,810 evaluations, almost five
times as many as the full matrix: 19,796 to reach an error of 1, and then about 5,300 per decade,
seven times as many. On the [axis-parallel ellipsoid](../axis_parallel_ellipsoid/) it was the
faster of the two. It still beats PSO, which also steps along the axes, by almost three times.
