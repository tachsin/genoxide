---
title: Trid
category: continuous
summary: Minimize the Trid function in 10 dimensions, a convex quadratic whose genes are coupled in a chain and whose minimum and bounds grow with the dimension, and compare how fast CMA-ES, sep-CMA-ES, DE, PSO and a GA close in on it.
reference: "Laguna, M. and Martí, R. (2005). Experimental testing of advanced scatter search designs for global optimization of multimodal functions. Journal of Global Optimization 33(2): 235-255."
reference_url: "https://doi.org/10.1007/s10898-004-1936-z"
optimum: "−n (n + 4) (n − 1) / 6 at xᵢ = i (n + 1 − i): −210 for n = 10"
languages: [rust, python]
order: 47
trace_note: "Recorded from another run: CMA-ES in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Trid

## The problem

The Trid function is a quadratic whose genes are coupled in a chain, to minimize:

```text
f(x) = Σᵢ₌₁ⁿ (xᵢ − 1)² − Σᵢ₌₂ⁿ xᵢ xᵢ₋₁,   each xᵢ in [−n², n²]
```

Here n = 10, so the bounds are [−100, 100].

Its origin is unknown: the earliest source found is Hedar's collection of global optimization test
problems, which Jamil and Yang (2013, functions 150 and 151, Trid 6 and Trid 10) credit. genoxide
takes the definition and the bounds from Laguna and Martí (2005, functions 24 and 25), whose second
sum prints xᵢxⱼ for xᵢxᵢ₋₁, and they are still to be checked against an original.

The minimum depends on n: −n (n + 4) (n − 1) / 6, at xᵢ = i (n + 1 − i). For n = 10, it's −210 at
(10, 18, 24, 28, 30, 30, 28, 24, 18, 10), and for n = 6, −50, as Laguna and Martí give; Jamil and
Yang give −200 for n = 10, a misprint. It's the only minimum: the Hessian is tridiagonal, with 2 on
the diagonal and −1 beside it, which is positive definite, and the gradient 2 (xᵢ − 1) − xᵢ₋₁ − xᵢ₊₁
is 0 there. genoxide's `problems::Trid` gives it as proven.

## What makes it hard

Not its shape, which is a bowl, but its scale and its coupling. The Hessian's eigenvalues are 2 − 2
cos(kπ/11) for k = 1 … 10, from 0.081 to 3.92: a condition number of 48, and the flattest direction
is the one along which all the genes rise and fall together, in a hump like the minimizer's. Every
gene is coupled to its neighbors, so no gene can be minimized alone. And the minimum is away from
the middle of the box, with genes up to 30.

## Representation

A `Real` genome of 10 genes, each in [−100, 100]: the point x itself. The fitness is f(x), to
minimize. The function is genoxide's `problems::Trid`, which brings its bounds and its minimum.

## Algorithm

Five algorithms, each with a budget of 10,000 evaluations per dimension, 100,000 in all, and a
target of 1e-8 above the minimum, from seed 1:

- CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195), which samples a
  population of 10 from a normal distribution and adapts its mean, its step size and its covariance
  matrix, from a step size of 0.3 of each gene's range and a random start;
- sep-CMA-ES (Ros and Hansen, 2008, PPSN X: 296-305), the same with a diagonal covariance matrix: a
  scale per gene but no correlations;
- differential evolution with genoxide's defaults, SHADE (Tanabe and Fukunaga, CEC 2013), with a
  population of 100;
- particle swarm optimization (Kennedy and Eberhart, 1995), 40 particles with Clerc and Kennedy's
  constriction coefficients and a global topology;
- a real-coded genetic algorithm: a population of 100, tournaments of 3, simulated binary crossover
  (Deb and Agrawal, 1995) with η = 15 and polynomial mutation with η = 20 at a rate of 1/10 per
  gene.

## Output

The first two lines give the dimension and the budget. Then a row per algorithm: the evaluations it
had used when its best error, the value minus −210, first reached 1, 1e-2, 1e-4, 1e-6 and 1e-8, and
the best error it found, to two significant digits. A dash is an error not reached. The function has
no `sin`, `cos` or `exp`, so the runs are the same on every platform, and in Python, `run` evaluates
the function in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/trid) plays back another run:
CMA-ES on the function in 2 dimensions, where the bounds are [−4, 4] and the minimum −2 at (2, 2),
so that the population can be drawn on its contour.

## Good results

The minimum is −210. CMA-ES reaches it to within 1e-8 after 2,180 evaluations, its full covariance
matrix learning the coupled shape of the bowl. sep-CMA-ES, which can't learn the coupling, takes
9,050, four times as many. PSO takes 29,440 and SHADE 33,600.

The genetic algorithm reaches an error of 1 after 57,074 evaluations and ends at 0.34: its crossover
and mutation work gene by gene, with steps that don't shrink with the error, on a bowl whose
flattest direction moves all the genes at once.
