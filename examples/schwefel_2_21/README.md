---
title: Schwefel 2.21
category: continuous
summary: Minimize the largest absolute value of 30 genes, where only the worst gene counts, and compare how fast CMA-ES, sep-CMA-ES, DE, PSO and a GA close in on the minimum.
reference: "Schwefel, H.-P. (1981). Numerical Optimization of Computer Models. Wiley. Problem 2.21."
reference_url: ""
optimum: "0 (at the origin)"
languages: [rust, python]
order: 44
family: Schwefel
trace_note: "Recorded from another run: CMA-ES in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Schwefel 2.21

## The problem

Schwefel's problem 2.21 is the largest absolute value of the genes, to minimize:

```text
f(x) = maxᵢ |xᵢ|,   each xᵢ in [−100, 100]
```

Its minimum is 0, at the origin. Here n = 30. It's Schwefel's (1981) problem 2.21; genoxide takes
the definition, the bounds and the dimension from Yao, Liu and Lin's (1999, f4) restatement, and
they are still to be checked against Schwefel's book.

## What makes it hard

The function is unimodal and convex, but only one gene counts at a time: the one farthest from 0. A
step that improves any other gene changes nothing, and a step that improves the worst gene helps
only until it passes the second worst. So the function has plateaus in every direction but a few,
and its level sets are cubes, with edges and corners where it isn't differentiable. To lower f by a
factor of 10, all 30 genes must shrink by 10 together.

An algorithm that ranks its samples, as all the ones here do, sees ties between samples that differ
only in their genes that don't count, and learns nothing from them.

## Representation

A `Real` genome of 30 genes, each in [−100, 100]: the point x itself. The fitness is f(x), to
minimize. The function is genoxide's `problems::Schwefel2_21`, which brings its bounds and its
minimum.

## Algorithm

Five algorithms, each with a budget of 10,000 evaluations per dimension, 300,000 in all, and a
target of 1e-8, from seed 1:

- CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195), which samples a
  population of 14 from a normal distribution and adapts its mean, its step size and its covariance
  matrix, from a step size of 0.3 of each gene's range and a random start;
- sep-CMA-ES (Ros and Hansen, 2008, PPSN X: 296-305), the same with a diagonal covariance matrix: a
  scale per gene but no correlations;
- differential evolution with genoxide's defaults, SHADE (Tanabe and Fukunaga, CEC 2013), with a
  population of 100;
- particle swarm optimization (Kennedy and Eberhart, 1995), 40 particles with Clerc and Kennedy's
  constriction coefficients and a global topology;
- a real-coded genetic algorithm: a population of 100, tournaments of 3, simulated binary crossover
  (Deb and Agrawal, 1995) with η = 15 and polynomial mutation with η = 20 at a rate of 1/30 per
  gene.

## Output

The first two lines give the dimension and the budget. Then a row per algorithm: the evaluations it
had used when its best error first reached 1, 1e-2, 1e-4, 1e-6 and 1e-8, and the best error it
found, to two significant digits. A dash is an error not reached. The function has no `sin`, `cos`
or `exp`, so the runs are the same on every platform, and in Python, `run` evaluates the function in
Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/schwefel-2-21) plays back another
run: CMA-ES on the function in 2 dimensions, so that the population can be drawn on its contour,
whose level sets are squares.

## Good results

The minimum is 0. CMA-ES reaches 1e-8 after 14,070 evaluations: 2,268 to get down to an error of 1,
then about 1,500 per decade of the remaining eight. sep-CMA-ES is nearly as fast, 16,058, since the
function needs no correlations between the genes, only one step size for all of them.

SHADE takes 85,300 evaluations, six times as many, and PSO 261,520, most of its budget. The genetic
algorithm reaches an error of 1 after 65,659 evaluations and ends at 0.11: its mutation changes one
gene in 30, which rarely lowers the maximum, and its steps don't shrink with the error.
