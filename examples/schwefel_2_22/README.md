---
title: Schwefel 2.22
category: continuous
summary: Minimize the sum plus the product of the absolute values of 30 genes, with a kink along every axis, and compare how fast CMA-ES, sep-CMA-ES, DE, PSO and a GA close in on the minimum.
reference: "Schwefel, H.-P. (1981). Numerical Optimization of Computer Models. Wiley. Problem 2.22."
reference_url: ""
optimum: "0 (at the origin)"
languages: [rust, python]
order: 45
family: Schwefel
trace_note: "Recorded from another run: CMA-ES in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Schwefel 2.22

## The problem

Schwefel's problem 2.22 adds the product of the genes' absolute values to their sum, to minimize:

```text
f(x) = Σ |xᵢ| + Π |xᵢ|,   each xᵢ in [−10, 10]
```

Its minimum is 0, at the origin. Here n = 30. It's Schwefel's (1981) problem 2.22; genoxide takes
the definition, the bounds and the dimension from Yao, Liu and Lin's (1999, f2) restatement, and
they are still to be checked against Schwefel's book. Jamil and Yang (2013, function 124) give
[−100, 100].

## What makes it hard

The sum is a cone: a pyramid with its tip at the origin, not differentiable along any plane where a
gene is 0. Near the origin it dominates, and a search has to bring each gene to 0 along a kink,
where a step that crosses 0 costs as much as it gains. Far from it, the product dominates: at a
random point of the box, Π |xᵢ| is around 10¹⁷, and at a corner 10³⁰, against a sum of at most 300.
The product also vanishes as soon as one gene is 0, so the function falls steeply towards every axis
plane there.

## Representation

A `Real` genome of 30 genes, each in [−10, 10]: the point x itself. The fitness is f(x), to
minimize. The function is genoxide's `problems::Schwefel2_22`, which brings its bounds and its
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

[The project page](https://tachsin.gr/projects/genoxide/examples/schwefel-2-22) plays back another
run: CMA-ES on the function in 2 dimensions, so that the population can be drawn on its contour.

## Good results

The minimum is 0. sep-CMA-ES reaches 1e-8 first, after 7,826 evaluations, and CMA-ES after 9,044:
the function is separable near the minimum, where the sum dominates, so a diagonal covariance matrix
is all it needs, and learns faster than a full one.

SHADE takes 49,900 evaluations and PSO 57,680. The genetic algorithm reaches an error of 1 after
10,404 evaluations, faster than SHADE and nearly as fast as PSO, but 1e-2 only after 110,881, and it
ends at 2.6e-3: its steps don't shrink with the error.
