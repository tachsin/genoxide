---
title: Matyas
category: continuous
summary: Minimize Matyas' function, a quadratic valley 25 times flatter along the diagonal than across it, with CMA-ES, DE, PSO and a GA from 30 seeds.
reference: "Jamil, M. and Yang, X.-S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation 4(2): 150-194."
reference_url: "https://doi.org/10.1504/IJMMNO.2013.055204"
optimum: "0 at (0, 0)"
languages: [rust, python]
order: 65
---

# Matyas

## The problem

Matyas' function is a quadratic in two variables, to minimize:

```text
f(x₁, x₂) = 0.26 (x₁² + x₂²) − 0.48 x₁x₂,   x₁, x₂ in [−10, 10]
```

Its origin is unknown: Jamil and Yang (2013, function 71) credit Hedar's collection of global
optimization test problems, and the name may refer to Matyas' random optimization (1965), which
couldn't be checked. genoxide takes the definition and the bounds from Jamil and Yang; Laguna and
Martí (2005, function 8) have the same function on [−5, 10]. Both are still to be checked against an
original.

The minimum is 0 at the origin: the Hessian is positive definite, so the function is a convex
quadratic with the origin its only minimum. genoxide's `problems::Matyas` gives it as proven.

## What makes it hard

The Hessian has the eigenvalues 1, across the diagonal x₁ = x₂, and 0.04, along it: the function is
a long valley on the diagonal, 25 times flatter along its floor than up its sides. At (1, 1), f is
0.04; at (1, −1), it's 1. At the corners (10, 10) and (−10, −10) it's only 4, while at (10, −10)
it's 100.

An algorithm that moves along the genes, one at a time or with independent steps, zigzags down the
valley: a step along x₁ alone climbs the side as much as it descends the floor. An algorithm that
learns the valley's direction can walk along it.

## Representation

A `Real` genome of 2 genes, each in [−10, 10]: the point (x₁, x₂) itself. The fitness is f, to
minimize. The function, its bounds and its minimum are genoxide's `problems::Matyas`.

## Algorithm

Four algorithms, each from seeds 1 to 30, each run stopping once its value is within 1e-8 of 0, or
after 10,000 evaluations:

- CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195), with genoxide's
  defaults: a population of 6 and a step size of 0.3 of each gene's range, from a random start. Its
  covariance matrix learns the valley's direction;
- differential evolution with genoxide's defaults, SHADE (Tanabe and Fukunaga, CEC 2013), with a
  population of 20: its difference vectors between members of the population point along the valley
  once the population lies in it;
- particle swarm optimization (Kennedy and Eberhart, 1995), 40 particles with Clerc and Kennedy's
  constriction coefficients and a global topology;
- a real-coded genetic algorithm: a population of 50, tournaments of 3, simulated binary crossover
  (Deb and Agrawal, 1995) with η = 15 and polynomial mutation with η = 20 at a rate of 1/2 per gene.

## Output

The first line gives the minimum, the seeds and the budget. Then a row per algorithm: how many of
the 30 runs reach the minimum and how many don't, and the median and largest number of evaluations
of the runs that reach it. In Python, `run` evaluates the function in Rust, so both versions print
the same table.

The page's plot shows the 30 runs of CMA-ES, each at its best point so far, over the function's
contour, with the minimum marked. A curve gives the best and the median run's error, on a
logarithmic axis.

[The project page](https://tachsin.gr/projects/genoxide/examples/matyas) plays this run back.

## Good results

A good result reaches 0 in every run. CMA-ES does, after a median of 279 evaluations and at most
402, about as many as on [Booth's function](../booth/): its covariance matrix makes the valley's
stretch irrelevant once learned. SHADE reaches it after a median of 1,060 evaluations and PSO after
2,540.

The genetic algorithm reaches 1e-8 in 7 of the 30 runs. Its median run ends at 1.7e-7: its crossover
and mutation work gene by gene with steps that don't shrink with the error, so it zigzags down the
flat floor of the valley.
