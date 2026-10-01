---
title: Schaffer F7
category: continuous
summary: Minimize Schaffer's F7, rings of ripples around the minimum, in 10 dimensions, with CMA-ES without and with restarts, DE, PSO and a GA from 10 seeds each.
reference: "Schaffer, J. D., Caruana, R. A., Eshelman, L. J. and Das, R. (1989). A study of control parameters affecting online performance of genetic algorithms for function optimization. Proceedings of the Third International Conference on Genetic Algorithms, Morgan Kaufmann: 51-60."
reference_url: ""
optimum: "0 (at the origin)"
languages: [rust, python]
order: 90
trace_note: "Recorded from another run: CMA-ES with IPOP restarts in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Schaffer F7

## The problem

Schaffer's F7 measures the distance of each pair of neighboring genes and adds ripples to it:

```text
f(x) = ((1 / (n − 1)) Σᵢ₌₁ⁿ⁻¹ √sᵢ (1 + sin²(50 sᵢ^(1/5))))²,   sᵢ = √(xᵢ² + xᵢ₊₁²)
each xᵢ in [−100, 100]
```

Its minimum is 0, at the origin. Here n = 10. Schaffer, Caruana, Eshelman and Das (1989) defined F7
in two dimensions; their paper couldn't be read. This n-dimensional form is BBOB's f17 (Hansen et
al. 2009), without the transformations BBOB applies to it, and the bounds are those of Schaffer's
F6. The CEC 2017 report (Awad et al. 2016, function 19) prints sin for sin², and scales its search
space to [−0.5, 0.5]. The original is still to be checked (issue #168).

## What makes it hard

Around the minimum, each pair's term is a cone, √sᵢ, with rings of ripples whose frequency grows as
sᵢ^(1/5): near the origin the rings crowd together, each a local minimum that holds a search whose
steps are smaller than the ring's width. Far from it, the cone dominates and leads inwards. The
terms of neighboring pairs share a gene, so the rings of one pair cut across those of the next.

## Representation

A `Real` genome of 10 genes, each in [−100, 100]: the point x itself. The fitness is f(x), to
minimize. The function is genoxide's `problems::SchafferF7`, which brings its bounds and its
minimum.

## Algorithm

Five algorithms, each from seeds 1 to 10, with a budget of 10,000 evaluations per dimension, 100,000
per run, and a target of 1e-8:

- CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195), which samples a
  population of 10 from a normal distribution and adapts its mean, its step size and its covariance
  matrix, from a step size of 0.3 of each gene's range and a random start;
- the same with IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776): a run that has
  converged starts again from a random point with twice the population;
- differential evolution with genoxide's defaults, SHADE (Tanabe and Fukunaga, CEC 2013), with a
  population of 100 and its restarts on stagnation;
- particle swarm optimization (Kennedy and Eberhart, 1995), 40 particles with Clerc and Kennedy's
  constriction coefficients and a global topology;
- a real-coded genetic algorithm: a population of 100, tournaments of 3, simulated binary crossover
  (Deb and Agrawal, 1995) with η = 15 and polynomial mutation with η = 20 at a rate of 1/10 per
  gene.

## Output

The first line gives the dimension, the seeds and the budget. Then a row per algorithm: how many of
its 10 runs reached the minimum, to within 1e-8, the median of their evaluations (a dash if none
did), and the median of every run's best error, to two significant digits. The function is evaluated
with genoxide's portable math, so the runs are the same on every platform, and in Python, `run`
evaluates it in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/schaffer-f7) plays back another
run: CMA-ES with IPOP restarts on the function in 2 dimensions, so that the population can be drawn
on its contour. It meets the target after 2,070 evaluations.

## Good results

The minimum is 0. CMA-ES with IPOP restarts reaches it in all 10 runs, after a median of 24,360
evaluations, and SHADE in all 10, after 58,450. Without restarts, CMA-ES ends in a ring in every
run, with a median error of 0.10. PSO reaches the minimum once, and its median run ends at 7.7e-5;
the genetic algorithm's at 2.4e-3.
