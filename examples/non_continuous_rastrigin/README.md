---
title: Non-continuous Rastrigin
category: continuous
summary: Minimize Rastrigin's function made flat between half-integers, in 10 dimensions, with CMA-ES without and with restarts, DE, PSO and a GA from 10 seeds each.
reference: "Liang, J. J., Qin, A. K., Suganthan, P. N. and Baskar, S. (2006). Comprehensive learning particle swarm optimizer for global optimization of multimodal functions. IEEE Transactions on Evolutionary Computation 10(3): 281-295."
reference_url: "https://doi.org/10.1109/TEVC.2005.857610"
optimum: "0 (at the origin)"
languages: [rust, python]
order: 94
trace_note: "Recorded from another run: CMA-ES with IPOP restarts in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Non-continuous Rastrigin

## The problem

The non-continuous Rastrigin function is Rastrigin's function of genes rounded to half-integers
away from the origin:

```text
f(x) = Σ (yᵢ² − 10 cos 2πyᵢ + 10),   yᵢ = xᵢ if |xᵢ| < 1/2, round(2xᵢ)/2 otherwise
each xᵢ in [−5.12, 5.12]
```

Its minimum is 0, at the origin. Here n = 10. It's the function f7 of Liang, Qin, Suganthan and
Baskar (2006), whose table II gives the bounds and the minimum. `round` rounds halves away from 0,
as MATLAB's does, which their experiments used. The CEC 2017 report composes it with BBOB's
transformations, which genoxide doesn't apply.

## What makes it hard

Outside [−0.5, 0.5], each gene's term takes only the values of Rastrigin's function at the
half-integers: flat steps, with Rastrigin's local minima at the integers. A search sees plateaus
with no slope to follow, and as many local minima as Rastrigin's function, about 10ⁿ in the box.
Only near the origin, where every gene is within 1/2, is the function smooth.

## Representation

A `Real` genome of 10 genes, each in [−5.12, 5.12]: the point x itself. The fitness is f(x), to
minimize. The function is genoxide's `problems::NonContinuousRastrigin`, which brings its bounds and
its minimum.

## Algorithm

Five algorithms, each from seeds 1 to 10, with a budget of 10,000 evaluations per dimension,
100,000 per run, and a target of 1e-8:

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

The first line gives the dimension, the seeds and the budget. Then a row per algorithm: how many
of its 10 runs reached the minimum, to within 1e-8, the median of their evaluations (a dash if none
did), and the median of every run's best error, to two significant digits. The function is
evaluated with genoxide's portable math, so the runs are the same on every platform, and in Python,
`run` evaluates it in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/non-continuous-rastrigin) plays back another run:
CMA-ES with IPOP restarts on the function in 2 dimensions, so that the population
can be drawn on its contour. Within 20,000 evaluations it ends at an error of 6.9e-4, near the
minimum but short of the target.

## Good results

The minimum is 0. SHADE reaches it in all 10 runs, after a median of 64,650 evaluations. No other
algorithm does within the budget: CMA-ES ends in a local minimum with a median error of 18, and 3.0
with IPOP restarts; PSO's median run ends at 4.0, four genes one integer away from 0. The genetic
algorithm gets near it, to a median error of 2.4e-5.
