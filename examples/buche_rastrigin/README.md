---
title: Büche-Rastrigin
category: continuous
summary: Minimize BBOB's Büche-Rastrigin function, an asymmetric Rastrigin, in 10 dimensions, with CMA-ES without and with restarts, DE, PSO and a GA from 10 seeds each.
reference: "Hansen, N., Finck, S., Ros, R. and Auger, A. (2009). Real-Parameter Black-Box Optimization Benchmarking 2009: Noiseless Functions Definitions. INRIA research report RR-6829."
reference_url: "https://hal.inria.fr/inria-00362633"
optimum: "0 (at the origin)"
languages: [rust, python]
order: 93
trace_note: "Recorded from another run: CMA-ES with IPOP restarts in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Büche-Rastrigin

## The problem

The Büche-Rastrigin function is Rastrigin's function of scaled and slightly bent genes, with a
penalty outside the box:

```text
f(x) = 10 (n − Σ cos 2πzᵢ) + Σ zᵢ² + 100 Σ max(0, |xᵢ| − 5)²,   zᵢ = sᵢ T_osz(xᵢ)
each xᵢ in [−5, 5]
```

T_osz is BBOB's oscillation, `sign(x) exp(x̂ + 0.049 (sin c₁x̂ + sin c₂x̂))` with x̂ = ln |x|,
c₁ = 10 and c₂ = 7.9 for positive x, 5.5 and 3.1 otherwise: the identity, with small smooth
wiggles. The scale sᵢ grows from 1 to √10 along the genes, and is ten times larger where xᵢ > 0
and i is odd. Its minimum is 0, at the origin. Here n = 10. It's BBOB's f4 (Hansen et al. 2009),
with its optimum at the origin and no offset, and BBOB's search domain.

## What makes it hard

Rastrigin's function has a local minimum near every integer point, roughly 10ⁿ of them in the box.
Here, on the positive side of the odd genes, the scale is ten times larger: the wells are ten times
narrower and the slope ten times steeper, so the landscape is lopsided. BBOB built it to deceive
search operators that are symmetric about the current point, which expect the minimum's basin to
look the same on both sides.

## Representation

A `Real` genome of 10 genes, each in [−5, 5]: the point x itself. The fitness is f(x), to
minimize. The function is genoxide's `problems::BucheRastrigin`, which brings its bounds and its
minimum.

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

[The project page](https://tachsin.gr/projects/genoxide/examples/buche-rastrigin) plays back another run:
CMA-ES with IPOP restarts on the function in 2 dimensions, so that the population
can be drawn on its contour. Within 20,000 evaluations it restarts 6 times, up to a population of
384, and ends at an error of 0.36: it doesn't reach the minimum in 2 dimensions either.

## Good results

The minimum is 0. SHADE reaches it in all 10 runs, after a median of 66,650 evaluations: its
differences between members of the population take the wells' spacing, and its restarts on
stagnation free it from the local minima it falls into. No other algorithm reaches it. CMA-ES ends
in a local minimum with a median error of 18, and 6.0 with IPOP restarts; PSO's median run ends at
7.5. The genetic algorithm, whose mutation changes one gene at a time, gets close, to a median error
of 2.2e-4, but doesn't reach 1e-8 within the budget.
