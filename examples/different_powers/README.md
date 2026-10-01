---
title: Different powers
category: continuous
summary: Minimize BBOB's different powers in 30 dimensions, exponents from 2 to 6 under a square root, as it is and shifted and rotated, and compare CMA-ES, sep-CMA-ES, DE, PSO and a GA.
reference: "Hansen, N., Finck, S., Ros, R. and Auger, A. (2009). Real-Parameter Black-Box Optimization Benchmarking 2009: Noiseless Functions Definitions. INRIA research report RR-6829."
reference_url: "https://hal.inria.fr/inria-00362633"
optimum: "0 (at the origin)"
languages: [rust, python]
order: 57
trace_note: "Recorded from another run: CMA-ES in 2 dimensions on the function rotated with seed 1, so that the population can be drawn on the function's contour."
---

# Different powers

## The problem

BBOB's different powers raises each gene's absolute value to a power from 2 to 6, and takes the
square root of the sum:

```text
f(x) = √(Σ |xᵢ|^(2 + 4 (i−1)/(n−1))),   i from 1 to n, each xᵢ in [−5, 5]
```

Its minimum is 0, at the origin. Here n = 30. It's BBOB's f14 (Hansen et al. 2009), which rotates
it, with its search domain. It isn't the sum of different powers of Molga and Smutnicki, with
powers 2 to n + 1 and no square root.

## What makes it hard

Near the minimum, the genes' sensitivities drift apart: an error of 1e-8, under the square root,
needs the first gene within 10⁻⁸ of 0, but allows the last one, with its sixth power, to be
10⁻²·⁷ ≈ 0.002 away. The closer the search gets, the more different the scales it needs, so a
search must keep adapting them. Shifted and rotated, every direction mixes the powers.

## Representation

A `Real` genome of 30 genes: the point x itself. The fitness is f(x), to minimize. The function is
genoxide's `problems::DifferentPowers`, which brings its bounds and its minimum, and the shifted and rotated
instance `problems::Rotated::new(problems::Shifted::new(function, 1), 1)`, which keeps them.

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

The second table runs the same algorithms on the function shifted and rotated, with genoxide's
`problems::Shifted` and `problems::Rotated` and seed 1: the minimum moves to a random point in the
middle 80% of the box, and an orthogonal matrix, drawn from normal numbers made orthonormal by
Gram-Schmidt as BBOB draws its rotations, turns the function about it. That's how the CEC and BBOB
suites use the function, with their own data; genoxide generates its instances instead.

## Output

The first line gives the dimension and the budget. Then two tables, the function as it is and shifted and rotated: a row per algorithm, the
evaluations it had used when its best error first reached each value of the heading, and the best
error it found, to two significant digits. A dash is an error not reached. The function is
evaluated with genoxide's portable math, so the runs are the same on every platform, and in Python,
`run` evaluates it in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/different-powers) plays back another
run: CMA-ES on the function in 2 dimensions, √(x₁² + x₂⁶), rotated with seed 1, so that the
population can be drawn on its contour. It meets the target after 714 evaluations.

## Good results

The minimum is 0. As it is, sep-CMA-ES reaches 1e-8 first, after 7,420 evaluations, then PSO
(23,080), SHADE (30,000) and CMA-ES (47,208); the genetic algorithm ends at 1.3e-6.

Shifted and rotated, only CMA-ES reaches 1e-8, after 48,734 evaluations, about as many as before.
sep-CMA-ES ends at 3.9e-5, SHADE at 1.1e-5, PSO at 1.6e-4 and the genetic algorithm at 3.1e-3:
their scales per gene, or steps along the axes, no longer fit.
