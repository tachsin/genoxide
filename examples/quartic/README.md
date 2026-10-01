---
title: Quartic
category: continuous
summary: Minimize De Jong's quartic function in 30 dimensions, without noise and with noise drawn from the genome, and compare CMA-ES, sep-CMA-ES, DE, PSO and a GA.
reference: "De Jong, K. A. (1975). An Analysis of the Behavior of a Class of Genetic Adaptive Systems. PhD thesis, University of Michigan."
reference_url: "https://hdl.handle.net/2027.42/4507"
optimum: "0 (at the origin, without noise)"
languages: [rust, python]
order: 52
trace_note: "Recorded from another run: CMA-ES in 2 dimensions, without noise, so that the population can be drawn on the function's contour."
---

# Quartic

## The problem

The quartic function is a weighted sum of fourth powers, to minimize; Yao, Liu and Lin add noise:

```text
f(x) = Σ i xᵢ⁴ + random[0, 1),   i from 1 to n, each xᵢ in [−1.28, 1.28]
```

Without the noise, its minimum is 0, at the origin. Here n = 30. It's De Jong's (1975) F4, with
Gaussian noise; genoxide takes this form, the uniform noise, the bounds and the dimension from Yao,
Liu and Lin (1999, f7), and is still to check De Jong's thesis (issue #168).

A fitness function in genoxide is deterministic: a copy of a genome inherits its fitness. So
`problems::Quartic::noisy` draws the noise from a generator seeded with the genome's bits: the same
genome always gets the same noise, and two genomes, however close, independent ones. Its minimum
isn't known, so its `optimum` is none.

## What makes it hard

Without noise, the function is unimodal and separable, but flat near the minimum: at 0.01 from 0 in
every gene, it's below 10⁻⁵. A search has to keep shrinking its steps on a slope that vanishes
faster than a parabola's.

With noise, every value is off by up to 1, far more than the quartic itself near the minimum. A
search that compares two points sees mostly their noise, and the best value it keeps is the one
whose noise happened to be small: the lowest of many draws, not the lowest quartic.

## Representation

A `Real` genome of 30 genes, each in [−1.28, 1.28]: the point x itself. The fitness is f(x), to
minimize. The functions are genoxide's `problems::Quartic::new` and `problems::Quartic::noisy`.

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

With noise, each algorithm runs to the end of its budget, since no target can be met.

## Output

The first line gives the dimension and the budget. Then a table for the function without noise: a
row per algorithm, the evaluations it had used when its best error first reached each value of the
heading, and the best error it found, to two significant digits. A dash is an error not reached.
Then, with noise, a row per algorithm: the best value it found, noise included, and the quartic at
that point without noise. The function is evaluated with genoxide's portable math, so the runs are
the same on every platform, and in Python, `run` evaluates it in Rust, so both versions print the
same.

[The project page](https://tachsin.gr/projects/genoxide/examples/quartic) plays back another run:
CMA-ES on the function in 2 dimensions without noise, x₁⁴ + 2x₂⁴, so that the population can be
drawn on its contour. It meets the target after 108 evaluations.

## Good results

Without noise, the minimum is 0. sep-CMA-ES reaches 1e-8 after 2,184 evaluations and CMA-ES after
2,436; PSO takes 12,440, SHADE 13,400 and the genetic algorithm 52,845.

With noise, PSO's best value is 0.0026, with a quartic of 0.0024 there, and SHADE's 0.0034 (a
quartic of 0.0033); the genetic algorithm ends at 0.011. Their best values are the quartic plus a
noise of about 10⁻⁴, the lowest of hundreds of thousands of draws. CMA-ES and sep-CMA-ES stop at
0.10: their step sizes adapt to differences between samples, which the noise dominates long before
the quartic is small. Yao, Liu and Lin report mean best values of 7.6·10⁻³ for their fast
evolutionary programming and 1.8·10⁻² for the classical one, after 3,000 generations of 100.
