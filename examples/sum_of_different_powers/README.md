---
title: Sum of different powers
category: continuous
summary: Minimize the sum of the genes' absolute values to powers from 2 to 31 in 30 dimensions, as it is and shifted and rotated, and compare how fast CMA-ES, sep-CMA-ES, DE, PSO and a GA close in on the minimum.
reference: "Molga, M. and Smutnicki, C. (2005). Test functions for optimization needs."
reference_url: ""
optimum: "0 (at the origin)"
languages: [rust, python]
order: 50
trace_note: "Recorded from another run: CMA-ES in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Sum of different powers

## The problem

The sum of different powers raises each gene's absolute value to a power one more than its index, to
minimize:

```text
f(x) = Σ |xᵢ|^(i+1),   i from 1 to n, each xᵢ in [−1, 1]
```

Its minimum is 0, at the origin. Here n = 30, so the powers run from 2 to 31. Its origin is
unknown: genoxide takes the definition and the bounds from Molga and Smutnicki (2005, section
2.8), and they're still to be checked against an original (issue #168). An early version of the
CEC 2017 report had it, shifted and rotated, as its function 2.

## What makes it hard

It's unimodal and separable, and each term is smallest at 0. But the terms differ widely in how much
they matter. Near the minimum, the first gene's term is a parabola, while the thirtieth's, |x|³¹,
is flat: at x₃₀ = 0.5 it's 5·10⁻¹⁰, and an error of 1e-8 allows x₃₀ up to 0.55. A search reaches
small values long before the later genes are near 0, and the flatter terms give it little to
follow.

Shifted and rotated, every direction mixes steep and flat terms, and the shapes that the steps
must learn are no longer along the axes.

## Representation

A `Real` genome of 30 genes, each in [−1, 1]: the point x itself. The fitness is f(x), to
minimize. The function is genoxide's `problems::SumOfDifferentPowers`, which brings its bounds and
its minimum.

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

[The project page](https://tachsin.gr/projects/genoxide/examples/sum-of-different-powers) plays back another
run: CMA-ES on the function in 2 dimensions, |x₁|² + |x₂|³, so that the population can be
drawn on its contour. It meets the target after 204 evaluations.

## Good results

The minimum is 0. On the function as it is, sep-CMA-ES reaches 1e-8 first, after 2,464
evaluations, then PSO (4,880), SHADE (6,500), CMA-ES (13,006) and the genetic algorithm (15,922):
every one gets there. The function is separable, and the methods that work a gene at a time, or
learn one scale per gene, are the fastest.

Shifted and rotated, CMA-ES takes about as long as before, 12,334 evaluations: its full covariance
matrix learns the rotation. SHADE needs five times as many, 32,200, and PSO 243,400. sep-CMA-ES
reaches 1e-6 after 10,220 evaluations but ends at 1.4e-8, and the genetic algorithm at 5.9e-8.
