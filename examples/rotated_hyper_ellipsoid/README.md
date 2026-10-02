---
title: Rotated hyper-ellipsoid
category: continuous
summary: Minimize the "rotated" hyper-ellipsoid in 30 dimensions, which is in fact axis-parallel, and then the same function truly rotated, with CMA-ES and sep-CMA-ES, DE, PSO and a GA.
reference: "Molga, M. and Smutnicki, C. (2005). Test functions for optimization needs."
reference_url: ""
optimum: "0 (at the origin)"
languages: [rust, python]
order: 53
trace_note: "Recorded from another run: CMA-ES in 2 dimensions on the function rotated with seed 1, so that the population can be drawn on the function's contour."
---

# Rotated hyper-ellipsoid

## The problem

The rotated hyper-ellipsoid, as Molga and Smutnicki (2005, section 2.3) define it, sums the sums of
squares of the first genes:

```text
f(x) = Σᵢ Σⱼ≤ᵢ xⱼ²,   each xᵢ in [−65.536, 65.536]
```

Its minimum is 0, at the origin. Here n = 30. Its origin is unknown; genoxide takes it, and its
bounds, from Molga and Smutnicki, and it's still to be checked against an original (issue #168).

Despite its name, it isn't rotated. Gene j appears in the n − j + 1 sums from i = j on, so

```text
f(x) = Σⱼ (n − j + 1) xⱼ²
```

an ellipsoid along the axes, with weights from 30 down to 1: genoxide's axis-parallel ellipsoid with
its genes reversed. Molga and Smutnicki describe Schwefel's problem 1.2, Σᵢ (Σⱼ≤ᵢ xⱼ)², whose
ellipsoids are rotated, but write this formula, which other collections repeat.

## What makes it hard

As written, little: it's a separable quadratic with a condition number of 30, which a search can
solve a gene at a time.

The example then rotates it, with genoxide's `problems::Rotated` and seed 1: an orthogonal matrix,
drawn from normal numbers made orthonormal by Gram-Schmidt as BBOB draws its rotations, turns the
function about its minimum. The ellipsoid's axes are no longer the genes' axes, and the genes
interact: a scale per gene no longer fits it, nor do steps along the axes. The plan of genoxide's
test problems suggested this page for comparing CMA-ES with a full and with a diagonal covariance
matrix.

## Representation

A `Real` genome of 30 genes, each in [−65.536, 65.536]: the point x itself. The fitness is f(x), to
minimize. The function is genoxide's `problems::RotatedHyperEllipsoid`, and its rotation
`problems::Rotated`, which keeps the bounds and the minimum.

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

The first line gives the dimension and the budget. Then two tables, the function as written and
rotated by an orthogonal matrix: a row per algorithm, the evaluations it had used when its best
error first reached each value of the heading, and the best error it found, to two significant
digits. A dash is an error not reached. The function is evaluated with genoxide's portable math, so
the runs are the same on every platform, and in Python, `run` evaluates it in Rust, so both versions
print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/rotated-hyper-ellipsoid) plays back
another run: CMA-ES on the function in 2 dimensions, 2x₁² + x₂², rotated with seed 1, so that the
population can be drawn on its contour. It meets the target after 342 evaluations.

## Good results

The minimum is 0. As written, sep-CMA-ES reaches 1e-8 first, after 4,886 evaluations, then CMA-ES
(7,140), PSO (27,520) and SHADE (31,900); the genetic algorithm ends at 1.1e-3.

Rotated, CMA-ES takes about as long, 7,294 evaluations: its full covariance matrix learns the
ellipsoid's axes, whichever they are. sep-CMA-ES takes 2.5 times as long, 12,180, since its diagonal
matrix can only scale the genes; with a condition number of 30, it still gets there. SHADE and PSO
slow down by two and three and a half times (59,000 and 96,600), and the genetic algorithm, which
recombines genes position by position, ends at 4.2.
