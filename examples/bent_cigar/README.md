---
title: Bent cigar
category: continuous
summary: Minimize a narrow ridge, a thousand times longer than wide, in 30 dimensions, as it is and shifted and rotated, and compare CMA-ES, sep-CMA-ES, DE, PSO and a GA.
reference: "Hansen, N., Finck, S., Ros, R. and Auger, A. (2009). Real-Parameter Black-Box Optimization Benchmarking 2009: Noiseless Functions Definitions. INRIA research report RR-6829."
reference_url: "https://hal.inria.fr/inria-00362633"
optimum: "0 (at the origin)"
languages: [rust, python]
order: 55
trace_note: "Recorded from another run: CMA-ES in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Bent cigar

## The problem

The bent cigar squares the first gene, and the others a million times more:

```text
f(x) = x₁² + 10⁶ Σᵢ₌₂ⁿ xᵢ²,   each xᵢ in [−100, 100]
```

Its minimum is 0, at the origin. Here n = 30. It's BBOB's f12 (Hansen et al. 2009), which bends it
with an asymmetric transformation and rotates it twice; this plain form and the bounds are the CEC
2014 (Liang, Qu and Suganthan 2013, function 2) and CEC 2017 (Awad et al. 2016, function 1) reports'
basic function, which those suites shift and rotate.

## What makes it hard

The function is a ridge: low only near the line along the first axis, a thousand times narrower than
long. A search has to find the ridge, then follow it to the minimum, with steps a thousand times
longer along it than across it, in a single direction. As it is, that direction is a gene's axis.
Shifted and rotated, it's a random one, and the search has to learn it.

## Representation

A `Real` genome of 30 genes: the point x itself. The fitness is f(x), to minimize. The function is
genoxide's `problems::BentCigar`, which brings its bounds and its minimum, and the shifted and
rotated instance `problems::Rotated::new(problems::Shifted::new(function, 1), 1)`, which keeps them.

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

The first line gives the dimension and the budget. Then two tables, the function as it is and
shifted and rotated: a row per algorithm, the evaluations it had used when its best error first
reached each value of the heading, and the best error it found, to two significant digits. A dash is
an error not reached. The function is evaluated with genoxide's portable math, so the runs are the
same on every platform, and in Python, `run` evaluates it in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/bent-cigar) plays back another run:
CMA-ES on the function in 2 dimensions, x₁² + 10⁶ x₂², not rotated (in 2 dimensions, it's the
high-conditioned elliptic function), so that the population can be drawn on its contour. It meets
the target after 756 evaluations.

## Good results

The minimum is 0. As it is, sep-CMA-ES reaches 1e-8 after 8,316 evaluations and CMA-ES after 13,482;
SHADE takes 42,800 and PSO 47,600. The genetic algorithm ends at 210.

Shifted and rotated, CMA-ES takes 13,902 evaluations, about as many: it learns the ridge's
direction. SHADE gets there after 171,700, four times as many as before, and the others fail:
sep-CMA-ES ends at 680, the genetic algorithm at 470 and PSO at 5,700.
