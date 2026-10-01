---
title: Rastrigin function
category: continuous
summary: Minimize a 30-dimensional function with a local minimum near every integer point.
reference: "Rastrigin, L. A. (1974). Systems of Extremal Control. Nauka, Moscow, in two dimensions (no DOI). The n-dimensional form as given by Mühlenbein, H., Schomisch, M. and Born, J. (1991). The parallel genetic algorithm as function optimizer. Parallel Computing 17(6-7): 619-632, function F6, to which the link points; first generalized by Rudolph, G. (1990). Globale Optimierung mit parallelen Evolutionsstrategien. Diplomarbeit, Universität Dortmund."
reference_url: "https://doi.org/10.1016/S0167-8191(05)80052-3"
optimum: "0 (at the origin)"
languages: [rust, python]
order: 55
trace_note: "Recorded from another run: L-SHADE in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Rastrigin function

## The problem

Rastrigin's function adds a cosine to a sphere:

```text
f(x) = 10n + Σ (xᵢ² − 10 cos 2πxᵢ),   each xᵢ in [−5.12, 5.12]
```

Rastrigin (1974, Systems of Extremal Control, Nauka) defined it in two dimensions. Rudolph (1990,
a Diplomarbeit at the University of Dortmund) generalized it to n dimensions, and this is the form
of Mühlenbein, Schomisch and Born (1991, F6). Its minimum is 0, at the origin. Here n = 30.

In one dimension, f(0) = 0. Near x = 1 there is a local minimum of about 1, and between them, at x =
0.5, a ridge of 20.25. Each integer from −5 to 5 has such a dip, deeper the nearer it is to 0.

## What makes it hard

The function is multimodal: there is a local minimum near every integer point of the box. With 11
integers per coordinate, the box holds 11³⁰ ≈ 1.7 × 10³¹ of them. A local search, or a small
population that contracts fast, settles in the first dip it finds.

What helps is the global structure: the sphere term makes the dips deeper towards the origin. Seen
from far enough, the function is a bowl.

## Representation

A `Real` genome of 30 genes, each in [−5.12, 5.12]: the point x itself. The fitness is f(x), to
minimize. The function is genoxide's `problems::Rastrigin`, which brings its bounds and its minimum.

## Algorithm

Two algorithms, each with a budget of 1,000,000 evaluations and a target of 1e-8.

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. It uses genoxide's
defaults: a population of 4 + ⌊3 ln 30⌋ = 14, a step size of 0.3 of each gene's range, and a random
start. With IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776), a run that has
converged starts again from a random point with twice the population. genoxide's docs recommend IPOP
for multimodal functions with a global structure, like this one: a larger population smooths out the
local minima.

L-SHADE (Tanabe and Fukunaga, 2014, IEEE CEC 2014: 1658-1665) is a differential evolution that
adapts its scale factor and crossover rate from successful trials. Its population starts at 18 times
the number of genes, 540, and shrinks linearly to 4 over the budget. genoxide's `De::l_shade` takes
L-SHADE's settings, so the example only gives it the budget.

## Output

One line per algorithm: the best value it found, to 6 decimals, and the evaluations it took. A run
stops as soon as it is within 1e-8 of the minimum, so a value of 0.000000 means that it met the
target. In Python, `run` evaluates the function in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/rastrigin) plays back another run: L-SHADE on Rastrigin in 2 dimensions, so that the
population can be drawn on the function's contour.

## Good results

The minimum is 0. Both algorithms reach the target: CMA-ES after about 180,000 evaluations, and
L-SHADE after about 410,000.
