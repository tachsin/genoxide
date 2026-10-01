---
title: Griewank
category: continuous
summary: Minimize a wide bowl with ripples that couple the genes, and see CMA-ES reach the minimum more often in more dimensions.
reference: "Griewank, A. O. (1981). Generalized descent for global optimization. Journal of Optimization Theory and Applications 34(1): 11-39."
reference_url: "https://doi.org/10.1007/BF00933356"
optimum: "0 (at the origin)"
languages: [rust, python]
order: 56
trace_note: "Recorded from another run: CMA-ES with IPOP restarts in 2 dimensions, with a budget of 50,000 evaluations, so that the population can be drawn on the function's contour."
---

# Griewank

## The problem

Griewank's function subtracts a product of cosines from a wide bowl:

```text
f(x) = 1 + Σ xᵢ² / 4000 − Π cos(xᵢ / √i),   i from 1 to n, each xᵢ in [−600, 600]
```

Its minimum is 0, at the origin, where the bowl is 1 and every cosine is 1. Griewank's own function
(1981), as Bosse and Bücker (2024, A piecewise smooth version of the Griewank function,
Optimization Methods and Software) restate it, is two-dimensional with the divisor 200. This
n-dimensional form, with the divisor 4000 and the bounds, is that of Mühlenbein, Schomisch and Born
(1991, F8) and of Yao, Liu and Lin (1999, f11). genoxide hasn't checked it against Griewank's paper
yet (issue #168).

The bowl, Σ xᵢ² / 4000, adds up to 90 per gene, at the bounds. The cosine of gene i has a period of
2π√i: 6.3 for the first gene, 34 for the thirtieth. The product is between −1 and 1, so the ripples
are at most 2 deep, against a bowl of up to 90 n.

## What makes it hard

Near the origin, the ripples dominate, and they make many local minima. The one nearest to the
origin is at x₁ ≈ ±π and x₂ ≈ ±π√2, with the other genes at 0: there, the first two cosines are both
−1, so their product is 1, as at the origin. Its value is about 3π² / 4000 ≈ 0.0074, the bowl alone.

This is how the product couples the genes. From that point, moving x₁ alone to 0 makes its cosine 1
while the second one is still −1: the product becomes −1, and f rises to about 2. The two genes have
to move together. A search that improves one gene at a time can't leave this minimum, and neither
can a population that has contracted around it.

Far from the origin, the bowl dominates. A product of many cosines, each between −1 and 1, is
usually close to 0, and the function there looks like the smooth bowl 1 + Σ xᵢ² / 4000. The more
genes, the more cosines in the product, and the smaller the ripples compared with the bowl. The
function gets easier in more dimensions, as the example shows.

## Representation

A `Real` genome of n genes, each in [−600, 600]: the point x itself. The fitness is f(x), to
minimize. The function is genoxide's `problems::Griewank`, which brings its bounds and its minimum.
The example runs n = 2, 5, 10, 20, 30 and 50: 10 and 30 are the usual test dimensions (the function
suite's and Yao, Liu and Lin's), and the others show the trend.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. It uses genoxide's
defaults: a population of 4 + ⌊3 ln n⌋, a step size of 0.3 of each gene's range (360), and a random
start. The adapted covariance lets it move several genes together, which the coupled minima need.

It runs in each dimension with seeds 1 to 10, twice:

- without restarts: a run that converges to a local minimum stays there;
- with IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776): a run that has converged
  starts again from a random point with twice the population. genoxide's docs recommend them for
  multimodal functions.

Each run has a budget of 10,000 evaluations per dimension, and at least 100,000 (in 2 and 5
dimensions, where the restarts need more than 10,000 per dimension), and a target of 1e-8.

## Output

The first line gives the number of seeds and the budget. Then a table has a row per dimension: of
the 10 runs without restarts and the 10 with IPOP restarts, how many met the target. In Python,
`run` evaluates the function in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/griewank) plays back another run:
CMA-ES with IPOP restarts on Griewank in 2 dimensions, so that the population can be drawn on the
function's contour. With a budget of 50,000 evaluations, it restarts 4 times, from a population of
6 to one of 96. Its best value falls from 0.092 to 0.0074, then 0.0028 and 0.0014, and it meets the
target after 12,786 evaluations.

## Good results

A good result is 10 of 10. Without restarts, CMA-ES reaches the minimum once in 10 in 2
dimensions, never in 5, 3 times in 10 in 10 dimensions, 8 times in 20 and 30, and 9 times in 50. The
runs that fail end in a local minimum near the origin: in 30 dimensions, seed 4 ends at 0.0074, with
x₁ = −3.14 and x₂ = 4.44, the minimum described above.

With IPOP restarts, it reaches the minimum every time, in every dimension. In 2 dimensions, where
the ripples, compared with the bowl, are the largest, it needs the most restarts: with seeds 1 to
100, it reaches the minimum every time, after at most 105,000 evaluations, but only 64 times with
20,000 evaluations, 10,000 per dimension, 89 times with 50,000 and 99 times with the example's
100,000. In 5 dimensions, it reaches it every time with seeds 1 to 100, after at most 82,000
evaluations, and 96 times with 50,000.
