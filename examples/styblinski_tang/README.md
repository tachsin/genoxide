---
title: Styblinski-Tang
category: continuous
summary: Minimize a separable 30-dimensional function whose every gene has two dips, the global one and a slightly shallower one nearly as wide.
reference: "Styblinski, M. A. and Tang, T.-S. (1990). Experiments in nonconvex optimization: stochastic approximation with function smoothing and simulated annealing. Neural Networks 3(4): 467-483."
reference_url: "https://doi.org/10.1016/0893-6080(90)90029-K"
optimum: "−1174.98 in 30 dimensions (−39.1662 n, at xᵢ ≈ −2.9035)"
languages: [rust, python]
order: 47
trace_note: "Recorded from another run: CMA-ES with IPOP restarts in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Styblinski-Tang

## The problem

The Styblinski-Tang function is a sum of the same quartic in each gene:

```text
f(x) = ½ Σ (xᵢ⁴ − 16xᵢ² + 5xᵢ),   each xᵢ in [−5, 5]
```

Styblinski and Tang (1990) used it to test stochastic approximation and simulated annealing.
genoxide's definition and bounds are as Jamil and Yang (2013, function 144) restate them; they
haven't yet been checked against the original paper. Here n = 30.

Each term, h(x) = ½ (x⁴ − 16x² + 5x), has two dips: its derivative, 2x³ − 16x + 2.5, is zero at
three points.

| x | h(x) | Point |
|---|---|---|
| −2.903534 | −39.166166 | the global minimum of the term |
| 0.156731 | 0.195612 | a local maximum, the ridge between the dips |
| 2.746803 | −25.029447 | a local minimum |

Without the term 5x, the dips would be mirror images, at ±2√2, and equally deep. The 5x tilts the
function: it lowers the left dip and raises the right one. The minimum of f is 30 times the
term's, −1174.984971, at xᵢ = −2.903534 in every gene. genoxide derives both from the formula.

## What makes it hard

The function is separable: each gene contributes its own term, and a gene's best value doesn't
depend on the others. But it is multimodal: each gene can sit in either dip, so the function has
2³⁰ ≈ 1.07 × 10⁹ local minima. Only one of them is global.

The wrong dip is deceptive. It is almost as wide as the right one: the ridge at 0.157 splits
[−5, 5] into 5.157 units for the left dip and 4.843 for the right, so a random value of a gene
lands in the wrong dip 48% of the time. It is also deep: a gene in it costs 14.1367 more than a
gene in the right one, while the ridge between them is 25.2 above the wrong dip and 39.4 above the
right. A search that contracts inside the wrong dip of a gene sees the ground rise in every
direction, and stays.

Each gene in the wrong dip adds exactly 14.1367 to the error, so an error is a count of wrong
genes: 113.09 is 8 genes.

## Representation

A `Real` genome of 30 genes, each in [−5, 5]: the point x itself. The fitness is f(x), to minimize.
The function is genoxide's `problems::StyblinskiTang`, which brings its bounds and its minimum. The
program runs it in 30 dimensions, genoxide's default: the chance that all 30 genes start in the
right dip is 0.516³⁰ ≈ 2 × 10⁻⁹.

## Algorithm

Three runs, each with a budget of 10,000 evaluations per dimension, 300,000 in all, and a target of
1e-8 above the minimum.

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. It uses genoxide's
defaults: a population of 4 + ⌊3 ln 30⌋ = 14, a step size of 0.3 of each gene's range, and a random
start. The first run has no restarts. The second has IPOP restarts (Auger and Hansen, 2005, IEEE
CEC 2005: 1769-1776): a run that has converged starts again from a random point with twice the
population. A larger population averages over more samples, so it sees the tilt that favors the
left dip before it contracts.

L-SHADE (Tanabe and Fukunaga, 2014, IEEE CEC 2014: 1658-1665) is a differential evolution that
adapts its scale factor and crossover rate from successful trials. Its crossover takes each gene
either from a mutant or from the parent. On a separable function that helps: a trial can move one
gene to the right dip and keep the others. Its population starts at 18 times the number of genes,
540, and shrinks linearly to 4 over the budget.

## Output

The first line gives the dimension, the minimum and the budget. Then one line per run: the error of
its best point, which is its value minus the minimum, to 4 decimals; the evaluations it took; and
how many of its 30 genes are more than 0.01 from −2.903534. A run stops as soon as it is within
1e-8 of the minimum, so an error of 0.0000 means that it met the target. In Python, `run` evaluates
the function in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/styblinski-tang) plays back
another run: CMA-ES with IPOP restarts on Styblinski-Tang in 2 dimensions, so that the population
can be drawn on the function's contour. Its first run, with 6 samples a generation, puts x₁ in the
wrong dip and stays at an error of 14.1367; the restart, with 12, finds the global minimum.

## Good results

The minimum is −1174.984971. CMA-ES without restarts ends at an error of 113.09 with the budget
spent: 8 of its 30 genes are in the wrong dip, at 2.7468, and the other 22 at the minimum. With
IPOP restarts, CMA-ES reaches the target after about 122,000 evaluations, and L-SHADE after about
168,000.
