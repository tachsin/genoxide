---
title: Michalewicz
category: continuous
summary: Minimize a 10-dimensional function of steep narrow valleys on flat plateaus, where each gene must find the best of its valleys.
reference: "Michalewicz, Z. (1992). Genetic Algorithms + Data Structures = Evolution Programs. Springer."
reference_url: ""
optimum: "−9.66015 in 10 dimensions (computed a gene at a time)"
languages: [rust, python]
order: 63
trace_note: "Recorded from another run: L-SHADE in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Michalewicz

## The problem

Michalewicz's function sums a term per gene, with a steepness m = 10:

```text
f(x) = −Σᵢ₌₁ⁿ sin(xᵢ) sin²ᵐ(i xᵢ² / π),   m = 10, each xᵢ in [0, π]
```

It comes from Michalewicz's book (1992). genoxide's definition, m and bounds are as Molga and
Smutnicki (2005, section 2.11) restate them; they haven't yet been checked against the book. Here
n = 10.

The minimum depends on n. The function is separable, so genoxide computes it a gene at a time: it
minimizes each term on its own, to the precision of an f64, which proves the value. That gives
−1.8013034 for n = 2, −4.6876582 for n = 5 and −9.6601517 for n = 10; Molga and Smutnicki give
−4.687 and −9.66. In 2 dimensions the minimum is at (2.20291, π/2).

## What makes it hard

The second factor, sin²⁰(i xᵢ² / π), is a sine raised to the 20th power. It is close to 0 almost
everywhere, and close to 1 only where the sine is near ±1. So each term is a flat plateau at 0,
cut by narrow valleys. Gene i has i of them, at xᵢ = π √((k + ½) / i) for k = 0, …, i − 1.

The valleys are narrow, and they get narrower along the genes: the one where x₁ has its minimum,
at 2.2029, is about 0.37 wide, and the one where x₁₀ has its minimum, at π/2, about 0.05 (counting
the points where the second factor is above ½). In each gene, 12 to 15% of [0, π] lies inside a
valley, and about two thirds of it is so flat that the term is within 0.01 of 0. A sample on the
plateau learns almost nothing about where the valleys are: few regions of the space are
informative.

The valleys aren't equally deep. Each is about sin(xᵢ) deep, so the best valley of a gene is the
one nearest π/2. A gene in another valley is a local minimum, and so is every combination: in 10
dimensions there are 1 × 2 × ⋯ × 10 = 10! = 3,628,800 local minima.

## Representation

A `Real` genome of 10 genes, each in [0, π]: the point x itself. The fitness is f(x), to minimize.
The function is genoxide's `problems::Michalewicz`, which brings its bounds and computes its
minimum. The program runs it in 10 dimensions, genoxide's default and the dimension of Molga and
Smutnicki's value, −9.66.

## Algorithm

Two algorithms, each with a budget of 10,000 evaluations per dimension, 100,000 in all, and a target
of 1e-8 above the minimum.

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. It uses genoxide's
defaults: a population of 4 + ⌊3 ln 10⌋ = 10, a step size of 0.3 of each gene's range, and a random
start. With IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776), a run that has
converged starts again from a random point with twice the population. Its samples change every
gene at once, and a larger population helps on functions whose local minima follow a global
trend. This one has no such trend: the plateaus between the valleys are flat.

L-SHADE (Tanabe and Fukunaga, 2014, IEEE CEC 2014: 1658-1665) is a differential evolution that
adapts its scale factor and crossover rate from successful trials. Its crossover takes each gene
either from a mutant or from the parent. On a separable function that helps: a trial can move one
gene into a better valley and keep the others where they are. Its population starts at 18 times the
number of genes, 180, and shrinks linearly to 4 over the budget.

## Output

The first line gives the dimension, the minimum and the budget. Then one line per algorithm: the
error of its best point, which is its value minus the minimum, to 4 decimals; whether the run
stopped at the target, within 1e-8 of the minimum, or with the budget spent; and how many of its 10
genes are more than 0.01 from the minimum's. In Python, `run` evaluates the function in Rust, so
both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/michalewicz) plays back another
run: L-SHADE on Michalewicz in 2 dimensions, so that the population can be drawn on the function's
contour.

## Good results

The minimum is −9.66015. L-SHADE reaches the target with 58,440 of its 100,000 evaluations. CMA-ES
with IPOP restarts ends at an error of 0.3306 with the budget spent. Four of its genes are in the
wrong valley: x₄ at 1.114 instead of 1.923 costs 0.0418, x₆ at 0.910 instead of π/2 costs 0.2114,
x₇ at 1.877 instead of 1.454 costs 0.0398, and x₉ at 1.283 instead of 1.656 costs 0.0376. The
other six are at the minimum.
