---
title: Bohachevsky 2
category: continuous
summary: Minimize Bohachevsky's second function, a bowl with a product of cosines at its center, with CMA-ES without and with restarts and DE from 30 seeds.
reference: "Bohachevsky, I. O., Johnson, M. E. and Stein, M. L. (1986). Generalized simulated annealing for function optimization. Technometrics 28(3): 209-217."
reference_url: "https://doi.org/10.1080/00401706.1986.10488128"
optimum: "0 at (0, 0)"
languages: [rust, python]
order: 77
family: Bohachevsky
tab: Bohachevsky 2
---

# Bohachevsky 2

## The problem

Bohachevsky's second function is a function of two variables to minimize:

```text
f(x₁, x₂) = x₁² + 2x₂² − 0.3 cos(3πx₁) cos(4πx₂) + 0.3,   x₁, x₂ in [−100, 100]
```

The three Bohachevsky functions are credited to Bohachevsky, Johnson and Stein (1986), who tested
their generalized simulated annealing on such functions; the paper couldn't be read, and whether it
has this function is unconfirmed. genoxide takes the definition and the bounds from Jamil and Yang's
(2013, function 18) restatement, which prints `0.3 cos(3πx₁) · 0.4 cos(4πx₂)` for the product;
Adorio (2005, MVF library, section 2.3) gives this form, on [−50, 50]. The definition is still to be
checked against the original.

The minimum is 0 at the origin, and it's the global minimum: 0.3 − 0.3 cos(3πx₁) cos(4πx₂) is at
least 0, and x₁² + 2x₂² is 0 only at the origin. genoxide's `problems::Bohachevsky2` gives it as
proven.

## What makes it hard

The bowl x₁² + 2x₂² dominates the box: at (100, 100), f is about 30,000, and the cosines change it
by less than 1. So from afar, the function is a smooth bowl, and every algorithm finds its bottom.
But near the origin the cosines win: a local minimum needs the bowl's slope, 2x₁ and 4x₂, to be
below the cosines' largest slopes, which it is only where |x₁| < 1.41 and |x₂| < 0.94, and there the
function is rippled. The product of cosines couples the genes: the ripple is a checkerboard, whose
dimples sit where both cosines are 1 or both are −1, so the nearest ones are diagonal from the
origin. Newton's method from a grid of points over [−1.5, 1.5]² finds 25 local minima, the nearest
to the origin 0.21831 at (±0.30916, ±0.22986), the four nearest, and 0.41293 at (±0.61861, 0).

So the search is easy until the last step: the ripple, a patch of less than 0.02% of the box, is
where a search that shrinks its steps too early settles in the wrong dimple.

## Representation

A `Real` genome of 2 genes, each in [−100, 100]: the point (x₁, x₂) itself. The fitness is f, to
minimize. The function, its bounds and its minimum are genoxide's `problems::Bohachevsky2`.

## Algorithm

Three algorithms, each from seeds 1 to 30, each run stopping once its value is within 1e-8 of 0, or
after 10,000 evaluations:

- CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195), with genoxide's
  defaults: a population of 6 and a step size of 0.3 of each gene's range, 60, from a random start;
- CMA-ES with IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776): a run that has
  converged starts again from a random point with twice the population;
- differential evolution with genoxide's defaults, SHADE (Tanabe and Fukunaga, CEC 2013), with a
  population of 20.

## Output

The first line gives the minimum, the seeds and the budget. Then a row per algorithm: how many of
the 30 runs reach the minimum and how many don't, and the median and largest number of evaluations
of the runs that reach it. In Python, `run` evaluates the function in Rust, so both versions print
the same table.

The page's plot shows the 30 runs of CMA-ES without restarts, each at its best point so far, over
the function's contour, with the minimum marked. A curve gives the best and the median run's error,
on a logarithmic axis. A run's best point is the best sample it has seen, which isn't always where
it converged.

[The project page](https://tachsin.gr/projects/genoxide/examples/bohachevsky2) plays this run
back.

## Good results

A good result reaches 0 in every run. Without restarts, 28 of the 30 runs of CMA-ES reach the
minimum, after a median of 396 evaluations. The other 2 settle in dimples around it: seed 16 in the
one at (−0.30916, −0.22986), with 0.21831, and seed 23 elsewhere, its best point, 0.147, being a
sample seen before it settled. With IPOP restarts, all 30 runs reach it, after a median of 402
evaluations and at most 1,254. SHADE reaches it in every run as well, after a median of 1,400
evaluations.
