---
title: Bohachevsky 1
category: continuous
summary: Minimize Bohachevsky's first function, a bowl with a ripple of cosines at its center, with CMA-ES without and with restarts and DE from 30 seeds.
reference: "Bohachevsky, I. O., Johnson, M. E. and Stein, M. L. (1986). Generalized simulated annealing for function optimization. Technometrics 28(3): 209-217."
reference_url: "https://doi.org/10.1080/00401706.1986.10488128"
optimum: "0 at (0, 0)"
languages: [rust, python]
order: 76
family: Bohachevsky
tab: Bohachevsky 1
---

# Bohachevsky 1

## The problem

Bohachevsky's first function is a function of two variables to minimize:

```text
f(x₁, x₂) = x₁² + 2x₂² − 0.3 cos(3πx₁) − 0.4 cos(4πx₂) + 0.7,   x₁, x₂ in [−100, 100]
```

The three Bohachevsky functions are credited to Bohachevsky, Johnson and Stein (1986), who tested
their generalized simulated annealing on such functions; the paper couldn't be read, and genoxide
takes the definition and the bounds from Jamil and Yang's (2013, function 17) restatement; Adorio
(2005, MVF library, section 2.3) gives the same function on [−50, 50]. The definition is still to be
checked against the original.

The minimum is 0 at the origin, and it's the global minimum: the cosine terms add 0.7 − 0.3
cos(3πx₁) − 0.4 cos(4πx₂) ≥ 0 to x₁² + 2x₂², and both parts are 0 only at the origin. genoxide's
`problems::Bohachevsky1` gives it as proven.

## What makes it hard

The bowl x₁² + 2x₂² dominates the box: at (100, 100), f is about 30,000, and the cosines change it
by less than 1. So from afar, the function is a smooth bowl, and every algorithm finds its bottom.
But near the origin the cosines win: a local minimum needs the bowl's slope, 2x₁ and 4x₂, to be
below the cosines' largest slopes, which it is only where |x₁| < 1.41 and |x₂| < 1.26, and there the
function is rippled. The cosines have periods 2/3 in x₁ and 1/2 in x₂, and are separable: the ripple
is a grid of dimples, and each gene's local minima don't depend on the other's. Newton's method from
a grid of points over [−1.5, 1.5]² finds 25 local minima, the nearest to the origin 0.41293 at
(±0.61861, 0), 0.46988 at (0, ±0.46953) and 0.88281 at (±0.61861, ±0.46953).

So the search is easy until the last step: the ripple, a patch of less than 0.02% of the box, is
where a search that shrinks its steps too early settles in the wrong dimple.

## Representation

A `Real` genome of 2 genes, each in [−100, 100]: the point (x₁, x₂) itself. The fitness is f, to
minimize. The function, its bounds and its minimum are genoxide's `problems::Bohachevsky1`.

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

[The project page](https://tachsin.gr/projects/genoxide/examples/bohachevsky1) plays this run
back.

## Good results

A good result reaches 0 in every run. Without restarts, 26 of the 30 runs of CMA-ES reach the
minimum, after a median of 423 evaluations. The other 4 settle in dimples around it: seed 20 in the
one at (0, −0.46953), with 0.46988, and seeds 10, 26 and 27 elsewhere, their best points, 0.073,
0.085 and 0.250, being samples seen before they settled. With IPOP restarts, all 30 runs reach it,
after a median of 438 evaluations and at most 1,344: a restart samples the ripple again with twice
the population, and one converges at the origin. SHADE reaches it in every run as well, after a
median of 1,360 evaluations: its population of 20 covers the ripple and its difference vectors
shrink as it gathers at the bottom.
