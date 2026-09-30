---
title: Three-hump camel
category: continuous
summary: Minimize the three-hump camel function, a global minimum between two local ones, with CMA-ES without and with restarts and DE from 30 seeds.
reference: "Jamil, M. and Yang, X.-S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation 4(2): 150-194."
reference_url: "https://doi.org/10.1504/IJMMNO.2013.055204"
optimum: "0 at (0, 0)"
languages: [rust, python]
order: 62
---

# Three-hump camel

## The problem

The three-hump camel function is a polynomial in two variables, to minimize:

```text
f(x₁, x₂) = 2x₁² − 1.05x₁⁴ + x₁⁶/6 + x₁x₂ + x₂²,   x₁, x₂ in [−5, 5]
```

Its origin is unknown: it's usually credited to Dixon and Szegö (1978) or to Branin (1972), neither
of which could be checked. genoxide takes the definition and the bounds from Jamil and Yang's (2013,
function 29) and Adorio's (2005, MVF library, section 2.7) restatements, which agree, and they are
still to be checked against an original.

The minimum is 0 at the origin, and it's the global minimum: for a given x₁, the lowest value over
x₂ is at x₂ = −x₁/2, where f is x₁² (1.75 − 1.05x₁² + x₁⁴/6), and the quadratic in x₁² has no real
root (1.05² < 4 · 1.75/6), so f is positive but at the origin. genoxide's `problems::ThreeHumpCamel`
gives it as proven.

## What makes it hard

The name counts the humps of its negative: the function has three minima, the global one at the
origin and two local ones, symmetric about it, at ±(1.74755, −0.87378) with 0.29864, where x₁'s
polynomial 1.75 − 1.05x₁² + x₁⁴/6 dips again. They lie at the bottom of the same long valley along
x₂ = −x₁/2, separated from the minimum by saddles only 0.88 high. Beyond them, the sixth power takes
over: at (5, 5), f is 2,048.

A local search that starts in the outer part of the valley ends in a local minimum, and a population
that shrinks before it has sampled the middle can too.

## Representation

A `Real` genome of 2 genes, each in [−5, 5]: the point (x₁, x₂) itself. The fitness is f, to
minimize. The function, its bounds and its minimum are genoxide's `problems::ThreeHumpCamel`.

## Algorithm

Three algorithms, each from seeds 1 to 30, each run stopping once its value is within 1e-8 of 0, or
after 10,000 evaluations:

- CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195), with genoxide's
  defaults: a population of 6 and a step size of 0.3 of each gene's range, 3, from a random start;
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

[The project page](https://tachsin.gr/projects/genoxide/examples/three-hump-camel) plays this run
back.

## Good results

A good result reaches 0 in every run. Without restarts, 28 of the 30 runs of CMA-ES reach the
minimum, after a median of 279 evaluations. The other 2 settle in a local minimum: seed 30 in the
one at (1.74755, −0.87378), with 0.29864, and seed 12 in the other, its best point, 0.161, being a
sample seen before it settled. With IPOP restarts, all 30 runs reach the minimum, after a median of
282 evaluations and at most 894. SHADE reaches it in every run as well, after a median of 950
evaluations.
