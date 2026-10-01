---
title: Booth
category: continuous
summary: Minimize Booth's function, a tilted quadratic bowl, with CMA-ES, DE, PSO and a GA from 30 seeds, and compare what each pays for eight digits of precision.
reference: "Jamil, M. and Yang, X.-S. (2013). A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation 4(2): 150-194."
reference_url: "https://doi.org/10.1504/IJMMNO.2013.055204"
optimum: "0 at (1, 3)"
languages: [rust, python]
order: 74
---

# Booth

## The problem

Booth's function is a sum of two squares of linear functions, to minimize:

```text
f(x₁, x₂) = (x₁ + 2x₂ − 7)² + (2x₁ + x₂ − 5)²,   x₁, x₂ in [−10, 10]
```

Its origin is unknown: genoxide takes the definition and the bounds from Jamil and Yang's (2013,
function 20) and Laguna and Martí's (2005, function 7) restatements, which agree, and they are
still to be checked against an original.

The minimum is 0 at (1, 3), where both linear functions are 0: the solution of x₁ + 2x₂ = 7 and
2x₁ + x₂ = 5. genoxide's `problems::Booth` gives it as proven.

## What makes it hard

Little: it's a convex quadratic, a bowl with elliptic level sets whose axes are turned by 45°. The
Hessian has the eigenvalues 2 (along x₁ = x₂) and 18 (across), a condition number of 9, so the bowl
is three times as long as it's wide, and neither axis of the ellipse is along a gene. There is one
minimum and no plateau: every algorithm should find it. The question is what each pays for the
last digits, since the example asks for an error of 1e-8, a distance of about 1e-4 from (1, 3).

## Representation

A `Real` genome of 2 genes, each in [−10, 10]: the point (x₁, x₂) itself. The fitness is f, to
minimize. The function, its bounds and its minimum are genoxide's `problems::Booth`.

## Algorithm

Four algorithms, each from seeds 1 to 30, each run stopping once its value is within 1e-8 of 0, or
after 10,000 evaluations:

- CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195), with genoxide's
  defaults: a population of 6 and a step size of 0.3 of each gene's range, from a random start. It
  learns the tilted ellipse in its covariance matrix;
- differential evolution with genoxide's defaults, SHADE (Tanabe and Fukunaga, CEC 2013), with a
  population of 20: current-to-pbest/1 mutation, binomial crossover, and F and CR adapted from
  their successes;
- particle swarm optimization (Kennedy and Eberhart, 1995), 40 particles with Clerc and Kennedy's
  constriction coefficients and a global topology;
- a real-coded genetic algorithm: a population of 50, tournaments of 3, simulated binary crossover
  (Deb and Agrawal, 1995) with η = 15 and polynomial mutation with η = 20 at a rate of 1/2 per
  gene.

## Output

The first line gives the minimum, the seeds and the budget. Then a row per algorithm: how many of
the 30 runs reach the minimum and how many don't, and the median and largest number of evaluations
of the runs that reach it. In Python, `run` evaluates the function in Rust, so both versions print
the same table.

The page's plot shows the 30 runs of CMA-ES, each at its best point so far, over the function's
contour, with the minimum marked. A curve gives the best and the median run's error, on a
logarithmic axis.

[The project page](https://tachsin.gr/projects/genoxide/examples/booth) plays this run back.

## Good results

A good result reaches 0 in every run. CMA-ES does, after a median of 312 evaluations and at most
414. SHADE does too, after 1,150, and PSO after 3,080: about ten times as many as CMA-ES, whose
steps shrink by a constant factor per generation once it has learned the ellipse.

The genetic algorithm reaches 1e-8 in 1 of the 30 runs. It finds the bowl's bottom as fast as the
others, but its polynomial mutation makes steps of a fixed share of the range, whatever the error:
the median run ends at 1.2e-6 and the worst at 2.3e-5, a hundred times above the target. On a
function this easy, the difference between the algorithms is only in how they refine.
