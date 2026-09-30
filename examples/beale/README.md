---
title: Beale
category: continuous
summary: Minimize Beale's function, a flat curved valley between walls that rise to 1.8·10⁵ at the corners, with CMA-ES without and with restarts, PSO and a GA from 30 seeds.
reference: "Beale, E. M. L. (1958). On an Iterative Method for Finding a Local Minimum of a Function of More than One Variable. Technical Report 25, Statistical Techniques Research Group, Princeton University."
reference_url: ""
optimum: "0 at (3, 0.5)"
languages: [rust, python]
order: 63
---

# Beale

## The problem

Beale's function is a sum of three squares in two variables, to minimize:

```text
f(x₁, x₂) = (1.5 − x₁ + x₁x₂)² + (2.25 − x₁ + x₁x₂²)² + (2.625 − x₁ + x₁x₂³)²,
x₁, x₂ in [−4.5, 4.5]
```

It comes from Beale's report of 1958, which couldn't be read: genoxide takes the definition and
the bounds from Jamil and Yang's (2013, function 10) and Laguna and Martí's (2005, function 6)
restatements, which agree, and they are still to be checked against the original.

The minimum is 0 at (3, 0.5), and it's the only point where all three squares are 0: the terms are
0 where x₁ (1 − x₂ᵏ) is 1.5, 2.25 and 2.625 for k = 1, 2, 3, and the ratios 1 + x₂ = 1.5 and
1 + x₂ + x₂² = 1.75 give x₂ = 0.5, then x₁ = 3. genoxide's `problems::Beale` gives it as proven.

## What makes it hard

The function rises from 0 at the minimum to between 169,681 and 181,854 at the four corners,
where the terms in x₂³ dominate. Half of the box is above 375, and a random point falls
below 1 with a probability of 1.4%. Near the minimum, the function is a valley that curves along
x₁ (1 − x₂) = 1.5 and is flat along its floor: the Hessian at (3, 0.5) has the eigenvalues 0.30
and 49.0, a condition number of 162.

Newton's method, started from each of 8,281 points of a grid over the box, finds (3, 0.5) and no
other local minimum inside the box. But on its edge there is one: along x₁ = −4.5, the function has
a minimum of 0.76207 at x₂ = 1.18643, where its slope in x₁ is 0.063, so it would go on falling if
x₁ could go below −4.5. A search that follows the valley the wrong way ends against the bound.

## Representation

A `Real` genome of 2 genes, each in [−4.5, 4.5]: the point (x₁, x₂) itself. The fitness is f, to
minimize. The function, its bounds and its minimum are genoxide's `problems::Beale`.

## Algorithm

Four algorithms, each from seeds 1 to 30, each run stopping once its value is within 1e-8 of 0, or
after 10,000 evaluations:

- CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195), which samples a
  population from a normal distribution and adapts its mean, step size and covariance matrix, with
  genoxide's defaults: a population of 6 and a step size of 0.3 of each gene's range, from a random
  start;
- CMA-ES with IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776): a run that has
  converged starts again from a random point with twice the population;
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

The page's plot shows the 30 runs of CMA-ES without restarts, each at its best point so far, over
the function's contour, with the minimum marked. A curve gives the best and the median run's error,
on a logarithmic axis.

[The project page](https://tachsin.gr/projects/genoxide/examples/beale) plays this run back.

## Good results

A good result reaches 0 in every run. CMA-ES with IPOP restarts does, after a median of 363
evaluations and at most 1,866. Without restarts, 29 of the 30 runs do, after a median of 360: the
other, seed 16, follows the valley to the bound and converges at the edge minimum, 0.76207 at
(−4.5, 1.18643), where it stays; the restarts of IPOP start such a run again.

PSO reaches the minimum in every run too, but after a median of 3,180 evaluations, nearly nine times
as many: its moves don't adapt their shape to the narrow valley. The genetic algorithm reaches 1e-8
in only 2 of the 30 runs: its operators find the valley and its floor, the median run ending at
3.2e-7, but their steps don't shrink with the error, and the last digits cost more than the budget.
One of its runs ends at the edge minimum as well.
