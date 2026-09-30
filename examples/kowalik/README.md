---
title: Kowalik
category: continuous
summary: Fit Kowalik and Osborne's rational model to 11 data points in 4 dimensions, a least squares problem with local minima on the bounds and poles inside them, with CMA-ES without and with restarts, DE and PSO from 30 seeds.
reference: "Kowalik, J. S. and Osborne, M. R. (1968). Methods for Unconstrained Optimization Problems. American Elsevier. As restated in Yao, X., Liu, Y. and Lin, G. (1999). Evolutionary programming made faster. IEEE Transactions on Evolutionary Computation 3(2): 82-102."
reference_url: "https://doi.org/10.1109/4235.771163"
optimum: "3.07486e-4 at (0.19283, 0.19084, 0.12312, 0.13577) (best known)"
languages: [rust, python]
order: 76
---

# Kowalik

## The problem

Kowalik's function is the sum of the squared errors of a rational model, fitted to 11 data points
(bᵢ, aᵢ), to minimize over its four parameters:

```text
f(x) = Σᵢ₌₁¹¹ (aᵢ − x₁ (bᵢ² + bᵢx₂) / (bᵢ² + bᵢx₃ + x₄))²,   x₁, …, x₄ in [−5, 5]
```

| i | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aᵢ | 0.1957 | 0.1947 | 0.1735 | 0.1600 | 0.0844 | 0.0627 | 0.0456 | 0.0342 | 0.0323 | 0.0235 | 0.0246 |
| 1/bᵢ | 0.25 | 0.5 | 1 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |

The data are Kowalik and Osborne's, from their book of 1968, which couldn't be read. genoxide takes
the function, the data and the bounds from Yao, Liu and Lin (1999, f15, table XI), who give the
reciprocals of the bᵢ: so b₆ = 1/6, b₉ = 1/12 and b₁₀ = 1/14 exactly. NIST's Statistical Reference
Datasets have the same problem as MGH09, after Moré, Garbow and Hillstrom (1981), with the bᵢ
rounded to 0.167, 0.0833 and 0.0714; with them, the minimum is NIST's certified 3.0750560385e-4.

The best known minimum is 3.0748598780560606e-4 at (0.1928334529825086, 0.19083623878262915,
0.12311729627785713, 0.13576598998153702), where the gradient is 0, computed to 40 digits by
Newton's method and rounded. Yao, Liu and Lin give ≈ 0.0003075 at (0.1928, 0.1908, 0.1231, 0.1358).
It's the best of the local minima that 3,000 local searches from random points reach, not proven
global, and genoxide's `problems::Kowalik` gives it as a best known value.

## What makes it hard

The fit is poor in three of its four directions: the Hessian at the minimum has the eigenvalues
0.0029, 0.018, 0.92 and 8.8, a condition number of 3,050. Near the minimum, a long flat valley lets
x₂, x₃ and x₄ change together while the error hardly does.

Away from it, the model misbehaves. A denominator bᵢ² + bᵢx₃ + x₄ is 0 on a plane that crosses the
box, where the model has a pole and f is infinite. And the bounds hold local minima of their own:
0.020363 at (−0.389, −5, 1.303, 5), on two bounds, and 0.0012232 at (0.283, −5, −4.780, −2.718), on
one, besides interior ones such as 0.0015940 at (0.234, −1.292, −0.836, −0.551) and 0.00042429 at
(0.225, −0.415, −0.025, −0.178). Of 3,000 local searches (L-BFGS-B, within the bounds) from random
points, 15% end at the minimum.

## Representation

A `Real` genome of 4 genes, each in [−5, 5]: the parameters x₁, …, x₄. The fitness is f, to
minimize. The function, its data, its bounds and its best known minimum are genoxide's
`problems::Kowalik`.

## Algorithm

Four algorithms, each from seeds 1 to 30, each run stopping once its value is within 1e-8 of the
best known minimum, or after 20,000 evaluations:

- CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195), with genoxide's
  defaults: a population of 8 and a step size of 0.3 of each gene's range, 3, from a random start;
- CMA-ES with IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776): a run that has
  converged starts again from a random point with twice the population;
- differential evolution with genoxide's defaults, SHADE (Tanabe and Fukunaga, CEC 2013), with a
  population of 20;
- particle swarm optimization (Kennedy and Eberhart, 1995), 40 particles with Clerc and Kennedy's
  constriction coefficients and a global topology.

## Output

The first line gives the best known minimum, the seeds and the budget. Then a row per algorithm: how
many of the 30 runs reach the best known minimum and how many don't, and the median and largest
number of evaluations of the runs that reach it. In Python, `run` evaluates the function in Rust, so
both versions print the same table.

The page's plot shows the run of CMA-ES with IPOP restarts from seed 6, one of those that end on the
bounds without restarts: each parameter on its range with the best known minimum marked, and a curve
of the best and the median error, on a logarithmic axis.

[The project page](https://tachsin.gr/projects/genoxide/examples/kowalik) plays this run back.

## Good results

A good result reaches the best known minimum in every run. CMA-ES with IPOP restarts does, after a
median of 2,124 evaluations and at most 8,888.

Without restarts, 19 of the 30 runs of CMA-ES reach it, after a median of 1,976 evaluations. The
other 11 converge into local minima: 6 at 0.020363 on the bounds, one at 0.0012232 on a bound, 2 at
0.0015940, one at 0.00042429 and one at 0.0047969. The restarts of IPOP start them again, from a
random point with a larger population, until one converges at the minimum. SHADE reaches it in 25
runs, after a median of 4,280 evaluations; the other 5 end on the bounds, at 0.0012232 and 0.020363,
or at another local minimum, 0.00072383.

PSO reaches it in only 1 run, after 17,560 evaluations. The 8 closest of its other runs end in the
valley, between 3.084e-4 and 3.95e-4, their x₂, x₃ and x₄ still moving along its flat floor when the
budget ends.
