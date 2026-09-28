---
title: Hartmann 6-D
category: continuous
summary: Minimize Hartmann's function in 6 dimensions, whose two deepest minima are nearly as deep and far apart, with CMA-ES from 30 seeds, without and with restarts.
reference: "Hartman, J. K. (1973). Some experiments in global optimization. Naval Research Logistics Quarterly 20(3): 569-576. Constants as tabulated in Dixon, L. C. W. and Szegö, G. P. (1978). The global optimisation problem: an introduction. In Towards Global Optimisation 2, North-Holland: 1-15."
reference_url: "https://doi.org/10.1002/nav.3800200316"
optimum: "−3.32237 at (0.20169, 0.15001, 0.47687, 0.27533, 0.31165, 0.65730) (best known)"
languages: [rust, python]
order: 59
family: Hartmann
tab: 6-D
---

# Hartmann 6-D

## The problem

Hartmann's function in 6 dimensions is a sum of four Gaussian wells, to minimize:

```text
f(x) = −Σᵢ₌₁⁴ cᵢ exp(−Σⱼ₌₁⁶ aᵢⱼ (xⱼ − pᵢⱼ)²),   each xⱼ in [0, 1]
```

with c = (1, 1.2, 3, 3.2) as in [3 dimensions](../hartmann3/), and

| i | aᵢ₁ … aᵢ₆ | pᵢ₁ … pᵢ₆ |
|---|---|---|
| 1 | 10, 3, 17, 3.5, 1.7, 8 | 0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886 |
| 2 | 0.05, 10, 17, 0.1, 8, 14 | 0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991 |
| 3 | 3, 3.5, 1.7, 10, 17, 8 | 0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650 |
| 4 | 17, 8, 0.05, 10, 0.1, 14 | 0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381 |

The function is Hartman's (1973); the constants are those that Dixon and Szegö (1978) tabulate.
Hartman's report (1972) defines the form with random constants, and Dixon and Szegö's book isn't
online: genoxide takes the constants from Yao, Liu and Lin's (1999, table XIII) reprint, but for
p₃₂, which they print as 0.1415. With 0.1415, the minimum is −3.32200 at (0.2017, 0.1468, 0.4767,
0.2753, 0.3117, 0.6573), which isn't the point Yao, Liu and Lin give themselves; with 0.1451, as
Jamil and Yang (2013) print it, the minimum is the one that later papers quote.

genoxide's `problems::Hartmann6` gives the minimum as −3.3223680114155147 at (0.20168951100670543,
0.15001069182345797, 0.476873974221897, 0.2753324304940561, 0.31165161660011326,
0.6573005340656204): the point where the gradient is 0, computed to 40 digits by Newton's method.
Later papers quote −3.32237 from Dixon and Szegö. It's the best known minimum, not proven global.

## What makes it hard

The function has two deep local minima, close in value and far apart:

| minimum | f | x₁ | x₂ | x₃ | x₄ | x₅ | x₆ |
|---|---|---|---|---|---|---|---|
| global | −3.32237 | 0.20169 | 0.15001 | 0.47687 | 0.27533 | 0.31165 | 0.65730 |
| other | −3.20316 | 0.40465 | 0.88244 | 0.84610 | 0.57399 | 0.13893 | 0.03850 |

They are 1.10 apart, differ in every gene, and halfway between them f is only −0.79. Nothing in the
function's shape leads from one to the other.

The other minimum is well 4 alone, the one with the largest weight, c₄ = 3.2. The global minimum is
well 3, with c₃ = 3, which well 1 deepens by 0.41 where they overlap. So the heaviest well isn't
where the minimum is, and the two differ by 0.119, 3.6% of the minimum. Local searches from 2,000
random points end at the global minimum two times in three, and at the other one otherwise.

## Representation

A `Real` genome of 6 genes, each in [0, 1]: the point x itself. The fitness is f(x), to minimize.
The function, its bounds and its best known minimum are genoxide's `problems::Hartmann6`.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195), which samples a
population from a normal distribution and adapts its mean, step size and covariance matrix, with
genoxide's defaults: a population of 4 + ⌊3 ln 6⌋ = 9, and a step size of 0.3 of each gene's
range, from a random start. It runs from seeds 1 to 30, twice:

- without restarts: a run converges into one of the basins, and stays there until its budget is
  spent;
- with IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776): a run that has converged
  starts again from a random point with twice the population.

Each run stops once its value is within 1e-6 of the best known minimum, or after 10,000
evaluations. A run is counted at the other minimum if its best value is within 1e-6 of −3.20316.

## Output

The first line gives the best known minimum, the seeds and the budget. Then a row per algorithm:
how many of the 30 runs reach the best known minimum, how many end at the other minimum and how many
elsewhere, and the median and largest number of evaluations of the runs that reach the minimum.
In Python, `run` evaluates the function in Rust, so both versions print the same table.

The page's plot is the run with IPOP restarts from the first seed whose run without restarts ends at
the other minimum, seed 5: each gene of its best point so far on [0, 1], with the best known
minimum's value marked, and a curve of the best and of the population's median value's distance
above the best known minimum, on a logarithmic axis. Its first descent ends at the other minimum,
where the best point's genes sit far from the marks and the curve stays at 0.119; after a restart,
the genes move to the marks.

[The project page](https://tachsin.gr/projects/genoxide/examples/hartmann6) plays this run back.

## Good results

A good result reaches −3.32237 every time. CMA-ES without restarts reaches it from 21 of the 30
seeds, after a median of 648 evaluations, and the other 9 runs end at −3.20316. With IPOP restarts,
all 30 runs reach it: the 21 do the same as without restarts, and the 9 others after one or two
restarts.

The share of runs that end at the other minimum isn't the seeds': over seeds 1 to 1,000, 21% of the
runs without restarts end there (30% of seeds 1 to 30). Those that restart once use 2,889 to 3,123
evaluations, and the two that restart twice, from seeds 23 and 28, 6,390 and 6,489.
