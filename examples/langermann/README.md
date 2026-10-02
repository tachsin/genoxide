---
title: Langermann
category: continuous
summary: Minimize Langermann's function in two dimensions, rings of ripples around five centers, whose deepest well is a narrow one beside a wider one nearly as deep, with CMA-ES, PSO and DE from 30 seeds.
reference: "Bersini, H., Dorigo, M., Langerman, S., Seront, G. and Gambardella, L. (1996). Results of the first international contest on evolutionary optimisation (1st ICEO). Proceedings of IEEE International Conference on Evolutionary Computation: 611-615. Two-dimensional form and constants as in Molga, M. and Smutnicki, C. (2005). Test functions for optimization needs."
reference_url: "https://doi.org/10.1109/ICEC.1996.542670"
optimum: "−4.15581 at (2.79340, 1.59723) (best known)"
languages: [rust, python]
order: 85
---

# Langermann

## The problem

Langermann's function is a sum of five damped ripples, each centered on a point aᵢ, to minimize:

```text
f(x) = Σᵢ₌₁⁵ cᵢ exp(−dᵢ/π) cos(π dᵢ),   dᵢ = (x₁ − aᵢ₁)² + (x₂ − aᵢ₂)²,   x₁, x₂ in [0, 10]
```

with c = (1, 2, 5, 2, 3) and the centers (3, 5), (5, 2), (2, 1), (1, 4) and (7, 9).

It comes from the first International Contest on Evolutionary Optimisation (Bersini, Dorigo,
Langerman, Seront and Gambardella, 1996), which couldn't be read; the contests' Langermann functions
have 5 and 10 dimensions, a minus sign in front of the sum, and centers in the organizers' code.
genoxide takes this two-dimensional form, its sign and its constants from Molga and Smutnicki (2005,
section 2.10), which gives no bounds, and the bounds [0, 10] from Surjanovic and Bingham's Virtual
Library of Simulation Experiments. With the contests' minus sign, the minimum would be −5.16213 at
(2.00299, 1.00610), which is this form's maximum.

The best known minimum is −4.155809291847785 at (2.7934022086450367, 1.5972325013283601): Newton's
method, started from the lowest points of a 2001 × 2001 grid over the box, finds it and, as the
next, −4.127576741310136 at (1.991205862734151, 1.9886198019478405). It's not proven global, and
genoxide's `problems::Langermann` gives it as a best known value.

## What makes it hard

Each center makes rings of ripples: cos(π dᵢ) changes sign each time the squared distance dᵢ grows
by 1, so the rings crowd together away from the center, while exp(−dᵢ/π) damps them. Where the rings
of different centers overlap, they interfere: on a 1001 × 1001 grid over the box, 2,412 points are
lower than their eight neighbors: the function has on the order of two thousand local minima.

The two deepest are neighbors, 0.89 apart, and nearly as deep: −4.15581 and −4.12758, both on the
first trough around the heaviest center, (2, 1) with c = 5, where cos(π d) is −1 at d ≈ 1. Both
wells are small: the area where f is below −4.0 is 0.017 in each, one part in 6,000 of the box, and
below −4.1, where the deepest is twice as wide as the other, 0.0036. A search that narrows down on
one of them, or on any of the other deep rings nearby, has to cross a ridge to get to the right one.

## Representation

A `Real` genome of 2 genes, each in [0, 10]: the point (x₁, x₂) itself. The fitness is f, to
minimize. The function, its bounds and its best known minimum are genoxide's `problems::Langermann`.

## Algorithm

Five algorithms, each from seeds 1 to 30, each run stopping once its value is within 1e-8 of the
best known minimum, or after 50,000 evaluations:

- CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195), with genoxide's
  defaults: a population of 6 and a step size of 0.3 of each gene's range, 3, from a random start;
- CMA-ES with IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776): a run that has
  converged starts again from a random point with twice the population;
- particle swarm optimization (Kennedy and Eberhart, 1995), 40 particles with Clerc and Kennedy's
  constriction coefficients, each pulled towards its own best position and the swarm's (a global
  topology);
- the same swarm with a ring topology: each particle is pulled towards the best of its two neighbors
  in a ring instead, so good positions spread slowly and the swarm explores longer;
- differential evolution with genoxide's defaults, SHADE (Tanabe and Fukunaga, CEC 2013), with a
  population of 20, and its restarts once the population has converged.

## Output

The first line gives the best known minimum, the seeds and the budget. Then a row per algorithm: how
many of the 30 runs reach the best known minimum and how many don't, and the median and largest
number of evaluations of the runs that reach it. In Python, `run` evaluates the function in Rust, so
both versions print the same table.

The page's plot shows the 30 runs of CMA-ES without restarts, each at its best point so far, over
the function's contour, with the best known minimum marked: they end scattered over the local minima
around it. A curve gives the best and the median run's error, on a logarithmic axis.

[The project page](https://tachsin.gr/projects/genoxide/examples/langermann) plays this run back.

## Good results

A good result reaches the best known minimum in every run. DE does, after a median of 3,330
evaluations and at most 16,838, and so does the ring swarm, after a median of 6,500 and at most
39,640.

CMA-ES without restarts never does: all 30 runs settle in local minima, their best points in the
second well, −4.12758, for seeds 8 and 20, in other troughs near the minimum for 11 more, between
−4.12 and −3.9, and the rest farther away, 6 of them near −2.19, around the center (7, 9). It
narrows its distribution quickly on the well that its first samples favor. With IPOP restarts, 25
runs reach the minimum, after a median of 5,568 evaluations; the other 5 have found the right well,
their best points between −4.1467 and −4.1557, when the budget ends. The swarm with a global
topology reaches it in 21 runs: 6 of the others gather in the second well, 2 around the local
minimum −2.194 near the center (7, 9), and one at −3.779.
