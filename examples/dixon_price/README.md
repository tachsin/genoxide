---
title: Dixon-Price
category: continuous
summary: Minimize the Dixon-Price function, a chain of curved valleys whose stationary point at 2/3 stalls nearly every search in 10 dimensions, with CMA-ES with restarts, DE and PSO from 30 seeds.
reference: "Dixon, L. C. W. and Price, R. C. (1989). Truncated Newton method for sparse unconstrained optimization using automatic differentiation. Journal of Optimization Theory and Applications 60(2): 261-275."
reference_url: "https://doi.org/10.1007/BF00940007"
optimum: "0 at xᵢ = 2^(−(2ⁱ − 2)/2ⁱ), and with xₙ negated"
languages: [rust, python]
order: 48
---

# Dixon-Price

## The problem

The Dixon-Price function chains each gene to the one before it, to minimize:

```text
f(x) = (x₁ − 1)² + Σᵢ₌₂ⁿ i (2xᵢ² − xᵢ₋₁)²,   each xᵢ in [−10, 10]
```

It's credited to Dixon and Price (1989), who tested a truncated Newton method on it; the paper
couldn't be read. genoxide takes the definition and the bounds from Jamil and Yang's (2013, function
48) and Laguna and Martí's (2005, function 37) restatements; Jamil and Yang's minimizer drops the
minus sign of its exponent, and Laguna and Martí's sum starts at i = 1, where x₀ is undefined. Both
are still to be checked against the original.

The minimum is 0, where every square is 0: x₁ = 1, and 2xᵢ² = xᵢ₋₁, so xᵢ = 2^(−(2ⁱ − 2)/2ⁱ): 1,
0.70711, 0.59460, 0.54525, …, towards 1/2. Each xᵢ must be positive for the next square to be 0,
except the last, which can take either sign: there are two minima, and genoxide's
`problems::DixonPrice` gives both, as proven.

## What makes it hard

The weights i grow along the chain, so the last genes' terms dominate: from a random start, a search
first makes each 2xᵢ² − xᵢ₋₁ small, and making all the genes small does that for all of them at
once. That leads to a trap. In 3 dimensions or more, (1/3, 0, …, 0) is a stationary point with the
value 2/3: the gradient is 0 there, and the Hessian positive semidefinite, singular only along xₙ.

It isn't a local minimum. The valley floor x₁ = (1 + 4x₂²)/3, xᵢ₊₁ = √(xᵢ/2), where the value is
(2/3) (1 − 2x₂²)², joins it to the minimum, going down all the way. But near the stationary point,
that floor is a cusp: xₙ grows as x₂^(1/2ⁿ⁻²), so in 10 dimensions, x₂ = 1e-6 already needs x₁₀ ≈
0.4. Moving along it means moving the last genes far while the first barely change, and searches
stall at 2/3 instead, the more often the more dimensions.

## Representation

A `Real` genome of n genes, each in [−10, 10]: the point x itself. The fitness is f, to minimize.
The function, its bounds and its minima are genoxide's `problems::DixonPrice`.

## Algorithm

Four algorithms, each from seeds 1 to 30, each run stopping once its value is within 1e-8 of 0, or
after 20,000 evaluations per dimension:

- CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) with IPOP restarts
  (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776), with genoxide's defaults: a population of 8 in
  5 dimensions and 10 in 10, a step size of 0.3 of each gene's range, and a restart with twice the
  population whenever a run has converged, in 5 and in 10 dimensions;
- differential evolution with genoxide's defaults, SHADE (Tanabe and Fukunaga, CEC 2013), with a
  population of 100 and restarts, in 10 dimensions;
- particle swarm optimization (Kennedy and Eberhart, 1995), 40 particles with Clerc and Kennedy's
  constriction coefficients and a global topology, in 10 dimensions;
- the same swarm with a ring topology, each particle pulled towards the best of its two neighbors
  instead of the swarm's best, in 10 dimensions.

## Output

The first line gives the minimum, the value at the stationary point, the seeds and the budget. Then
a row per algorithm and dimension: how many of the 30 runs reach the minimum, how many end at the
stationary point, to within 1e-8, and how many elsewhere, and the median and largest number of
evaluations of the runs that reach the minimum. In Python, `run` evaluates the function in Rust, so
both versions print the same table.

The page's plot shows the run of the ring swarm in 10 dimensions from seed 2, one of those that
reach the minimum: each gene on its range with the minimum marked, and a curve of the best and the
median error, on a logarithmic axis.

[The project page](https://tachsin.gr/projects/genoxide/examples/dixon-price) plays this run back.

## Good results

In 5 dimensions, CMA-ES with IPOP restarts reaches the minimum in every run, after a median of 3,184
evaluations and at most 21,288. In 10 dimensions it never does: all 30 runs, restarts and all, end
at the stationary point, as do all 30 runs of SHADE.

The swarms do better, since they don't converge on the stationary point as fast: the global one
reaches the minimum in 3 runs, after a median of 11,400 evaluations, and the ring in 13, after a
median of 34,880. The 2 runs of each that end elsewhere are on the valley floor just below 2/3, at
0.6666655 to 0.6666661, with x₁ = 1/3, x₂ between 0.0014 and 0.0020, and the later genes up the
floor's cusp, the last near ±0.485.

So in 10 dimensions, none of these methods reaches the minimum in every run: the best, the ring
swarm, does in 13 of 30. In 30 dimensions, the default, with 10,000 evaluations per dimension, a
separate test found it harder still: every run of CMA-ES, with and without restarts, and of SHADE
ended at the stationary point, the ring swarm reached the minimum in 1 of 30 runs, and a genetic
algorithm came within 1e-3 of it in 2.
