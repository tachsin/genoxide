---
title: Rosenbrock
category: continuous
summary: Minimize a 30-dimensional function whose minimum lies at the end of a narrow curved valley.
reference: "Rosenbrock, H. H. (1960). An automatic method for finding the greatest or least value of a function. The Computer Journal 3(3): 175-184."
reference_url: "https://doi.org/10.1093/comjnl/3.3.175"
optimum: "0 (at (1, …, 1))"
languages: [rust, python]
order: 55
trace_note: "Recorded from another run: CMA-ES in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Rosenbrock

## The problem

Rosenbrock's function sums the same term over each pair of neighboring genes:

```text
f(x) = Σᵢ₌₁ⁿ⁻¹ [100 (xᵢ₊₁ − xᵢ²)² + (xᵢ − 1)²],   each xᵢ in [−30, 30]
```

Rosenbrock (1960) defined it in two dimensions, with no bounds, to test his method of
minimization. This is the chained form in n dimensions, with the bounds of Yao, Liu and Lin (1999,
IEEE Transactions on Evolutionary Computation 3(2): 82-102, function f5). It isn't the "extended"
form, which sums over the separate pairs x₁x₂, x₃x₄, and so on. The minimum is 0, at (1, …, 1).
Here n = 30, Yao, Liu and Lin's dimension.

In two dimensions, the first part of the term, 100 (x₂ − x₁²)², is 0 on the parabola x₂ = x₁². That
parabola is the floor of a valley. Along the floor, f is (x₁ − 1)², which falls by only 4 from x₁ =
−1 to x₁ = 1. Across the floor, f rises fast: a step of 0.1 away from (1, 1) in x₂ costs 1.
Rosenbrock started his searches from (−1.2, 1), where f is 24.2.

## What makes it hard

Finding the valley is easy: the steep walls lead to it. Following it is hard. The floor is narrow
and bends, so a direction that goes along the floor at one point leaves it a little further on.
An algorithm that steps along the axes, or in all directions alike, must take steps small enough
to stay inside the valley, and needs many of them to travel along it.

In n dimensions the terms form a chain: each gene wants to be the square of the one before it,
xᵢ₊₁ ≈ xᵢ², and x₁ wants to be 1. The genes can't be solved one at a time. Yao, Liu and Lin count
the function as unimodal, but for n from 4 to 30 it also has a local minimum (Shang and Qiu, 2006,
Evolutionary Computation 14(1): 119-126), with x₁ near −1 from n = 6 on.

## Representation

A `Real` genome of 30 genes, each in [−30, 30]: the point x itself. The fitness is f(x), to
minimize. The function is genoxide's `problems::Rosenbrock`, which brings its bounds and its
minimum.

## Algorithm

Three algorithms, each with a budget of 10,000 evaluations per dimension, 300,000 in all, and a
target of 1e-8 above the minimum.

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. The covariance
matrix is what suits this function: the distribution stretches along the directions of past
successful steps, which here are the directions of the valley, and it keeps turning as the valley
bends. It uses genoxide's defaults: a population of 4 + ⌊3 ln 30⌋ = 14, a step size of 0.3 of each
gene's range, a random start, and no restarts.

L-SHADE (Tanabe and Fukunaga, 2014, IEEE CEC 2014: 1658-1665) is a differential evolution that
adapts its scale factor and crossover rate from successful trials. Its steps are differences
between members of the population, so they line up with the valley once the population does. Its
population starts at 18 times the number of genes, 540, and shrinks linearly to 4 over the budget.

Particle swarm optimization (Kennedy and Eberhart, 1995, Proceedings of ICNN'95: 1942-1948) moves
40 particles, each pulled towards its own best point and the swarm's best, with Clerc and
Kennedy's constriction coefficients (2002, IEEE Transactions on Evolutionary Computation 6(1):
58-73). It has no model of the valley's direction.

## Output

The first line gives the dimension, the minimum and the budget. Then one line per algorithm: the
error of its best point, which is its value minus the minimum, to 4 decimals; the evaluations it
took; and how many of its 30 genes are more than 0.01 from 1. A run stops as soon as it is within
1e-8 of the minimum, so an error of 0.0000 means that it met the target. In Python, `run`
evaluates the function in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/rosenbrock) plays back another
run: CMA-ES on Rosenbrock in 2 dimensions, so that the population can be drawn on the function's
contour.

## Good results

The minimum is 0. CMA-ES reaches the target after about 47,000 evaluations, and L-SHADE after
about 250,000: the adapted covariance matrix makes CMA-ES five times faster.

PSO ends at an error of 6.79 with the budget spent. Its best point is on the valley floor, but far
along it from the minimum: x₁ to x₁₂ are within 0.01 of 1, and from there each gene is about the
square of the one before, down to x₂₁ ≈ 0.75, x₂₄ ≈ 0.12 and x₃₀ ≈ 0. The swarm found the valley
and moved along it too slowly to reach the end.
