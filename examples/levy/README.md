---
title: Levy
category: continuous
summary: Minimize a 30-dimensional function with many local minima around a broad bowl, where a converged search needs a restart.
reference: "Levy, A. V. and Montalvo, A. (1985). The tunneling algorithm for the global minimization of functions. SIAM Journal on Scientific and Statistical Computing 6(1): 15-29."
reference_url: "https://doi.org/10.1137/0906002"
optimum: "0 (at (1, …, 1))"
languages: [rust, python]
order: 46
trace_note: "Recorded from another run: L-SHADE in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Levy

## The problem

Levy's function first maps each gene xᵢ to wᵢ = 1 + (xᵢ − 1) / 4, and then adds three kinds of
term:

```text
f(x) = sin²(πw₁)
     + Σᵢ₌₁ⁿ⁻¹ (wᵢ − 1)² [1 + 10 sin²(πwᵢ + 1)]
     + (wₙ − 1)² [1 + sin²(2πwₙ)],              each xᵢ in [−10, 10]
```

Its minimum is 0, at (1, …, 1), where every wᵢ is 1. Here n = 30.

The function is usually credited to Levy and Montalvo (1985), but several functions carry Levy's
name. This is the form of Surjanovic and Bingham's Virtual Library of Simulation Experiments, with
sin²(πwᵢ + 1) and the bounds [−10, 10]. Laguna and Martí (2005, function 38) print xₙ instead of
wₙ in the last sine; genoxide uses wₙ, like the other terms. The Levy-Montalvo functions as Yao,
Liu and Lin (1999, f12 and f13) restate them have sin²(πyᵢ₊₁) instead, of which πwᵢ + 1 may be a
misreading. The minimum is 0 at (1, …, 1) in every form. genoxide hasn't yet checked its form
against the original paper.

## What makes it hard

Each gene has terms of its own, so the genes don't interact. A middle term is a product: a bowl,
(wᵢ − 1)², times a factor that oscillates between 1 and 11 and repeats every 4 units of xᵢ.
Across [−10, 10], each middle gene has five local minima besides the global one at 1; the first
gene has four, and the last, whose factor repeats every 2 units, seven. In 30 dimensions that makes
5 × 6²⁸ × 8 ≈ 2.5 × 10²³ minima, one of them global. Far from the minimum, the dips are deep. Near
it, they are shallow: the nearest one, at xᵢ ≈ −0.093, has a value of 0.0895, and its rim, at xᵢ ≈
0.15, is only 0.004 higher.

The broad bowl helps: seen from far enough, the function falls towards (1, …, 1). But a search that
has contracted onto one dip can't see the bowl any more. It then needs a step large enough to cross
the rim, and a converged search no longer makes such steps.

## Representation

A `Real` genome of 30 genes, each in [−10, 10]: the point x itself. The fitness is f(x), to
minimize. The function is genoxide's `problems::Levy`, which brings its bounds and its minimum. The
program runs it in 30 dimensions, genoxide's default and Yao, Liu and Lin's dimension: in 2
dimensions, every algorithm here solves it within a few hundred to a few thousand evaluations.

## Algorithm

Three runs, each with a budget of 10,000 evaluations per dimension, 300,000 in all, and a target of
1e-8 above the minimum.

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. It uses genoxide's
defaults: a population of 4 + ⌊3 ln 30⌋ = 14, a step size of 0.3 of each gene's range, and a random
start. The first run has no restarts. The second has IPOP restarts (Auger and Hansen, 2005, IEEE
CEC 2005: 1769-1776): a run that has converged starts again from a random point with twice the
population. genoxide's docs recommend IPOP for multimodal functions, and the comparison over
seeds shows why.

L-SHADE (Tanabe and Fukunaga, 2014, IEEE CEC 2014: 1658-1665) is a differential evolution that
adapts its scale factor and crossover rate from successful trials. Its population starts at 18 times
the number of genes, 540, and shrinks linearly to 4 over the budget, so it explores widely before it
contracts.

## Output

The first line gives the dimension, the minimum and the budget. Then one line per run: the error of
its best point, which is its value minus the minimum, to 4 decimals; the evaluations it took; and
how many of its 30 genes are more than 0.01 from 1. A run stops as soon as it is within 1e-8 of the
minimum, so an error of 0.0000 means that it met the target. In Python, `run` evaluates the
function in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/levy) plays back another run:
L-SHADE on Levy in 2 dimensions, so that the population can be drawn on the function's contour.

## Good results

The minimum is 0. With seed 1, CMA-ES without restarts reaches the target after 5,138 evaluations,
before it converges anywhere else, so the run with IPOP restarts is the same run. That's luck: with
seeds 1 to 25, CMA-ES without restarts reaches the target 6 times, and the other 19 runs end with
the budget spent, with 1 to 6 genes in dips, at errors of 0.0895 to 2.73. At 0.0895, 29 genes are
at 1, and one is at −0.093, in the nearest dip. Its step size shrank as the other genes converged,
and it can't cross the rim, however low. With IPOP restarts, CMA-ES reaches the target with all 25
seeds, after 4,592 to 32,508 evaluations. L-SHADE reaches it after about 100,000.
