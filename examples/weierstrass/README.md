---
title: Weierstrass
category: continuous
summary: Minimize the Weierstrass function, continuous but nowhere smooth, in 10 dimensions, with CMA-ES without and with restarts, DE, PSO and a GA from 10 seeds each.
reference: "Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y.-P., Auger, A. and Tiwari, S. (2005). Problem Definitions and Evaluation Criteria for the CEC 2005 Special Session on Real-Parameter Optimization. Nanyang Technological University and KanGAL report 2005005."
reference_url: "https://github.com/P-N-Suganthan/CEC2005"
optimum: "0 (at the origin)"
languages: [rust, python]
order: 95
trace_note: "Recorded from another run: CMA-ES with IPOP restarts in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Weierstrass

## The problem

The Weierstrass function sums cosines of growing frequency and shrinking amplitude:

```text
f(x) = Σᵢ Σₖ aᵏ cos(2π bᵏ (xᵢ + 0.5)) − n Σₖ aᵏ cos(π bᵏ),   a = 0.5, b = 3, k from 0 to 20
each xᵢ in [−0.5, 0.5]
```

Its minimum is 0, at the origin: each gene's sum is at least −Σ aᵏ, reached where every cosine is
−1, at the integers, and the second term is −n Σ aᵏ, since every bᵏ is odd. Here n = 10. It's the
function F11 of the CEC 2005 report (Suganthan et al. 2005), shifted and rotated there, with its
constants and bounds; Liang et al. (2006) and the CEC 2014 report have the same form, and BBOB's
f16 another one. It's named after Weierstrass's (1872) continuous, nowhere-differentiable function.

## What makes it hard

Each gene's sum is a fractal: ripples on ripples, each three times faster and half as high as the
last, down to 3²⁰ ≈ 3.5·10⁹ periods per unit. The function is continuous but differentiable only on
a set of points, and has local minima at every scale. A search that has found the right valley at
one scale still has to find it at every finer one.

## Representation

A `Real` genome of 10 genes, each in [−0.5, 0.5]: the point x itself. The fitness is f(x), to
minimize. The function is genoxide's `problems::Weierstrass`, which brings its bounds and its
minimum.

## Algorithm

Five algorithms, each from seeds 1 to 10, with a budget of 10,000 evaluations per dimension,
100,000 per run, and a target of 1e-8:

- CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195), which samples a
  population of 10 from a normal distribution and adapts its mean, its step size and its covariance
  matrix, from a step size of 0.3 of each gene's range and a random start;
- the same with IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776): a run that has
  converged starts again from a random point with twice the population;
- differential evolution with genoxide's defaults, SHADE (Tanabe and Fukunaga, CEC 2013), with a
  population of 100 and its restarts on stagnation;
- particle swarm optimization (Kennedy and Eberhart, 1995), 40 particles with Clerc and Kennedy's
  constriction coefficients and a global topology;
- a real-coded genetic algorithm: a population of 100, tournaments of 3, simulated binary crossover
  (Deb and Agrawal, 1995) with η = 15 and polynomial mutation with η = 20 at a rate of 1/10 per
  gene.

## Output

The first line gives the dimension, the seeds and the budget. Then a row per algorithm: how many
of its 10 runs reached the minimum, to within 1e-8, the median of their evaluations (a dash if none
did), and the median of every run's best error, to two significant digits. The function is
evaluated with genoxide's portable math, so the runs are the same on every platform, and in Python,
`run` evaluates it in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/weierstrass) plays back another run:
CMA-ES with IPOP restarts on the function in 2 dimensions, so that the population
can be drawn on its contour. It meets the target after 816 evaluations.

## Good results

The minimum is 0. CMA-ES with IPOP restarts reaches it in all 10 runs, after a median of 10,195
evaluations, and SHADE in all 10, after 42,300. PSO reaches it 8 times, after 17,720. CMA-ES without
restarts reaches it once, and its median run ends at 2.6e-3; the genetic algorithm never does, its
median run ending at 1.2e-2.
