---
title: Penalized 2
category: continuous
summary: Minimize Yao, Liu and Lin's second penalized function in 30 dimensions, with CMA-ES without and with restarts, DE, PSO and a GA from 10 seeds each.
reference: "Yao, X., Liu, Y. and Lin, G. (1999). Evolutionary programming made faster. IEEE Transactions on Evolutionary Computation 3(2): 82-102."
reference_url: "https://doi.org/10.1109/4235.771163"
optimum: "0 (at (1, …, 1))"
languages: [rust, python]
order: 92
family: Penalized
tab: Penalized 2
trace_note: "Recorded from another run: CMA-ES with IPOP restarts in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Penalized 2

## The problem

Yao, Liu and Lin's second generalized penalized function has sines of three times the genes, and a
penalty beyond ±5:

```text
f(x) = 0.1 {sin²(3πx₁) + Σᵢ₌₁ⁿ⁻¹ (xᵢ − 1)² [1 + sin²(3πxᵢ₊₁)] + (xₙ − 1)² [1 + sin²(2πxₙ)]}
       + Σ u(xᵢ, 5, 100, 4),   each xᵢ in [−50, 50]
u(x, a, k, m) = k (|x| − a)^m if |x| > a, and 0 otherwise
```

Its minimum is 0, at (1, …, 1), where every term is 0. Here n = 30. It's Yao, Liu and Lin's (1999)
f13, whose appendix gives this definition, the bounds and the dimension; their table I drops the
square of the last term's (xₙ − 1), without which the function would have no minimum there. The
function is usually credited to Levy and Montalvo's tunneling papers (1985), which weren't read.

## What makes it hard

The sines put a local minimum near every point where their arguments are whole multiples of π: a
grid of shallow wells over the box, the more of them the more genes. The wells' depth scales with
the squares in front of the sines, so they are shallower the nearer the minimum, and the squares
lead towards it. The penalty u is 0 inside [−10, 10] (Penalized 1) or [−5, 5] (Penalized 2), and
rises as a fourth power outside: a wall that keeps the search away from the bounds of [−50, 50].

## Representation

A `Real` genome of 30 genes, each in [−50, 50]: the point x itself. The fitness is f(x), to
minimize. The function is genoxide's `problems::Penalized2`, which brings its bounds and its
minimum.

## Algorithm

Five algorithms, each from seeds 1 to 10, with a budget of 10,000 evaluations per dimension, 300,000
per run, and a target of 1e-8:

- CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195), which samples a
  population of 14 from a normal distribution and adapts its mean, its step size and its covariance
  matrix, from a step size of 0.3 of each gene's range and a random start;
- the same with IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776): a run that has
  converged starts again from a random point with twice the population;
- differential evolution with genoxide's defaults, SHADE (Tanabe and Fukunaga, CEC 2013), with a
  population of 100 and its restarts on stagnation;
- particle swarm optimization (Kennedy and Eberhart, 1995), 40 particles with Clerc and Kennedy's
  constriction coefficients and a global topology;
- a real-coded genetic algorithm: a population of 100, tournaments of 3, simulated binary crossover
  (Deb and Agrawal, 1995) with η = 15 and polynomial mutation with η = 20 at a rate of 1/30 per
  gene.

## Output

The first line gives the dimension, the seeds and the budget. Then a row per algorithm: how many of
its 10 runs reached the minimum, to within 1e-8, the median of their evaluations (a dash if none
did), and the median of every run's best error, to two significant digits. The function is evaluated
with genoxide's portable math, so the runs are the same on every platform, and in Python, `run`
evaluates it in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/penalized2) plays back another run:
CMA-ES with IPOP restarts on the function in 2 dimensions, so that the population can be drawn on
its contour. It meets the target after 396 evaluations.

## Good results

The minimum is 0. CMA-ES with IPOP restarts reaches it in all 10 runs, after a median of 6,776
evaluations, and without restarts in 9 of 10, after 6,762. SHADE reaches it every time, after
30,250, and PSO 7 times, after 26,840. The genetic algorithm never does: its median run ends at
1.1e-5.
