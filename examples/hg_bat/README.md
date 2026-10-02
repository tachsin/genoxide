---
title: HGBat
category: continuous
summary: Minimize HGBat, a groove that curves around to the minimum, in 10 dimensions, with CMA-ES without and with restarts, DE, PSO and a GA from 10 seeds each.
reference: "Liang, J. J., Qu, B. Y. and Suganthan, P. N. (2013). Problem Definitions and Evaluation Criteria for the CEC 2014 Special Session and Competition on Single Objective Real-Parameter Numerical Optimization. Technical report 201311, Zhengzhou University and Nanyang Technological University."
reference_url: "https://github.com/P-N-Suganthan/CEC2014"
optimum: "0 (at (−1, …, −1))"
languages: [rust, python]
order: 98
trace_note: "Recorded from another run: CMA-ES with IPOP restarts in 2 dimensions, so that the population can be drawn on the function's contour."
---

# HGBat

## The problem

HGBat is HappyCat's relative, with the difference of two squares in the first term:

```text
f(x) = |(Σ xᵢ²)² − (Σ xᵢ)²|^(1/2) + (½ Σ xᵢ² + Σ xᵢ) / n + ½,   each xᵢ in [−5, 5]
```

Its minimum is 0, at (−1, …, −1), the only one: the second part is Σ (xᵢ + 1)² / (2n), 0 only there,
where the first is 0 too. Here n = 10. genoxide takes it from the CEC 2014 report (Liang, Qu and
Suganthan 2013, function 12), which scales its search space [−100, 100] by 5/100, to [−5, 5], and
gives no other source; it's usually credited to Beyer and Finck too, whose paper couldn't be read.

## What makes it hard

The first term is 0 where ‖x‖² = |Σ xᵢ|, on two spheres through the origin, one of them through
(−1, …, −1), and rises as a square root away from them: a groove whose floor curves around to the
minimum, with a gentle slope along it. As on HappyCat, a search falls into the groove at once, then
has to follow a curving direction with small steps across it and large ones along it.

## Representation

A `Real` genome of 10 genes, each in [−5, 5]: the point x itself. The fitness is f(x), to minimize.
The function is genoxide's `problems::HgBat`, which brings its bounds and its minimum.

## Algorithm

Five algorithms, each from seeds 1 to 10, with a budget of 10,000 evaluations per dimension, 100,000
per run, and a target of 1e-8:

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

The first line gives the dimension, the seeds and the budget. Then a row per algorithm: how many of
its 10 runs reached the minimum, to within 1e-8, the median of their evaluations (a dash if none
did), and the median of every run's best error, to two significant digits. The function is evaluated
with genoxide's portable math, so the runs are the same on every platform, and in Python, `run`
evaluates it in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/hg-bat) plays back another run:
CMA-ES with IPOP restarts on the function in 2 dimensions, so that the population can be drawn on
its contour. Within 20,000 evaluations it ends at an error of 0.010, in the groove: it doesn't reach
the target in 2 dimensions either.

## Good results

No algorithm reaches the minimum to within 1e-8. SHADE comes closest, with a median error of 0.13;
PSO ends at 0.20, the genetic algorithm at 0.30, CMA-ES with IPOP restarts at 0.34 and without
restarts at 0.45. They reach the groove and stall in it, as on HappyCat.

A larger budget doesn't change that. CMA-ES with IPOP restarts from seeds 1 to 5, with 1,000,000
evaluations each, ten times the page's budget, ends at errors of 0.11 to 0.33 (0.29, 0.11, 0.33,
0.25 and 0.29). Nelder-Mead started from each of those points, with an initial step of 0.01 of each
range, converges after 1,347 to 2,021 evaluations at 0.10 to 0.25 (0.25, 0.10, 0.20, 0.15 and 0.15):
better, but nowhere near the minimum.

That matches the function's record. The CEC 2014 competition's winner, L-SHADE, didn't reach the
minimum either: on the competition's shifted and rotated HGBat (its F14) in 10 dimensions, with
100,000 evaluations, its best of 51 runs ended at an error of 4.5·10⁻² and its median at 7.6·10⁻²
(Tanabe, R. and Fukunaga, A. S. (2014). Improving the search performance of SHADE using linear
population size reduction. 2014 IEEE Congress on Evolutionary Computation: 1658-1665, table I,
[doi:10.1109/CEC.2014.6900380](https://doi.org/10.1109/CEC.2014.6900380)).
