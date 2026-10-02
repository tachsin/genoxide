---
title: HappyCat
category: continuous
summary: Minimize HappyCat, a groove that curves around to the minimum, in 10 dimensions, with CMA-ES without and with restarts, DE, PSO and a GA from 10 seeds each.
reference: "Beyer, H.-G. and Finck, S. (2012). HappyCat: a simple function class where well-known direct search algorithms do fail. Parallel Problem Solving from Nature, PPSN XII, LNCS 7491: 367-376."
reference_url: "https://doi.org/10.1007/978-3-642-32937-1_37"
optimum: "0 (at (−1, …, −1))"
languages: [rust, python]
order: 97
trace_note: "Recorded from another run: CMA-ES with IPOP restarts in 2 dimensions, so that the population can be drawn on the function's contour."
---

# HappyCat

## The problem

HappyCat adds a slope to the distance from a sphere of radius √n:

```text
f(x) = |Σ xᵢ² − n|^(1/4) + (½ Σ xᵢ² + Σ xᵢ) / n + ½,   each xᵢ in [−5, 5]
```

Its minimum is 0, at (−1, …, −1), the only one: the second part is Σ (xᵢ + 1)² / (2n), 0 only there,
where the first is 0 too. Here n = 10. It's Beyer and Finck's (2012) function, which couldn't be
read: its parameter α shapes the groove, and its experiments use α = 1/8, which would be this
function's 1/4 if α is the exponent of (Σ xᵢ² − n)², as it's usually written; that couldn't be
confirmed. genoxide takes the definition from the CEC 2014 report (Liang, Qu and Suganthan 2013,
function 11), which cites Beyer and Finck and scales its search space [−100, 100] by 5/100, to [−5,
5]. The original is still to be checked (issue #168).

## What makes it hard

The first term is 0 on the sphere Σ xᵢ² = n and rises steeply, as a fourth root, away from it: a
narrow groove around the sphere. The slope along the groove is gentle, and the minimum lies in the
groove. A search falls into the groove at once, then has to follow it around the sphere; its steps
must be small across the groove and large along it, a direction that curves as it goes. Beyer and
Finck built it as a simple function on which well-known direct search methods fail: they stall in
the groove. The shape of its contour in two dimensions gave it its name.

## Representation

A `Real` genome of 10 genes, each in [−5, 5]: the point x itself. The fitness is f(x), to minimize.
The function is genoxide's `problems::HappyCat`, which brings its bounds and its minimum.

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

[The project page](https://tachsin.gr/projects/genoxide/examples/happy-cat) plays back another run:
CMA-ES with IPOP restarts on the function in 2 dimensions, so that the population can be drawn on
its contour. Within 20,000 evaluations it ends at an error of 1.7e-4, in the groove near the
minimum: it doesn't reach the target in 2 dimensions either.

## Good results

No algorithm reaches the minimum to within 1e-8: that's the function's point. CMA-ES with IPOP
restarts comes closest, with a median error of 5.4·10⁻³; CMA-ES without restarts ends at 9.4·10⁻²,
the genetic algorithm at 7.8·10⁻², SHADE at 0.10 and PSO at 0.14. They all reach the groove, and
stall in it, short of the minimum.

A larger budget doesn't change that. CMA-ES with IPOP restarts from seeds 1 to 5, with 1,000,000
evaluations each, ten times the page's budget, ends at errors of 8.0·10⁻⁴ to 2.8·10⁻³ (1.9·10⁻³,
2.8·10⁻³, 8.2·10⁻⁴, 1.2·10⁻³ and 8.0·10⁻⁴). Nelder-Mead started from each of those points, with an
initial step of 0.01 of each range, converges after 2,584 to 2,735 evaluations without improving
any of them.

That matches the function's reputation, as Beyer and Finck's title says: "a simple function class
where well-known direct search algorithms do fail". The CEC 2014 competition's winner, L-SHADE,
didn't reach the minimum either: on the competition's shifted and rotated HappyCat (its F13) in 10
dimensions, with 100,000 evaluations, its best of 51 runs ended at an error of 1.6·10⁻² and its
median at 5.3·10⁻² (Tanabe, R. and Fukunaga, A. S. (2014). Improving the search performance of SHADE
using linear population size reduction. 2014 IEEE Congress on Evolutionary Computation: 1658-1665,
table I, [doi:10.1109/CEC.2014.6900380](https://doi.org/10.1109/CEC.2014.6900380)).
