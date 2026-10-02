---
title: Katsuura
category: continuous
summary: Minimize Katsuura's function, rugged everywhere, in 10 dimensions, with CMA-ES with IPOP restarts from 10 seeds, against CMA-ES without restarts, DE, PSO and a GA.
reference: "Hansen, N., Finck, S., Ros, R. and Auger, A. (2009). Real-Parameter Black-Box Optimization Benchmarking 2009: Noiseless Functions Definitions. INRIA research report RR-6829."
reference_url: "https://hal.inria.fr/inria-00362633"
optimum: "0 (at the origin, and wherever every gene is a multiple of 1/2)"
languages: [rust, python]
order: 96
trace_note: "Recorded from another run: CMA-ES with IPOP restarts in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Katsuura

## The problem

Katsuura's function multiplies, over the genes, terms that measure how far the gene's binary digits
are from whole numbers:

```text
f(x) = (10 / n²) Πᵢ (1 + i Σⱼ₌₁³² |2ʲxᵢ − round(2ʲxᵢ)| / 2ʲ)^(10 / n^1.2) − 10 / n²
each xᵢ in [−5, 5]
```

Its minimum is 0, wherever every gene is a multiple of 1/2: then every 2ʲxᵢ is a whole number, and
every factor is 1. There are 21ⁿ such points in the box, among them the origin. Here n = 10. It's
BBOB's f23 (Hansen et al. 2009), "based on the idea" of Katsuura (1991, The American Mathematical
Monthly 98(5): 411-416, not read), without BBOB's rotation and scaling, and its penalty outside [−5,
5]; the CEC 2014 report has the same basic function.

## What makes it hard

Each factor is a continuous function of its gene, with kinks at every multiple of 2⁻³³, rugged at
every scale down to 2⁻³², and the product couples the genes. The landscape is highly repetitive,
with global minima on a grid of spacing 1/2 and local minima everywhere between: a search that has
found a good region gains little from its neighborhood.

The grid of global minima includes the bounds, ±5. A search that pushes genes onto the bounds, as
PSO does when a particle would leave the box and stops at the bound, lands on global minima without
searching. BBOB avoids this by rotating and shifting the function; genoxide's `problems::Shifted`
does the same.

## Representation

A `Real` genome of 10 genes, each in [−5, 5]: the point x itself. The fitness is f(x), to minimize.
The function is genoxide's `problems::Katsuura`, which brings its bounds and its minimum.

## Algorithm

The main method is CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195)
with IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776). It samples a population of
10 from a normal distribution and adapts its mean, its step size and its covariance matrix, from a
step size of 0.3 of each gene's range and a random start; a run that has converged starts again from
a random point with twice the population. It runs from seeds 1 to 10, with a budget of 50,000
evaluations per dimension, 500,000 per run, and a target of 1e-8.

Four contrasts run from the same seeds with the budget of the other functions' pages, 10,000
evaluations per dimension, 100,000 per run:

- CMA-ES without restarts, which stops once it has converged (`cmaes::Restarts::Stop`): sampling on
  around its point to the end of the budget wouldn't change its best;
- differential evolution with genoxide's defaults, SHADE (Tanabe and Fukunaga, CEC 2013), with a
  population of 100 and its restarts on stagnation;
- particle swarm optimization (Kennedy and Eberhart, 1995), 40 particles with Clerc and Kennedy's
  constriction coefficients and a global topology;
- a real-coded genetic algorithm: a population of 100, tournaments of 3, simulated binary crossover
  (Deb and Agrawal, 1995) with η = 15 and polynomial mutation with η = 20 at a rate of 1/10 per
  gene.

Each generation's points are evaluated in parallel, which gives the same results on any number of
threads.

## Output

The first line gives the dimension and the seeds. Then a row per algorithm: its budget, how many of
its 10 runs reached the minimum, to within 1e-8, the median of their evaluations (a dash if none
did), and the median of every run's best error, to two significant digits. Two lines follow: the
fewest and most evaluations the main method needed, and how many of PSO's runs ended on a corner of
the box. The function is evaluated with genoxide's portable math, so the runs are the same on every
platform, and in Python, `run` evaluates it in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/katsuura) plays back another run:
CMA-ES with IPOP restarts on the function in 2 dimensions, so that the population can be drawn on
its contour. It meets the target after 990 evaluations, at a multiple of 1/2.

## Good results

The minimum is 0. CMA-ES with IPOP restarts reaches it in all 10 runs, after a median of 230,100
evaluations, from 41,990 (seed 3) to 456,820 (seed 10): each restart is a fresh try with a larger
population, and one of them eventually lands in a minimum's basin and follows it down to 1e-8. With
the other pages' budget of 100,000, it reached it only once, from seed 3.

The contrasts show why it takes that many. CMA-ES without restarts converges to a local minimum, at
a median error of 0.10. SHADE comes closest of the others, to a median error of 5.8e-6, and the
genetic algorithm ends at 3.7e-4. PSO reaches 0 in all 10 runs, after a median of 380 evaluations,
but only through the bounds: its particles head out of the box, stop at ±5 in every gene, and all 10
runs end on a corner, a global minimum (from seed 1, at (5, 5, −5, 5, −5, 5, −5, 5, 5, −5)). On the
function shifted with genoxide's `problems::Shifted` and seed 1, whose minima are no longer on the
bounds, PSO's runs from seeds 1 and 2 end at 0.021.
