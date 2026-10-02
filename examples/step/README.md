---
title: Step
category: continuous
summary: Minimize a sphere of flat steps in 30 dimensions, whose gradient is 0 almost everywhere, and compare how fast CMA-ES, sep-CMA-ES, DE, PSO and a GA reach the minimum.
reference: "Yao, X., Liu, Y. and Lin, G. (1999). Evolutionary programming made faster. IEEE Transactions on Evolutionary Computation 3(2): 82-102."
reference_url: "https://doi.org/10.1109/4235.771163"
optimum: "0 (on the cube [−0.5, 0.5)ⁿ)"
languages: [rust, python]
order: 51
trace_note: "Recorded from another run: CMA-ES in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Step

## The problem

The step function rounds each gene to the nearest integer before squaring it:

```text
f(x) = Σ ⌊xᵢ + 0.5⌋²,   each xᵢ in [−100, 100]
```

Its minimum is 0, on the whole cube [−0.5, 0.5)ⁿ, where every gene rounds to 0. Here n = 30. It's
Yao, Liu and Lin's (1999) f6, whose table I and appendix give this definition, the bounds, the
dimension and the minimum. De Jong's (1975) F3, which it's often credited to, is another step
function, Σ ⌊xᵢ⌋ on [−5.12, 5.12]⁵, with its minimum at a corner.

## What makes it hard

The function is a sphere made of flat terraces: its value is a whole number, and it changes only
where a gene crosses a half-integer. Its gradient is 0 almost everywhere, and a small step changes
nothing: a search sees many points of equal value and has to cross plateaus without a direction.
Yao, Liu and Lin chose it for that: their classical evolutionary programming, whose steps were
small, ended far from the minimum, and Cauchy steps reached it. Near the minimum, the error counts
the genes off by one step: an error of 4 is four genes at ±1.

## Representation

A `Real` genome of 30 genes, each in [−100, 100]: the point x itself. The fitness is f(x), to
minimize. The function is genoxide's `problems::Step`, which brings its bounds and its minimum.

## Algorithm

Five algorithms, each with a budget of 10,000 evaluations per dimension, 300,000 in all, and the
target 0, from seed 1:

- CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195), which samples a
  population of 14 from a normal distribution and adapts its mean, its step size and its covariance
  matrix, from a step size of 0.3 of each gene's range and a random start;
- sep-CMA-ES (Ros and Hansen, 2008, PPSN X: 296-305), the same with a diagonal covariance matrix: a
  scale per gene but no correlations;
- differential evolution with genoxide's defaults, SHADE (Tanabe and Fukunaga, CEC 2013), with a
  population of 100;
- particle swarm optimization (Kennedy and Eberhart, 1995), 40 particles with Clerc and Kennedy's
  constriction coefficients and a global topology;
- a real-coded genetic algorithm: a population of 100, tournaments of 3, simulated binary crossover
  (Deb and Agrawal, 1995) with η = 15 and polynomial mutation with η = 20 at a rate of 1/30 per
  gene.

## Output

The first line gives the dimension and the budget. Then a table: a row per algorithm, the
evaluations it had used when its best error first reached each value of the heading, and the best
error it found, to two significant digits. A dash is an error not reached. The errors are whole
numbers here, so the columns are 1000, 100, 10, 1 and 0. The function is evaluated with genoxide's
portable math, so the runs are the same on every platform, and in Python, `run` evaluates it in
Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/step) plays back another run:
CMA-ES on the function in 2 dimensions, so that the population can be drawn on its contour. It
reaches the minimum after 114 evaluations.

## Good results

The minimum is 0. sep-CMA-ES reaches it after 1,848 evaluations and CMA-ES after 1,862: from a step
size of 60, a third of the range, the plateaus are small next to their steps, and they shrink their
steps no faster than they close in. SHADE reaches it after 11,600 evaluations and the genetic
algorithm after 25,390. PSO stops at an error of 4, four genes one step from 0: once the swarm has
gathered on a plateau, its particles slow down and see no better point near them.
