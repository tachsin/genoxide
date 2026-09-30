---
title: Powell
category: continuous
summary: Minimize Powell's singular function in 24 dimensions, convex but with a singular Hessian at the minimum, and compare how fast CMA-ES, sep-CMA-ES, DE, PSO and a GA close in on it.
reference: "Powell, M. J. D. (1962). An iterative method for finding stationary values of a function of several variables. The Computer Journal 5(2): 147-151."
reference_url: "https://doi.org/10.1093/comjnl/5.2.147"
optimum: "0 (at the origin)"
languages: [rust, python]
order: 49
trace_note: "Recorded from another run: CMA-ES in 4 dimensions, Powell's own."
---

# Powell

## The problem

Powell's singular function is a sum of two squares and two fourth powers of four genes, to minimize;
in more dimensions, the sum of the same over blocks of four:

```text
f(x) = Σ over blocks (x₁, x₂, x₃, x₄) of
       (x₁ + 10x₂)² + 5 (x₃ − x₄)² + (x₂ − 2x₃)⁴ + 10 (x₁ − x₄)⁴,   each xᵢ in [−4, 5]
```

Its minimum is 0, at the origin. Here n = 24, six blocks.

Powell (1962) defined it in 4 dimensions, with the start (3, −1, 0, 1), where f = 215, and no
bounds; the paper couldn't be read. genoxide takes the formula from Steihaug and Suleiman (2013,
Journal of Global Optimization 56(3): 845-853), who restate it from Powell, and the extension to
blocks of four and the bounds from Laguna and Martí (2005, function 36, with n = 24). Jamil and Yang
(2013, function 91) print (x₂ − x₃)⁴ for (x₂ − 2x₃)⁴, and give the start as the minimizer.

The origin is the only point where every term is 0: x₁ = −10x₂, x₃ = x₄, x₂ = 2x₃ and x₁ = x₄ leave
only 0. genoxide's `problems::Powell` gives it as proven.

## What makes it hard

The function is convex, but its Hessian at the minimum is singular: the fourth powers have no
curvature at 0, so in each block, two directions (along which x₁ + 10x₂ and x₃ − x₄ stay 0) are
flat to second order. There the function falls only as the fourth power of the distance, so an
error of 1e-8 needs the genes within about 1e-2 of the minimum along those directions, and within
1e-4 along the others. Methods that rely on a quadratic model converge only linearly, as Steihaug
and Suleiman show for Newton's method.

The squares also couple the genes of each block with weights up to 10, so the level sets are long
and oblique. The blocks don't interact: a search that learns one block's shape can use it for all
six.

## Representation

A `Real` genome of 24 genes, each in [−4, 5]: the point x itself. The fitness is f(x), to minimize.
The function is genoxide's `problems::Powell`, which brings its bounds and its minimum.

## Algorithm

Five algorithms, each with a budget of 10,000 evaluations per dimension, 240,000 in all, and a
target of 1e-8, from seed 1:

- CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195), which samples a
  population of 13 from a normal distribution and adapts its mean, its step size and its covariance
  matrix, from a step size of 0.3 of each gene's range and a random start;
- sep-CMA-ES (Ros and Hansen, 2008, PPSN X: 296-305), the same with a diagonal covariance matrix:
  a scale per gene but no correlations;
- differential evolution with genoxide's defaults, SHADE (Tanabe and Fukunaga, CEC 2013), with a
  population of 100;
- particle swarm optimization (Kennedy and Eberhart, 1995), 40 particles with Clerc and Kennedy's
  constriction coefficients and a global topology;
- a real-coded genetic algorithm: a population of 100, tournaments of 3, simulated binary
  crossover (Deb and Agrawal, 1995) with η = 15 and polynomial mutation with η = 20 at a rate of
  1/24 per gene.

## Output

The first two lines give the dimension and the budget. Then a row per algorithm: the evaluations it
had used when its best error first reached 1, 1e-2, 1e-4, 1e-6 and 1e-8, and the best error it
found, to two significant digits. A dash is an error not reached. The function has no `sin`, `cos`
or `exp`, so the runs are the same on every platform, and in Python, `run` evaluates the function
in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/powell) plays back another run:
CMA-ES on the function in 4 dimensions, Powell's own, each gene on its range with the minimum
marked.

## Good results

The minimum is 0. CMA-ES reaches 1e-8 after 19,864 evaluations: 1,716 to get down to an error of 1,
then about 2,300 per decade. Its full covariance matrix learns the oblique squares, and its step
size keeps shrinking along the flat directions.

sep-CMA-ES is faster at first, reaching 1e-2 after 2,665 evaluations, but then slows: it reaches
1e-6 only after 104,260 evaluations and ends at 1.7e-7, since a diagonal matrix can't line up with
the oblique squares. SHADE reaches 1e-8 after 79,400 evaluations. PSO reaches 1e-4 after 59,640
and ends at 4.7e-6, and the genetic algorithm reaches 1e-2 after 119,908 and ends at 3.3e-3: along
the flat directions, a small error still means genes far from 0, and their steps don't shrink with
the error.
