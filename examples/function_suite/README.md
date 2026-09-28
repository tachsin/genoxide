---
title: Function suite
category: continuous
summary: CMA-ES, SHADE and PSO on twelve classic test functions in 10 dimensions, with the error to each known minimum.
reference: "Yao, X., Liu, Y. and Lin, G. (1999). Evolutionary programming made faster. IEEE Transactions on Evolutionary Computation 3(2): 82-102."
reference_url: "https://doi.org/10.1109/4235.771163"
optimum: "an error of 0 on each function"
languages: [rust, python]
order: 63
---

# Function suite

## The problem

Twelve test functions of real variables, each to minimize, in 10 dimensions. All are scalable: they
are defined for any number of variables. Most definitions and bounds are those of Yao, Liu and Lin
(1999), who used them to compare evolutionary algorithms. Each comes from genoxide's `problems`,
with its bounds, its minimum and its original reference; the [docs of
`genoxide::problems`](https://docs.rs/genoxide/latest/genoxide/problems/) give each formula and its
source.

| Function | Bounds | Minimum (n = 10) | Shape |
|---|---|---|---|
| Sphere | [−100, 100] | 0 | a round bowl |
| Axis-parallel ellipsoid | [−5.12, 5.12] | 0 | a bowl stretched along each axis |
| Schwefel's problem 1.2 | [−100, 100] | 0 | a bowl with strongly interacting genes |
| Zakharov | [−5, 10] | 0 | a bowl with strongly interacting genes |
| Rosenbrock | [−30, 30] | 0 | a narrow curved valley |
| Rastrigin | [−5.12, 5.12] | 0 | a local minimum near every integer point |
| Ackley | [−32, 32] | 0 | a nearly flat outer region around a deep hole, covered in local minima |
| Griewank | [−600, 600] | 0 | a wide bowl with a product of cosines that couples the genes |
| Schwefel's problem 2.26 | [−500, 500] | −4189.83 | the best local minima far apart, the global one near a corner |
| Levy | [−10, 10] | 0 | sine terms that make many local minima |
| Styblinski-Tang | [−5, 5] | −391.66 | a local minimum of each term near 2.75, the global one near −2.90 |
| Michalewicz | [0, π] | −9.66 | steep narrow valleys on flat plateaus |

## What makes it hard

The functions test different things. Yao, Liu and Lin divide theirs into unimodal functions,
multimodal functions with many local minima, whose number grows exponentially with the dimension,
and low-dimensional functions with only a few local minima.
The first five here test how fast an algorithm converges, and whether it copes with genes of
different scales (the ellipsoid), genes that interact (Schwefel 1.2, Zakharov) and a valley that
bends (Rosenbrock). The first four have a single minimum. Rosenbrock's function, which Yao, Liu and
Lin count as unimodal, also has a second, local minimum for n from 4 to 30 (Shang and Qiu, 2006,
Evolutionary Computation 14(1): 119-126). It has x₁ ≈ −0.78 for n = 4, and x₁ near −1 from n = 6
on, as in the 10 dimensions here. The other seven are multimodal, and test whether an
algorithm escapes local minima.

## Representation

A `Real` genome of 10 genes, within each function's bounds: the point x itself. The fitness is the
function's value, to minimize.

## Algorithm

Three algorithms, each with genoxide's defaults:

- CMA-ES (Hansen and Ostermeier, 2001) with IPOP restarts (Auger and Hansen, 2005): a converged run
  starts again from a random point with twice the population.
- SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78), a differential evolution that adapts its
  scale factor and crossover rate from successful trials, with its published population of 100.
  genoxide adds restarts, once the population has converged or after 200 generations without
  progress.
- Particle swarm optimization (Kennedy and Eberhart, 1995, Proceedings of ICNN'95: 1942-1948) with
  40 particles, a global topology, and Clerc and Kennedy's constriction coefficients (2002, IEEE
  Transactions on Evolutionary Computation 6(1): 58-73). It has no restarts.

Each run has a budget of 10,000 evaluations per dimension, 100,000 in all. It stops early within
1e-8 of the known minimum. There is one run per algorithm and function, with one seed: the table
shows what each run found, not an average.

## Output

The first line gives the dimension and the budget. Then a table has a row per function and a column
per algorithm. Each cell is the error of that run: the best value it found minus the known minimum,
to two significant digits. An error below 1e-8 means the run met the target and stopped early. In
Python, `run` evaluates the functions in Rust, so both versions print the same table on one
platform. The functions call the platform's `sin`, `cos` and `exp`, whose last bit differs between
operating systems, and runs this long amplify it: on Windows or macOS a few cells differ from the
table in `output.txt`, which is the Linux output.

[The project page](https://tachsin.gr/projects/genoxide/examples/function-suite) plays these runs back.

## Good results

The best possible error is 0, and an error below 1e-8 counts as solved. All three solve the unimodal
functions, except PSO on Rosenbrock. SHADE solves all twelve. CMA-ES solves the multimodal ones
except Michalewicz and Schwefel 2.26, where it stays far from the minimum. PSO, which has no
restarts, solves only Ackley and Levy among the multimodal functions.
