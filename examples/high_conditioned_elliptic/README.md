---
title: High-conditioned elliptic
category: continuous
summary: Minimize an ellipsoid with a condition number of 10⁶ in 30 dimensions, as it is and shifted and rotated as CEC 2005's F3, and compare CMA-ES, sep-CMA-ES, DE, PSO and a GA.
reference: "Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y.-P., Auger, A. and Tiwari, S. (2005). Problem Definitions and Evaluation Criteria for the CEC 2005 Special Session on Real-Parameter Optimization. Nanyang Technological University and KanGAL report 2005005."
reference_url: "https://github.com/P-N-Suganthan/CEC2005"
optimum: "0 (at the origin)"
languages: [rust, python]
order: 54
trace_note: "Recorded from another run: CMA-ES in 2 dimensions on the function rotated with seed 1, so that the population can be drawn on the function's contour."
---

# High-conditioned elliptic

## The problem

The high-conditioned elliptic function is an ellipsoid whose weights grow geometrically from the
first gene to the last:

```text
f(x) = Σ (10⁶)^((i−1)/(n−1)) xᵢ²,   i from 1 to n, each xᵢ in [−100, 100]
```

Its minimum is 0, at the origin. Here n = 30. It's the function F3 of the CEC 2005 report
(Suganthan et al. 2005), which shifts and rotates it, and gives these bounds; the CEC 2014 and
2017 reports have the same basic function, and BBOB's f2 and f10 (Hansen et al. 2009) the same
ellipsoid with an oscillation that genoxide doesn't apply.

## What makes it hard

The weights run from 1 to 10⁶: the ellipsoid's axes from 1 to 1,000 in length, a condition number
of 10⁶. Along the first gene the function is a million times flatter than along the last. A search
with one step size for every direction either crawls along the flat axes or overshoots along the
steep ones: it must learn a scale per direction.

As it is, those directions are the genes' axes, and a scale per gene is enough. Shifted and
rotated, as in CEC 2005, they're 30 random directions: only a full covariance matrix, with its
465 parameters, can learn them.

## Representation

A `Real` genome of 30 genes: the point x itself. The fitness is f(x), to minimize. The function is
genoxide's `problems::HighConditionedElliptic`, which brings its bounds and its minimum, and the shifted and rotated
instance `problems::Rotated::new(problems::Shifted::new(function, 1), 1)`, which keeps them.

## Algorithm

Five algorithms, each with a budget of 10,000 evaluations per dimension, 300,000 in all, and a
target of 1e-8, from seed 1:

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

The second table runs the same algorithms on the function shifted and rotated, with genoxide's
`problems::Shifted` and `problems::Rotated` and seed 1: the minimum moves to a random point in the
middle 80% of the box, and an orthogonal matrix, drawn from normal numbers made orthonormal by
Gram-Schmidt as BBOB draws its rotations, turns the function about it. That's how the CEC and BBOB
suites use the function, with their own data; genoxide generates its instances instead.

## Output

The first line gives the dimension and the budget. Then two tables, the function as it is and shifted and rotated: a row per algorithm, the
evaluations it had used when its best error first reached each value of the heading, and the best
error it found, to two significant digits. A dash is an error not reached. The function is
evaluated with genoxide's portable math, so the runs are the same on every platform, and in Python,
`run` evaluates it in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/high-conditioned-elliptic) plays back another
run: CMA-ES on the function in 2 dimensions, x₁² + 10⁶ x₂², rotated with seed 1, so that the
population can be drawn on its contour. It meets the target after 654 evaluations.

## Good results

The minimum is 0. As it is, sep-CMA-ES reaches 1e-8 first, after 9,016 evaluations: a scale per
gene fits the ellipsoid. PSO takes 35,320, CMA-ES 39,242 (most of them, 36,036, to reach an error of
1, while its covariance matrix learns the scales), and SHADE 40,100. The genetic algorithm ends at
0.92.

Shifted and rotated, as CEC 2005's F3, CMA-ES takes the same 39,550 evaluations: for its full
covariance matrix, a rotation changes nothing. Every other algorithm fails: sep-CMA-ES ends at
5.6·10⁴, SHADE at 2.7·10³, PSO at 1.2·10⁶ and the genetic algorithm at 4.8·10⁶.
