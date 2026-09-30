---
title: CEC 2006 g01
category: constrained
summary: A concave quadratic in 13 variables under 9 linear inequalities, the first problem of the CEC 2006 competition, solved with SHADE and Deb's feasibility rules.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "−15 at (1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1), proven"
languages: [rust, python]
order: 97
family: "CEC 2006"
tab: g01
---

# CEC 2006 g01

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24 test
problems, g01 to g24, with their best known solutions and rules for comparing algorithms. g01 is
the first. The report takes it from Floudas and Pardalos (1990, A Collection of Test Problems for
Constrained Global Optimization Algorithms, LNCS 455, Springer). It has no physical meaning: the
variables are x1 to x13, and the report names the constraints g1 to g9.

Minimize

```text
f(x) = 5 (x1 + x2 + x3 + x4) − 5 (x1² + x2² + x3² + x4²) − (x5 + x6 + … + x13)
```

subject to nine linear inequalities, each g(x) ≤ 0:

```text
g1 = 2x1 + 2x2 + x10 + x11 − 10      g4 = −8x1 + x10      g7 = −2x4 − x5 + x10
g2 = 2x1 + 2x3 + x10 + x12 − 10      g5 = −8x2 + x11      g8 = −2x6 − x7 + x11
g3 = 2x2 + 2x3 + x11 + x12 − 10      g6 = −8x3 + x12      g9 = −2x8 − x9 + x12
```

x1 to x9 and x13 lie in [0, 1], x10 to x12 in [0, 100]. The minimum is −15, at x1 = … = x9 = 1,
x10 = x11 = x12 = 3 and x13 = 1. It is proven. Six constraints are active there, met with
equality: g1, g2, g3, g7, g8 and g9.

The report's rules give each run 500,000 evaluations. A run succeeds when it finds a feasible
solution within 0.0001 of the minimum.

## What makes it hard

The minimum lies on the boundary of the feasible region, where six constraints meet. A search has
to approach that corner without crossing any of the six boundaries.

The feasible region is small. The report estimates its share of the box from random points:
0.0111 %. Most random solutions break some constraint, since x10 to x12 range up to 100 but g1 to
g3 keep their sums below 10.

The objective is concave. Each term 5 xi − 5 xi² of x1 to x4 is a downward parabola, and the rest
is linear. A concave function over a region bounded by planes has its minima at the corners of the
region, and it can have local minima at many of them. A search that settles into the wrong corner
has to cross the whole region to leave it.

## Representation

A `Real` genome of 13 genes within the bounds above. The fitness is the value f(x) and the total
constraint violation, the sum of max(0, g(x)) over the nine inequalities: 0 for a feasible
solution.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights.

## Algorithm

SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78), a differential evolution that adapts its
scale factor and crossover rate from successful trials, with genoxide's defaults: its published
population of 100, and a restart after 200 generations without progress. Differential evolution
builds each trial from differences between solutions, so its steps shrink as the population
gathers at the corner.

The run has the report's budget of 500,000 evaluations, and seed 1. It stops early once its best
solution is feasible and within 1e-8 of the minimum, relative to its size: at f ≤ −15 + 1.5e-7.

Other algorithms of genoxide do worse, in four runs each with the same budget and target. CMA-ES
without restarts ends at local minima, between −13.8 and −12.6. With restarts from a growing
population (IPOP), it meets the target after 70,000 to 166,000 evaluations, and L-SHADE, whose
population shrinks over the budget, after 81,000 to 89,000. A GA with simulated binary crossover
and polynomial mutation ends between −14.998 and −14.996, and particle swarm optimization between
−12 and −7.

## Output

The first line names the run. The second gives the best value, whether it's feasible, and the
evaluations the run took. The third gives the best solution, x1 to x13, and the fourth the
constraints active at it: those with |g(x)| ≤ 1e-6. In Python, `run` evaluates the problem in
Rust, so both versions print the same.

The page's plot shows each variable on its range, and each constraint's state: violated, active or
satisfied. Its curve shows the error f − f* of the best feasible solution, and of the population's
median, on a log scale. The best's curve begins at the first feasible solution, and the median's
once half the population is feasible.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g01) plays this run back.

## Good results

A gap of 0 to the minimum −15 is the best possible. SHADE finds its first feasible solution after
1,100 evaluations, with f = −2.93, and meets the 1e-8 target after 36,000. Its solution is the
report's, to 4 decimals, with the same six active constraints: g1, g2, g3, g7, g8 and g9. The error
falls by about a factor of 10 every 4,000 evaluations. With seeds 2 to 10, every run meets the
target, after 35,600 to 38,400 evaluations.
