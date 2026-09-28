---
title: CEC 2006 g16
category: constrained
summary: A nonlinear function of 5 variables, computed through a chain of 17 intermediate quantities, under 38 inequality constraints, solved by CMA-ES with IPOP restarts and Deb's feasibility rules.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "−1.90515525853479 (best known)"
languages: [rust, python]
order: 92
---

# CEC 2006 g16

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24
problems, g01 to g24, from the literature, with their best known solutions and rules for comparing
algorithms. g16 is the sixteenth, the report's equations 32 to 34 (pages 7 to 10). The report takes
it from Himmelblau (1972, Applied Nonlinear Programming, McGraw-Hill).

There are 5 variables:

```text
x1 in [704.4148, 906.3855]    x2 in [68.6, 288.88]    x3 in [0, 134.75]
x4 in [193, 287.0966]         x5 in [25, 84.1988]
```

The objective isn't one formula. The report computes 17 quantities y1 to y17, and 17 helpers c1
to c17, one after the other, each from the variables and the ones before it. The first few:

```text
y1 = x2 + x3 + 41.6          c1 = 0.024 x4 − 4.62          y2 = 12.5 / c1 + 12
c2 = 0.0003535 x1² + 0.5311 x1 + 0.08705 y2 x1             c3 = 0.052 x1 + 78 + 0.002377 y2 x1
y3 = c2 / c3                 y4 = 19 y3                     ...
```

and so on to y17, with divisions and products all along. The objective is a weighted sum of some
of them:

```text
f(x) = 0.000117 y14 + 0.1365 + 0.00002358 y13 + 0.000001502 y16 + 0.0321 y12 + 0.004324 y5
       + 0.0001 c15 / c16 + 37.48 y2 / c12 − 0.0000005843 y17
```

There are 38 inequality constraints, each g(x) ≤ 0. Four are direct:

```text
g1 = 0.28 / 0.72 · y5 − y4                g2 = x3 − 1.5 x2
g3 = 3496 y2 / c12 − 21                   g4 = 110.6 + y1 − 62212 / c17
```

and the other 34 keep each of y1 to y17 between a lower and an upper limit: g5 and g6 for y1,
g7 and g8 for y2, and so on to g37 and g38 for y17. genoxide's `G16` has the full chain, from the
report's equation 34.

The best known value is f* = −1.90515525853479, at x = (705.174537070090537, 68.6,
102.899999999999991, 282.324931593660324, 37.5841164258054832). It isn't proven optimal.

## What makes it hard

The feasible region is small: of 10 million random points in the box, 2,022 were feasible, about
0.02 %. The chain makes most constraints nonlinear functions of all the variables, and couples
them: moving one variable changes many y's at once.

The minimum is a corner where many boundaries meet. At the report's x*, five constraints are
active: g2, g3, g4, g5 and g36, the upper limit of y16. x2 is also at its lower bound, 68.6. That
is more conditions than variables. g2 holds when x3 = 1.5 x2, and g5 when y1 = x2 + x3 + 41.6
reaches its lower limit 213.1, x2 + x3 = 171.5: with x2 at 68.6, both give x3 = 102.9. The
report's table 3 counts four active constraints at x*; genoxide's docs found five.

Other corners nearby are almost as good. At one of them, a little above f*, nearly every step is
blocked by a constraint or goes uphill, and a search has to find the narrow edge along which f
still falls. Some searches converge there instead.

The constraints also have very different scales. g36 is in the units of y16, which is about
140,000 at the optimum, and g38 in those of y17, in the millions, while g2 is in the units of x2
and x3, below 300.

## Representation

A `Real` genome of 5 genes, x1 to x5, within the report's bounds. genoxide's
`problems::cec2006::G16` is the fitness: the value f(x) and the total constraint violation, the
sum of max(0, g(x)) over the 38 inequalities, 0 for a feasible solution.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights,
which matters with constraints of such different scales.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. It uses genoxide's
defaults: a population of 4 + ⌊3 ln 5⌋ = 8, a step size of 0.3 of each gene's range and a random
start. With IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776), a run that has
converged starts again from a random point, with twice the population. A sample outside the bounds
is drawn again, up to 100 times, and then clipped to them. Deb's rules rank the samples.

The run has the report's budget of 500,000 evaluations, and stops once its best solution is
feasible with an error f(x) − f* of at most 1e-8, an absolute error. The report counts a run as
successful with an error of at most 1e-4; the example asks for more.

Why CMA-ES: its covariance matrix can stretch the samples along a narrow direction, such as an edge
between active constraints, and it needs few evaluations in 5 variables. With seeds 1 to 25,
IPOP-CMA-ES met the target on all 25 runs, after a median of 12,400 evaluations (6,008 to 24,208).
Without restarts, CMA-ES met it on 18: the other 7 converged at nearby corners, between 2.5e-5 and
3.2e-3 above f*, and 4 of them missed the report's 1e-4. SHADE (Tanabe and Fukunaga, 2013, IEEE CEC
2013: 71-78), genoxide's default differential evolution, also met the target on all 25, after a
median of 37,500 evaluations (35,200 to 41,200), three times as many.

## Output

The first line names the run. The second gives what stopped it, after how many evaluations and
restarts, the error f(x) − f* and whether the best solution is feasible: "< 1e-8" means the run met
its target. The third gives when the best solution was first feasible, and when its error first
met the report's criterion of success. The fourth compares f(x) with f*, to 6 significant digits,
and the fifth gives the solution, to 6 digits. The last names the active constraints, those with
|g| ≤ 1e-4, and counts the others. The threshold is larger than on other pages because of the
scales: the run stops with g4 and g36 about 3e-5 from their boundaries. In Python, `run` evaluates
the problem in Rust, so both versions print the same.

The page shows each variable on its range, and each of the 38 constraints' state, as a grid. Its
curve shows the error f − f* of the best feasible solution, and of the population's median, on a
log scale. The best's curve begins at the first feasible solution, and the median's once half the
population is feasible.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g16) plays this run back.

## Good results

A good run is feasible and ends within 1e-4 of f*, the report's success. A value below f* is
possible, since f* is only the best known, but the runs here end just above it.

The recorded run finds its first feasible solution after 424 evaluations. After about 2,000, it
reaches a corner with x2 at 68.6, x3 at 102.9 and x5 ≈ 34.13, where g2, g3, g5 and g36 are active
but g4 has slack: 8.8e-4 above f*, a failure by the report's criterion. It stays there for about
3,000 evaluations, and then moves on, x5 growing to about 37.58, where g4 is active too. It is
within 1e-4 of f* after 7,496 evaluations, and meets its target after 11,824, without a restart.

The solution is x = (705.175, 68.6000, 102.900, 282.325, 37.5841), the best known point to 6
digits, with the same five active constraints, g2, g3, g4, g5 and g36, and x2 at its lower bound.
The other 33 constraints have slack.
