---
title: CEC 2006 g23
category: constrained
summary: A linear cost in 9 variables under 2 bilinear inequalities and 4 equalities, a blending model whose best known solution uses the equality tolerance, solved by CMA-ES with restarts and Deb's feasibility rules.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "−400.055099999999584 (best known, with the equalities met within 0.0001)"
languages: [rust, python]
order: 119
family: "CEC 2006"
tab: g23
---

# CEC 2006 g23

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24 test
problems, g01 to g24, with their best known solutions and rules for comparing algorithms. g23 is the
twenty-third, the report's equations 47 and 48 (pages 14 and 15). The report takes it from Xia's
collection of global optimization test problems (the report's reference 10). It has 9 variables:

```text
x1, x2, x6 in [0, 300]    x3, x5, x7 in [0, 100]    x4, x8 in [0, 200]    x9 in [0.01, 0.03]
```

Minimize

```text
f(x) = −9 x5 − 15 x8 + 6 x1 + 16 x2 + 10 (x6 + x7)
```

subject to two inequalities, each g(x) ≤ 0, and four equalities, each h(x) = 0:

```text
g1 = x9 x3 + 0.02 x6 − 0.025 x5
g2 = x9 x4 + 0.02 x7 − 0.015 x8
h1 = x1 + x2 − x3 − x4
h2 = 0.03 x1 + 0.01 x2 − x9 (x3 + x4)
h3 = x3 + x6 − x5
h4 = x4 + x7 − x8
```

The constraints have the form of a blending model. Read that way, x1 and x2 flow into a pool (h1)
and leave it as x3 and x4, and x9 is the pool's concentration of some component, 3 % in x1 and 1 %
in x2 (h2). The pool's outflows join x6 and x7 to make x5 and x8 (h3 and h4), whose concentrations
may not exceed 2.5 % and 1.5 % (g1 and g2, with x6 and x7 at 2 %). f is the cost of the inflows less
the value of the products. Every constraint but h1, h3 and h4 is bilinear, a product of x9 with a
flow: the feasible set is not convex.

The report counts an equality as met when |h(x)| ≤ 0.0001, and so does genoxide's `G23`. The best
known value is f* = −400.055099999999584, at the report's x*:

```text
x = (0.0051, 99.9947, 9.0e−18, 99.9999, 0.0001, 2.8e−14, 100, 200, 0.0100000100000100008)
```

where each equality is at the tolerance, |h| = 0.0001, g2 is active and g1 is −2.5·10⁻⁶. The report
prints x* with 8 numbers for 9 variables: a comma is missing in the last, printed as
"2000.0100000100000100008", which is x8 = 200 and x9 = 0.0100000100000100008. With them, x*
evaluates to f*. The value is the best known, not proven optimal.

The tolerance matters here. The point x2 = x4 = x7 = 100, x8 = 200, x9 = 0.01, the rest 0, meets
every constraint exactly, with g2 active, and f = −15·200 + 16·100 + 10·100 = −400. The best known
solution is 0.0551 below it, by using the tolerance of every equality.

## What makes it hard

The feasible region has no volume: random points never meet four equalities. The report's table 3
gives a feasible share of 0.0000 %. A search starts outside, guided only by the violation, and must
reach a thin layer around a 5-dimensional surface in 9 dimensions.

On that surface, the best known solution is a corner: x3 and x6 at their lower bound 0, x8 at its
upper bound 200, x9 at its lower bound 0.01, g2 active and each equality at the edge of its
tolerance. The bilinear constraints make the region curved, with other corners: runs of CMA-ES
without restarts converged at f = −100.05, 300 above f*, and in corners near the best known one, up
to 0.005 above it.

## Representation

A `Real` genome of 9 genes, x1 to x9, within the bounds above. genoxide's `problems::cec2006::G23`
is the fitness: the value f(x) and the total constraint violation, the sum of max(0, g(x)) over the
inequalities and of max(0, |h(x)| − 0.0001) over the equalities, 0 for a feasible solution.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights.
Before the first feasible solution, the search is a minimization of the violation, which leads to
the region.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. It uses genoxide's
defaults, a population of 4 + ⌊3 ln 9⌋ = 10, a step size of 0.3 of each gene's range and a random
start, with IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776): when a run converges,
the next starts from a random point with twice the population. A sample outside the bounds is drawn
again, up to 100 times, and then clipped to them. Deb's rules rank the samples.

The run has the report's budget of 500,000 evaluations, and stops once its best solution is feasible
with an absolute error f(x) − f* of at most 1e-8. The report counts a run as successful with an
error of at most 1e-4; the example asks for more.

Why CMA-ES with restarts: with 25 seeds, it met the target on every run, after a median of 132,040
evaluations (from 64,180 to 199,350). Without restarts, 17 of the 25 met it; the other eight ended
short of it, three at −100.05 and the others from 4·10⁻⁸ to 0.005 above f*. SHADE (Tanabe and
Fukunaga, 2013, IEEE CEC 2013: 71-78), genoxide's default differential evolution, met the target on
all 25 too, but after a median of 153,800 evaluations (at most 276,100). L-SHADE, whose population
shrinks over the budget, met it on all 25 as well, after a median of 213,949.

## Output

The first line names the run. The second gives what stopped it, after how many evaluations, the
error f(x) − f* and whether the best solution is feasible: "< 1e-8" means the run met its target.
The third gives the evaluations to the first feasible solution, and to an error of 1e-4, the
report's criterion of success. The fourth gives the restarts and the population of each run. The
fifth compares f(x) with f*, to 6 significant digits. The sixth gives the solution, with the genes
near 0 in scientific notation, and the last the constraints: for g1 and g2, "active" on the boundary
(|g| ≤ 1e-6), else the value of g, negative when it's satisfied; for h1 to h4, "active" when the
equality is met within the tolerance, else by how much |h| exceeds it. In Python, `run` evaluates
the problem in Rust, so both versions print the same.

The page's plot shows each variable on its range, and each constraint's state: violated, active or
satisfied (an equality met within the tolerance shows as active). Its curve shows the error f − f*
of the best feasible solution, and of the population's median, on a log scale. The best's curve
begins at the first feasible solution, and the median's once half the population is feasible.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g23) plays this run back.

## Good results

A good run is feasible and ends within 1e-4 of f*, the report's success. CMA-ES with restarts meets
the target of 1e-8 with every seed tried.

Seed 1 finds its first feasible solution after 5,210 evaluations. Its first run, with a population
of 10, is within the report's criterion after 56,810 evaluations, and brings the error to 1.0·10⁻⁷
after about 65,000, just short of the target, where it converges. The restart, with a population of
20, starts again from a random point: its population is mostly infeasible at first, and then passes
the first run's best, and meets the target after 112,810. The solution is the report's x* to 5
digits, with g2 and every equality at its limit.
