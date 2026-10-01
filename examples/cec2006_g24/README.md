---
title: CEC 2006 g24
category: constrained
summary: A linear function of 2 variables under 2 quartic inequalities, whose feasible region is two parts joined at a point, with three local minima, solved by CMA-ES with restarts and Deb's feasibility rules.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "−5.50801327159536 (proven)"
languages: [rust, python]
order: 140
family: "CEC 2006"
tab: g24
---

# CEC 2006 g24

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24 test
problems, g01 to g24, with their best known solutions and rules for comparing algorithms. g24 is the
last, the report's equations 49 and 50 (page 15). The report takes it from Floudas et al. (1999,
Handbook of Test Problems in Local and Global Optimization, Kluwer). It has two variables, x1 in [0,
3] and x2 in [0, 4].

Minimize

```text
f(x) = −x1 − x2
```

subject to two inequalities, each g(x) ≤ 0:

```text
g1 = −2 x1⁴ + 8 x1³ − 8 x1² + x2 − 2
g2 = −4 x1⁴ + 32 x1³ − 88 x1² + 96 x1 + x2 − 36
```

Each is an upper bound on x2 that depends on x1. Written that way, they are simpler than they look:

```text
g1:  x2 ≤ 2 (x1 (x1 − 2))² + 2
g2:  x2 ≤ 4 ((x1 − 1)(x1 − 3))²
```

The first bound is at least 2 everywhere. The second is 0 at x1 = 1 and at x1 = 3, where only x2 = 0
is feasible. The report says the feasible region consists of two disconnected sub-regions: they are
the parts left and right of x1 = 1, and they meet at the single point (1, 0).

The minimum is f* = −5.50801327159536, at x* = (2.32952019747762, 3.17849307411774), where both
constraints are active. The report prints x* without the comma, as
"2.329520197477623.17849307411774"; the two numbers add up to −f*. genoxide's docs mark the minimum
as proven, from the shape above: at each x1, the best x2 is the least of the two bounds and 4, so f
is a function of x1 alone, and its least value is where the two bounds cross, at a root of x1⁴ − 12
x1³ + 40 x1² − 48 x1 + 17. The report's x1 is that root to 14 digits.

## What makes it hard

Little, for most algorithms. The report's table 3 gives its feasible share as 79.6556 %, but that is
the share where g1 alone holds. With both constraints, the share is 44.21 %, computed for this page
by integrating the bounds above. Either way, random points are often feasible.

What remains is a trap for a local search. Along x1, f has three local minima, each where the two
bounds on x2 cross:

```text
x1 = 0.6116, x2 = 3.4421, f = −4.0537   (the left part)
x1 = 1.5996, x2 = 2.8204, f = −4.4200
x1 = 2.3295, x2 = 3.1785, f = −5.5080   (the global minimum)
```

Each sits in a corner between the two curved bounds. A search that settles into one of the other two
corners has to cross a region where f gets worse to reach the third.

## Representation

A `Real` genome of 2 genes, x1 in [0, 3] and x2 in [0, 4]. genoxide's `problems::cec2006::G24` is
the fitness: the value f(x) and the total constraint violation, the sum of max(0, g(x)) over the two
constraints, 0 for a feasible solution.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. It uses genoxide's
defaults, a population of 4 + ⌊3 ln 2⌋ = 6, a step size of 0.3 of each gene's range and a random
start, with IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776): when a run converges,
the next starts from a random point with twice the population. A sample outside the bounds is drawn
again, up to 100 times, and then clipped to them. Deb's rules rank the samples.

The run has the report's budget of 500,000 evaluations, and stops once its best solution is feasible
with an absolute error f(x) − f* of at most 1e-8. The report counts a run as successful with an
error of at most 1e-4; the example asks for more.

Why the restarts: without them, CMA-ES met the target on 22 of 25 seeds, after a median of 1,104
evaluations. The other three converged to the two other corners, twice at f = −4.4200 and once at
−4.0537, and stayed there. With IPOP, all 25 met the target, after a median of 1,104 evaluations
(from 930 to 3,306). SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78), genoxide's default
differential evolution, met it on all 25 too, but after a median of 15,600; L-SHADE after a median
of 6,948.

## Output

The first line names the run. The second gives what stopped it, after how many evaluations, the
error f(x) − f* and whether the best solution is feasible: "< 1e-8" means the run met its target.
The third gives the evaluations to the first feasible solution, and to an error of 1e-4, the
report's criterion of success. The fourth gives the restarts and the population of each run. The
fifth compares f(x) with f*, to 6 significant digits. The sixth gives the solution, and the last the
two constraints: "active" for a constraint on its boundary (|g| ≤ 1e-6), else the value of g,
negative when it's satisfied. In Python, `run` evaluates the problem in Rust, so both versions print
the same.

The page's plot shows each variable on its range, and each constraint's state: violated, active or
satisfied. Its curve shows the error f − f* of the best feasible solution, and of the population's
median, on a log scale. The best's curve begins at the first feasible solution, and the median's
once half the population is feasible.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g24) plays this run back.

## Good results

A good run is feasible and ends within 1e-4 of f*, the report's success. CMA-ES with restarts meets
the target of 1e-8 with every seed tried.

Seed 1 needs no restart. Its first generation of 6 samples has a feasible one. The run meets the
report's criterion after 522 evaluations and the target after 954. The solution is the report's x*
to 6 digits, with both constraints active.
