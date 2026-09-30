---
title: CEC 2006 g06
category: constrained
summary: A cubic in 2 variables whose feasible region is a thin crescent between two circles, solved by CMA-ES with Deb's feasibility rules.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "−6961.81387558015 (proven)"
languages: [rust, python]
order: 82
family: "CEC 2006"
tab: g06
---

# CEC 2006 g06

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24 problems,
g01 to g24, from the literature, with their best known solutions and rules for comparing
algorithms. g06 is the sixth. The report takes it from Floudas and Pardalos (1990, A Collection of
Test Problems for Constrained Global Optimization Algorithms, LNCS 455, Springer).

There are 2 variables, x1 in [13, 100] and x2 in [0, 100]. The problem is

```text
minimize   f(x) = (x1 − 10)³ + (x2 − 20)³
subject to g1(x) = −(x1 − 5)² − (x2 − 5)² + 100 ≤ 0
           g2(x) = (x1 − 6)² + (x2 − 5)² − 82.81 ≤ 0
```

g1 keeps x outside the circle of radius 10 around (5, 5), and g2 inside the circle of radius 9.1
around (6, 5). The second circle is smaller and shifted to the right, so the two overlap except for
a thin crescent on the right, between x1 = 14.095 and x1 = 15.1. That crescent is the feasible
region. The bound x1 ≥ 13 leaves it whole.

f grows with both variables, so the minimum lies at the crescent's lower tip, where the two circles
meet. There, (x1 − 5)² − (x1 − 6)² = 100 − 82.81 gives x1 = 14.095 exactly. The minimum is
f* = −6961.81387558015, at x = (14.095, 0.8429607892154795668), with both constraints active.
genoxide's docs mark it as proven.

## What makes it hard

The feasible region is tiny. The report estimates each problem's feasible share of the box from
random points: 0.0066 % for g06. A random sample is feasible about once in 15,000 draws, so a search
must first find the crescent, guided by how far its samples are from it.

Then the minimum is a sharp corner. The crescent narrows to a point at its tip, and the objective
pulls toward it, along both boundaries at once. Near the tip, most steps leave the crescent through
one of its sides.

## Representation

A `Real` genome of 2 genes, x1 and x2, within the report's bounds. genoxide's
`problems::cec2006::G06` is the fitness: the value f(x) and the total constraint violation,
max(0, g1(x)) + max(0, g2(x)), 0 for a feasible solution.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights.
Before the first feasible solution, the search is a minimization of the violation, which leads to
the crescent.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. It uses genoxide's
defaults: a population of 4 + ⌊3 ln 2⌋ = 6, a step size of 0.3 of each gene's range, a random start
and no restarts. A sample outside the bounds is drawn again, up to 100 times, and then clipped to
them. Deb's rules rank the samples.

The run has the report's budget of 500,000 evaluations, and stops once its best solution is
feasible with an error f(x) − f* of at most 1e-8. The report counts a run as successful with an
error of at most 1e-4; the example asks for more.

Why CMA-ES: its covariance matrix can stretch the samples along the crescent, and its step size
shrinks as the crescent narrows. With 25 seeds, CMA-ES met the target on every run, after a median
of 2,046 evaluations (at most 2,478). SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78),
genoxide's default differential evolution, met it on all 25 too, but after a median of 35,600
evaluations (at most 38,700).

## Output

The first line names the run. The second gives what stopped it, after how many evaluations, the
error f(x) − f* and whether the best solution is feasible: "< 1e-8" means the run met its target.
The third gives the evaluations to the first feasible solution, and to an error of 1e-4, the
report's criterion of success. The fourth compares f(x) with f*, to 6 significant digits. The fifth
gives the solution, and the last the two constraints: "active" for a constraint on its boundary
(|g| ≤ 1e-6), else the value of g, negative when it's satisfied. In Python, `run` evaluates the
problem in Rust, so both versions print the same.

The page shows each variable on its range, and each constraint's state.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g06) plays this run back.

## Good results

A good run is feasible and ends within 1e-4 of f*, the report's success. CMA-ES goes further: with
a target of 1e-10 instead of 1e-8, it still met it with all 25 seeds.

The run's first solution in the crescent comes after 102 evaluations. The run then follows the
crescent down to its tip: it meets the report's criterion after 1,260 evaluations and its target
after 1,842. The solution is the tip, x1 = 14.0950 and x2 = 0.842961, with both constraints active.
