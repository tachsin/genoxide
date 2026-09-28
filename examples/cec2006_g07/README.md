---
title: CEC 2006 g07
category: constrained
summary: A convex quadratic in 10 variables under 3 linear and 5 nonlinear inequalities, six of them active at the minimum, solved by CMA-ES with restarts and Deb's feasibility rules.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "24.30620906818 (proven)"
languages: [rust, python]
order: 83
---

# CEC 2006 g07

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24 test
problems, g01 to g24, with their best known solutions and rules for comparing algorithms. g07 is
the seventh, the report's equations 16 and 17 (page 5). The report takes it from Hock and
Schittkowski (1981, Test Examples for Nonlinear Programming Codes, Lecture Notes in Economics and
Mathematical Systems 187, Springer). It has no physical meaning: the variables are x1 to x10, and
the report names the constraints g1 to g8.

Minimize

```text
f(x) = x1² + x2² + x1 x2 − 14 x1 − 16 x2 + (x3 − 10)² + 4 (x4 − 5)² + (x5 − 3)²
       + 2 (x6 − 1)² + 5 x7² + 7 (x8 − 11)² + 2 (x9 − 10)² + (x10 − 7)² + 45
```

subject to eight inequalities, each g(x) ≤ 0, three linear and five quadratic:

```text
g1 = −105 + 4 x1 + 5 x2 − 3 x7 + 9 x8
g2 = 10 x1 − 8 x2 − 17 x7 + 2 x8
g3 = −8 x1 + 2 x2 + 5 x9 − 2 x10 − 12
g4 = 3 (x1 − 2)² + 4 (x2 − 3)² + 2 x3² − 7 x4 − 120
g5 = 5 x1² + 8 x2 + (x3 − 6)² − 2 x4 − 40
g6 = x1² + 2 (x2 − 2)² − 2 x1 x2 + 14 x5 − 6 x6
g7 = 0.5 (x1 − 8)² + 2 (x2 − 4)² + 3 x5² − x6 − 30
g8 = −3 x1 + 6 x2 + 12 (x9 − 8)² − 7 x10
```

Each variable lies in [−10, 10]. The minimum is f* = 24.30620906818, at x* = (2.171996,
2.363683, 8.773926, 5.095984, 0.9906548, 1.430574, 1.321644, 9.828726, 8.280092, 8.375927) to 7
digits. Six constraints are active there, g1 to g6; g7 and g8 have slack. The report's x* exceeds
g1 by 6·10⁻¹⁴, from rounding, as the report says.

genoxide's docs mark the minimum as proven: the objective and all eight constraints are convex, so
the feasible region is convex and a local minimum is the global one.

## What makes it hard

Not local minima: a convex problem has none. The difficulty is the feasible region and the corner
of it where the minimum lies.

The region is small. The report estimates each problem's feasible share of the box from random
points: 0.0003 % for g07, about 3 points in a million. A sample of 10 million random points, drawn
for this page, had 8 feasible ones. A search must first find the region, guided only by how far its
samples are from it.

Then six constraints are active at the minimum. The unconstrained minimum of the quadratic lies
outside the region, so the search presses against six curved and flat boundaries at once. A step
that improves f tends to cross one of them, and Deb's rules rank any infeasible sample below every
feasible one. The search has to learn the directions that stay inside, and they change as it
approaches the corner.

## Representation

A `Real` genome of 10 genes, x1 to x10, within [−10, 10]. genoxide's `problems::cec2006::G07` is
the fitness: the value f(x) and the total constraint violation, the sum of max(0, g(x)) over the
eight constraints, 0 for a feasible solution.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights.
Before the first feasible solution, the search is a minimization of the violation, which leads to
the region.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. It uses genoxide's
defaults, a population of 4 + ⌊3 ln 10⌋ = 10, a step size of 0.3 of each gene's range and a random
start, with IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776): when a run converges,
the next starts from a random point with twice the population. A sample outside the bounds is drawn
again, up to 100 times, and then clipped to them. Deb's rules rank the samples.

The run has the report's budget of 500,000 evaluations, and stops once its best solution is
feasible with an absolute error f(x) − f* of at most 1e-8. The report counts a run as successful
with an error of at most 1e-4; the example asks for more.

Why CMA-ES: it learns the correlations between the variables, and so the directions that stay
inside the corner of six active constraints. With 25 seeds, it met the target on every run, after
a median of 30,760 evaluations (from 22,650 to 51,990). Why the restarts: without them, 24 of the
25 runs are the same, but one converged at an error of 1.5e-8, just short of the target, and then
kept sampling around that point until the budget ran out; IPOP restarts it. SHADE (Tanabe and
Fukunaga, 2013, IEEE CEC 2013: 71-78), genoxide's default differential evolution, met the target
on all 25 too, but after a median of 71,300 evaluations (at most 75,800). L-SHADE, whose
population shrinks over the budget, met it on 24, after a median of 94,741.

## Output

The first line names the run. The second gives what stopped it, after how many evaluations, the
error f(x) − f* and whether the best solution is feasible: "< 1e-8" means the run met its target.
The third gives the evaluations to the first feasible solution, and to an error of 1e-4, the
report's criterion of success. The fourth gives the restarts and the population of each run. The
fifth compares f(x) with f*, to 6 significant digits. The sixth gives the solution, and the last
the eight constraints: "active" for a constraint on its boundary (|g| ≤ 1e-6), else the value of
g, negative when it's satisfied. In Python, `run` evaluates the problem in Rust, so both versions
print the same.

The page's plot shows each variable on its range, and each constraint's state: violated, active or
satisfied. Its curve shows the error f − f* of the best feasible solution, and of the population's
median, on a log scale. The best's curve begins at the first feasible solution, and the median's
once half the population is feasible.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g07) plays this run back.

## Good results

A good run is feasible and ends within 1e-4 of f*, the report's success. CMA-ES meets the target
of 1e-8 with every seed tried.

Seed 1 needs no restart. The run finds its first feasible solution after 350 evaluations, 35
generations of 10. It then follows the boundaries toward the corner: the error is 0.18 after 3,850
evaluations and 0.0024 after 15,370. The run meets the report's criterion after 18,550 evaluations
and the target after 33,540. The solution is the report's x* to 5 or 6 digits, with the same six
active constraints, g1 to g6. g7 = −6.148 and g8 = −50.02 have slack.

The median's curve has gaps: in some generations, fewer than half the samples are feasible, as
the population straddles the boundaries of the corner.
