---
title: CEC 2006 g09
category: constrained
summary: A polynomial of degree 6 in 7 variables under 4 nonlinear inequalities, two of them active at the minimum, solved by CMA-ES with Deb's feasibility rules.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "680.630057374402 (proven)"
languages: [rust, python]
order: 85
family: "CEC 2006"
tab: g09
---

# CEC 2006 g09

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24 test
problems, g01 to g24, with their best known solutions and rules for comparing algorithms. g09 is
the ninth, the report's equations 20 and 21 (pages 5 and 6). The report takes it from Hock and
Schittkowski (1981, Test Examples for Nonlinear Programming Codes, Lecture Notes in Economics and
Mathematical Systems 187, Springer). It has no physical meaning: the variables are x1 to x7, and
the report names the constraints g1 to g4.

Minimize

```text
f(x) = (x1 − 10)² + 5 (x2 − 12)² + x3⁴ + 3 (x4 − 11)² + 10 x5⁶ + 7 x6² + x7⁴
       − 4 x6 x7 − 10 x6 − 8 x7
```

subject to four inequalities, each g(x) ≤ 0:

```text
g1 = −127 + 2 x1² + 3 x2⁴ + x3 + 4 x4² + 5 x5
g2 = −282 + 7 x1 + 3 x2 + 10 x3² + x4 − x5
g3 = −196 + 23 x1 + x2² + 6 x6² − 8 x7
g4 = 4 x1² + x2² − 3 x1 x2 + 2 x3² + 5 x6 − 11 x7
```

Each variable lies in [−10, 10]. The minimum is f* = 680.630057374402, at x* = (2.330499,
1.951372, −0.4775414, 4.365726, −0.6244870, 1.038131, 1.594227) to 7 digits. Two constraints are
active there, g1 and g4; g2 and g3 have slack. The report's x* exceeds g1 by 4·10⁻¹⁶, from
rounding. genoxide's docs mark the minimum as proven.

## What makes it hard

The report estimates each problem's feasible share of the box from random points: 0.5121 % for
g09. A sample of 10 million random points, drawn for this page, had 0.5258 % feasible ones. That's
enough for a search to find the region soon; the difficulty is the scaling.

The terms have degrees 2, 4 and 6, so the variables act on very different scales. Over the box, the
term 10 x5⁶ alone ranges from 0 to 10⁷, while (x1 − 10)² stays below 400. Near the minimum it's
the other way: x5 is −0.62, where 10 x5⁶ is 0.6 and flat, while x2's term 5 (x2 − 12)² is 505 and
steep. A search with the same step in every direction either overshoots in the steep ones or
crawls in the flat ones.

The constraint g1 moves the minimum far from the objective's own. Without constraints, x2 = 12 and
x4 = 11 would be best, but g1 holds 3 x2⁴ + 4 x4² below about 127, and pushes x2 down to 1.95 and
x4 to 4.37. The search ends pressed against g1 and g4, two curved boundaries.

## Representation

A `Real` genome of 7 genes, x1 to x7, within [−10, 10]. genoxide's `problems::cec2006::G09` is the
fitness: the value f(x) and the total constraint violation, the sum of max(0, g(x)) over the four
constraints, 0 for a feasible solution.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. It uses genoxide's
defaults: a population of 4 + ⌊3 ln 7⌋ = 9, a step size of 0.3 of each gene's range, a random start
and no restarts. A sample outside the bounds is drawn again, up to 100 times, and then clipped to
them. Deb's rules rank the samples.

The run has the report's budget of 500,000 evaluations, and stops once its best solution is
feasible with an absolute error f(x) − f* of at most 1e-8. The report counts a run as successful
with an error of at most 1e-4; the example asks for more.

Why CMA-ES: its covariance matrix learns a scale for each direction, the steep and the flat ones,
and the correlations along the two active boundaries. With 25 seeds, it met the target on every
run, after a median of 9,045 evaluations (from 7,857 to 11,583). None converged early, so restarts
wouldn't change a run. SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78), genoxide's default
differential evolution, met the target on all 25 too, but after a median of 46,700 evaluations (at
most 48,700), and L-SHADE, whose population shrinks over the budget, after a median of 43,549.

## Output

The first line names the run. The second gives what stopped it, after how many evaluations, the
error f(x) − f* and whether the best solution is feasible: "< 1e-8" means the run met its target.
The third gives the evaluations to the first feasible solution, and to an error of 1e-4, the
report's criterion of success. The fourth compares f(x) with f*, to 6 significant digits. The fifth
gives the solution, and the last the four constraints: "active" for a constraint on its boundary
(|g| ≤ 1e-6), else the value of g, negative when it's satisfied. In Python, `run` evaluates the
problem in Rust, so both versions print the same.

The page's plot shows each variable on its range, and each constraint's state: violated, active or
satisfied. Its curve shows the error f − f* of the best feasible solution, and of the population's
median, on a log scale. The best's curve begins at the first feasible solution, and the median's
once half the population is feasible.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g09) plays this run back.

## Good results

A good run is feasible and ends within 1e-4 of f*, the report's success. CMA-ES meets the target
of 1e-8 with every seed tried.

The run finds its first feasible solution after 45 evaluations, 5 generations of 9. The error is
4.2 after 873 evaluations and 0.38 after 2,601, and from there falls, on average, by a factor of
10 every 900 evaluations: the run meets the report's criterion
after 6,723 evaluations and the target after 9,684. The solution is the report's x* to 4 or 5
digits, with g1 and g4 active. g2 = −252.6 and g3 = −144.9 have slack.
