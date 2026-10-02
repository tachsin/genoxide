---
title: CEC 2006 g10
category: constrained
summary: A linear objective in 8 badly scaled variables under 3 linear and 3 bilinear inequalities, all six active at the minimum, solved by CMA-ES with restarts and Deb's feasibility rules.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "7049.24802052867 (proven)"
languages: [rust, python]
order: 126
family: "CEC 2006"
tab: g10
---

# CEC 2006 g10

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24 test
problems, g01 to g24, with their best known solutions and rules for comparing algorithms. g10 is
the tenth, the report's equation 22 and the constraints below it (page 6). The report takes it from
Hock and Schittkowski (1981, Test Examples for Nonlinear Programming Codes, Lecture Notes in
Economics and Mathematical Systems 187, Springer). The report states it as a mathematical problem,
and so do genoxide's docs: they don't give the variables a meaning.

Minimize

```text
f(x) = x1 + x2 + x3
```

subject to six inequalities, each g(x) ≤ 0, three linear and three bilinear:

```text
g1 = −1 + 0.0025 (x4 + x6)
g2 = −1 + 0.0025 (x5 + x7 − x4)
g3 = −1 + 0.01 (x8 − x5)
g4 = −x1 x6 + 833.33252 x4 + 100 x1 − 83333.333
g5 = −x2 x7 + 1250 x5 + x2 x4 − 1250 x4
g6 = −x3 x8 + 1250000 + x3 x5 − 2500 x5
```

x1 lies in [100, 10000], x2 and x3 in [1000, 10000], and x4 to x8 in [10, 1000]. The minimum is
f* = 7049.24802052867, at x* = (579.3067, 1359.971, 5109.971, 182.0177, 295.6012, 217.9823,
286.4165, 395.6012) to 7 digits. All six constraints are active there, to 10⁻¹⁰: the report's
table 3 counts six, while its text names only g1, g2 and g3. genoxide's docs mark the minimum as
proven.

## What makes it hard

The feasible region is tiny. The report estimates each problem's feasible share of the box from
random points: 0.0010 % for g10, 10 points in a million. A sample of 10 million random points,
drawn for this page, had 49 feasible ones.

The problem is badly scaled. The variables span ranges from 990 to 9,900 wide. The constraints g1
to g3 are of order 1, while the terms of g4 to g6 are of order 10⁵ to 10⁶, such as x3 x8 ≈ 2·10⁶
at x*. The total violation adds them up, so before the first feasible solution, g4 to g6 dominate
the search.

The objective sees only x1 to x3; x4 to x8 matter only through the constraints. The minimum lies
where all six boundaries meet, and the objective falls slowly along them: near x*, a solution can
move x1 by 0.1, with x2 and x3 following along the boundaries, and change f by less than 1e-4. The
search follows a long, narrow valley toward a point it can't see from far.

## Representation

A `Real` genome of 8 genes, x1 to x8, within the report's bounds. genoxide's
`problems::cec2006::G10` is the fitness: the value f(x) and the total constraint violation, the
sum of max(0, g(x)) over the six constraints, 0 for a feasible solution.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. It uses genoxide's
defaults, a population of 4 + ⌊3 ln 8⌋ = 10, a step size of 0.3 of each gene's range and a random
start, with IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776): when a run converges,
the next starts from a random point with twice the population. A sample outside the bounds is drawn
again, up to 100 times, and then clipped to them. Deb's rules rank the samples.

The run has the report's budget of 500,000 evaluations, and stops once its best solution is
feasible with an error f(x) − f* of at most 1e-8 relative to |f*|, as the page of g01 does: at an
error of 7.0e-5, below the report's criterion of success, 1e-4.

Why a relative target: the error of 1e-8 that the other pages ask for is 1.4·10⁻¹² of f* here. The
last steps toward it follow the narrow valley, and cost many evaluations. With 25 seeds and IPOP,
CMA-ES met an absolute 1e-8 in all 25 runs, but after a median of 127,660 evaluations, about twice
as many as the relative target takes. With BIPOP restarts (Hansen, 2009, GECCO '09 companion:
2389-2396), which alternate large and small populations, it met it in all 25 too, after the same
median.

Why CMA-ES: it starts with a step in proportion to each variable's range, and its covariance matrix
learns the valley. With the relative target and 25 seeds, it met the target on every run, after a
median of 65,540 evaluations (from 38,490 to 136,140). Without restarts, 3 of the 25 runs
converged early, at errors of 1.7e-3, 3.5e-3 and 7.3e-5, and stayed there. SHADE (Tanabe and
Fukunaga, 2013, IEEE CEC 2013: 71-78), genoxide's default differential evolution, met the target on
all 25 too, after a median of 66,200 evaluations, but a slow run took 405,400. L-SHADE, whose
population shrinks over the budget, met it on 22.

## Output

The first line names the run. The second gives what stopped it, after how many evaluations, the
error f(x) − f* and whether the best solution is feasible. The third gives the evaluations to the
first feasible solution, and to an error of 1e-4, the report's criterion of success. The fourth
gives the restarts and the population of each run. The fifth compares f(x) with f*, to 6
significant digits. The sixth gives the solution, and the last the six constraints: "active" for a
constraint on its boundary (|g| ≤ 1e-6), else the value of g, negative when it's satisfied. In
Python, `run` evaluates the problem in Rust, so both versions print the same.

The page's plot shows each variable on its range, and each constraint's state: violated, active or
satisfied. Its curve shows the error f − f* of the best feasible solution, and of the population's
median, on a log scale. The best's curve begins at the first feasible solution, and the median's
once half the population is feasible.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g10) plays this run back.

## Good results

A good run is feasible and ends within 1e-4 of f*, the report's success. With restarts, CMA-ES
meets the relative target with every seed tried.

The run finds its first feasible solution after 410 evaluations. With its first population of 10,
it brings the error to 110 after 3,850 evaluations, 20 after 12,810 and 0.1 after 32,010, meets the
report's criterion after 40,930 evaluations and the target after 41,050, at an error of 6.8e-5,
without a restart.

The solution is x* to 3 or 4 digits: x2 is 1359.68 against x*'s 1359.97, a gap that changes f by
less than 1e-4, since x3 makes up for it. g1 to g3 are active. g4 to g6 are between −2.5e-4 and
−1.6e-4: not active by the 1e-6 rule, but their terms are of order 10⁵ to 10⁶, so these are
relative slacks of 10⁻¹⁰ to 10⁻⁹.
