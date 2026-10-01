---
title: CEC 2006 g04
category: constrained
summary: Himmelblau's nonlinear problem, a quadratic in 5 variables under 6 nonlinear constraints, solved by CMA-ES with Deb's feasibility rules.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "−30665.53867178332 (proven)"
languages: [rust, python]
order: 120
family: "CEC 2006"
tab: g04
---

# CEC 2006 g04

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24 problems,
g01 to g24, from the literature, with their best known solutions and rules for comparing
algorithms. g04 is the fourth. The report takes it from Himmelblau (1972, Applied Nonlinear
Programming, McGraw-Hill), and it's often called Himmelblau's nonlinear problem. The report states
it as a mathematical problem, and so do genoxide's docs: they don't give the variables a meaning.

There are 5 variables, x1 to x5, with x1 in [78, 102], x2 in [33, 45], and x3, x4 and x5 in
[27, 45]. The problem is

```text
minimize   f(x) = 5.3578547 x3² + 0.8356891 x1 x5 + 37.293239 x1 − 40792.141
subject to 0 ≤ u(x) ≤ 92
           90 ≤ v(x) ≤ 110
           20 ≤ w(x) ≤ 25
where      u(x) = 85.334407 + 0.0056858 x2 x5 + 0.0006262 x1 x4 − 0.0022053 x3 x5
           v(x) = 80.51249 + 0.0071317 x2 x5 + 0.0029955 x1 x2 + 0.0021813 x3²
           w(x) = 9.300961 + 0.0047026 x3 x5 + 0.0012547 x1 x3 + 0.0019085 x3 x4
```

The report writes the three pairs of limits as six constraints g(x) ≤ 0: g1 = u − 92, g2 = −u,
g3 = v − 110, g4 = 90 − v, g5 = w − 25 and g6 = 20 − w. Each is nonlinear, a product of two
variables or a square.

The minimum is f* = −30665.53867178332, at x = (78, 33, 29.9952560256815985, 45,
36.7758129057882073). genoxide's docs mark it as proven. Some engineering papers use a variant with
0.00026 in u for the report's 0.0006262; genoxide implements the report's form.

## What makes it hard

Not much, by the standard of the collection. The report estimates each problem's feasible share of
the box from random points: 52.1 % for g04, against 0.0066 % for g06. The objective is a quadratic.

The difficulty is at the end. The minimum is a corner: x1 and x2 are at their lower bounds, x4 at
its upper bound, and g1 and g6 are active, u = 92 and w = 20. Five conditions fix the five
variables. A search has to reach that corner exactly, pressed against three bounds and two curved
constraints at once, and the error can't fall below a small value unless every one of them is
nearly met.

## Representation

A `Real` genome of 5 genes, x1 to x5, within the report's bounds. genoxide's
`problems::cec2006::G04` is the fitness: the value f(x) and the total constraint violation, the sum
of max(0, g(x)) over the six constraints, 0 for a feasible solution.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. It uses genoxide's
defaults: a population of 4 + ⌊3 ln 5⌋ = 8, a step size of 0.3 of each gene's range, a random start
and no restarts. A sample outside the bounds is drawn again, up to 100 times, and then clipped to
them. Deb's rules rank the samples.

The run has the report's budget of 500,000 evaluations, and stops once its best solution is
feasible with an error f(x) − f* of at most 1e-8. The report counts a run as successful with an
error of at most 1e-4; the example asks for more.

Why CMA-ES: it adapts its step size and learns the correlations between the variables, so its
samples narrow down on the corner at a steady rate. With 25 seeds, CMA-ES met the target on every
run, after a median of 4,328 evaluations (at most 5,864). SHADE (Tanabe and Fukunaga, 2013, IEEE
CEC 2013: 71-78), genoxide's default differential evolution, met it on all 25 too, but after a
median of 43,800 evaluations (at most 46,400), ten times as many.

## Output

The first line names the run. The second gives what stopped it, after how many evaluations, the
error f(x) − f* and whether the best solution is feasible: "< 1e-8" means the run met its target.
The third gives the evaluations to the first feasible solution, and to an error of 1e-4, the
report's criterion of success. The fourth compares f(x) with f*, to 6 significant digits. The fifth
gives the solution, and the last the six constraints: "active" for a constraint on its boundary
(|g| ≤ 1e-6), else the value of g, negative when it's satisfied. In Python, `run` evaluates the
problem in Rust, so both versions print the same.

The page shows each variable on its range, and each constraint's state.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g04) plays this run back.

## Good results

A good run is feasible and ends within 1e-4 of f*, the report's success. CMA-ES goes further: with
a target of 1e-10 instead of 1e-8, it still met it with all 25 seeds.

The run's first samples are already feasible, as half the box is. It meets the report's criterion
after 2,712 evaluations and its target after 4,744. The solution is the minimum's corner: x1 = 78,
x2 = 33 and x4 = 45, with u = 92 at its upper limit (g1) and w = 20 at its lower limit (g6). v is
98.84, inside [90, 110], so g2 to g5 have slack.
