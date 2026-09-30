---
title: CEC 2006 g11
category: constrained
summary: The point of a parabola nearest to (0, 1), a quadratic in 2 variables under one equality constraint, solved by CMA-ES with Deb's feasibility rules.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "0.7499 (3/4 − 0.0001) at (±0.707036, 0.5) for the report's tolerance 0.0001, proven"
languages: [rust, python]
order: 87
family: "CEC 2006"
tab: g11
---

# CEC 2006 g11

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24 problems,
g01 to g24, from the literature, with their best known solutions and rules for comparing algorithms.
g11 is the eleventh, the report's equations 23 and 24 (page 6). The report takes it from Koziel and
Michalewicz (1999, Evolutionary algorithms, homomorphous mappings, and constrained parameter
optimization, Evolutionary Computation 7(1): 19-44).

There are 2 variables, x1 and x2, each in [−1, 1]. The problem is

```text
minimize   f(x) = x1² + (x2 − 1)²
subject to h(x) = x2 − x1² = 0
```

f is the squared distance from the point (0, 1), and the equality keeps x on the parabola x2 = x1².
The problem asks for the points of the parabola nearest to (0, 1). On the parabola, with u = x1²,
f = u + (1 − u)², which is least at u = 1/2: the two points (±1/√2, 1/2), with f = 3/4.

Real-valued samples almost never meet an equality exactly. The report counts an equality as met when
|h(x)| ≤ 0.0001, and so does genoxide's `G11`. The feasible set becomes a thin band around the
parabola, and its upper edge, x2 = x1² + 0.0001, is a little nearer to (0, 1). There,
f = u + (1 − u − 0.0001)², least at u = 1/2 − 0.0001. The minimum is f* = 3/4 − 0.0001 = 0.7499, at
x = (±0.707036, 0.5), the report's value. genoxide derives it from the definition, and it's proven.

## What makes it hard

The feasible band is a curve 0.0002 thick in x2, over the 2 units of x1. Its area is about
2 × 0.0002 = 0.0004, 0.01 % of the box's 4: a random point is feasible about once in 10,000 draws.
A search has to find the band, guided by the violation, and then move along it.

The band is curved, so a straight step along it leaves it unless the step is short. Deb's rules rank
a solution that leaves the band below every feasible one, and so a search moves along the parabola
only in small steps.

f is also flat near the minimum. Along the band's edge, f − f* grows with the square of the distance
from the minimum: an error of 1e-8 allows x1 and x2 to differ from the minimum by about 1e-4. The
printed solution differs by 5e-5 and 7e-5.

## Representation

A `Real` genome of 2 genes, x1 and x2, in [−1, 1]. genoxide's `problems::cec2006::G11` is the
fitness: the value f(x) and the constraint violation, max(0, |h(x)| − 0.0001), 0 for a feasible
solution. The tolerance is the report's, `genoxide::problems::cec2006::EQUALITY_TOLERANCE`.
`G11::with_tolerance` changes it, and the minimum with it: 3/4 − δ for a tolerance δ.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. It uses genoxide's
defaults: a population of 4 + ⌊3 ln 2⌋ = 6, a step size of 0.3 of each gene's range, a random start
and no restarts. Deb's rules rank the samples.

The run has the report's budget of 500,000 evaluations, and stops once its best solution is feasible
with an error f(x) − f* of at most 1e-8, an absolute error. The report counts a run as successful
with an error of at most 1e-4; the example asks for more.

Why CMA-ES: once its best samples lie in the band, the steps it learns from lie along the band, so
its distribution stretches along the parabola and flattens across it. Its step size shrinks as the
band's curvature demands. With seeds 1 to 25, in the same budget:

| Algorithm | Runs that met the target | Evaluations (median, range) |
|---|---|---|
| CMA-ES | 25 of 25 | 3,642 (846 to 5,814) |
| SHADE | 16 of 25 | 156,296 (36,100 to 331,990) |
| L-SHADE | 14 of 25 | 19,083 (6,480 to 26,042) |

SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78) is genoxide's default differential
evolution, and L-SHADE (Tanabe and Fukunaga, 2014, IEEE CEC 2014: 1658-1665) its variant with a
population that shrinks over the budget. They build a trial from the difference of two solutions.
Between two points of a curved band, that difference points off the band, so most trials are
infeasible and lose to their parents. SHADE's nine other runs ended with errors up to 7.9e-5, within
the report's success but short of the target. L-SHADE's eleven others ended with errors up to 0.051.

## Output

The first line names the run. The second gives what stopped it, after how many evaluations, the
error f(x) − f* and whether the best solution is feasible: "< 1e-8" means the run met its target.
The third gives the evaluations to the first feasible solution, and to an error of 1e-4, the
report's criterion of success. The fourth compares f(x) with f*, to 6 significant digits. The fifth
gives the solution, and the last the value of h at it, and whether it's within the tolerance. In
Python, `run` evaluates the problem in Rust, so both versions print the same.

The page's plot shows each variable on its range, and the constraint's state from its violation,
max(0, |h| − 0.0001): violated until the best solution reaches the band, and met from then on. Its
curve shows the error f − f* of the best feasible solution, and of the population's median, on a log
scale.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g11) plays this run back.

## Good results

A good run is feasible and ends within 1e-4 of f*, the report's success. f* is proven, so no run can
end below it.

The recorded run finds its first feasible solution after 108 evaluations, near the bottom of the
parabola: after 200 evaluations, its best is x = (−0.11, 0.01), with an error of 0.24. It then
climbs along the parabola in small steps: the error is 0.13 after 2,300 evaluations and 0.01 after
4,200. It meets the report's criterion after 4,776 evaluations and its target after 5,376. The
solution is x = (−0.706988, 0.499932), 1e-4 from the minimum at (−0.707036, 0.5), with h = 0.0001:
on the upper edge of the band, where f is least. The run ends at the minimum with x1 < 0. With seeds
1 to 25, 16 runs end there, and 9 at the other one, with x1 > 0.
