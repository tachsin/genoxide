---
title: CEC 2006 g13
category: constrained
summary: The exponential of a product of 5 variables under 3 nonlinear equality constraints, with a local minimum that traps most single runs, solved by CMA-ES with IPOP restarts and Deb's feasibility rules.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "0.053941514041898 (best known, with the equalities met within 0.0001)"
languages: [rust, python]
order: 89
---

# CEC 2006 g13

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24 problems,
g01 to g24, from the literature, with their best known solutions and rules for comparing algorithms.
g13 is the thirteenth, the report's equations 26 and 27 (pages 6-7). The report takes it from Hock
and Schittkowski (1981, Test Examples for Nonlinear Programming Codes, Lecture Notes in Economics
and Mathematical Systems 187, Springer).

There are 5 variables, x1 and x2 in [−2.3, 2.3], and x3 to x5 in [−3.2, 3.2]. The problem is

```text
minimize   f(x) = exp(x1 x2 x3 x4 x5)
subject to h1(x) = x1² + x2² + x3² + x4² + x5² − 10 = 0
           h2(x) = x2 x3 − 5 x4 x5 = 0
           h3(x) = x1³ + x2³ + 1 = 0
```

h1 puts x on a sphere of radius √10, h2 ties x4 x5 to x2 x3, and h3 ties x2 to x1. Three equalities
in five variables leave a surface of two dimensions. f is least where the product x1 x2 x3 x4 x5 is
most negative.

Real-valued samples almost never meet an equality exactly. The report counts an equality as met when
|h(x)| ≤ 0.0001, and so does genoxide's `G13`: the surface becomes a thin shell. The best known
value is f* = 0.053941514041898, at x = (−1.71714224003, 1.59572124049468, 1.8272502406271,
−0.763659881912867, −0.76365986736498), the report's. It meets the equalities within the tolerance
only, and genoxide's docs mark it as a best known value, not a proven optimum. As printed, the
report's x* exceeds the tolerance by 3e-15 in h2, from rounding its digits.

The minimum isn't unique. Changing the signs of two of x3, x4 and x5 leaves the product, h1 and |h2|
as they were, so every solution has three twins: x = (−1.717, 1.596, 1.827, 0.764, 0.764) is as good
as the report's.

## What makes it hard

The feasible region has no volume without the tolerance, and hardly any with it: three shells, each
0.0002 thick in its h, must meet. A random solution is practically never feasible. A search must
first reach the shell, guided only by the violation, and then move along it in small steps, since
the equalities are nonlinear and a straight step leaves the shell.

The shell also holds a local minimum, with f ≈ 0.4388 at about x = (−0.699, −0.870, 2.790, 0.697,
−0.697), and its twins. There, x1 and x2 are both negative, and the product is −0.82 instead of
−2.92. A search that has converged there would have to cross worse solutions along the shell to
leave it.

## Representation

A `Real` genome of 5 genes, x1 to x5, within the report's bounds. genoxide's
`problems::cec2006::G13` is the fitness: the value f(x) and the total constraint violation,
max(0, |h1| − 0.0001) + max(0, |h2| − 0.0001) + max(0, |h3| − 0.0001), 0 for a feasible solution.
The tolerance is the report's, `genoxide::problems::cec2006::EQUALITY_TOLERANCE`.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. Its covariance
matrix can learn the directions of the shell, so its samples spread along it rather than across it.
It starts with genoxide's defaults: a population of 4 + ⌊3 ln 5⌋ = 8, a step size of 0.3 of each
gene's range, and a random start. Deb's rules rank the samples.

A single run of CMA-ES often ends at the local minimum. IPOP restarts (Auger and Hansen, 2005, IEEE
CEC 2005: 1769-1776) start a new run from a random point, with twice the population, whenever one
has converged. A larger population sees more of the shell before it converges, and more runs mean
more chances to start near the global minimum.

The run has the report's budget of 500,000 evaluations, and stops once its best solution is feasible
with an error f(x) − f* of at most 1e-8, an absolute error. The report counts a run as successful
with an error of at most 1e-4; the example asks for more.

With seeds 1 to 25, in the same budget:

| Algorithm | Runs that met the target | Evaluations (median, range) |
|---|---|---|
| CMA-ES with IPOP restarts | 24 of 25 | 115,312 (43,560 to 343,088) |
| CMA-ES | 8 of 25 | 51,120 (43,560 to 80,856) |
| L-SHADE | 0 of 25 | |
| SHADE | 0 of 25 | |

Without restarts, the other 17 runs of CMA-ES ended at the local minimum, with an error of 0.385.
With IPOP restarts, one run, with seed 16, was still there when the budget ran out. The 24 others
found all four twins of the minimum. SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78),
genoxide's default differential evolution, and L-SHADE (Tanabe and Fukunaga, 2014, IEEE CEC 2014:
1658-1665), its variant with a shrinking population, build a trial from the difference between two
solutions. On a curved shell, that difference points off it, and the trials are infeasible. SHADE
ended 4 runs infeasible and 21 with errors from 0.73 to 0.95; L-SHADE ended all 25 feasible, with
errors from 0.054 to 0.69.

## Output

The first line names the run. The second gives what stopped it, after how many evaluations, the
error f(x) − f* and whether the best solution is feasible: "< 1e-8" means the run met its target.
The third gives the evaluations to the first feasible solution, and to an error of 1e-4, the
report's criterion of success. The fourth compares f(x) with f*, to 6 significant digits. The fifth
gives the solution, and the last |h| of each equality, met when it's at most 0.0001. In Python,
`run` evaluates the problem in Rust, so both versions print the same.

The page's plot shows each variable on its range, and each constraint's state from its violation,
max(0, |h| − 0.0001): violated until the best solution reaches the shell, and met from then on. Its
curve shows the error f − f* of the best feasible solution, and of the population's median, on a log
scale. The median's curve has gaps: in about half of the generations, half or more of the samples
fall outside the shell.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g13) plays this run back.

## Good results

A good run is feasible and ends within 1e-4 of f*, the report's success. A value below f* is
possible, since f* is only the best known, but the runs here end just above it.

The recorded run finds its first feasible solution after 896 evaluations. Its first three runs, with
8, 16 and 32 samples per generation, converge to the local minimum, with an error of 0.385. Each
restart shows in the curve: the median jumps up, while the best stays. The fourth run, with 64
samples, starts after 140,576 evaluations and finds the global minimum's basin: the error falls to
0.11 after 233,500 evaluations and 0.003 after 266,300. The run meets the report's criterion after
273,248 evaluations and its target after 277,152. The solution is x = (−1.71710, 1.59567, 1.82732,
0.763702, 0.763627), a twin of the report's, with x4 and x5 positive. It sits on or near the edges
of the tolerance: |h1| and |h2| are 0.0001, and |h3| is 0.000099.
