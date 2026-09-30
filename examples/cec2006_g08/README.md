---
title: CEC 2006 g08
category: constrained
summary: A ratio of sines in 2 variables with several peaks inside a small lens between two parabolas, solved by CMA-ES with restarts and Deb's feasibility rules.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "−0.0958250414180359 (proven)"
languages: [rust, python]
order: 84
family: "CEC 2006"
tab: g08
---

# CEC 2006 g08

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24 test
problems, g01 to g24, with their best known solutions and rules for comparing algorithms. g08 is
the eighth, the report's equations 18 and 19 (page 5). The report takes it from Koziel and
Michalewicz (1999, Evolutionary algorithms, homomorphous mappings, and constrained parameter
optimization, Evolutionary Computation 7(1): 19-44). It has no physical meaning: the variables are
x1 and x2, and the report names the constraints g1 and g2.

The source maximizes a function; the report, and genoxide, minimize its negative:

```text
f(x) = −sin³(2π x1) sin(2π x2) / (x1³ (x1 + x2))
```

subject to two inequalities, each g(x) ≤ 0:

```text
g1 = x1² − x2 + 1          (x2 lies above the parabola x2 = x1² + 1)
g2 = 1 − x1 + (x2 − 4)²    (x1 lies right of the parabola x1 = 1 + (x2 − 4)²)
```

Both variables lie in [0, 10]. At x1 = 0, the lower bound, f is 0/0, so genoxide's fitness is
invalid there, worse than any valid solution.

The minimum is f* = −0.0958250414180359, at x* = (1.2279713526, 4.2453733661). No constraint is
active there: the minimum lies inside the feasible region.

## What makes it hard

The two parabolas cross at (1, 4) and (2, 5), and the feasible region is the lens between them:
x1 from 1 to 2, x2 from 3 to 5. The report estimates each problem's feasible share of the box from
random points: 0.8560 % for g08. A sample of 10 million random points, drawn for this page, had
0.8654 % feasible ones.

The objective is multimodal. Its numerator repeats with period 1 in each variable, and its
denominator x1³ (x1 + x2) grows with both, so the box holds a grid of peaks that shrink from the
origin outwards. The lens contains parts of several. Besides the minimum, runs of this page ended
at two local minima: (1.734, 4.746), where f = −0.0291, a peak inside the lens, and (1.674, 3.802),
where f = −0.0258, on the boundary of g1, the part of a peak that lies inside it. A search that
settles on one of them has to leave the peak to find the minimum.

## Representation

A `Real` genome of 2 genes, x1 and x2, within [0, 10]. genoxide's `problems::cec2006::G08` is the
fitness: the value f(x) and the total constraint violation, max(0, g1(x)) + max(0, g2(x)), 0 for a
feasible solution.

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

The run has the report's budget of 500,000 evaluations, and stops once its best solution is
feasible with an absolute error f(x) − f* of at most 1e-8. The report counts a run as successful
with an error of at most 1e-4; the example asks for more.

Why the restarts: this is the problem where they matter. With 25 seeds and no restarts, CMA-ES met
the target on 20 runs, after a median of 300 evaluations. The other 5 converged to a local minimum,
four to (1.734, 4.746) and one to (1.674, 3.802), and sampled around it until the budget ran out;
their best solutions, found on the way, had errors from 0.02 to 0.07. With IPOP restarts, all 25
met the target: those 5 after one restart with a population of 12, after at most 2,046
evaluations. The median is 312.

SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78), genoxide's default differential
evolution, met the target on all 25 runs too, after a median of 4,600 evaluations (at most 5,100),
and L-SHADE, whose population shrinks over the budget, after a median of 2,700. SHADE's 100
individuals spread over the whole lens; CMA-ES's 6 samples follow one peak, which is faster when
it's the right one.

## Output

The first line names the run. The second gives what stopped it, after how many evaluations, the
error f(x) − f* and whether the best solution is feasible: "< 1e-8" means the run met its target.
The third gives the evaluations to the first feasible solution, and to an error of 1e-4, the
report's criterion of success. The fourth gives the restarts and the population of each run. The
fifth compares f(x) with f*, to 6 significant digits. The sixth gives the solution, and the last
the two constraints: "active" for a constraint on its boundary (|g| ≤ 1e-6), else the value of g,
negative when it's satisfied. In Python, `run` evaluates the problem in Rust, so both versions
print the same.

The page's plot shows each variable on its range, and each constraint's state: violated, active or
satisfied. Its curve shows the error f − f* of the best feasible solution, and of the population's
median, on a log scale. The best's curve begins at the first feasible solution, and the median's
once half the population is feasible.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g08) plays this run back.

## Good results

A good run is feasible and ends within 1e-4 of f*, the report's success. With restarts, CMA-ES
meets the target of 1e-8 with every seed tried.

Seed 1 needs no restart. Its first 6 samples already include a feasible one, with an error of
0.14. The run climbs the minimum's peak: it meets the report's criterion after 132 evaluations and
the target after 312, 52 generations of 6. The solution is x* to 4 decimals, x1 = 1.22798 and
x2 = 4.24540, inside the lens: g1 = −1.737 and g2 = −0.1678. The peak is flat at its top, so an
error of 1e-8 still leaves the solution about 3·10⁻⁵ from x* in x2.
