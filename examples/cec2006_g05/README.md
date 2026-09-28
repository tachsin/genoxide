---
title: CEC 2006 g05
category: constrained
summary: A cubic in 4 variables under 3 nonlinear equality constraints, whose feasible set is a thin tube around a curve, solved by CMA-ES with Deb's feasibility rules.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "5126.4967140071 (best known, with the equalities met within 0.0001)"
languages: [rust, python]
order: 81
---

# CEC 2006 g05

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24 problems,
g01 to g24, from the literature, with their best known solutions and rules for comparing
algorithms. g05 is the fifth. The report takes it from Hock and Schittkowski (1981, Test Examples
for Nonlinear Programming Codes, Lecture Notes in Economics and Mathematical Systems 187, Springer).

There are 4 variables, x1 and x2 in [0, 1200], and x3 and x4 in [−0.55, 0.55]. The problem is

```text
minimize   f(x) = 3 x1 + 0.000001 x1³ + 2 x2 + (0.000002 / 3) x2³
subject to g1(x) = −x4 + x3 − 0.55 ≤ 0
           g2(x) = −x3 + x4 − 0.55 ≤ 0
           h3(x) = 1000 sin(−x3 − 0.25) + 1000 sin(−x4 − 0.25) + 894.8 − x1 = 0
           h4(x) = 1000 sin(x3 − 0.25) + 1000 sin(x3 − x4 − 0.25) + 894.8 − x2 = 0
           h5(x) = 1000 sin(x4 − 0.25) + 1000 sin(x4 − x3 − 0.25) + 1294.8 = 0
```

g1 and g2 say that x3 and x4 differ by at most 0.55. h3 and h4 fix x1 and x2 from x3 and x4, and
h5 ties x4 to x3. The feasible set is a curve: one degree of freedom left of four. f grows with x1
and x2, so the minimum is the point of the curve where they're smallest, together.

Real-valued samples almost never meet an equality exactly. The report counts an equality as met
when |h(x)| ≤ 0.0001, and so does genoxide's `G05`: the curve becomes a thin tube.
The best known value is f* = 5126.4967140071, at x = (679.945148297028709, 1026.06697600004691,
0.118876369094410433, −0.396233485215178260), found by Koziel and Michalewicz (1999, Evolutionary
Computation 7(1): 19-44). It meets the equalities within the tolerance only, and genoxide's docs
mark it as a best known value, not a proven optimum.

## What makes it hard

The feasible region has no volume without the tolerance, and hardly any with it. The report
estimates each problem's feasible share of the box from random points, and gives 0.0000 % for g05,
to its four decimals. A search must first reach the tube from outside, guided only by the
violation.

Once inside, it has to stay there. The tube is 0.0001 wide in each h, and the sines bend it, so a
step in almost any direction leaves it. A search can only move along the curve, in small steps.

The best solutions also sit on the tube's wall. Moving off the curve, within the tolerance, lowers
f a little, and the runs here end with each |h| at 0.0001, on the edge of what counts as met.

## Representation

A `Real` genome of 4 genes, x1 to x4, within the report's bounds. genoxide's
`problems::cec2006::G05` is the fitness: the value f(x) and the total constraint violation,
max(0, g1) + max(0, g2) + max(0, |h3| − 0.0001) + max(0, |h4| − 0.0001) + max(0, |h5| − 0.0001),
0 for a feasible solution.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. It uses genoxide's
defaults: a population of 4 + ⌊3 ln 4⌋ = 8, a step size of 0.3 of each gene's range, a random start
and no restarts. A sample outside the bounds is drawn again, up to 100 times, and then clipped to
them. Deb's rules rank the samples.

The run has the report's budget of 500,000 evaluations, and stops once its best solution is
feasible with an error f(x) − f* of at most 1e-8. The report counts a run as successful with an
error of at most 1e-4; the example asks for more.

Why CMA-ES: its covariance matrix can learn the direction of the tube, so its samples spread along
the curve rather than across it. With 25 seeds, CMA-ES met the target on 24 runs, after a median of
21,728 evaluations (at most 82,880). The other run converged 8.7e-8 above f*, still a success by the
report's criterion; with IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776), all 25
met the target. SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78), genoxide's default
differential evolution, met it on 24 of 25 runs, after a median of 148,000 evaluations (at most
246,100). The other run found a feasible solution 45 above f*, and then restarted every 200
generations without finding a better one. Without restarts (`de::Restarts::Never`, or `restarts="never"` in Python), SHADE
met the target on all 25, after a median of 150,600 evaluations.

## Output

The first line names the run. The second gives what stopped it, the error f(x) − f* and whether the
best solution is feasible: "< 1e-8" means the run met its target. The third compares f(x) with f*,
to 6 significant digits. The fourth gives the solution, to 4 significant digits, and the last the
five constraints. An inequality is "active" on its boundary (|g| ≤ 1e-6), else the line gives g,
negative when it's satisfied. An equality met within the tolerance is always "active", else the line
gives how far |h| exceeds 0.0001. In Python, `run` evaluates the problem in Rust, so both versions
print the same.

The equalities call the platform's `sin`, whose last bit can differ between operating systems. A run
can then take a different path, and stop after a different number of evaluations at a slightly
different point of the tube. The example prints only what doesn't depend on it: no evaluation
counts, and the solution to 4 digits. With seeds 1 to 12, x1 ended between 679.944 and 679.946.
This run is the same on Windows and Linux.

The page shows each variable on its range, and each constraint's state.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g05) plays this run back.

## Good results

A good run is feasible and ends within 1e-4 of f*, the report's success. A value below f* is
possible, since f* is only the best known, but the runs here end just above it.

The recorded run finds its first feasible solution after about 750 evaluations, at f ≈ 5379, 250
above f*. It then follows the tube down, in steps that the tube keeps small: f is about 5200 after
20,000 evaluations, and within 1e-4 of f* after about 41,000. It meets its target soon after. The
solution is x = (679.9, 1026, 0.1189, −0.3962), the best known point to 4 digits. g1 and g2 have
slack, and the three equalities are met, each with |h| at 0.0001, on the edge of the tolerance.
