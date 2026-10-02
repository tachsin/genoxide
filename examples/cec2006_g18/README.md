---
title: CEC 2006 g18
category: constrained
summary: The area of a hexagon in 9 variables whose diagonals are at most 1, under 13 inequality constraints, with a local optimum that traps a third of the runs, solved by CMA-ES with IPOP restarts and Deb's feasibility rules.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "−0.866025403784439, −√3/2 (best known)"
languages: [rust, python]
order: 134
family: "CEC 2006"
tab: g18
---

# CEC 2006 g18

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24
problems, g01 to g24, from the literature, with their best known solutions and rules for comparing
algorithms. g18 is the eighteenth, the report's equations 37 and 38 (page 11). The report takes it
from Himmelblau (1972, Applied Nonlinear Programming, McGraw-Hill). The source maximizes; the
report, and genoxide, minimize the negated value.

There are 9 variables, x1 to x8 in [−10, 10] and x9 in [0, 20]. The problem is

```text
minimize   f(x) = −0.5 (x1 x4 − x2 x3 + x3 x9 − x5 x9 + x5 x8 − x6 x7)
subject to g1  = x3² + x4² − 1 ≤ 0                g8  = (x3 − x7)² + (x4 − x8)² − 1 ≤ 0
           g2  = x9² − 1 ≤ 0                      g9  = x7² + (x8 − x9)² − 1 ≤ 0
           g3  = x5² + x6² − 1 ≤ 0                g10 = x2 x3 − x1 x4 ≤ 0
           g4  = x1² + (x2 − x9)² − 1 ≤ 0         g11 = −x3 x9 ≤ 0
           g5  = (x1 − x5)² + (x2 − x6)² − 1 ≤ 0  g12 = x5 x9 ≤ 0
           g6  = (x1 − x7)² + (x2 − x8)² − 1 ≤ 0  g13 = x6 x7 − x5 x8 ≤ 0
           g7  = (x3 − x5)² + (x4 − x6)² − 1 ≤ 0
```

The formulas have a geometric reading, derived here from the definition. Take six points in the
plane: A = (x1, x2), B = (x3, x4), C = (0, x9), D = (x5, x6), E = (x7, x8) and the origin O. Then
−f is the shoelace formula for the area of the hexagon A-B-C-D-E-O. g1 to g9 keep the distance
between two of the points at most 1: O and B, O and C, O and D, A and C, A and D, A and E, B and
D, B and E, C and E. These are the hexagon's nine diagonals, the pairs of vertices that aren't
neighbors. g10 to g13 keep the triangles O-A-B, O-B-C, O-C-D and O-D-E turning the same way.

The best known value is f* = −0.866025403784439, −√3/2 to its digits, at the report's x*. It isn't
proven optimal. g1, g3, g4, g6, g7 and g9 are active there.

## What makes it hard

No constraint bounds the hexagon's six sides, and the shoelace formula doesn't require a hexagon
whose sides don't cross. The best known solutions use this. Their active constraints are the sides
of two equilateral triangles with sides of 1, A-C-E and B-D-O. The hexagon A-B-C-D-E-O jumps from
one triangle to the other, its sides cross five times, and the formula adds up the two triangles'
areas: 2 · √3/4 = √3/2. No figure whose points are all within 1 of each other has that much area:
even a disk of diameter 1 has π/4 ≈ 0.785.

A proper hexagon competes with it. At f = −0.674981, 0.191 above f*, lies a convex hexagon whose
sides don't cross and whose 15 distances are all at most 1. It's a local optimum, far from the best
known solutions: x9 is 1 there, and 0.6 at the report's x*. Of 25 runs of CMA-ES without restarts,
9 converged to it and stayed there.

The best known solutions are also not unique. The two triangles can shift against each other: at
the report's x*, A, B and C are 0.6 from D, E and O, and at the run's solution 0.83. The feasible
region is small too: of 10 million random points in the box, none was feasible.

## Representation

A `Real` genome of 9 genes, x1 to x9, within the report's bounds. genoxide's
`problems::cec2006::G18` is the fitness: the value f(x) and the total constraint violation, the
sum of max(0, g(x)) over the 13 inequalities, 0 for a feasible solution.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. It uses genoxide's
defaults: a population of 4 + ⌊3 ln 9⌋ = 10, a step size of 0.3 of each gene's range and a random
start. With IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776), a run that has
converged starts again from a random point, with twice the population. A sample outside the bounds
is drawn again, up to 100 times, and then clipped to them. Deb's rules rank the samples.

The run has the report's budget of 500,000 evaluations, and stops once its best solution is
feasible with an error f(x) − f* of at most 1e-8, an absolute error. The report counts a run as
successful with an error of at most 1e-4; the example asks for more.

Why IPOP restarts: they let a run that converged to the local optimum start again elsewhere, with
a larger population each time. With seeds 1 to 25, IPOP-CMA-ES met the target on all 25
runs, after a median of 18,480 evaluations (12,050 to 192,960). Without restarts, CMA-ES met it on
15; of the other 10, 9 stayed at the local optimum and one at −0.5, 0.366 above f*. L-SHADE (Tanabe
and Fukunaga, 2014, IEEE CEC 2014: 1658-1665), genoxide's differential evolution with a population
that shrinks over the budget, also met the target on all 25, after a median of 86,540 evaluations
and at most 99,405: slower, but more even. SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78)
met it on 24 of 25, after a median of 74,700; the other run stayed at the local optimum. The
example shows CMA-ES, whose run with seed 1 misses the trap and needs no restart.

## Output

The first line names the run. The second gives what stopped it, after how many evaluations and
restarts, the error f(x) − f* and whether the best solution is feasible: "< 1e-8" means the run met
its target. The third gives when the best solution was first feasible, and when its error first
met the report's criterion of success. The fourth compares f(x) with f*, to 6 significant digits,
and the fifth gives the solution, to 4 digits. The last names the active constraints, those with
|g| ≤ 1e-6, and counts the others. In Python, `run` evaluates the problem in Rust, so both versions
print the same.

The page shows each variable on its range, and each of the 13 constraints' state. Its curve shows
the error f − f* of the best feasible solution, and of the population's median, on a log scale.
The best's curve begins at the first feasible solution, and the median's once half the population
is feasible.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g18) plays this run back.

## Good results

A good run is feasible and ends within 1e-4 of f*, the report's success. A value below f* is
possible, since f* is only the best known, but the runs here end just above it.

The recorded run finds its first feasible solution after 460 evaluations. Its first CMA-ES run, of
10 samples, finds the two triangles without coming near the local optimum, where x9 is 1: from then
on, its best solution's x9 stays below 0.86. It is within 1e-4 of f* after 5,310 evaluations, stays
about 5.2e-5 above f* from about 7,000 to 11,000 evaluations, and meets its target after 20,250,
without a restart.

The solution is x = (−0.08706, −0.1688, 0.8193, −0.5734, −0.08696, −0.9962, 0.8192, 0.2539,
0.8274), another best known solution than the report's, with the same six active constraints: g1,
g3, g4, g6, g7 and g9. A and D, B and E, C and O are 0.83 apart, further than at the report's x*:
the two triangles are shifted against each other along C-O.
