---
title: CEC 2006 g12
category: constrained
summary: A quadratic in 3 variables whose feasible region is 729 disjoint spheres, a trap for a search that settles in one, solved by SHADE with Deb's feasibility rules.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "−1 at (5, 5, 5) (proven)"
languages: [rust, python]
order: 128
family: "CEC 2006"
tab: g12
---

# CEC 2006 g12

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24 problems,
g01 to g24, from the literature, with their best known solutions and rules for comparing algorithms.
g12 is the twelfth, the report's equation 25 and the constraint below it (page 6). The report takes
it from Koziel and Michalewicz (1999, Evolutionary algorithms, homomorphous mappings, and
constrained parameter optimization, Evolutionary Computation 7(1): 19-44).

There are 3 variables, x1 to x3, each in [0, 10]. The source maximizes (100 − d²)/100, with d the
distance from x to the box's center (5, 5, 5); the report, and genoxide, minimize its negative:

```text
minimize   f(x) = −(100 − (x1 − 5)² − (x2 − 5)² − (x3 − 5)²) / 100
subject to g(x) = (x1 − p)² + (x2 − q)² + (x3 − r)² − 0.0625 ≤ 0
           for some p, q, r in {1, 2, …, 9}
```

The constraint puts a sphere of radius 0.25 around each of the 9³ = 729 points (p, q, r) of a grid
with spacing 1. A solution is feasible inside any of them. The spheres don't touch: their centers
are 1 apart, and their radii add up to 0.5. genoxide computes g from the nearest center, rounding
each variable to the nearest of 1 to 9, since the sum of squares is least there.

f is least at the box's center, (5, 5, 5), which is also the center of a sphere. The minimum there
is f* = −1, with g = −0.0625: the constraint isn't active. genoxide's docs mark it as proven.

## What makes it hard

Finding a feasible solution is easy. The spheres fill 729 × (4/3)π 0.25³ = 47.7 of the box's volume
of 1,000: 4.77 %, so about one random point in 21 is feasible.

The difficulty is getting from one sphere to another. Between two spheres, every point is
infeasible, and Deb's rules rank it below every feasible solution. A search that has gathered in one
sphere sees nothing better nearby: its best is the point of that sphere nearest to (5, 5, 5), and
every small step from there makes things worse. Only a step long enough to land inside a better
sphere helps. The points in between don't lead there: they are infeasible, and rank below the
search's own best, whatever their value.

## Representation

A `Real` genome of 3 genes, x1 to x3, in [0, 10]. genoxide's `problems::cec2006::G12` is the
fitness: the value f(x) and the constraint violation, max(0, g(x)), 0 inside a sphere.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights.

## Algorithm

SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78) is genoxide's default differential
evolution. It keeps 100 solutions, and builds each trial from the difference of two of them, added
to a solution between the current one and one of the best. A trial replaces its parent if it's not
worse. The scale of the difference and the rate of crossover adapt from the trials that succeed. The
run uses genoxide's defaults, with the restarts that genoxide adds to SHADE.

The run has the report's budget of 500,000 evaluations, and stops once its best solution is feasible
with an error f(x) − f* of at most 1e-8, an absolute error. The report counts a run as successful
with an error of at most 1e-4; the example asks for more.

Why SHADE: its population is spread over many spheres at once, and the difference between two of its
solutions in different spheres is a step of whole grid units, which can carry a trial from one
sphere into another. CMA-ES, which solves g11 and g14, fails here without restarts. Its samples
gather in one sphere, and its step size shrinks to fit it; then no sample reaches another. With
seeds 1 to 25, in the same budget:

| Algorithm | Runs that met the target | Evaluations (median, range) |
|---|---|---|
| SHADE | 25 of 25 | 8,800 (7,400 to 9,400) |
| L-SHADE | 25 of 25 | 5,552 (4,536 to 6,506) |
| CMA-ES with IPOP restarts | 25 of 25 | 25,123 (3,801 to 86,590) |
| CMA-ES | 0 of 25 | |

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) converged in a sphere
other than the center's in every run. With seed 17, it ended at (5.82, 5.82, 5), on the edge of the
sphere around (6, 6, 5), 1.16 from (5, 5, 5), with an error of 0.0135. In all 25 runs, its best
ended in another sphere, with errors of 0.0056 to 0.094: not a single sample landed in the center's
sphere. IPOP restarts (Auger and
Hansen, 2005, IEEE CEC 2005: 1769-1776) start a new run from a random point, with twice the
population, whenever one has converged; with enough of them, one run lands in the center's sphere.
L-SHADE (Tanabe and Fukunaga, 2014, IEEE CEC 2014: 1658-1665) starts with 18 × 3 = 54 solutions and
shrinks the population over the budget. These runs end before it has shrunk much, and it's faster
only because its population is smaller than SHADE's 100.

## Output

The first line names the run. The second gives what stopped it, after how many evaluations, the
error f(x) − f* and whether the best solution is feasible: "< 1e-8" means the run met its target.
The third gives the evaluations to the first feasible solution, and to an error of 1e-4, the
report's criterion of success. The fourth compares f(x) with f*, to 6 decimals. The fifth gives the
solution, and the last the nearest of the 729 centers and the value of g, at most 0 inside that
center's sphere. In Python, `run` evaluates the problem in Rust, so both versions print the same.

The page's plot shows each variable on its range, and the constraint's value g: at or below 0 inside
a sphere, −0.0625 at its center. Its curve shows the error f − f* of the best feasible solution, and
of the population's median, on a log scale.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g12) plays this run back.

## Good results

A good run ends inside the sphere around (5, 5, 5): at any other sphere's best point, the error is
at least 0.0056, at (5, 5, 4.25) and the like, 0.75 from the center. Within the center's sphere, the
error is the squared distance from (5, 5, 5), divided by 100: an error of 1e-8 allows a distance of
0.001.

The recorded run's first population already holds feasible solutions, with a best error of 0.19. Its
best moves to the sphere around (5, 6, 5) after 200 evaluations, with an error of 0.0062, and stays
there. After 2,300 evaluations, a trial moves x2 from 5.78 to 4.94, a step of almost one grid unit,
into the center's sphere: the error drops to 0.00027. The population follows: its median is feasible
from 900 evaluations on, and falls from 0.25 to 3e-7 by the end. The run meets the report's
criterion after 2,900 evaluations and its target after 9,300. The solution is x = (4.99930, 5.00010,
4.99940), 0.001 from the center, with g = −0.062499, well inside the sphere.
