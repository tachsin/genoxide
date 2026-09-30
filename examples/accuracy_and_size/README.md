---
title: Accuracy against size
category: genetic programming
summary: The trade-off between a formula's error and its size on Nguyen-7, ln(x + 1) + ln(x² + 1), found by NSGA-II on trees as a Pareto front.
reference: "Uy, N. Q., Hoai, N. X., O'Neill, M., McKay, R. I. and Galván-López, E. (2011). Semantically-based crossover in genetic programming: application to real-valued symbolic regression. Genetic Programming and Evolvable Machines 12(2): 91-119."
reference_url: https://doi.org/10.1007/s10710-010-9121-2
optimum: "The formula, ln(x + 1) + ln(x² + 1), is a tree of 13 nodes; this run's front doesn't reach it, and shows the best error found for each size instead"
languages: [rust]
order: 259
---

# Accuracy against size

## The problem

Symbolic regression of Nguyen-7 (Uy et al. 2011): find a formula for y in x from the points of

y = ln(x + 1) + ln(x² + 1),

at 20 values of x drawn at random from [0, 2], with Koza's function set (addition, subtraction,
multiplication, protected division, sine, cosine, the exponential and a protected logarithm, and x).
The formula is a tree of 13 nodes, plog(x + x/x) + plog(x·x + x/x), the 1s built from x / x. But
genetic programming rarely finds it: the [All twelve](../nguyen_all/) tab of the Nguyen problems
shows the usual search ending on trees of hundreds of nodes that fit the points to three or four
decimals.

That's the common case in symbolic regression on real data, where there is no exact formula to
find: every error can be lowered a little more by a larger tree. The question is then not which
formula, but which trade-off. Minimizing the error and the size together, as two objectives, answers
it in one run: the Pareto front, the smallest error found for each size, from the single variable x
to trees of 50 nodes, where the small ones are formulas a person can read.

There's no Python version: the Python package has no genetic programming yet.

## What makes it hard

A search on the error alone grows its trees (bloat): each small improvement adds nodes, and a larger
tree fits the training points better whether or not it's closer to the formula. The second
objective stops that, but it pulls the other way: a population that keeps small trees has fewer
large ones to recombine, and the most accurate end of the front is less accurate than a
single-objective search would reach.

## Representation

Trees of `gp::regression::problems::Nguyen7`'s primitives (`add`, `sub`, `mul`, the protected
division `pdiv`, `sin`, `cos`, `exp`, the protected logarithm `plog` and `x`), with Koza's limits
and initialization in `gp::Gp`: depth at most 17, at most 1024 nodes, ramped half-and-half with
depths 2 to 6.

The two objectives, both minimized: the root mean squared error on the 20 points after linear
scaling (`gp::regression::Regression`: the error of a + b × tree, with a and b fitted by least
squares, so a tree only needs the shape), and the number of nodes. A tree with a value that isn't
finite at a point is invalid.

## Algorithm

NSGA-II (`Nsga2`, Deb et al. 2002) on the trees, with a population of 1000, subtree crossover at a
rate of 0.9 and subtree mutation at a rate of 0.1, for 200 generations. NSGA-II keeps the trees of
the best non-dominated fronts, the most spread-out first, and removes duplicates, so the population
stays spread over the sizes instead of converging on one.

## Output

The first two lines give the problem and the setting. Then comes the Pareto front, one line per
point, from the smallest tree to the most accurate: its size, its RMSE on the training points and
on 100 test points from [0, 2] that the search never saw, and, up to 25 nodes, the formula with its
scaling. Trees that differ only in the order of their arguments have the same error and size: the
front lists one of them. Two points that look alike (sizes 1 and 3, x and plog(exp(x))) differ in the
last digits of their errors, by rounding.

The project page plays the run back: the front of each generation, as the logarithm of the error
against the size.

## Good results

This run's front has 27 points, from x alone (size 1, RMSE 0.022) to a tree of 50 nodes (RMSE
1.1 × 10⁻⁴). Along it:

- 7 nodes cut the error of a straight line to a third: −0.8709 + 1.2597 × (cos(cos(cos(cos(x)))) + x), RMSE
  0.0074;
- 13 nodes cut it tenfold, to 0.0020;
- 25 nodes reach 4.2 × 10⁻⁴ on the training and 6.7 × 10⁻⁴ on the test points.

Beyond about 25 nodes, the training error keeps falling, to 1.1 × 10⁻⁴, but the test error rises, to
2 × 10⁻³: the larger trees fit the 20 points more closely and the formula less. The front shows
where that starts, which a search on the error alone can't.

The exact formula, 13 nodes, isn't on the front. In 12 more runs I tried, seeds 2 to 4 with these
settings and 9 with a population of 2000, 500 generations or a mutation rate of 0.3, the best error
stayed between 6 × 10⁻⁵ and 5 × 10⁻⁴, without exact recovery.
