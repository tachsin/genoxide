---
title: Koza's quartic
category: genetic programming
summary: Find the formula x⁴ + x³ + x² + x from 20 points of it, by genetic programming.
reference: "Koza, J. R. (1992). Genetic Programming: On the Programming of Computers by Means of Natural Selection. MIT Press."
reference_url: https://www.genetic-programming.com/gpbook1toc.html
optimum: "x⁴ + x³ + x² + x, exactly (an RMSE of 0, up to rounding)"
languages: [rust, python]
order: 255
---

# Koza's quartic

## The problem

Symbolic regression: given points (x, y), find a formula for y in x, not just its coefficients but
its shape. Here the points come from the quartic polynomial

y = x⁴ + x³ + x² + x,

at 20 values of x drawn at random from [−1, 1]. Koza (1992) used it to introduce genetic programming
for symbolic regression, and it became the problem most papers tested on: McDermott et al. (2012,
Genetic programming needs better benchmarks, GECCO 2012: 791-798) found it the most used of all,
and argued that it's a toy that says little about a method. It stays the classic first problem, as
here.

The formula is found when it's recovered exactly: an error at the level of rounding, both on the 20
training points and on 101 test points spread evenly over [−1, 1] that the search never sees.

The Python version builds the same islands with `gx.gp` and evaluates the trees in Rust
(`problem.regression(linear_scaling=False)`): it prints the same output and writes the same
trace.

## What makes it hard

Nothing tells the search what shape the formula has. It builds formulas from Koza's function set,
addition, subtraction, multiplication, protected division (1 where the denominator is 0), sine,
cosine, the exponential and a protected logarithm (ln |a|, and 0 at 0), and the variable x. There
are no constants: a formula needs x / x for a 1.

Many formulas come close without being it. Sines, cosines and exponentials of x fit a smooth curve
on [−1, 1] to within a few thousandths, and a search can settle there: the error is small but not
zero, and the formula grows as it adds corrections. Only the polynomial itself, in some
arrangement, has an error at rounding level.

## Representation

A tree of a genetic program (`gp::Tree`): the functions at the inner nodes, x at the leaves, stored
as a flat array in prefix order. The problem is `gp::regression::problems::Koza1`: its primitives
are `gp::regression::Math` values in a `gp::PrimitiveSet` of one type (`add`, `sub`, `mul`, the
protected division `pdiv`, `sin`, `cos`, `exp` and the protected logarithm `plog`), and its data
the 20 training and 101 test points. `gp::Gp` sets Koza's limits and initialization: trees of depth at most 17 (the root at depth
0) and at most 1024 nodes, and ramped half-and-half with depths 2 to 6. The initial population
of each island is `Gp::ramped_half_and_half`: Koza's even division, the same number of trees of each
depth, half by the full method and half by grow, without duplicates.

The fitness is the root mean squared error on the 20 points, minimized: `gp::regression::Regression`
without linear scaling, so the tree itself has to fit the points. The trees are evaluated on
all 20 points at once, a column of values per node (`Tree::evaluate_columns`), with genoxide's
`math` functions, so the values are the same bits on every platform. A tree with a value that isn't
finite (an exponential that overflows) is invalid.

## Algorithm

Eight genetic algorithms of 500 trees each, as islands in a ring that pass their two best trees on
every 10 generations. Each has tournaments of 7, subtree crossover at a rate of 0.9 (Koza's, with
the crossover points at function nodes 90% of the time, and within the limits), and subtree
mutation at a rate of 0.1 (a random node's subtree replaced by one grown to a depth of at most 4).
The run stops at an RMSE of 10⁻¹⁰ times the standard deviation of the 20 values, or after 200
generations.

Islands, because a single population often settles on a near miss: with one population of 500
(Koza's size) and the same settings, 9 of the 20 seeds I tried found the formula in 200
generations. Separate islands settle on different near misses, and the formula found on one spreads
to the others.

## Output

The first two lines give the problem and the setting. Then come how the run stopped, after how many
generations and evaluations, the RMSE of the best tree on the training and the test points, and the
tree itself, with its size and depth, written as genoxide writes trees (`Tree::display`): each
function by its name with its arguments in parentheses.

[The project page](https://tachsin.gr/projects/genoxide/examples/koza-quartic) plays this run back.

## Good results

The optimum is the quartic itself, recovered exactly. The run of `output.txt` found it after 4
generations and 17,061 evaluations, as

add(add(x, mul(x, x)), mul(add(x, mul(x, x)), mul(x, x))),

which is (x + x²) + (x + x²)·x² = x⁴ + x³ + x² + x, in 15 nodes. Its RMSE, about 10⁻¹⁶ on the
training and the test points, is the rounding of a different order of operations.

Over seeds 1 to 100 (the islands' seeds, with the same data), 96 runs recovered the formula, after
7 generations and 27,000 evaluations in the median, and 22 generations at most. The other 4 ended
after 200 generations on near misses of 131 to 385 nodes.
