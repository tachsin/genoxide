---
title: Nguyen-1
category: genetic programming
summary: Find the formula x³ + x² + x from 20 points of it, by genetic programming, the first of Nguyen's twelve symbolic regression problems.
reference: "Uy, N. Q., Hoai, N. X., O'Neill, M., McKay, R. I. and Galván-López, E. (2011). Semantically-based crossover in genetic programming: application to real-valued symbolic regression. Genetic Programming and Evolvable Machines 12(2): 91-119."
reference_url: https://doi.org/10.1007/s10710-010-9121-2
optimum: "x³ + x² + x, exactly (an RMSE of 0, up to rounding)"
languages: [rust]
order: 258
family: Nguyen
tab: Nguyen-1
---

# Nguyen-1

## The problem

Symbolic regression: given points (x, y), find a formula for y in x. Here the points come from

y = x³ + x² + x,

at 20 values of x drawn at random from [−1, 1]. It's the first of the twelve problems of Uy et al.
(2011), known as the Nguyen problems: polynomials, sines and cosines, logarithms, a square root and
four functions of two variables, all with the same function set. McDermott et al. (2012, Genetic
programming needs better benchmarks, GECCO 2012: 791-798) list them among the problems most used
to compare genetic programming methods, together with Koza's quartic.

The formula is found when it's recovered exactly: an error at the level of rounding, both on the 20
training points and on 100 test points drawn from [−1, 1] that the search never sees.

The problem's name, target, sampling and function set are as McDermott et al. (2012) restate them,
not yet checked against the paper. Its points come from a fixed seed of genoxide's random stream:
another library's 20 points differ, and so can how hard the problem is (the
[All twelve](../nguyen_all/) tab shows how much).

There's no Python version: the Python package has no genetic programming yet.

## What makes it hard

Little, for genetic programming: it's the smallest of the Nguyen polynomials, and a tree of 11
nodes, x·x·x + (x + x·x), is enough. The function set is Koza's: addition, subtraction,
multiplication, protected division (1 where the denominator is 0), sine, cosine, the exponential and
a protected logarithm (ln |a|, and 0 at 0), and the variable x, without constants. As with Koza's
quartic, sines, cosines and exponentials of x fit the smooth curve closely without being it, and a
search can settle on such a near miss; here that's rare.

## Representation

A tree of a genetic program (`gp::Tree`): the functions at the inner nodes, x at the leaves, stored
as a flat array in prefix order. The problem is `gp::regression::problems::Nguyen1`: its primitives
are `gp::regression::Math` values in a `gp::PrimitiveSet` of one type (`add`, `sub`, `mul`, the
protected division `pdiv`, `sin`, `cos`, `exp` and the protected logarithm `plog`), and its data the
20 training and 100 test points. `gp::Gp` sets Koza's limits and initialization: trees of depth at
most 17 and at most 1024 nodes, and ramped half-and-half with depths 2 to 6, the initial population
of each island by `Gp::ramped_half_and_half` (Koza's even division, without duplicates).

The fitness is the root mean squared error on the 20 points, minimized: `gp::regression::Regression`
without linear scaling, so the tree itself has to fit the points. With linear scaling (the error of
a + b × tree, a and b fitted by least squares), the search recovers the formula less often: in 13
of 20 runs I tried, against every one without. Scaling makes any tree with the right shape up to a
shift and a stretch as good as the formula, and many trees of sines and exponentials come close to
that shape. The trees are evaluated on all 20 points at once (`Tree::evaluate_columns`), with
genoxide's `math` functions, so the values are the same bits on every platform. A tree with a value
that isn't finite is invalid.

## Algorithm

The same as Koza's quartic: eight genetic algorithms of 500 trees each, as islands in a ring that
pass their two best trees on every 10 generations. Each has tournaments of 7, subtree crossover at a
rate of 0.9 (with the crossover points at function nodes 90% of the time, within the limits), and
subtree mutation at a rate of 0.1. The run stops at an RMSE of 10⁻¹⁰ times the standard deviation
of the 20 values, or after 200 generations.

## Output

The first two lines give the problem and the setting. Then come how the run stopped, after how many
generations and evaluations, the RMSE of the best tree on the training and the test points, and the
tree itself, with its size and depth, written as genoxide writes trees (`Tree::display`): each
function by its name with its arguments in parentheses.

## Good results

The optimum is the polynomial itself, recovered exactly. The run of `output.txt` found it after 4
generations and 16,671 evaluations, as

add(mul(x, mul(x, x)), add(x, mul(x, x))),

which is x·x·x + (x + x·x) = x³ + x² + x, in 11 nodes, with an RMSE of about 10⁻¹⁷ on the training
and 10⁻¹⁶ on the test points, the rounding of a different order of operations.

Over 100 runs (the islands' seeds 100 s + 0 to 7 for s = 1 to 100, the first being `output.txt`'s,
with the same data), all 100 recovered the formula, after 3 generations and 13,500 evaluations in
the median, and 7 generations at most.
