---
title: Nguyen-9
category: genetic programming
summary: Find the formula sin(x) + sin(y²) of two variables from 100 points of it, by genetic programming.
reference: "Uy, N. Q., Hoai, N. X., O'Neill, M., McKay, R. I. and Galván-López, E. (2011). Semantically-based crossover in genetic programming: application to real-valued symbolic regression. Genetic Programming and Evolvable Machines 12(2): 91-119."
reference_url: https://doi.org/10.1007/s10710-010-9121-2
optimum: "sin(x) + sin(y²), exactly (an RMSE of 0, up to rounding)"
languages: [rust]
order: 258
family: Nguyen
tab: Nguyen-9
---

# Nguyen-9

## The problem

Symbolic regression in two variables: find a formula for z in x and y from the points of

z = sin(x) + sin(y²),

at 100 points (x, y) drawn at random from the square [−1, 1]². It's the ninth of the twelve Nguyen
problems (Uy et al. 2011), the first of their four of two variables. The formula is found when it's
recovered exactly: an error at the level of rounding on the 100 training points and on 500 test
points from the square that the search never sees.

The problem's name, target, sampling and function set are as McDermott et al. (2012) restate them,
not yet checked against the paper. Its points come from a fixed seed of genoxide's random stream;
another library's differ.

There's no Python version: the Python package has no genetic programming yet.

## What makes it hard

A second variable doubles the leaves a tree can have and multiplies the shapes it can take. But the
formula is a sum of one term in each variable, 7 nodes, and a search can improve each term apart:
a tree with sin(x) right and the y term wrong is already better than one without either. That makes
it one of the easiest Nguyen problems for genetic programming, as the page's results show; the
[All twelve](../nguyen_all/) tab compares it with the others.

## Representation

A tree of a genetic program (`gp::Tree`), of the primitives of `gp::regression::problems::Nguyen9`:
Koza's function set (`add`, `sub`, `mul`, the protected division `pdiv`, `sin`, `cos`, `exp`, the
protected logarithm `plog`) and the variables `x` and `y`. `gp::Gp` sets Koza's limits and
initialization: depth at most 17, at most 1024 nodes, and ramped half-and-half with depths 2 to 6.

The fitness is the root mean squared error on the 100 points, minimized, without linear scaling
(`gp::regression::Regression`): the tree itself has to fit the points. The trees are evaluated on
all 100 points at once, a column of values per node (`Tree::evaluate_columns`), with genoxide's
`math` functions, so the values are the same bits on every platform.

## Algorithm

The same as the Nguyen-1 page and Koza's quartic: eight genetic algorithms of 500 trees each, as
islands in a ring that pass their two best trees on every 10 generations, each with tournaments of
7, subtree crossover at a rate of 0.9 and subtree mutation at a rate of 0.1. The run stops at an RMSE
of 10⁻¹⁰ times the standard deviation of the 100 values, or after 200 generations.

## Output

The first two lines give the problem and the setting. Then come how the run stopped, after how many
generations and evaluations, the RMSE of the best tree on the training and the test points, and the
tree itself, with its size and depth, written as genoxide writes trees (`Tree::display`).

On the project page, the curve of the best tree is drawn along the diagonal x = y of the square,
beside the target's: a slice of the surface, since a curve can't show all of it.

## Good results

The optimum is the formula itself, recovered exactly. The run of `output.txt` found it after 7
generations and 25,727 evaluations, as

add(sin(mul(y, y)), sin(x)),

the formula in 7 nodes, with an RMSE of 0 on the training and the test points.

Over 100 runs (the islands' seeds 100 s + 0 to 7 for s = 1 to 100, the first being `output.txt`'s,
with the same data), all 100 recovered the formula, after 5 generations and 18,700 evaluations in
the median, and 11 generations at most.
