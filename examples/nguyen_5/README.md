---
title: Nguyen-5
category: genetic programming
summary: Find the formula sin(x²) cos(x) − 1 from 20 points of it, by genetic programming with linear scaling, which supplies the constant that the function set lacks.
reference: "Uy, N. Q., Hoai, N. X., O'Neill, M., McKay, R. I. and Galván-López, E. (2011). Semantically-based crossover in genetic programming: application to real-valued symbolic regression. Genetic Programming and Evolvable Machines 12(2): 91-119."
reference_url: https://doi.org/10.1007/s10710-010-9121-2
optimum: "sin(x²) cos(x) − 1, exactly (an RMSE of 0, up to rounding)"
languages: [rust]
order: 258
family: Nguyen
tab: Nguyen-5
---

# Nguyen-5

## The problem

Symbolic regression: find a formula for y in x from the points of

y = sin(x²) cos(x) − 1,

at 20 values of x drawn at random from [−1, 1]. It's the fifth of the twelve Nguyen problems (Uy et
al. 2011), the first with sines and cosines. The formula is found when it's recovered exactly: an
error at the level of rounding on the 20 training points and on 100 test points from [−1, 1] that
the search never sees.

The problem's name, target, sampling and function set are as McDermott et al. (2012) restate them,
not yet checked against the paper. Its points come from a fixed seed of genoxide's random stream;
another library's differ.

There's no Python version: the Python package has no genetic programming yet.

## What makes it hard

The function set, Koza's (addition, subtraction, multiplication, protected division, sine, cosine,
the exponential and a protected logarithm, and x), has no constants. The shape sin(x²) cos(x) is a
tree of 6 nodes, but the − 1 has to be built too, from x / x (a protected division that is 1
everywhere) or cos(x − x), and a search on the tree's own error has to find both at once. Without
the − 1, the best trees fit the curve shifted by one, and their error stays large.

## Representation

A tree of a genetic program (`gp::Tree`), of the primitives of `gp::regression::problems::Nguyen5`:
`add`, `sub`, `mul`, the protected division `pdiv`, `sin`, `cos`, `exp`, the protected logarithm
`plog` and `x`. `gp::Gp` sets Koza's limits and initialization: depth at most 17, at most 1024
nodes, and ramped half-and-half with depths 2 to 6.

The fitness is the root mean squared error on the 20 points after linear scaling
(`gp::regression::Regression`, with scaling on, its default): the error of a + b × tree, with a and
b fitted to the points by least squares for each tree (Keijzer 2003, Improving symbolic regression
with interval arithmetic and linear scaling, EuroGP 2003: 70-82). The search then only needs the
shape, sin(x²) cos(x); the fitted a = −1 and b = 1 give the rest. Without scaling, the same search
recovered the formula in 8 of 20 runs I tried; with it, in 94 of 100. A tree with a value that isn't
finite is invalid.

## Algorithm

The same as the Nguyen-1 page and Koza's quartic: eight genetic algorithms of 500 trees each, as
islands in a ring that pass their two best trees on every 10 generations, each with tournaments of
7, subtree crossover at a rate of 0.9 and subtree mutation at a rate of 0.1. The run stops at an RMSE
of 10⁻¹⁰ times the standard deviation of the 20 values, or after 200 generations.

## Output

The first two lines give the problem and the setting. Then come how the run stopped, after how many
generations and evaluations, the RMSE of the best tree on the training and the test points after
its scaling, and the tree with its scaling, as `a + b * (tree)` (`Regression::display`).

## Good results

The optimum is the formula itself, recovered exactly. The run of `output.txt` found it after 2
generations and 10,670 evaluations, as

−1 + 1 × (mul(cos(x), sin(mul(mul(x, x), pdiv(x, x))))),

that is cos(x) · sin(x · x · 1) − 1, in 11 nodes, with an RMSE of 0 on the training and the test
points: the scaling's intercept and slope are exactly −1 and 1.

Over 100 runs (the islands' seeds 100 s + 0 to 7 for s = 1 to 100, the first being `output.txt`'s,
with the same data), 94 recovered the formula, after 2 generations and 10,600 evaluations in the
median, and 7 generations at most. The other 6 ended after 200 generations on near misses of 166 to
283 nodes, with test errors from 3 × 10⁻⁵ to 0.2.
