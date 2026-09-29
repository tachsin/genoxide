---
title: "|x| by strongly typed GP"
category: genetic programming
summary: Find the absolute value |x| from 20 points of it, with a comparison that returns a Boolean and a conditional that takes one, by strongly typed genetic programming.
reference: "Montana, D. J. (1995). Strongly typed genetic programming. Evolutionary Computation 3(2): 199-230."
reference_url: https://doi.org/10.1162/evco.1995.3.2.199
optimum: "|x| exactly (an RMSE of 0, up to rounding)"
languages: [rust]
order: 257
---

# |x| by strongly typed GP

## The problem

Symbolic regression again, as in [Koza's quartic](../koza_quartic/): given points (x, y), find a
formula for y. Here y = |x|, at 20 values of x drawn at random from [−1, 1], and the formula is found
when it's recovered exactly: an error at the level of rounding on the 20 training points and on 101
test points spread evenly over [−1, 1].

The absolute value has a kink at 0, and no polynomial has one. It takes a decision: −x where x is
negative, else x. So the formulas are built from two types of values, real numbers and Booleans:
`less` compares two reals and returns a Boolean, `if` takes a Boolean and two reals, and `and`,
`or` and `not` combine Booleans. This is strongly typed genetic programming (Montana 1995): every
function says what types its arguments and its result have, and the search only makes programs in
which they fit.

There's no Python version: the Python package has no genetic programming yet.

## What makes it hard

Without types, a function set has to make every function accept every value: Koza (1992) avoided
predicates such as a comparison that returns a Boolean, as Montana notes. Types let the set say
what a program means: a comparison goes where a condition goes, and a number never does. With types,
generation, crossover and mutation have to keep every program well typed: a subtree is only ever
replaced by one of its own type.

The problem itself is small. Close fits without a decision miss by far more than rounding (x² has
an RMSE of 0.18 over [−1, 1]), and recovering |x| exactly needs the comparison at 0 and the two
branches right.

## Representation

A tree of a genetic program (`gp::Tree`) over a `gp::PrimitiveSet` of two types, `real` and
`bool`: add, sub and mul of two reals; less(a, b), true when a < b; and, or and not of Booleans;
if(c, a, b), a when c is true, else b; the variable x; and ephemeral random constants, the integers
from −2 to 2, which a real can be. Trees return a real. `gp::Gp` sets Koza's limits and
initialization: depth at most 17 and at most 1024 nodes, ramped half-and-half with depths 2 to 6,
the initial population divided evenly without duplicates (`Gp::ramped_half_and_half`).

The fitness is the root mean squared error on the 20 points, minimized. Each tree is evaluated at
each point with `Tree::evaluate`, on a stack of values of an enum with a real and a Boolean
variant: the set's types guarantee which one each function gets.

## Algorithm

A genetic algorithm of 1000 trees, with double tournaments (fitness tournaments of 7, the smaller
of their two winners chosen with probability 0.7) against bloat, subtree crossover at a rate of 0.9,
and at a rate of 0.1 one of four mutations: subtree (weight 0.5), point (0.3), hoist (0.1) and
shrink (0.1). Every operator keeps the trees typed. The run stops at an RMSE of 10⁻¹⁰ times the
standard deviation of the 20 values, or after 100 generations.

## Output

The first two lines give the problem and the setting. Then come how the run stopped, after how many
generations and evaluations, the RMSE of the best tree on the training and the test points, and the
tree itself, with its size and depth.

[The project page](https://tachsin.gr/projects/genoxide/examples/abs-typed) plays this run back.

## Good results

The optimum is |x| itself, recovered exactly. The run of `output.txt` found it after 2 generations
and 2600 evaluations, as

if(less(x, 0.0), mul(-1.0, x), sub(x, 0.0)),

which is −x where x < 0, else x: |x|, with an error of 0 on the training and the test points.

Over seeds 1 to 100 (the seed of the algorithm and of the initial population, with the same data),
all 100 runs recovered |x|: after 3 generations and 3285 evaluations in the median, and 12
generations at most. The expressions found had 20 nodes in the median and 110 at most; some reach
the same function by longer routes, such as 2 · if(x < 0, −x, 0) + x.
