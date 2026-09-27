---
title: 0/1 knapsack
category: constrained
summary: Choose the items with the highest total value whose total weight fits a capacity.
reference: "Martello, S. and Toth, P. (1990). Knapsack Problems: Algorithms and Computer Implementations. Wiley."
reference_url: "https://silvano333.github.io/kp.html"
optimum: "625 (total value)"
languages: [rust, python]
order: 20
---

# 0/1 knapsack

## The problem

The 0/1 knapsack problem has items, each with a weight and a value, and a knapsack with a capacity.
It chooses the items with the highest total value whose total weight is at most the capacity. Each
item is taken whole or not at all. Martello and Toth (1990) treat it and its variants in a book.

The example has 20 items and a capacity of 400. The first ten are the test data of Martello and
Toth's program for the multiple knapsack problem (Algorithm 632, 1985, ACM Transactions on
Mathematical Software 11(2): 135-140), where one knapsack of capacity 165 holds a value of at most
309. The other ten items and the capacity are this example's own. The first item weighs 23 and is
worth 92; the eighth weighs 85 and is worth 84. All 20 together weigh 896, more than twice the
capacity.

## What makes it hard

The problem is NP-hard (Martello and Toth, 1990), though dynamic programming solves it in time
proportional to the number of items times the capacity. With 20 items, there are 2²⁰ = 1,048,576
selections, and 356,115 of them (34%) fit.

The capacity constraint binds. The best selection is unique and weighs 395 of 400, and adding any
item to it breaks the limit. Taking items in order of value per unit of weight, while they fit,
gives 604, not the optimum.

## Representation

A `Binary` genome of 20 bits, a bit per item: 1 takes the item. Every bit string is a selection, but
not every selection fits.

The fitness function returns two numbers: the total value, and how far the weight exceeds the
capacity (0 when it fits). genoxide compares them with Deb's feasibility rules (Deb, 2000, Computer
Methods in Applied Mechanics and Engineering 186: 311-338). A selection that fits beats one that
doesn't. Two that fit compare by value. Two that don't compare by their excess weight. So overweight
selections still guide the search: the less overweight, the better. There is no penalty weight to
tune.

## Algorithm

A genetic algorithm:

- a population of 60;
- tournament selection of size 3; genoxide's guide recommends tournaments with constraints, since
  roulette selection gives infeasible selections no weight;
- two-point crossover, which swaps a segment of items between the parents;
- bit-flip mutation at a rate of 1/20 per bit, one flip per child on average.

The optimum isn't given to the run. It stops after 200 generations without improvement, or after
2,000 generations.

## Output

The first line lists the chosen items, numbered from 0. The second gives their total value and
weight, and the capacity. The third gives the evaluations, and the optimum that dynamic programming
finds for comparison.

The run stops only after 200 generations without improvement, so its last 200 generations find
nothing better, and their evaluations count too.

[The project page](https://tachsin.gr/projects/genoxide/examples/knapsack) plays this run back.

## Good results

The optimum is 625: 13 items that weigh 395. The run finds it, and the dynamic programming line
confirms it.
