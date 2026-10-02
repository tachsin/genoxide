---
title: 0/1 knapsack
category: constrained
summary: Choose the items with the highest total value whose total weight fits a capacity, on an instance of 50 items drawn from one of Pisinger's classes, checked against dynamic programming.
reference: "Pisinger, D. (2005). Where are the hard knapsack problems? Computers & Operations Research 32(9): 2271-2284."
reference_url: https://doi.org/10.1016/j.cor.2004.03.002
optimum: "20849 (total value, by dynamic programming)"
languages: [rust, python]
order: 20
---

# 0/1 knapsack

## The problem

The 0/1 knapsack problem has items, each with a weight and a value (a profit), and a knapsack with
a capacity. It chooses the items with the highest total value whose total weight is at most the
capacity; each item is taken whole or not at all. Martello and Toth (1990, *Knapsack Problems:
Algorithms and Computer Implementations*, Wiley) treat it and its variants in a book.

The instance is drawn from the first of Pisinger's (2005) classes of generated instances, the
uncorrelated one: 50 items whose weights and values are drawn independently and uniformly from 1
to R = 1000. The capacity is half the total weight, rounded down, as instance 50 of Pisinger's
series of 100 has it: c = ⌊50/101 Σ w⌋ (his eq. 5), 12,572 of 25,397. The instance is genoxide's
`problems::binary::Knapsack::generator(KnapsackClass::Uncorrelated, 50).seed(1)`: its items are
drawn from seed 1 with genoxide's portable random numbers, the same on every platform and in Python.
The first item weighs 613 and is worth 705; the second weighs 858 and is worth 45.

## What makes it hard

The problem is NP-hard, though only weakly: dynamic programming over the capacities solves it in
time proportional to the number of items times the capacity (Bellman's recursion), and genoxide's
`optimum` does, exactly. With 50 items there are 2⁵⁰ ≈ 10¹⁵ selections, far too many to try.

The capacity binds: the best selection weighs 12,551 of 12,572, and taking any further item breaks
the limit: the lightest item left out weighs 209. Pisinger's classes are hard in different ways.
Uncorrelated instances are "generally easy to solve" for exact algorithms; in the strongly
correlated class, whose values are the weights plus R/10, sorting by value per unit of weight is
sorting by weight, and the continuous optimum is far from the integer one.

## Representation

A `Binary` genome of 50 bits, a bit per item: 1 takes the item. Every bit string is a selection, but
not every selection fits.

The fitness has two numbers: the total value, and how far the weight exceeds the capacity (0 when
it fits), `constraint::at_most(weight, capacity)`. genoxide compares them with Deb's feasibility
rules (Deb, 2000, Computer Methods in Applied Mechanics and Engineering 186: 311-338). A selection
that fits beats one that doesn't. Two that fit compare by value. Two that don't compare by their
excess weight. So overweight selections still guide the search: the less overweight, the better.
There is no penalty weight to tune.

## Algorithm

A genetic algorithm:

- a population of 200;
- tournament selection of size 3; genoxide's guide recommends tournaments with constraints, since
  roulette selection gives infeasible selections no weight;
- uniform crossover, which takes each item's bit from either parent;
- bit-flip mutation at a rate of 1/50 per bit, one flip per child on average.

It stops as soon as it reaches the optimum, which dynamic programming finds beforehand; or, if it
never does, after 200 generations without improvement, or after 2,000 generations. A run from seed
1, then runs from seeds 1 to 20. As a contrast, the same runs on an instance of Pisinger's strongly
correlated class, 50 items from seed 1.

## Output

The first line gives the instance: its total weight and capacity. The next three give the run from
seed 1: the chosen items, numbered from 0, their total value and weight, and the evaluations, with
the optimum that dynamic programming finds. The last two lines give the runs from seeds 1 to 20:
how many reach the optimum, and after how many evaluations (the median); and how many reach the
optimum of the strongly correlated instance. Copies of a parent inherit its score, so a generation
of 200 needs fewer than 200 evaluations. The problem is evaluated in Rust in both languages, so both
print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/knapsack) plays this run back.

## Good results

The optimum is 20,849: 32 items that weigh 12,551. The run from seed 1 finds it after 6,864
evaluations, and so do all 20 runs from seeds 1 to 20, after a median of 6,936. Over seeds 1 to 200,
every run found it, within 16,536 evaluations; on the instances of seeds 1 to 5, 997 of 1,000 runs
(200 each) did.

The strongly correlated instance, optimum 16,020, is another matter: the same GA reaches it from 2
of the 20 seeds, and the other runs stall from 1 to 115 below it, within 0.7%. Pisinger's hard
classes are hard for genetic algorithms too.
