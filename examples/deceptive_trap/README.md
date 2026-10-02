---
title: Deceptive trap
category: binary
summary: Maximize 10 blocks of 4 bits, each a trap function that leads away from its optimum, with a genetic algorithm whose two-point crossover keeps the blocks together, against uniform crossover and hill climbing.
reference: "Deb, K. and Goldberg, D. E. (1993). Analyzing deception in trap functions. Foundations of Genetic Algorithms 2: 93-108."
reference_url: https://doi.org/10.1016/B978-0-08-094832-4.50012-X
optimum: "40 (all ones)"
languages: [rust, python]
order: 12
---

# Deceptive trap

## The problem

A trap function scores a block of k bits by its number of ones u (Deb and Goldberg, 1993, equation
1):

```text
f(u) = a (z − u) / z          for u ≤ z
       b (u − z) / (k − z)    otherwise
```

It falls from a at u = 0 to 0 at the slope change z, and rises to b > a at u = k. Ackley (1987, *A
Connectionist Machine for Genetic Hillclimbing*, section 3.3.3) introduced it with a = 8k, b = 10k
and z = ⌊3k/4⌋; Deb and Goldberg analyze any a, b and z. Here a = k − 1, b = k and z = k − 1, so a
block of k = 4 bits scores 3, 2, 1 and 0 for 0 to 3 ones, and 4 for all four. The string has 10
consecutive blocks, bits 0 to 3, 4 to 7 and so on, 40 bits, and scores the sum of its blocks: 40
at most, for the string of all ones, and 30 for the string of all zeros. The function is genoxide's
`problems::binary::Trap::new(10, 4)`.

## What makes it hard

Every step up within a block leads to all zeros, except the last one: from 3 ones, the fourth gives
4, and anything else less. A block with u ones below 4 gains by losing a one. Deb and Goldberg prove
that a trap is *fully deceptive* (every schema of order below k, averaged over the strings it
contains, favors the all-zeros block) if and only if all schemata of order k − 1 are (Theorem 1),
and give the condition on r = a/b (inequality 16). With a = k − 1, b = k and z = k − 1, r = 3/4 is
above their limit (k − 1)/(2k − 3) = 3/5 for k = 4 (eq. 20): the blocks are fully deceptive.
genoxide's tests check this by averaging over every schema. A hill climber takes each block to its
attractor, all zeros, unless the block starts with all ones, or with three and the missing bit is
the first to flip.

The 10 blocks are independent, so a search that finds the optimum of each block, and keeps it,
solves the whole. A genetic algorithm can, if crossover passes whole blocks from parent to child
and selection keeps the complete ones; the blocks are tight, 4 adjacent bits, so a point crossover
rarely cuts one.

## Representation

A `Binary` genome of 40 bits is the string itself. The fitness is the sum of the blocks' values, to
maximize.

## Algorithm

A genetic algorithm:

- a population of 1000, enough to hold several copies of each block's optimum from the start;
- tournament selection of size 4;
- two-point crossover, which cuts the string at two points: of the 39 places it can cut, 9 are
  between blocks, and a cut elsewhere breaks one block;
- bit-flip mutation at a rate of 1/40 per bit;
- the default generational scheme, which keeps the best individual.

A run from seed 1 stops at 40, or after 300 generations; then runs from seeds 1 to 20. Two
contrasts run from the same seeds: the same GA with uniform crossover, which takes each bit from
either parent and so breaks the blocks apart, and hill climbing, `LocalSearch` flipping one bit at a
time and keeping the change if it's no worse, for 10,000 evaluations.

## Output

The table follows the run from seed 1: each generation's best score, its population's median score
and the blocks of the best string that are all ones. Then the generation and evaluations at which
it reaches 40. The last three lines give the runs from seeds 1 to 20: how many reach 40 with
two-point crossover and after how many generations (the median), how many with uniform crossover
and the median number of blocks of ones they end with, and hill climbing's median best and blocks
of ones. The function is evaluated in Rust in both languages, so both print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/deceptive-trap) plays back the run
from seed 1: the first 16 strings of the population, blocks of ones spreading through them.

## Good results

The optimum is 40. The run from seed 1 reaches it at generation 15, after 14,687 evaluations, and so
do all 20 runs from seeds 1 to 20, after a median of 15 generations. Over seeds 1 to 300, every run
reached it, within 95 generations.

The contrasts don't. Uniform crossover reaches it from none of the 20 seeds in 300 generations: its
runs end with a median of 3 blocks of ones of the 10. Hill climbing
ends at a median of 31: one block of ones and nine at the deceptive attractor, 3 each.
