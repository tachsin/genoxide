---
title: Royal road R2
category: binary
summary: Maximize 8 blocks of 8 ones and their pairs, quadruples and whole, each scoring when complete, with the genetic algorithm of Mitchell, Forrest and Holland's settings.
reference: "Mitchell, M., Forrest, S. and Holland, J. H. (1992). The royal road for genetic algorithms: fitness landscapes and GA performance. Proceedings of the First European Conference on Artificial Life: 245-254."
optimum: "256 (all ones)"
languages: [rust, python]
order: 14
family: Royal road
tab: R2
---

# Royal road R2

## The problem

R2 is the royal road function of Mitchell, Forrest and Holland's (1992) Figure 1, which Forrest and
Mitchell later called R2. It's a sum over 15 schemas s, each scoring its order, its number of
defined bits, c_s = order(s), when a string is an instance of it:

```text
F(x) = Σₛ c_s σ_s(x),   σ_s(x) = 1 if x is an instance of s, 0 otherwise
```

The schemas form a tree over 64 bits: 8 blocks of 8 consecutive ones (c = 8), the 4 pairs of
adjacent blocks, 16 ones each (c = 16), the 2 halves of 32 ones (c = 32) and the whole string
(c = 64). The string of all ones is an instance of all 15 and scores 8 · 8 + 4 · 16 + 2 · 32 + 64
= 256. A string whose first 16 bits are ones scores 8 + 8 + 16 = 32. The function is genoxide's
`problems::binary::RoyalRoad::r2()`; [R1](../royal_road_r1/) is its lowest level alone.

## What makes it hard

The intermediate schemas were meant as stepping stones: a genetic algorithm would combine two
complete blocks into a pair, two pairs into a half, and climb the tree by crossover. A hill climber
gains nothing from them: a step that doesn't complete or break a block changes neither R1 nor R2,
so a climber that keeps changes that are no worse takes the same steps on both. The difficulty is
R1's: a block short of complete scores nothing, and a string that completes one spreads through the
population, its zeros elsewhere hitchhiking with it.

Mitchell et al. ran their genetic algorithm 50 times: it found the optimum after 590 generations on
average (with a standard error of 50), 542 at the median, and 1,022 without crossover (their Table
1). Their iterated hill climber never found it in 256,000 evaluations; later, Mitchell, Holland and
Forrest (1994) found random-mutation hill climbing faster than the GA on R1.

## Representation

A `Binary` genome of 64 bits is the string itself. The fitness is R2, to maximize.

## Algorithm

A genetic algorithm with Mitchell et al.'s settings:

- a population of 128;
- tournament selection of size 2, in place of their fitness-proportionate selection with sigma
  scaling (at most 1.5 expected offspring), which genoxide doesn't have;
- single-point crossover at a rate of 0.7 per pair of parents;
- bit-flip mutation with probability 0.005 per bit;
- the default generational scheme, which keeps the best individual.

A run from seed 1 stops at 256, or after 256,000 evaluations, the paper's 2,000 generations of 128;
then runs from seeds 1 to 50, as in the paper. For comparison, random-mutation hill climbing
(`LocalSearch` flipping one bit at a time and keeping the change if it's no worse) from seeds 1 to
200, as on [R1](../royal_road_r1/).

## Output

The table follows the run from seed 1: the generations at which its best score improves. Then the
generations and evaluations at which it reaches 256. The last three lines give the 50 runs of the
GA, how many reach 256 and the mean and median of their generations, with the paper's for
comparison, and the 200 runs of hill climbing, with their evaluations. The function is evaluated in
Rust in both languages, so both print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/royal-road-r2) plays back the run
from seed 1: the first 16 strings of the population, as blocks of ones complete and spread.

## Good results

The optimum is 256. The run from seed 1 reaches it at generation 69, and all 50 runs from seeds 1 to
50 do, after 661 generations on average and 478 at the median, close to the paper's 590 and 542.
Over seeds 1 to 300, every run reached it within 256,000 evaluations, after 27,296 at the median
(genoxide doesn't evaluate a child that is a copy of its parent again). Random-mutation hill
climbing reaches it in all 200 runs after 6,793 evaluations on average and 6,144 at the median, the
same as on R1, step for step: on R2 too, it needs about a quarter of the GA's evaluations.
