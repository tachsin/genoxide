---
title: Royal road R1
category: binary
summary: Maximize 8 blocks of 8 ones that score only when complete, with random-mutation hill climbing, which Mitchell, Holland and Forrest found nearly ten times faster on it than their genetic algorithm.
reference: "Mitchell, M., Holland, J. H. and Forrest, S. (1994). When will a genetic algorithm outperform hill climbing? Advances in Neural Information Processing Systems 6: 51-58."
reference_url: https://proceedings.neurips.cc/paper_files/paper/1993/hash/ab88b15733f543179858600245108dd8-Abstract.html
optimum: "64 (all ones)"
languages: [rust, python]
order: 13
family: Royal road
tab: R1
---

# Royal road R1

## The problem

R1 is a sum over 8 schemas, each a block of 8 consecutive bits that must all be ones, s₁ =
11111111∗∗…∗ to s₈ = ∗∗…∗11111111, each with the coefficient cᵢ = 8, its order (Mitchell, Holland
and Forrest, 1994, Figure 1):

```text
R1(x) = Σᵢ cᵢ δᵢ(x),   δᵢ(x) = 1 if x is an instance of sᵢ, 0 otherwise
```

A string scores 8 for each complete block: a string with two complete blocks scores 16, and the
string of 64 ones, the optimum, 64. The function is genoxide's `problems::binary::RoyalRoad::r1()`.
Its hierarchical sibling, [R2](../royal_road_r2/), scores the blocks' combinations too.

## What makes it hard

The royal road functions were designed to be easy for a genetic algorithm: crossover would combine
complete blocks into strings with more of them, a "royal road" to the optimum, while a hill climber
would have to set 8 bits at once to complete a block. Both expectations failed (Forrest and
Mitchell, 1993, as summarized by Mitchell et al.). A block short of complete scores nothing, so a string's
fitness says nothing about its 7 of 8 ones: the search drifts on plateaus. In a genetic algorithm,
a string with a new block takes over the population, and the zeros next to the block hitchhike with
it, slowing the discovery of the blocks beside it.

A hill climber that flips one bit at a time and keeps any change that's no worse drifts freely
inside the incomplete blocks without losing the complete ones. Mitchell et al. analyze it: the
expected time to find one block of K ones is slightly above 2ᴷ, 301.2 evaluations for K = 8, and to
find N blocks about E(K, 1) N (1 + 1/2 + … + 1/N), 6,549 evaluations for R1 (their equation 1).

## Representation

A `Binary` genome of 64 bits is the string itself. The fitness is R1, to maximize.

## Algorithm

Random-mutation hill climbing (RMHC), Mitchell et al.'s: start from a random string, flip one bit
chosen at random, and keep the change if the string is no worse. In genoxide, that's `LocalSearch`
with `BitFlip::count(1)` as its neighbor, one neighbor per step, and the default acceptance of
neighbors that are no worse. A run from seed 1 stops at 64, or after 256,000 evaluations, the
paper's limit; then 200 runs, seeds 1 to 200, as in the paper's Table 1.

For comparison, a genetic algorithm with the paper's population of 128, single-point crossover at a
rate of 0.7 and mutation of 0.005 per bit (Mitchell, Forrest and Holland, 1992), with tournament
selection of size 2 in place of their fitness-proportionate selection with sigma scaling, from seeds
1 to 50 with the same limit.

## Output

The first lines follow the run from seed 1: the evaluations at which it completes a block, its
score rising by 8, and the evaluations to 64. Then two lines for the 200 runs of RMHC, how many
reach 64 and the mean and median of their evaluations, with the paper's for comparison, and two for
the 50 runs of the genetic algorithm. The function is evaluated in Rust in both languages, so both
print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/royal-road-r1) plays back the run
from seed 1: the string, its blocks filling with ones one after another.

## Good results

The optimum is 64. The run from seed 1 reaches it after 10,101 evaluations, and every one of the 200
runs does, after 6,793 evaluations on average, with a median of 6,144. Mitchell et al.'s 200 runs
took 6,179 on average (with a standard error of 186) and 5,775 at the median, against their expected
6,549. The genetic algorithm reaches 64 from all 50 seeds too, after 34,303 evaluations on average,
five times as many as hill climbing; the paper's GA took 61,334 (and genoxide doesn't evaluate a
child that is a copy of its parent again).
