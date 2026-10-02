---
title: NK landscape
category: binary
summary: Maximize a rugged NK landscape of 20 bits, each interacting with 4 others, with iterated local search, checked against the optimum found by evaluating every string.
reference: "Kauffman, S. A. and Weinberger, E. D. (1989). The NK model of rugged fitness landscapes and its application to maturation of the immune response. Journal of Theoretical Biology 141(2): 211-245."
reference_url: https://doi.org/10.1016/S0022-5193(89)80019-0
optimum: "0.742541 (landscape seed 1, by exhaustive search)"
languages: [rust, python]
order: 15
---

# NK landscape

## The problem

Kauffman and Weinberger's NK model scores a string of N bits as the mean of N contributions. The
contribution wᵢ of bit i depends on its own value and on those of K other bits, its neighbors:
each of the 2^(K+1) combinations gets a value drawn independently and uniformly from (0, 1), and

```text
W(x) = (1/N) Σᵢ wᵢ(xᵢ, x_{i₁}, …, x_{i_K})
```

(their equation 1). K tunes the landscape: at K = 0 the bits are independent and there's a single
optimum; as K grows towards N − 1, the landscape gets more rugged, with more local optima and lower
ones, until at K = N − 1 every one-bit change gives a new random fitness.

Here N = 20 and K = 4, and each bit's 4 neighbors are drawn at random from the other 19 (their
Table 2; "adjacent" neighbors, a bit's flanking ones on a circle, are their Table 1). The landscape
is genoxide's `problems::binary::NkLandscape::new(20, 4, Neighborhood::Random, 1)`: its neighbors
and its 20 tables of 32 values are drawn from seed 1 with genoxide's portable random numbers, so
it's the same landscape on every platform and in Python. Its contributions are odd multiples of 2⁻⁵³,
added up exactly, so the fitness is the same to the bit everywhere.

## What makes it hard

The optimum isn't known in closed form: each landscape has its own. genoxide's `optimum` computes it
exactly, here by evaluating all 2²⁰ = 1,048,576 strings, changing one bit at a time in Gray code
order so that only the contributions that depend on it change (a hundredth of a second). For
adjacent neighbors it uses dynamic programming instead, which solves landscapes of hundreds of bits.

With K = 4, a bit's best value depends on its neighbors' values, which depend on theirs: changing
one bit changes its own contribution and those of the bits it bears on, five on average. The
landscape has many local optima, strings that no single flip improves. A search that only climbs
stops at one of them.

## Representation

A `Binary` genome of 20 bits is the string itself. The fitness is W, to maximize.

## Algorithm

Iterated local search: `LocalSearch` flips one bit at a time and keeps the change if the string is
no worse, and after 100 steps without a better best, restarts from the best string, changed by 5
random flips, enough to leave its basin and few enough to keep most of it. A run from seed 1 stops
at the optimum, or after 500,000 evaluations.

Then, on the landscapes of seeds 1 to 5, iterated local search and, as a contrast, a genetic
algorithm run from seeds 1 to 20: a population of 500, tournaments of 2, two-point crossover and
bit-flip mutation at 1/20 per bit, for 200 generations.

## Output

The first lines give the landscape and its optimum, with the string that reaches it, bit 0 first.
The table follows the run from seed 1: the evaluations at which its best improves, to 6 decimals,
and then the string it ends with. The last table gives, for each landscape, its optimum, how many
runs of iterated local search from seeds 1 to 20 reach it and their median evaluations, and how many
runs of the GA do. The landscapes are evaluated in Rust in both languages, so both print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/nk-landscape) plays back the run
from seed 1: the string, as single flips climb and restarts kick it.

## Good results

The optimum of the landscape of seed 1 is 0.742541. The run from seed 1 reaches it after 927
evaluations, at the same string as exhaustive search. On the 5 landscapes, iterated local search
reaches the optimum from all 20 seeds, after a median of 1,109 to 4,910 evaluations. Over landscapes
1 to 10 and seeds 1 to 100 each, it reached the optimum in 997 of 1,000 runs within 200,000
evaluations.

The genetic algorithm reaches it on the easier landscapes but not the others: from 5 of 20 seeds on
landscape 2 and 6 of 20 on landscape 5, where its other runs end below the optimum within 200
generations. Each landscape is a new instance, so a method has to be judged over several.
