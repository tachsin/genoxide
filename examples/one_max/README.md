---
title: OneMax
category: binary
summary: Find the bit string with the most ones, the "hello world" of genetic algorithms.
reference: "Ackley, D. H. (1987). A Connectionist Machine for Genetic Hillclimbing. Kluwer Academic Publishers."
reference_url: https://doi.org/10.1007/978-1-4613-1997-9
optimum: "500 (all ones)"
languages: [rust, python]
order: 10
---

# OneMax

## The problem

OneMax scores a bit string by the number of its ones. Ackley (1987) used it as one of the test
functions of his genetic hill climber. The string 10110010 scores 4. Here the strings have 500 bits,
so the best score is 500, for the string of all ones.

## What makes it hard

Little, by design: it's the baseline that other problems are compared with. Each bit counts on its
own, and the bits don't interact. Every string other than the optimum gains a one when a single zero
flips, so there are no local optima. What the problem measures is how fast an algorithm climbs.

The slow part is the end. At a flip rate of 1/500 per bit, a mutation flips a given zero with
probability 0.2%, and it can flip a one at the same time. For the (1+1) evolutionary algorithm,
which keeps one string and flips each bit with probability 1/n, the expected number of evaluations
is of the order of n log n (Droste, Jansen and Wegener, 2002, Theoretical Computer Science 276(1-2):
51-81).

## Representation

A `Binary` genome of 500 bits is the solution itself: there is nothing to decode. The fitness is the
genome's number of ones, to maximize.

## Algorithm

A genetic algorithm with the pieces that genoxide's guide (`AGENTS.md`) lists for yes/no choices:

- a population of 100;
- tournament selection of size 3, within the guide's usual 2 to 5;
- uniform crossover, which takes each bit from either parent with probability 0.5, at genoxide's
  default crossover rate of 0.9;
- bit-flip mutation at a rate of 1/500 per bit, one flip per child on average;
- the default generational scheme, which keeps the best individual (elitism 1).

The bits don't interact, so there are no blocks of neighboring bits for a point crossover to keep
together; uniform crossover mixes the parents' ones freely. The run stops at 500 ones, or after
10,000 generations.

## Output

The first line is a table header. Each row gives a generation, every 50th, and the best count in the
population. Generation 0 is the random population. A random string has 250 ones on average, give or
take about 11, so the best of 100 has a little under 280. The count then climbs fast, and slows down
as the last zeros get rare.

The last line gives the final count, the generations and the evaluations. The evaluations are fewer
than 100 per generation: a child equal to its parent inherits its fitness and isn't evaluated again.

[The project page](https://tachsin.gr/projects/genoxide/examples/one-max) plays this run back.

## Good results

The optimum is 500 ones. The run reaches it after about 120 generations and 10,000 evaluations.
