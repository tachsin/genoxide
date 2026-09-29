---
title: Koza's 11-multiplexer
category: genetic programming
summary: Find the Boolean function that uses 3 address bits to select one of 8 data bits, right in all 2048 cases, by genetic programming.
reference: "Koza, J. R. (1992). Genetic Programming: On the Programming of Computers by Means of Natural Selection. MIT Press."
reference_url: https://www.genetic-programming.com/gpbook1toc.html
optimum: "All 2048 cases right"
languages: [rust]
order: 256
---

# Koza's 11-multiplexer

## The problem

A multiplexer has address bits and data bits, and outputs the data bit that the address selects.
The 11-multiplexer has 3 address bits, a0 to a2, and 8 data bits, d0 to d7: with a2 a1 a0 = 110,
the output is d6. Given only its truth table, the 2048 combinations of the 11 inputs with the right
output for each, the task is to find a Boolean function that computes it.

Koza (1992) used it to show that genetic programming finds programs for problems whose solution has
a hierarchical structure: the 11-multiplexer is 7 conditionals, a decision on a2 over decisions on
a1 over decisions on a0. With 4000 programs and at most 51 generations, his run found a correct one
in generation 9. The critique of GP benchmarks (McDermott et al. 2012, Genetic programming needs
better benchmarks, GECCO 2012: 791-798) lists it among the overused problems; it stays GP's classic
Boolean test, with an exact optimum.

There's no Python version: the Python package has no genetic programming yet.

## What makes it hard

The search sees only how many of the 2048 cases a program gets right. A program that returns one
data bit is right in 1152 cases: the 256 where that bit is selected, and half the others. Building
on that needs the address bits combined in the right way, and a program that gets 1536 cases right
can be far from one that gets them all. Programs also grow: the parts that don't change the output
pile up (bloat), and the search slows down as they do.

## Representation

A tree of a genetic program (`gp::Tree`) of Koza's primitives, from `gp::boolean::Multiplexer`:
the functions and, or, not and if (if(a, b, c) is b when a is true, else c), and the 11 inputs as
terminals. One type, Boolean. `gp::Gp` sets Koza's limits and initialization: depth at most 17 and
at most 1024 nodes, and ramped half-and-half with depths 2 to 6, the initial population divided
evenly among the depths and the two methods without duplicates (`Gp::ramped_half_and_half`).

The fitness is the number of cases a tree gets wrong, Koza's standardized fitness, minimized: 0 is
the optimum. Trees are evaluated on 64 cases at once, a bit per case in a 64-bit word, so all 2048
cases take 32 passes over the tree.

## Algorithm

A genetic algorithm of 4000 trees, Koza's population size. Parents are chosen by double tournament
(Luke and Panait 2002): the winners of two tournaments of 7 on fitness meet in a tournament of
size, where the smaller wins with probability 0.7 (D = 1.4, the setting Luke and Panait (2006)
found best), which holds bloat back. Subtree crossover at a rate of 0.9 (Koza's, points at
function nodes 90% of the time, within the limits) and, at a rate of 0.1, one of four mutations
(`gp::Mutations`): subtree (weight 0.5), point (0.3), hoist (0.1) and shrink (0.1). The run stops
when every case is right, or after 50 generations, the 51 of Koza's runs with the initial one.

## Output

The first two lines give the problem and the setting. Then come how the run stopped, after how many
generations and evaluations, how many cases the best tree gets right, and the tree itself, with its
size and depth, written as genoxide writes trees (`Tree::display`).

[The project page](https://tachsin.gr/projects/genoxide/examples/multiplexer-11) plays this run
back.

## Good results

The optimum is a function right in all 2048 cases. The run of `output.txt` found one after 15
generations and 51,907 evaluations, in 40 nodes. Read from the root, it tests a0, then a1 and a2,
and returns the selected data bit; some of its tests repeat one made above them, which is how
evolved programs look before they're simplified.

Over seeds 1 to 100 (the seed of the algorithm and of the initial population), all 100 runs found a
correct function within 50 generations: after 14 generations and 49,675 evaluations in the median,
and 21 generations at most. The functions found had 57 nodes in the median and 152 at most.

Without the pressure on size (D = 1, so the size tournament is a coin toss and the selection a
tournament of 7 on fitness), 91 of the 100 runs found one within 50 generations, after 19
generations in the median, and the functions found had 148 nodes in the median and 782 at most.
