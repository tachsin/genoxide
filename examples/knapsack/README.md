---
title: 0/1 knapsack
category: constrained
summary: Choose the items with the highest total value whose total weight fits a capacity.
reference: "Martello, S. and Toth, P. (1990). Knapsack Problems: Algorithms and Computer Implementations. Wiley."
reference_url: null
optimum: "OPTIMUM_PLACEHOLDER (total value)"
languages: [rust, python]
order: 20
---

# 0/1 knapsack

The 0/1 knapsack problem chooses items, each with a weight and a value, to maximize the total
value without exceeding a capacity: here 20 items and a capacity of 400. A bit per item selects
it, and the fitness function returns the value and how far the weight exceeds the capacity, so
Deb's feasibility rules rank feasible selections first and infeasible ones by their excess. A
genetic algorithm with tournament selection, two-point crossover and bit-flip mutation runs until
200 generations pass without improvement. The example prints the items it chose, their value and
weight, and the optimum found by dynamic programming.
