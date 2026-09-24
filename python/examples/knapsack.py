"""A 0/1 knapsack with a weight limit as a constraint: the fitness function returns the value and
how far over the limit the selection is, and feasible solutions rank above infeasible ones."""

import numpy as np

import genoxide as gx

rng = np.random.default_rng(0)
weights = rng.integers(1, 30, size=50)
values = rng.integers(1, 100, size=50)
capacity = weights.sum() // 3


def knapsack(bits):
    weight = weights[bits].sum()
    return float(values[bits].sum()), float(max(0, weight - capacity))


ga = gx.Ga(
    gx.Binary(50),
    population_size=100,
    select=gx.Tournament(2),
    crossover=gx.UniformCrossover(),
    mutation=gx.BitFlip(rate=1 / 50),
    seed=1,
)
result = ga.run(knapsack, generations=300)
chosen = result.best_genome
print(
    f"value {result.best_fitness:.0f}, weight {weights[chosen].sum()} of {capacity}, "
    f"{chosen.sum()} items"
)
