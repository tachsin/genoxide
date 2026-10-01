"""0/1 knapsack: choose items with the highest total value that fit in the knapsack.

Shows a constraint with Deb's feasibility rules: the fitness function returns the value and how
far the weight exceeds the capacity, so overweight selections still guide the search towards the
feasible ones. The result is checked against the optimum found by dynamic programming.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/knapsack/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

# (weight, value): the first ten are the test data of Martello and Toth's Algorithm 632 (ACM TOMS
# 11(2), 1985), the other ten this example's own
ITEMS = [
    (23, 92),
    (31, 57),
    (29, 49),
    (44, 68),
    (53, 60),
    (38, 43),
    (63, 67),
    (85, 84),
    (89, 87),
    (82, 72),
    (12, 31),
    (17, 29),
    (41, 52),
    (35, 38),
    (27, 44),
    (58, 61),
    (19, 26),
    (46, 55),
    (71, 70),
    (33, 41),
]
CAPACITY = 400
WEIGHTS = np.array([weight for weight, _ in ITEMS])
VALUES = np.array([value for _, value in ITEMS])


def value(selection):
    """The total value, and how much the weight exceeds the capacity (0 if the items fit)."""
    weight = WEIGHTS[selection].sum()
    return float(VALUES[selection].sum()), float(max(0, weight - CAPACITY))


def optimum():
    """The best value that fits, by dynamic programming over the capacities."""
    best = [0] * (CAPACITY + 1)
    for weight, value in ITEMS:
        for capacity in range(CAPACITY, weight - 1, -1):
            best[capacity] = max(best[capacity], best[capacity - weight] + value)
    return best[CAPACITY]


ga = gx.Ga(
    gx.Binary(len(ITEMS)),
    population_size=200,
    select=gx.Tournament(3),
    crossover=gx.PointCrossover(2),
    mutation=gx.BitFlip(rate=1 / len(ITEMS)),
    seed=7,
)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(ITEMS, CAPACITY, optimum())
# stops at the optimum that dynamic programming finds, or once the search stalls
result = ga.run(
    value,
    target=optimum(),
    stagnation=200,
    generations=2_000,
    on_generation=trace.on_generation,
)

best = result.best_genome
print(f"items {np.flatnonzero(best).tolist()}")
print(f"value {VALUES[best].sum()}, weight {WEIGHTS[best].sum()} of {CAPACITY}")
print(f"after {result.evaluations} evaluations; the optimum is {optimum()}")
trace.write()
