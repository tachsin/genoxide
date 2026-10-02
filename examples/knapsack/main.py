"""0/1 knapsack: choose items with the highest total value that fit in the knapsack, on an
instance of 50 items drawn from Pisinger's uncorrelated class.

Shows a constraint with Deb's feasibility rules: the fitness is the value and how far the weight
exceeds the capacity, so overweight selections still guide the search towards the feasible ones.
The instance is genoxide's problems.binary.Knapsack, generated from seed 1, which run evaluates in
Rust, and whose optimum dynamic programming finds. A run from seed 1 stops at it; then runs from
seeds 1 to 20 count how often the GA reaches it, and, as a contrast, how often it reaches the
optimum of an instance of Pisinger's strongly correlated class, which is harder.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/knapsack/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

ITEMS = 50
SEEDS = 20


def ga(knapsack, seed):
    """The genetic algorithm from ``seed``."""
    return gx.Ga(
        knapsack.genome,
        population_size=200,
        select=gx.Tournament(3),
        crossover=gx.UniformCrossover(),
        mutation=gx.BitFlip(rate=1 / ITEMS),
        seed=seed,
    )


def run(knapsack, seed, optimum, on_generation=None):
    """A run from ``seed``: it stops at the optimum that dynamic programming finds, or once the
    search stalls."""
    return ga(knapsack, seed).run(
        knapsack,
        target=optimum,
        stagnation=200,
        generations=2_000,
        on_generation=on_generation,
    )


def seeds(knapsack, optimum):
    """The runs from seeds 1 to 20 that reach the optimum, and their median evaluations."""
    evaluations = sorted(
        result.evaluations
        for result in (run(knapsack, seed, optimum) for seed in range(1, SEEDS + 1))
        if result.stop_reason == "target"
    )
    middle = len(evaluations) // 2
    if not evaluations:
        return 0, float("nan")
    if len(evaluations) % 2:
        return len(evaluations), float(evaluations[middle])
    return len(evaluations), (evaluations[middle - 1] + evaluations[middle]) / 2


knapsack = gx.problems.binary.Knapsack(ITEMS, "uncorrelated", seed=1)
optimum = knapsack.optimum.value
weights, values = knapsack.weights, knapsack.profits
print(
    f"{ITEMS} uncorrelated items (R = 1000, seed 1): total weight {weights.sum()}, "
    f"capacity {knapsack.capacity}"
)

# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(knapsack, optimum)
result = run(knapsack, 1, optimum, trace.on_generation)
best = result.best_genome
print(f"items {np.flatnonzero(best).tolist()}")
print(f"value {values[best].sum()}, weight {weights[best].sum()} of {knapsack.capacity}")
print(
    f"after {result.evaluations} evaluations; the optimum, by dynamic programming, is "
    f"{optimum:.0f}"
)

reached, median = seeds(knapsack, optimum)
print(
    f"\nseeds 1 to {SEEDS}: {reached} reach {optimum:.0f}, after a median of {median:.1f} "
    "evaluations"
)
strong = gx.problems.binary.Knapsack(ITEMS, "strongly_correlated", seed=1)
strong_optimum = strong.optimum.value
reached, _ = seeds(strong, strong_optimum)
print(
    f"contrast, {ITEMS} strongly correlated items (seed 1): {reached} of {SEEDS} reach its "
    f"optimum, {strong_optimum:.0f}"
)
trace.write()
