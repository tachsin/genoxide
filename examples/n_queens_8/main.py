"""N-Queens: place N queens on an N×N board so that no two attack each other.

A permutation genome puts one queen in each row and each column (queen ``row`` is in column
``order[row]``), so only the diagonals can conflict. Permutations have no position-wise crossover,
so this uses (μ+λ) with swap mutation only. The fitness function takes a generation at a time.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/n_queens_8/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

N = 8
ROWS = np.arange(N)


def conflicts(orders):
    """The number of pairs of queens on the same diagonal, for a genome per row."""
    # a separate range of 2N counters for each genome
    offsets = 2 * N * np.arange(len(orders))[:, None]
    pairs = 0
    for diagonal in (ROWS + N - orders, ROWS + orders):
        counts = np.bincount((diagonal + offsets).ravel(), minlength=2 * N * len(orders))
        pairs = pairs + (counts * (counts - 1) // 2).reshape(len(orders), 2 * N).sum(axis=1)
    return pairs


ga = gx.Ga(
    gx.Permutation(N),
    population_size=20,
    select=gx.Tournament(2),
    crossover=gx.NoCrossover(),
    mutation=gx.SwapMutation(),
    scheme=gx.MuPlusLambda(20),
    objective="minimize",
    seed=1,
)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(N)
result = ga.run(
    conflicts, batch=True, target=0, generations=50_000, on_generation=trace.on_generation
)

print(
    f"{result.best_fitness:.0f} conflicts after {result.generations} generations and "
    f"{result.evaluations} evaluations"
)
print(f"columns {result.best_genome.tolist()}")
trace.write()
