"""N-Queens with 64 queens: a permutation puts one queen per row and per column, and tabu search
removes the diagonal conflicts."""

import numpy as np

import genoxide as gx

n = 64
rows = np.arange(n)


def conflicts(order):
    """Queens that share a diagonal with another queen."""
    return float(2 * n - len(np.unique(order - rows)) - len(np.unique(order + rows)))


search = gx.LocalSearch(
    gx.Permutation(n),
    neighbor=gx.SwapMutation(),
    neighbors=32,
    acceptance=gx.Tabu(20),
    objective="minimize",
    seed=1,
)
result = search.run(conflicts, target=0, evaluations=1_000_000)
print(f"{result.best_fitness:.0f} conflicts after {result.evaluations} evaluations")
