"""Rastrigin: minimize a real-valued function with many local minima, in 30 dimensions.

Compares CMA-ES with IPOP restarts (a population that doubles at each restart) and L-SHADE
(differential evolution with a population that shrinks over the budget). The global minimum is 0,
at the origin. The fitness function takes a generation at a time.

    python examples/rastrigin/main.py
"""

import numpy as np

import genoxide as gx

DIMENSIONS = 30
BUDGET = 1_000_000


def rastrigin(x):
    """x has a genome per row."""
    return 10 * x.shape[1] + np.sum(x * x - 10 * np.cos(2 * np.pi * x), axis=1)


real = gx.Real((-5.12, 5.12), length=DIMENSIONS)
for name, algorithm in (
    ("CMA-ES", gx.Cmaes(real, restarts="ipop", objective="minimize", seed=1)),
    ("L-SHADE", gx.De(real, l_shade=BUDGET, objective="minimize", seed=1)),
):
    result = algorithm.run(rastrigin, batch=True, target=1e-8, evaluations=BUDGET)
    print(f"{name}: {result.best_fitness:.6f} after {result.evaluations} evaluations")
