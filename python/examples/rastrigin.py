"""Rastrigin in 30 dimensions: CMA-ES with IPOP restarts, and L-SHADE, with a generation per call
of a vectorized numpy function."""

import numpy as np

import genoxide as gx


def rastrigin(x):
    """x has a genome per row."""
    return 10 * x.shape[1] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)


genome = gx.Real((-5.12, 5.12), length=30)
budget = 1_000_000
for algorithm in (
    gx.Cmaes(genome, restarts="ipop", objective="minimize", seed=1),
    gx.De(genome, l_shade=budget, objective="minimize", seed=1),
):
    result = algorithm.run(rastrigin, batch=True, target=1e-8, evaluations=budget)
    print(
        f"{type(algorithm).__name__}: {result.best_fitness:.3g} after {result.evaluations} "
        f"evaluations, {result.seconds:.2f} s"
    )
