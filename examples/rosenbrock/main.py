"""Rosenbrock: minimize a function whose minimum lies at the end of a narrow curved valley, in 30
dimensions.

Compares CMA-ES (which learns the valley's direction), L-SHADE (differential evolution with a
population that shrinks over the budget) and particle swarm optimization. The global minimum is
0, at (1, …, 1). The function is genoxide's problems.Rosenbrock, which run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/rosenbrock/main.py
"""

import genoxide as gx
import numpy as np

from trace import record_2d

DIMENSIONS = 30
BUDGET = 10_000 * DIMENSIONS

problem = gx.problems.Rosenbrock(DIMENSIONS)
optimum = problem.optimum
target = optimum.value + 1e-8
print(
    f"Rosenbrock in {DIMENSIONS} dimensions: minimum {optimum.value:.4f}, "
    f"{BUDGET} evaluations at most"
)
for name, algorithm in (
    ("CMA-ES", gx.Cmaes(problem.genome, objective="minimize", seed=1)),
    ("L-SHADE", gx.De(problem.genome, l_shade=BUDGET, objective="minimize", seed=1)),
    ("PSO", gx.Pso(problem.genome, population_size=40, objective="minimize", seed=1)),
):
    result = algorithm.run(problem, target=target, evaluations=BUDGET)
    # the error to the minimum (rounding can put a solution a few ulps below it), and how many
    # genes are more than 0.01 from the minimum's
    error = max(result.best_fitness - optimum.value, 0.0)
    off = int(np.sum(np.abs(result.best_genome - optimum.solutions[0]) > 0.01))
    print(
        f"{name}: error {error:.4f} after {result.evaluations} evaluations, "
        f"{off} of {DIMENSIONS} genes off by more than 0.01"
    )

# with GENOXIDE_TRACE=<file>, a trace for the plot on the example's page, of a separate run in
# 2 dimensions: the plot is the function's contour
record_2d()
