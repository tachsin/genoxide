"""Ackley: minimize a function with a deep central hole in a nearly flat, rippled plain, in 30
dimensions.

Compares CMA-ES with IPOP restarts and particle swarm optimization with two topologies: every
particle following the whole swarm's best, and every particle following the best of its two
neighbors on a ring. The global minimum is 0, at the origin. The function is genoxide's
problems.Ackley, which run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/ackley/main.py
"""

import genoxide as gx

from trace import record_2d

DIMENSIONS = 30
BUDGET = 10_000 * DIMENSIONS

problem = gx.problems.Ackley(DIMENSIONS)
target = problem.optimum.value + 1e-8
for name, algorithm in (
    ("CMA-ES", gx.Cmaes(problem.genome, restarts="ipop", objective="minimize", seed=1)),
    ("PSO, global", gx.Pso(problem.genome, population_size=40, objective="minimize", seed=1)),
    (
        "PSO, ring",
        gx.Pso(problem.genome, population_size=40, ring=1, objective="minimize", seed=1),
    ),
):
    result = algorithm.run(problem, target=target, evaluations=BUDGET)
    print(f"{name}: {result.best_fitness:.6f} after {result.evaluations} evaluations")

# with GENOXIDE_TRACE=<file>, a trace for the plot on the example's page, of a separate run in
# 2 dimensions: the plot is the function's contour
record_2d()
