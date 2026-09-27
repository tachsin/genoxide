"""Schwefel 2.26: minimize a deceptive function whose best local minima are far apart, in 30
dimensions.

Compares CMA-ES with IPOP restarts, particle swarm optimization with a ring topology and L-SHADE
(differential evolution with a population that shrinks over the budget). The global minimum is
−418.98 per gene, where every gene is 420.97, near the upper bound. The function is genoxide's
problems.Schwefel2_26, which run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/schwefel_2_26/main.py
"""

import genoxide as gx

from trace import record_2d

DIMENSIONS = 30
BUDGET = 10_000 * DIMENSIONS

problem = gx.problems.Schwefel2_26(DIMENSIONS)
print(f"minimum: {problem.optimum.value:.2f}")
target = problem.optimum.value + 1e-8
# the global minimizer of each gene, 420.97
best_gene = problem.optimum.solutions[0][0]
for name, algorithm in (
    ("CMA-ES", gx.Cmaes(problem.genome, restarts="ipop", objective="minimize", seed=1)),
    ("PSO", gx.Pso(problem.genome, population_size=40, ring=1, objective="minimize", seed=1)),
    ("L-SHADE", gx.De(problem.genome, l_shade=BUDGET, objective="minimize", seed=1)),
):
    result = algorithm.run(problem, target=target, evaluations=BUDGET)
    near = sum(abs(x - best_gene) < 1.0 for x in result.best_genome)
    print(
        f"{name}: {result.best_fitness:.2f}, with {near} of {DIMENSIONS} genes within 1 of "
        f"{best_gene:.2f}"
    )

# with GENOXIDE_TRACE=<file>, a trace for the plot on the example's page, of a separate run in
# 2 dimensions: the plot is the function's contour
record_2d()
