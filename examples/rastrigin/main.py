"""Rastrigin: minimize a real-valued function with many local minima, in 30 dimensions.

Compares CMA-ES with IPOP restarts (a population that doubles at each restart) and L-SHADE
(differential evolution with a population that shrinks over the budget). The global minimum is 0,
at the origin. The function is genoxide's problems.Rastrigin, which run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/rastrigin/main.py
"""

import genoxide as gx

from trace import record_2d

DIMENSIONS = 30
BUDGET = 1_000_000

problem = gx.problems.Rastrigin(DIMENSIONS)
target = problem.optimum.value + 1e-8
for name, algorithm in (
    ("CMA-ES", gx.Cmaes(problem.genome, restarts="ipop", objective="minimize", seed=1)),
    ("L-SHADE", gx.De(problem.genome, l_shade=BUDGET, objective="minimize", seed=1)),
):
    result = algorithm.run(problem, target=target, evaluations=BUDGET)
    print(f"{name}: {result.best_fitness:.6f} after {result.evaluations} evaluations")

# with GENOXIDE_TRACE=<file>, a trace for the plot on the example's page, of a separate run in
# 2 dimensions: the plot is the function's contour
record_2d()
