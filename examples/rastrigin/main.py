"""Rastrigin: minimize a real-valued function with many local minima, in 30 dimensions.

Compares CMA-ES with IPOP restarts (a population that doubles at each restart) and L-SHADE
(differential evolution with a population that shrinks over the budget). The global minimum is 0,
at the origin. The function is genoxide's problems.Rastrigin, which run evaluates in Rust.

    python examples/rastrigin/main.py
"""

import genoxide as gx

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
