"""Griewank: minimize a wide bowl with ripples that couple the genes, in 2 to 50 dimensions.

Runs CMA-ES with 10 seeds in each dimension, without restarts and with IPOP restarts (a population
that doubles at each restart), and counts the runs that reach the global minimum, 0 at the origin.
Without restarts, CMA-ES reaches it more often in more dimensions. The function is genoxide's
problems.Griewank, which run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/griewank/main.py
"""

import genoxide as gx

from trace import record_2d

DIMENSIONS = [2, 5, 10, 20, 30, 50]
EVALUATIONS_PER_DIMENSION = 10_000
# the budget in few dimensions, where 10,000 per dimension is too little for IPOP's restarts
MINIMUM_BUDGET = 100_000
SEEDS = 10


def solved(problem, restarts):
    """The runs of the seeds that reach the target."""
    target = problem.optimum.value + 1e-8
    budget = max(EVALUATIONS_PER_DIMENSION * problem.dimensions, MINIMUM_BUDGET)
    solved = 0
    for seed in range(1, SEEDS + 1):
        cmaes = gx.Cmaes(problem.genome, restarts=restarts, objective="minimize", seed=seed)
        result = cmaes.run(problem, target=target, evaluations=budget)
        if result.stop_reason == "target":
            solved += 1
    return solved


print(
    f"Runs of CMA-ES within 1e-8 of the minimum, of {SEEDS}, with {EVALUATIONS_PER_DIMENSION} "
    f"evaluations per dimension, {MINIMUM_BUDGET} at least"
)
print(f"{'dimensions':>10}{'no restarts':>14}{'IPOP':>14}")
for dimensions in DIMENSIONS:
    problem = gx.problems.Griewank(dimensions)
    never, ipop = solved(problem, "never"), solved(problem, "ipop")
    print(f"{dimensions:>10}{f'{never}/{SEEDS}':>14}{f'{ipop}/{SEEDS}':>14}")

# with GENOXIDE_TRACE=<file>, a trace for the plot on the example's page, of a separate run in
# 2 dimensions: the plot is the function's contour
record_2d()
