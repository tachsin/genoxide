"""Schwefel 2.26: minimize a deceptive function whose best local minima are far apart, in 30
dimensions.

L-SHADE (differential evolution with a population that shrinks over the budget) reaches the global
minimum, −418.98 per gene, where every gene is 420.97, near the upper bound. CMA-ES with IPOP
restarts and particle swarm optimization with a ring topology, for contrast, stay far from it. The
function is genoxide's problems.Schwefel2_26, which run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/schwefel_2_26/main.py
"""

import genoxide as gx

from trace import record_2d

DIMENSIONS = 30
BUDGET = 10_000 * DIMENSIONS

problem = gx.problems.Schwefel2_26(DIMENSIONS)
minimum = problem.optimum.value
print(f"minimum: {minimum:.2f}, {BUDGET} evaluations at most")
target = minimum + 1e-8
# the global minimizer of each gene, 420.97
best_gene = problem.optimum.solutions[0][0]


def scientific(value):
    """Two significant digits, as Rust writes them: 3.6e-9."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def report(name, algorithm):
    """Runs the algorithm and prints its best value, its error and its genes near 420.97."""
    result = algorithm.run(problem, target=target, evaluations=BUDGET)
    # rounding can put a solution a few ulps below the minimum
    error = max(result.best_fitness - minimum, 0.0)
    near = sum(abs(x - best_gene) < 1.0 for x in result.best_genome)
    print(
        f"{name}: {result.best_fitness:.2f}, error {scientific(error)}, {near} of {DIMENSIONS} "
        f"genes within 1 of {best_gene:.2f}"
    )


report("L-SHADE", gx.De(problem.genome, l_shade=BUDGET, objective="minimize", seed=1))
print("for contrast, two algorithms that stay far from it:")
report(
    "CMA-ES with IPOP", gx.Cmaes(problem.genome, restarts="ipop", objective="minimize", seed=1)
)
report(
    "PSO on a ring",
    gx.Pso(problem.genome, population_size=40, ring=1, objective="minimize", seed=1),
)

# with GENOXIDE_TRACE=<file>, a trace for the plot on the example's page, of a separate run in
# 2 dimensions: the plot is the function's contour
record_2d()
