"""Nguyen-1: find the formula x^3 + x^2 + x from 20 points of it, by genetic programming.

The first of Nguyen's twelve symbolic regression problems (Uy et al. 2011): trees of Koza's
functions (+, -, x, protected division, sin, cos, exp and a protected logarithm) and the
variable x, evolved by a genetic algorithm with subtree crossover and mutation, fitted to the
root mean squared error on the points. The run stops at exact recovery: an error at the level of
rounding, on the 20 training points and on 100 test points in [-1, 1], with the expression
printed.

The trees and their errors are computed in Rust (``gx.gp``), with genoxide's portable math, so
the run is the Rust example's, to the bit, on every platform.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/nguyen_1/main.py
"""

import genoxide as gx

from trace import Trace

# the islands, their trees, and the generations between migrations
ISLANDS = 8
POPULATION = 500
INTERVAL = 10


def scientific(value):
    """``value`` with 2 decimals and an exponent, as Rust's ``{:.2e}`` writes it: 8.76e-11,
    0.00e0."""
    if value != value:
        return "NaN"
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa}e{int(exponent)}"


# Nguyen's function set and sampling: 20 training points uniform in [-1, 1], from a fixed seed,
# and 100 test points from another; the RMSE of the tree itself, without linear scaling
problem = gx.gp.regression.problems.Nguyen1()
regression = problem.regression(linear_scaling=False)
dataset = problem.dataset()
training, test = dataset.training, dataset.test
# exact recovery: an error of at most 1e-10 of the values' standard deviation
tolerance = 1e-10 * training.deviation()
print("Nguyen-1: x^3 + x^2 + x from 20 points in [-1, 1]")
print(f"{ISLANDS} islands of {POPULATION} trees, until the RMSE is at most {scientific(tolerance)}")
print()

# Koza's limits and initialization: depth 17, ramped half-and-half of depths 2 to 6
gp = gx.gp.Gp(problem.primitives())
islands = []
for island in range(ISLANDS):
    seed = 100 + island
    islands.append(
        gx.Ga(
            gp,
            population_size=POPULATION,
            # Koza's even division among the depths and methods, without duplicates
            initial_genomes=gp.ramped_half_and_half(POPULATION, seed),
            select=gx.Tournament(7),
            crossover=gx.gp.SubtreeCrossover(),
            mutation=gx.gp.SubtreeMutation(),
            crossover_rate=0.9,
            mutation_rate=0.1,
            objective="minimize",
            seed=seed,
        )
    )
islands = gx.Islands(islands, interval=INTERVAL, migrants=2)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace("nguyen_1", problem, regression, (-1.0, 1.0))
result = islands.run(
    regression, target=tolerance, generations=200, on_generation=trace.on_generation
)

best = result.best_genome


def rmse(sample):
    error = regression.error(best, sample)
    return float("nan") if error is None else error


print(
    f"{result.stop_reason.capitalize()} after {result.generations} generations and "
    f"{result.evaluations} evaluations"
)
print(f"RMSE on the 20 training points: {scientific(rmse(training))}")
print(f"RMSE on the 100 test points:    {scientific(rmse(test))}")
print()
print(f"the expression, {len(best)} nodes of depth {best.depth}:")
print(best)
trace.write()
