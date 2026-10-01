"""Nguyen-1 to 12: all twelve of Nguyen's symbolic regression problems, by genetic programming,
with and without linear scaling.

Each problem is run twice with the same search, the one of the Nguyen-1, 5 and 9 pages: eight
islands of 500 trees of Koza's functions, with subtree crossover and mutation, for at most 200
generations. Once on the RMSE of the tree itself, and once after linear scaling (a + b x tree,
with a and b fitted by least squares). A problem is recovered when the error is at the level of
rounding on the training and the test points; the table gives the generations it took, or the
test error of the best tree when it wasn't.

The trees and their errors are computed in Rust (``gx.gp``), in parallel and with genoxide's
portable math, so the runs are the Rust example's, to the bit, on every platform.

    python examples/nguyen_all/main.py
"""

import genoxide as gx

# the islands, their trees, and the generations between migrations
ISLANDS = 8
POPULATION = 500
INTERVAL = 10


def scientific(value):
    """``value`` with 1 decimal and an exponent, as Rust's ``{:.1e}`` writes it: 6.3e-2."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def run(problem, regression):
    """A run of the search on the problem, as "recovered at 7" or "test RMSE 1.2e-3"."""
    dataset = problem.dataset()
    training, test = dataset.training, dataset.test
    # exact recovery: an error of at most 1e-10 of the values' standard deviation
    tolerance = 1e-10 * training.deviation()
    # Koza's limits and initialization: depth 17, ramped half-and-half of depths 2 to 6
    gp = gx.gp.Gp(problem.primitives())
    islands = []
    for island in range(ISLANDS):
        seed = 100 + island
        islands.append(
            gx.Ga(
                gp,
                population_size=POPULATION,
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
    # parallel evaluation: the same results as without, sooner
    result = islands.run(regression, target=tolerance, generations=200, parallel=True)
    test_error = regression.error(result.best_genome, test)
    recovered = test_error is not None and test_error <= 1e-10 * test.deviation()
    if result.stop_reason == "target" and recovered:
        return f"recovered at {result.generations}"
    if test_error is not None:
        return f"test RMSE {scientific(test_error)}"
    return "test not finite"


print(f"Nguyen-1 to 12, {ISLANDS} islands of {POPULATION} trees for at most 200 generations")
print("recovered: the generation of exact recovery; else the best tree's test RMSE")
print()
print(f"{'problem':<10} {'target':<32} {'tree itself':>18} {'linear scaling':>18}")
for problem in gx.gp.regression.problems.all():
    if not problem.name.startswith("Nguyen"):
        continue
    without = run(problem, problem.regression(linear_scaling=False))
    with_scaling = run(problem, problem.regression())
    print(f"{problem.name:<10} {problem.formula:<32} {without:>18} {with_scaling:>18}")
