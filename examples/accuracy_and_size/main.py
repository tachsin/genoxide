"""Accuracy against size: the trade-off between a formula's error and its size on Nguyen-7,
ln(x + 1) + ln(x^2 + 1), by NSGA-II on trees.

Genetic programming rarely recovers Nguyen-7 exactly: the search settles on trees of hundreds of
nodes that fit the points to a few decimals. Minimizing the error and the size together, as two
objectives, gives instead the whole trade-off at once, the Pareto front: for each size, the
smallest error found, from a constant to large and accurate trees, with the small ones readable.

Both objectives are computed in Rust (``gx.gp.WithSize``), in parallel and with genoxide's
portable math, so the run is the Rust example's, to the bit, on every platform.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/accuracy_and_size/main.py
"""

import genoxide as gx

from trace import Trace

POPULATION = 1000
GENERATIONS = 200


def scientific(value):
    """``value`` with 2 decimals and an exponent, as Rust's ``{:.2e}`` writes it: 2.15e-2."""
    if value != value:
        return "NaN"
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa}e{int(exponent)}"


# Nguyen's function set and sampling: 20 training points uniform in [0, 2], from a fixed seed,
# and 100 test points from another; the RMSE after linear scaling
problem = gx.gp.regression.problems.Nguyen7()
regression = problem.regression()
test = problem.dataset().test
print("Nguyen-7: ln(x + 1) + ln(x^2 + 1) from 20 points in [0, 2]")
print(
    f"NSGA-II, {POPULATION} trees for {GENERATIONS} generations, minimizing the RMSE and the size"
)
print()

# Koza's limits and initialization: depth 17, ramped half-and-half of depths 2 to 6
gp = gx.gp.Gp(problem.primitives())
nsga2 = gx.Nsga2(
    gp,
    objectives=["minimize", "minimize"],
    population_size=POPULATION,
    initial_genomes=gp.ramped_half_and_half(POPULATION, 1),
    crossover=gx.gp.SubtreeCrossover(),
    mutation=gx.gp.SubtreeMutation(),
    crossover_rate=0.9,
    mutation_rate=0.1,
    seed=1,
)
# the two objectives: the training RMSE, and the number of nodes; a tree whose value isn't finite
# at a point is invalid
objectives = gx.gp.WithSize(regression)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace()
result = nsga2.run(
    objectives, generations=GENERATIONS, parallel=True, on_generation=trace.on_generation
)

# the front, from the smallest tree to the most accurate: one tree per point, since trees that
# differ only in the order of their arguments have the same error and size
valid = [
    (tree, float(values[0]))
    for tree, values in zip(result.front_genomes, result.front_objectives)
    if values[0] == values[0]
]
front = []
for tree, error in sorted(valid, key=lambda pair: (len(pair[0]), pair[1])):
    if not front or (len(front[-1][0]), front[-1][1]) != (len(tree), error):
        front.append((tree, error))
print(f"the Pareto front after {result.evaluations} evaluations: {len(front)} points")
print(f"{'size':>5} {'RMSE':>9} {'test RMSE':>9}  a + b * (expression), up to 25 nodes")
for tree, error in front:
    test_error = regression.error(tree, test)
    scaling = regression.scaling(tree)
    expression = ""
    if scaling is not None and len(tree) <= 25:
        intercept, slope = scaling
        sign = "-" if slope < 0.0 else "+"
        expression = f"{intercept:.4f} {sign} {abs(slope):.4f} * ({tree})"
    test_text = scientific(float("nan") if test_error is None else test_error)
    print(f"{len(tree):>5} {scientific(error):>9} {test_text:>9}  {expression}")
trace.write()
