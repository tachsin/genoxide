"""Hartmann 3-D: find the local minima of Hartmann's function in 3 dimensions by restarting a local
search from random points.

Each search is a hill climber with Gaussian steps; it ends in the minimum whose basin it started
in. The searches that end close together are grouped, and the table gives each group's best point
and value. The function, its bounds and its best known minimum come from genoxide's
problems.Hartmann3, which run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/hartmann3/main.py
"""

import genoxide as gx

from trace import Trace

SEARCHES = 30
STEPS = 1_000


def significant(value):
    """The value rounded to 6 significant digits."""
    return float(f"{value:.5e}")


problem = gx.problems.Hartmann3()
optimum = problem.optimum
genome = problem.genome
# per group of searches that ended close together: its best point and value, and its searches
minima = []
# with GENOXIDE_TRACE=<file>, a trace of the searches for the plot on the example's page
trace = Trace([genome.bounds] * genome.length, optimum)
for seed in range(1, SEARCHES + 1):
    search = gx.LocalSearch(
        genome,
        neighbor=gx.GaussianMutation(0.001, rate=1.0),
        neighbors=10,
        acceptance=gx.Improving(),
        objective="minimize",
        seed=seed,
    )
    result = search.run(problem, generations=STEPS, on_generation=trace.on_generation)
    point = result.best_genome.tolist()
    value = result.best_fitness
    # the same minimum: within 0.02 in every gene (the bounds are [0, 1])
    close = (
        minimum
        for minimum in minima
        if all(abs(point[i] - minimum["point"][i]) <= 0.02 for i in range(3))
    )
    minimum = next(close, None)
    if minimum is None:
        minima.append({"point": point, "value": value, "searches": 1})
    else:
        minimum["searches"] += 1
        if value < minimum["value"]:
            minimum["point"], minimum["value"] = point, value
# by value, to 6 significant digits, then by the first gene
minima.sort(key=lambda minimum: (significant(minimum["value"]), minimum["point"][0]))

print(f"{SEARCHES} local searches from random points in [0, 1]^3, {STEPS} steps each")
print("minimum reached          searches  best value")
for minimum in minima:
    x1, x2, x3 = minimum["point"]
    is_global = "  global" if abs(minimum["value"] - optimum.value) < 1e-3 else ""
    print(
        f"({x1:.3f}, {x2:.3f}, {x3:.3f})  {minimum['searches']:>8}  "
        f"{minimum['value']:>10.5f}{is_global}"
    )
trace.write()
