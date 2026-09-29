"""Himmelblau: find the four global minima of a two-dimensional function by restarting a local
search from random points.

Each search is a hill climber with Gaussian steps; it ends in the minimum whose basin it started
in. The known minima come from genoxide's problems.Himmelblau.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/himmelblau/main.py
"""

import math

import numpy as np

import genoxide as gx

from trace import Trace

SEARCHES = 20
STEPS = 3_000


def scientific(value):
    """Two significant digits, e.g. 1.2e-7."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


problem = gx.problems.Himmelblau()
minima = problem.optimum.solutions
# per known minimum: the searches that ended nearest it, and the best and worst values they
# reached
found = [[0, math.inf, 0.0] for _ in minima]
# with GENOXIDE_TRACE=<file>, a trace of the searches for the plot on the example's page
trace = Trace(minima)
for seed in range(1, SEARCHES + 1):
    search = gx.LocalSearch(
        problem.genome,
        neighbor=gx.GaussianMutation(0.0005, rate=1.0),
        neighbors=10,
        acceptance=gx.Improving(),
        objective="minimize",
        seed=seed,
    )
    result = search.run(problem, generations=STEPS, on_generation=trace.on_generation)
    nearest = int(np.argmin(np.linalg.norm(minima - result.best_genome, axis=1)))
    found[nearest][0] += 1
    found[nearest][1] = min(found[nearest][1], result.best_fitness)
    found[nearest][2] = max(found[nearest][2], result.best_fitness)

print(f"{SEARCHES} local searches from random points in [-5, 5] x [-5, 5], {STEPS} steps each")
print("minimum                  searches  values reached")
for minimum, (searches, best, worst) in zip(minima, found):
    print(
        f"({minimum[0]:>9.6f}, {minimum[1]:>9.6f})  {searches:>8}  "
        f"{scientific(best)} to {scientific(worst)}"
    )
trace.write()
