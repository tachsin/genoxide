"""Nelder-Mead with restarts: find the four global minima of Himmelblau's function with one search
that starts again from a random point each time its simplex has converged.

Each run converges to the minimum of the basin it starts in; twenty runs land in all four. The
known minima come from genoxide's problems.Himmelblau.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/nelder_mead_himmelblau/main.py
"""

import math

import numpy as np

import genoxide as gx

from trace import Trace

RESTARTS = 19


def scientific(value):
    """Two significant digits, e.g. 1.2e-7."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


problem = gx.problems.Himmelblau()
minima = problem.optimum.solutions
nelder_mead = gx.NelderMead(problem.genome, restarts=RESTARTS, objective="minimize", seed=1)
# the best vertex of each run when it converged, and its value
ends = []
# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace(minima)


def control(algorithm, progress):
    if algorithm.converged:
        end = progress.population[0]
        trace.end(end)
        ends.append((end, float(progress.scores[0])))


result = nelder_mead.run(
    problem, evaluations=100_000, on_generation=trace.on_generation, control=control
)
assert result.stop_reason == "converged"

# per known minimum: the runs that ended nearest it, and the best and worst values they reached
found = [[0, math.inf, 0.0] for _ in minima]
for end, value in ends:
    nearest = int(np.argmin(np.linalg.norm(minima - end, axis=1)))
    found[nearest][0] += 1
    found[nearest][1] = min(found[nearest][1], value)
    found[nearest][2] = max(found[nearest][2], value)

print(
    f"{len(ends)} runs of Nelder-Mead in [-5, 5] x [-5, 5]: one, then {RESTARTS} restarts from "
    "random points"
)
print("minimum                  runs  values reached")
for minimum, (runs, best, worst) in zip(minima, found):
    print(
        f"({minimum[0]:>9.6f}, {minimum[1]:>9.6f})  {runs:>4}  "
        f"{scientific(best)} to {scientific(worst)}"
    )
print(f"{result.evaluations} evaluations in all, {result.evaluations / len(ends):.1f} per run")
trace.write()
