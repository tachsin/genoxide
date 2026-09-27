"""Kursawe: minimize two objectives whose Pareto front is in disconnected pieces, with SPEA2 and
NSGA-II.

Kursawe's problem in 3 variables, from genoxide's problems.Kursawe; run evaluates it in Rust. Its
front isn't known in closed form, so the example compares the two algorithms' fronts by their
hypervolume, and counts the pieces each finds: a new piece starts where f₁ grows by more than
0.2 between two neighbors on the front.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/kursawe/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

REFERENCE = [-14.0, 1.0]


def report(name, front):
    """Prints the size of the front, its pieces and its hypervolume."""
    ordered = front[np.argsort(front[:, 0], kind="stable")]
    # along a piece, f₁ grows by at most about 0.13 between neighbors, and f₂ falls; the gaps
    # between the pieces of the true front are 0.25 to 0.92 wide in f₁, with f₂ nearly unchanged
    pieces = 1 + int((np.diff(ordered[:, 0]) > 0.2).sum())
    volume = gx.indicators.hypervolume(front, REFERENCE)
    print(f"{name:<8} {len(front)} solutions in {pieces} pieces, hypervolume {volume:.4f}")


problem = gx.problems.Kursawe(3)
settings = dict(
    objectives=problem.objectives,
    population_size=100,
    crossover=gx.SimulatedBinaryCrossover(15),
    mutation=gx.PolynomialMutation(20, rate=1 / 3),
    seed=1,
)
# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace(REFERENCE)
spea2 = gx.Spea2(problem.genome, **settings)
result = spea2.run(problem, generations=250, on_generation=trace.fronts("SPEA2"))
report("SPEA2", result.front_objectives)
nsga2 = gx.Nsga2(problem.genome, **settings)
result = nsga2.run(problem, generations=250, on_generation=trace.fronts("NSGA-II"))
report("NSGA-II", result.front_objectives)
trace.write()
