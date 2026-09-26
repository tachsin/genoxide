"""Kursawe: minimize two objectives whose Pareto front is in disconnected pieces, with SPEA2 and
NSGA-II.

Kursawe's problem in 3 variables, from genoxide's problems.Kursawe; run evaluates it in Rust. Its
front isn't known in closed form, so the example compares the two algorithms' fronts by their
hypervolume, and counts the pieces each finds: a new piece starts where two neighbors on the
front are more than 0.5 apart.

    python examples/kursawe/main.py
"""

import numpy as np

import genoxide as gx

REFERENCE = [-14.0, 1.0]


def report(name, front):
    """Prints the size of the front, its pieces and its hypervolume."""
    ordered = front[np.argsort(front[:, 0], kind="stable")]
    gaps = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    pieces = 1 + int((gaps > 0.5).sum())
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
spea2 = gx.Spea2(problem.genome, **settings)
report("SPEA2", spea2.run(problem, generations=250).front_objectives)
nsga2 = gx.Nsga2(problem.genome, **settings)
report("NSGA-II", nsga2.run(problem, generations=250).front_objectives)
