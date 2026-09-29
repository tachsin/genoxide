"""CTP1: minimize two objectives over two variables subject to two constraints, with
a front two thirds of which lie on the boundaries of two constraints.

NSGA-II with the settings of the report's experiments, a population of 100 for 500
generations. Prints how many solutions of the final front are feasible, how many pieces of
the optimal front they reach, their IGD+ to it and their hypervolume.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on
the example's page, with trace.py.

    python examples/ctp1/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace, scaled, whole_front_hypervolume

problem = gx.problems.Ctp1()
# how close a solution must come to a piece of the optimal front, in scaled objectives, to reach it
REACH = 0.02


def pieces(front):
    """The pieces of a front sorted by f₁, split where neighbors are more than 0.01 apart."""
    gaps = np.flatnonzero(np.hypot(*np.diff(front, axis=0).T) > 0.01) + 1
    return np.split(front, gaps)


def report(name, generations, result):
    """The feasible solutions of a run's front: how many of them, how many pieces of the optimal
    front they reach, their IGD+ to it and their hypervolume, as a share of the whole front's."""
    front = result.front_objectives[result.front_violations == 0]
    size = len(result.front_objectives)
    feasible = "all feasible" if len(front) == size else f"{len(front)} feasible"
    print(f"{name}, {generations} generations: {size} solutions on the front, {feasible}")
    optimal = problem.optimal_front(2000)
    found = scaled(front)
    parts = pieces(optimal)
    reached = sum(
        any(np.hypot(*(found - p).T).min() <= REACH for p in scaled(piece)) for piece in parts
    )
    print(f"  pieces of the optimal front reached: {reached} of {len(parts)}")
    distance = gx.indicators.igd_plus(found, scaled(optimal))
    volume = gx.indicators.hypervolume(found, [1.1, 1.1])
    whole = whole_front_hypervolume()
    print(
        f"  IGD+ {distance:.5f}, hypervolume {volume:.4f}, "
        f"{100 * volume / whole:.2f}% of the whole front's {whole:.4f}"
    )


def nsga2():
    """NSGA-II with the settings of the report's experiments: a population of 100, SBX and
    polynomial mutation with η = 20, crossover at 0.9 and mutation at 1/n per gene."""
    return gx.Nsga2(
        problem.genome,
        objectives=problem.objectives,
        population_size=100,
        crossover=gx.SimulatedBinaryCrossover(20),
        mutation=gx.PolynomialMutation(20, rate=0.5),
        seed=1,
    )


# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
result = nsga2().run(problem, generations=500, on_generation=trace.on_generation)
report("NSGA-II", 500, result)
trace.write()
