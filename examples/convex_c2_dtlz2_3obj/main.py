"""Convex C2-DTLZ2 with 3 objectives: minimize three objectives over 12 variables in [0, 1], convex
DTLZ2 with a cylinder around the diagonal that cuts a hole in its front.

NSGA-III with the settings of Jain and Deb (2014), for 250 generations. Prints how many
solutions of the final front are feasible, how many of the points where the 91 reference
directions meet the front they reach, and their IGD+ to those points and hypervolume.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on
the example's page, with trace.py.

    python examples/convex_c2_dtlz2_3obj/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace, scaled, targets

problem = gx.problems.ConvexC2Dtlz2()
# how close a solution must come to a target point, in scaled objectives, to reach it
REACH = 0.02


def nsga3():
    """NSGA-III with the settings of Jain and Deb (2014): the 91 reference directions of Das and
    Dennis's method with 12 divisions, a population of 92, SBX with η = 30 at a rate of 1, and
    polynomial mutation with η = 20 at a rate of 1/n per gene for n genes."""
    return gx.Nsga3(
        problem.genome,
        objectives=problem.objectives,
        reference_directions=gx.das_dennis(3, 12),
        population_size=92,
        crossover=gx.SimulatedBinaryCrossover(30),
        crossover_rate=1.0,
        mutation=gx.PolynomialMutation(20, rate=1 / problem.dimensions),
        seed=1,
    )


def report(name, generations, front, size):
    """The feasible solutions of a run's front: how many of them, how many target points they
    reach, their IGD+ to the target points and their hypervolume, as a share of the target
    points'."""
    feasible = "all feasible" if len(front) == size else f"{len(front)} feasible"
    print(f"{name}, {generations} generations: {size} solutions on the front, {feasible}")
    found = scaled(front)
    points = targets()
    reached = sum(np.sqrt(((found - t) ** 2).sum(axis=1)).min() <= REACH for t in points)
    print(f"  target points reached: {reached} of {len(points)}")
    distance = gx.indicators.igd_plus(found, points)
    volume = gx.indicators.hypervolume(found, [1.1] * 3)
    theirs = gx.indicators.hypervolume(points, [1.1] * 3)
    print(
        f"  IGD+ {distance:.5f}, hypervolume {volume:.4f}, "
        f"{100 * volume / theirs:.2f}% of the target points' {theirs:.4f}"
    )


# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(problem)
result = nsga3().run(problem, generations=250, on_generation=trace.on_generation)
front = result.front_objectives[result.front_violations == 0]
report("NSGA-III", 250, front, len(result.front_objectives))
trace.write()
