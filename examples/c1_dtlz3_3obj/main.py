"""C1-DTLZ3 with 3 objectives: minimize three objectives over 12 variables in [0, 1], DTLZ3 with an
infeasible shell between the radii 4 and 9 around the origin.

NSGA-III with the settings of Jain and Deb (2014), for 1,500 generations, twice: with Deb's
rules for the constraint, and on the objectives alone, the constraint checked on the final
front. Prints, for each, how many solutions of the final front are feasible, how many of the
points where the 91 reference directions meet the front they reach, and their IGD+ to those
points and hypervolume.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of the second run for the plot on
the example's page, with trace.py.

    python examples/c1_dtlz3_3obj/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace, scaled, targets

problem = gx.problems.C1Dtlz3()
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


# NSGA-III with Deb's rules: a feasible solution beats an infeasible one
result = nsga3().run(problem, generations=1500)
front = result.front_objectives[result.front_violations == 0]
report("NSGA-III with Deb's rules", 1500, front, len(result.front_objectives))

# NSGA-III on the objectives alone: the constraint ignored in the search, and checked on the final
# front; with GENOXIDE_TRACE=<file>, a trace of this run for the plot on the example's page
trace = Trace(problem)
result = nsga3().run(
    lambda genomes: problem.evaluate(genomes)[0],
    batch=True,
    generations=1500,
    on_generation=trace.on_generation,
)
objectives, violations = problem.evaluate(result.front_genomes)
report("NSGA-III on the objectives alone", 1500, objectives[violations == 0], len(objectives))
trace.write()
