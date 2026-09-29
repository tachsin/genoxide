"""DTLZ3 with 3 objectives: minimize three conflicting objectives over 12 variables in [0, 1],
whose Pareto front is the positive eighth of the unit sphere, behind 3¹⁰ − 1 local fronts.

NSGA-III twice: with Deb and Jain's (2014) settings, 91 reference directions from Das and Dennis's
method with 12 divisions, a population of 92 and 1,000 generations; and with 703 directions, 36
divisions, as many solutions, and 600 generations. Prints each final front's size, hypervolume,
IGD+ to 1,035 points of the optimal front, and how far its farthest solution is from the front.
run evaluates the problem in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of the second run for the plot on the
example's page, with trace.py.

    python examples/dtlz3_3obj/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

VARIABLES = 12

# the reference point of the hypervolume: 1.1 times the nadir point (1, 1, 1)
REFERENCE = [1.1, 1.1, 1.1]

problem = gx.problems.Dtlz3(objectives=3, variables=VARIABLES)


def run(name, divisions, population, generations, on_generation=None):
    """Runs NSGA-III with the directions of Das and Dennis's method with ``divisions``, and
    reports its front after ``generations``."""
    nsga3 = gx.Nsga3(
        problem.genome,
        objectives=problem.objectives,
        reference_directions=gx.das_dennis(3, divisions),
        population_size=population,
        crossover=gx.SimulatedBinaryCrossover(30),
        mutation=gx.PolynomialMutation(20, rate=1 / VARIABLES),
        seed=1,
    )
    result = nsga3.run(problem, generations=generations, on_generation=on_generation)

    front = result.front_objectives
    volume = gx.indicators.hypervolume(front, REFERENCE)
    # IGD+ to 1,035 points spread evenly over the sphere
    distance = gx.indicators.igd_plus(front, problem.optimal_front(1000))
    # the objectives are a point at distance 1 + g from the origin: g is 0 on the front
    farthest = max(0.0, (np.sqrt(np.sum(front * front, axis=1)) - 1).max())
    print(
        f"{name:<14} {len(front)} solutions, hypervolume {volume:.4f}, IGD+ {distance:.4f}, "
        f"largest g {farthest:.5f}"
    )


# Deb and Jain's settings: 91 directions and a population of 92, the multiple of 4 above
run("91 directions", 12, 92, 1000)

# with GENOXIDE_TRACE=<file>, a trace of this run for the plot on the example's page
trace = Trace(REFERENCE)
# 703 directions, and a solution for each
run("703 directions", 36, None, 600, trace.on_generation)

# the whole front's hypervolume: 1.1³ minus the eighth of the unit ball, π/6
print("the whole front: hypervolume 0.8074")
trace.write()
