"""DTLZ2 with 3 objectives: minimize three conflicting objectives over 12 variables in [0, 1], whose
Pareto front is the positive eighth of the unit sphere.

NSGA-III twice: with Deb and Jain's (2014) settings, 91 reference directions from Das and Dennis's
method with 12 divisions and a population of 92; and with 703 directions, 36 divisions, and as
many solutions. Both run for 250 generations. Prints each final front's size, hypervolume and IGD+
to 1,035 points of the optimal front. The fitness function takes a generation at a time.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of the second run for the plot on the
example's page, with trace.py.

    python examples/dtlz2_3obj/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

VARIABLES = 12


def dtlz2(x):
    """x has a genome per row; the result, a row of objective values per genome."""
    radius = 1 + np.sum((x[:, 2:] - 0.5) * (x[:, 2:] - 0.5), axis=1)
    angle = x[:, :2] * np.pi / 2
    return np.column_stack(
        [
            radius * np.cos(angle[:, 0]) * np.cos(angle[:, 1]),
            radius * np.cos(angle[:, 0]) * np.sin(angle[:, 1]),
            radius * np.sin(angle[:, 0]),
        ]
    )


def hypervolume(front, reference):
    """The volume that the front dominates below the reference point, in slices between the
    values of the last objective: each slice is the area that the points below it dominate."""
    points = front[np.all(front < reference, axis=1)]
    points = points[np.argsort(points[:, 2], kind="stable")]
    tops = np.append(points[1:, 2], reference[2])
    volume = 0.0
    for index, top in enumerate(tops):
        below = points[: index + 1, :2]
        below = below[np.lexsort((below[:, 1], below[:, 0]))]
        area, ceiling = 0.0, reference[1]
        for x, y in below:
            if y < ceiling:
                area += (reference[0] - x) * (ceiling - y)
                ceiling = y
        volume += (top - points[index, 2]) * area
    return volume


# 1,035 points spread evenly over the sphere, for IGD+
OPTIMAL = gx.problems.Dtlz2(objectives=3, variables=VARIABLES).optimal_front(1000)


def run(name, divisions, population, on_generation=None):
    """Runs NSGA-III with the directions of Das and Dennis's method with ``divisions`` for 250
    generations, and reports its front."""
    nsga3 = gx.Nsga3(
        gx.Real((0.0, 1.0), length=VARIABLES),
        objectives=["minimize"] * 3,
        reference_directions=gx.das_dennis(3, divisions),
        population_size=population,
        crossover=gx.SimulatedBinaryCrossover(30),
        mutation=gx.PolynomialMutation(20, rate=1 / VARIABLES),
        seed=1,
    )
    result = nsga3.run(dtlz2, batch=True, generations=250, on_generation=on_generation)

    front = result.front_objectives
    volume = hypervolume(front, np.array([1.1, 1.1, 1.1]))
    distance = gx.indicators.igd_plus(front, OPTIMAL)
    print(f"{name:<14} {len(front)} solutions, hypervolume {volume:.4f}, IGD+ {distance:.4f}")


# Deb and Jain's settings: 91 directions and a population of 92, the multiple of 4 above
run("91 directions", 12, 92)

# with GENOXIDE_TRACE=<file>, a trace of this run for the plot on the example's page
trace = Trace()
# 703 directions, and a solution for each
run("703 directions", 36, None, trace.on_generation)

# the whole front's hypervolume, with the reference point (1.1, 1.1, 1.1): 1.1³ minus the eighth
# of the unit ball, π/6
print("the whole front: hypervolume 0.8074")
trace.write()
