"""DTLZ2 with 3 objectives: minimize three conflicting objectives over 12 variables in [0, 1], whose
Pareto front is the positive eighth of the unit sphere.

NSGA-III with the 91 reference directions of Das and Dennis's method with 12 divisions, a
population of 92 and 250 generations, as in Deb and Jain (2014). Prints the size of the final
front and its hypervolume. The fitness function takes a generation at a time.

    python examples/dtlz2_3obj/main.py
"""

import numpy as np

import genoxide as gx

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


nsga3 = gx.Nsga3(
    gx.Real((0.0, 1.0), length=VARIABLES),
    objectives=["minimize"] * 3,
    reference_directions=gx.das_dennis(3, 12),
    population_size=92,
    crossover=gx.SimulatedBinaryCrossover(30),
    mutation=gx.PolynomialMutation(20, rate=1 / VARIABLES),
    seed=1,
)
result = nsga3.run(dtlz2, batch=True, generations=250)

# the hypervolume of the front, with the reference point (1.1, 1.1, 1.1); the whole front's is
# 1.1³ minus the eighth of the unit ball, π/6
front = result.front_objectives
volume = hypervolume(front, np.array([1.1, 1.1, 1.1]))
print(f"{len(front)} solutions on the front, hypervolume {volume:.4f} (the whole front: 0.8074)")
