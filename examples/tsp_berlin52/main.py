"""Travelling salesman: the shortest round trip through the 52 locations in Berlin of TSPLIB's
berlin52, whose optimal tour has length 7542.

A permutation genome is the order of the visits. Local search with inversion neighbors (a random
2-opt move: a reversed segment of the tour) and simulated annealing, which also accepts worse
tours, less and less often as the temperature cools.

    python examples/tsp_berlin52/main.py
"""

import numpy as np

import genoxide as gx

# the coordinates of the locations, from berlin52.tsp
LOCATIONS = np.array([
    (565.0, 575.0), (25.0, 185.0), (345.0, 750.0), (945.0, 685.0),
    (845.0, 655.0), (880.0, 660.0), (25.0, 230.0), (525.0, 1000.0),
    (580.0, 1175.0), (650.0, 1130.0), (1605.0, 620.0), (1220.0, 580.0),
    (1465.0, 200.0), (1530.0, 5.0), (845.0, 680.0), (725.0, 370.0),
    (145.0, 665.0), (415.0, 635.0), (510.0, 875.0), (560.0, 365.0),
    (300.0, 465.0), (520.0, 585.0), (480.0, 415.0), (835.0, 625.0),
    (975.0, 580.0), (1215.0, 245.0), (1320.0, 315.0), (1250.0, 400.0),
    (660.0, 180.0), (410.0, 250.0), (420.0, 555.0), (575.0, 665.0),
    (1150.0, 1160.0), (700.0, 580.0), (685.0, 595.0), (685.0, 610.0),
    (770.0, 610.0), (795.0, 645.0), (720.0, 635.0), (760.0, 650.0),
    (475.0, 960.0), (95.0, 260.0), (875.0, 920.0), (700.0, 500.0),
    (555.0, 815.0), (830.0, 485.0), (1170.0, 65.0), (830.0, 610.0),
    (605.0, 625.0), (595.0, 360.0), (1340.0, 725.0), (1740.0, 245.0),
])
OPTIMUM = 7542

# TSPLIB's EUC_2D distance: the Euclidean distance, rounded to the nearest integer
dx = LOCATIONS[:, None, 0] - LOCATIONS[None, :, 0]
dy = LOCATIONS[:, None, 1] - LOCATIONS[None, :, 1]
DISTANCES = np.floor(np.sqrt(dx * dx + dy * dy) + 0.5)


def tour_length(order):
    return float(DISTANCES[order, np.roll(order, -1)].sum())


search = gx.LocalSearch(
    gx.Permutation(len(LOCATIONS)),
    neighbor=gx.InversionMutation(),
    acceptance=gx.Annealing(initial_temperature=100.0, cooling=0.99996),
    objective="minimize",
    seed=1,
)
result = search.run(tour_length, target=OPTIMUM, evaluations=200_000)

print(
    f"tour length {result.best_fitness:.0f} after {result.evaluations} evaluations "
    f"(the optimum: {OPTIMUM})"
)
# the tour from location 1, numbered from 1 as in TSPLIB
order = result.best_genome
tour = np.roll(order, -np.flatnonzero(order == 0)[0]) + 1
print(f"tour {tour.tolist()}")
