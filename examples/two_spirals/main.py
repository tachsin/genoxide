"""Two spirals: evolve the 2,545 weights of a neural network that tells two interleaved spirals
apart, by OpenAI's evolution strategy.

Lang and Witbrock's (1988) benchmark: 194 points on two spirals that wind three times around the
origin, 97 each, one the mirror image of the other through the origin. A network with two hidden
layers of 48 tanh units (2,545 weights with the biases) outputs a value in [-1, 1] for a point;
its sign is the spiral. The fitness is the mean squared error to the targets 1 and -1, minimized
by OpenEs from small random weights, until the network classifies every point.

The network runs in Rust (``gx.nn.Mlp.forward``), and the points use genoxide's portable sine and
cosine (``gx.math``), so the run is the Rust example's, to the bit, on every platform.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/two_spirals/main.py
"""

import math

import numpy as np

import genoxide as gx

from trace import Trace

# the points of each spiral
PER_SPIRAL = 97
# the largest radius, to which the coordinates are scaled: the points lie in [-1, 1]²
RADIUS = 6.5


def spiral_points():
    """Lang and Witbrock's points, scaled to [-1, 1]², a row each, and their targets 1 and -1:
    point i of the first spiral at the angle i π / 16 and the radius 6.5 (104 - i) / 104, and its
    mirror image through the origin on the second."""
    points, targets = [], []
    for i in range(PER_SPIRAL):
        angle = i * math.pi / 16.0
        radius = RADIUS * (104 - i) / 104.0
        x, y = radius * gx.math.sin(angle) / RADIUS, radius * gx.math.cos(angle) / RADIUS
        points += [[x, y], [-x, -y]]
        targets += [1.0, -1.0]
    return np.array(points), np.array(targets)


def round_half_up(value):
    """``value``, 0 or more, rounded to the nearest whole number, halves up, as Rust rounds it
    (Python's ``round`` rounds halves to even)."""
    floor = math.floor(value)
    return floor + 1 if value - floor >= 0.5 else floor


points, targets = spiral_points()
# 2 inputs, two hidden layers of 48 tanh units and a tanh output, with biases
mlp = gx.nn.Mlp([2, 48, 48, 1], "tanh", output_activation="tanh")


def classified(weights):
    """The points on the right side of 0."""
    outputs = mlp.forward(weights, points)[:, 0]
    return int(np.count_nonzero(outputs * targets > 0.0))


def error(weights):
    """The mean squared error to the targets, the squares added in order, as Rust adds them."""
    differences = mlp.forward(weights, points)[:, 0] - targets
    return float(np.cumsum(differences * differences)[-1]) / len(points)


# small random weights to start from: large ones saturate the tanh units
initial = gx.Real((-0.25, 0.25), length=mlp.parameters).random_genome(1)
open_es = gx.OpenEs(
    mlp.representation((-3.0, 3.0)),
    population_size=100,
    sigma=0.01,
    optimizer=gx.Adam(0.01),
    evaluate_mean=True,
    initial_mean=initial,
    parallel_breeding=True,
    objective="minimize",
    seed=1,
)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(mlp, points, targets)
solution = None


def on_generation(progress):
    """Records the generation, and stops the run once the generation's best network classifies
    every point."""
    global solution
    trace.record(progress)
    # the generation's best, the first of equals; an invalid score is NaN
    best = int(np.nanargmin(progress.scores))
    weights = progress.population[best]
    if classified(weights) == len(points):
        solution = (weights, float(progress.scores[best]), progress.evaluations)
        return False
    return True


result = open_es.run(error, evaluations=1_000_000, parallel=True, on_generation=on_generation)

if solution is None:
    print(
        f"not solved after {result.evaluations} evaluations: "
        f"{classified(result.best_genome)} of {len(points)} points classified"
    )
else:
    weights, mean_squared_error, evaluations = solution
    print(
        f"all {len(points)} points classified after {evaluations} evaluations in "
        f"{result.generations} generations, by a network of {mlp.parameters} weights"
    )
    print(f"mean squared error {mean_squared_error:.6f}")
    print()
    # the network's decision over [-1, 1]², a character per cell: # for the first spiral's side
    # and . for the second's, the points as A and B
    columns, rows = 61, 31
    cells = np.array(
        [
            [-1.0 + 2.0 * column / (columns - 1), 1.0 - 2.0 * row / (rows - 1)]
            for row in range(rows)
            for column in range(columns)
        ]
    )
    outputs = mlp.forward(weights, cells)[:, 0].reshape(rows, columns)
    grid = [["#" if output > 0.0 else "." for output in line] for line in outputs]
    for (x, y), target in zip(points, targets):
        column = round_half_up((x + 1.0) / 2.0 * (columns - 1))
        row = round_half_up((1.0 - y) / 2.0 * (rows - 1))
        grid[row][column] = "A" if target > 0.0 else "B"
    for line in grid:
        print("".join(line))
    trace.write()
