"""Continuation: the smallest ball around 420 points in 10 dimensions, its center found by
minimizing a smoothed largest distance, (Σ dᵢ^2p)^(1/2p), for p = 2, 4, 8 and 16, each stage from
the last, with Adam's state kept between them.

The points are made so that the center is known exactly, and the last stage ends at it. Then the
same stages keeping only the point, and p = 16 from the start, as contrasts.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/continuation/main.py
"""

import math

import numpy as np

import genoxide as gx

from trace import Trace

# the dimensions, and the points near the center
D = 10
NEAR = 400
# how far from the center the near points are, at most
SPREAD = 0.3
# the stages' p = 2^k, for k = 1 to 4
POWERS = (1, 2, 3, 4)
# Adam's learning rate, and the largest component of the gradient at which a stage has converged
RATE = 0.05
TOLERANCE = 1e-11


def points():
    """The center c, and the points: c ± eᵢ on every axis i, the farthest from c, and NEAR points
    within SPREAD of it, each coordinate moved by up to SPREAD / √D."""
    center = np.array([(i + 1) / D - 0.5 for i in range(D)])
    axes = []
    for i in range(D):
        for sign in (1.0, -1.0):
            point = center.copy()
            point[i] += sign
            axes.append(point)
    offsets = gx.Real((0.0, 1.0), length=NEAR * D).random_genome(1).reshape(NEAR, D)
    near = center + (SPREAD / math.sqrt(D)) * offsets
    return center, np.vstack([np.array(axes), near])


CENTER, POINTS = points()


def squared(x):
    """|x − a|² for every point a, each summed in the order of the genes."""
    differences = x - POINTS
    # a cumulative sum adds one term after the other, as the Rust example's loop
    return np.cumsum(differences * differences, axis=1)[:, -1]


def smoothed(x, k):
    """The smoothed largest squared distance (Σ gᵢ^p)^(1/p), gᵢ = |x − aᵢ|², p = 2^k, and its
    gradient, in the order of the Rust example's operations, so that both take the same steps.
    With M the largest gᵢ, it's M (Σ rᵢ^p)^(1/p) with rᵢ = gᵢ / M ≤ 1, which can't overflow; the
    powers are k squarings and the root k square roots, which round the same on every platform.
    The gradient is (Σ rᵢ^p)^(1/p − 1) Σ rᵢ^(p−1) 2 (x − aᵢ)."""
    g = squared(x)
    most = float(np.max(g))
    r = g / most
    power = r.copy()
    for _ in range(k):
        power = power * power
    total = float(np.cumsum(power)[-1])
    weights = np.where(r > 0.0, power / np.where(r > 0.0, r, 1.0), 0.0)
    root = total
    for _ in range(k):
        root = math.sqrt(root)
    terms = (2.0 * weights)[:, None] * (x - POINTS)
    gradient = np.cumsum(terms, axis=0)[-1] * (root / total)
    return most * root, gradient


def distance(x):
    """The largest difference of a coordinate from the center's."""
    return float(np.max(np.abs(x - CENTER)))


def largest(x):
    """The largest distance from x to a point."""
    return math.sqrt(float(np.max(squared(x))))


def scientific(value):
    """One digit after the point, e.g. 1.2e-7."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


trace = Trace()


def run(keep, powers, line):
    """Adam through the stages of ``powers`` from the same start, keeping ``keep`` between them,
    until the last stage has converged; each step recorded in the trace as run ``line``."""
    stage = {"index": 0}
    ended = []

    def on_stage(index):
        stage["index"] = index

    def on_stage_finished(finished, point):
        ended.append(distance(point))

    def record(running, progress):
        p = 1 << powers[stage["index"]]
        trace.record(line, progress.generation, distance(progress.population[0]), p)

    adam = gx.FirstOrder(
        gx.Real((-2.0, 2.0), length=D),
        step="adam",
        learning_rate=RATE,
        initial_genome=[-1.5] * D,
        gradient_tolerance=TOLERANCE,
        objective="minimize",
    )
    continuation = gx.Continuation(
        adam,
        stages=len(powers),
        on_stage=on_stage,
        keep=keep,
        on_stage_finished=on_stage_finished,
    )
    last = {}

    def keep_last(running, progress):
        record(running, progress)
        last["point"] = progress.population[0]

    result = continuation.run(
        lambda x: smoothed(x, powers[stage["index"]]),
        gradient=True,
        generations=100_000,
        control=keep_last,
    )
    assert result.stop_reason == "converged"
    stages = [
        (
            1 << powers[finished.index],
            finished.generations,
            finished.evaluations,
            math.sqrt(finished.best_fitness),
            ended[finished.index],
        )
        for finished in result.stages
    ]
    return stages, result, last["point"]


print(
    f"The smallest ball around {2 * D + NEAR} points in {D} dimensions: {2 * D} at distance 1 "
    f"from its center, {NEAR} within {SPREAD} of it"
)
print("Adam through 4 stages of the smoothed largest distance, each from the last")
stages, kept, point = run("state", POWERS, 0)
print(" stage   p  steps  evaluations  smoothed distance  distance to the center")
for index, (p, steps, evaluations, value, gap) in enumerate(stages):
    print(
        f"{index + 1:>6}  {p:>2}  {steps:>5}  {evaluations:>11}  {value:>17.10f}  "
        f"{scientific(gap):>22}"
    )
error = distance(point)
print(
    f"converged after {kept.generations} steps and {kept.evaluations} evaluations: the center "
    f"within {scientific(error)}, the largest distance 1 + {scientific(largest(point) - 1.0)}"
)
assert error < 1e-10

# the contrasts: the same stages keeping only the point, and p = 16 alone
_, only_point, point = run("point", POWERS, 1)
print(
    f"the point only kept between stages: {only_point.generations} steps and "
    f"{only_point.evaluations} evaluations, the center within {scientific(distance(point))}"
)
_, cold, point = run("state", POWERS[3:], 2)
print(
    f"p = 16 from the start: {cold.generations} steps and {cold.evaluations} evaluations, the "
    f"center within {scientific(distance(point))}"
)
trace.write()
