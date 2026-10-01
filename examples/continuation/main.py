"""Continuation: a tilted Rastrigin function in 10 dimensions, Σ (xᵢ − aᵢ)² + 10 (1 − cos 2πxᵢ),
minimized through 6 stages of its Gaussian smoothing, σ from 0.6 (convex) to 0 (the function
itself), each stage from the last, with L-BFGS-B.

The function is separable, so its global minimum is computed exactly, gene by gene, and the last
stage ends at it. Then, as contrasts, σ = 0 from the same start, which a local method can't take
out of the nearest basin, and the stages with L-BFGS-B's curvature pairs kept.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/continuation/main.py
"""

import math

import numpy as np

import genoxide as gx

from trace import Trace

# the centers of the quadratic, off the cosine's lattice, and the cosine's amplitude
CENTERS = np.array([1.3, -0.7, 2.2, -1.6, 0.35, 3.25, -2.8, 0.7, -0.3, 1.8])
A = 10.0
# the stages' smoothing: 0.6 is convex (above 0.517), 0 the function itself
SIGMAS = (0.6, 0.4, 0.3, 0.2, 0.1, 0.0)
# every gene starts here
START = -3.0
# L-BFGS-B's largest projected gradient component at which a stage has converged
TOLERANCE = 1e-10
PI, TAU = math.pi, math.tau


def smoothed(x, s):
    """The function smoothed by a Gaussian of σ, E[f(x + σz)] for z standard normal, and its
    gradient. E[cos 2π(x + σz)] = e^(−2π²σ²) cos 2πx and E[(x + σz − a)²] = (x − a)² + σ², so each
    gene's term is (x − a)² + σ² + A (1 − e^(−2π²σ²) cos 2πx), convex once 4π²A e^(−2π²σ²) < 2.
    With genoxide's portable cos, sin and exp, and the terms summed one after the other, the same
    bits as the Rust example."""
    e = float(gx.math.exp(np.array([-2.0 * PI * PI * s * s]))[0])
    d = x - CENTERS
    terms = d * d + s * s + A * (1.0 - e * gx.math.cos(TAU * x))
    gradient = 2.0 * d + 2.0 * PI * A * e * gx.math.sin(TAU * x)
    # a cumulative sum adds one term after the other, as the Rust example's loop
    return float(np.cumsum(terms)[-1]), gradient


def value(x, s):
    """The function smoothed by σ, without the gradient."""
    return smoothed(x, s)[0]


def global_minimum():
    """The global minimum, gene by gene: in each basin around an integer k near the center a,
    where the term is convex (|x − k| ≤ 1/4), the root of its derivative 2(x − a) + 2πA sin 2πx by
    bisection to the last bit, and of those the lowest. A gene's basins are bisected side by side,
    as an array."""
    minimum = []
    for a in CENTERS:
        k = np.arange(math.floor(a) - 3, math.ceil(a) + 4, dtype=float)
        low, high = k - 0.25, k + 0.25
        for _ in range(100):
            middle = 0.5 * (low + high)
            rising = 2.0 * (middle - a) + 2.0 * PI * A * gx.math.sin(TAU * middle) > 0.0
            high = np.where(rising, middle, high)
            low = np.where(rising, low, middle)
        x = 0.5 * (low + high)
        terms = (x - a) * (x - a) + A * (1.0 - gx.math.cos(TAU * x))
        # the first of the lowest, as the Rust loop keeps it
        minimum.append(x[int(np.argmin(terms))])
    return np.array(minimum)


def distance(x, exact):
    """The largest difference of a gene from the global minimum's."""
    return float(np.max(np.abs(x - exact)))


def scientific(value):
    """One digit after the point, e.g. 1.2e-7."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


trace = Trace()


def run(sigmas, keep_pairs, exact, line):
    """L-BFGS-B through the stages of ``sigmas`` from the same start, keeping its pairs between
    them or not, until the last stage has converged; each round recorded in the trace as run
    ``line``."""
    stage = {"index": 0}
    ended = []
    last = {}

    def on_stage(index):
        stage["index"] = index

    def on_stage_finished(finished, point):
        ended.append(distance(point, exact))

    def record(running, progress):
        point = progress.population[0]
        trace.record(line, progress.generation, distance(point, exact), sigmas[stage["index"]])
        last["point"] = point

    lbfgsb = gx.Lbfgsb(
        gx.Real((-5.0, 5.0), length=len(CENTERS)),
        initial_genome=[START] * len(CENTERS),
        gradient_tolerance=TOLERANCE,
        keep_pairs=keep_pairs,
        objective="minimize",
    )
    continuation = gx.Continuation(
        lbfgsb, stages=len(sigmas), on_stage=on_stage, on_stage_finished=on_stage_finished
    )
    result = continuation.run(
        lambda x: smoothed(x, sigmas[stage["index"]]),
        gradient=True,
        evaluations=10_000,
        control=record,
    )
    assert result.stop_reason == "converged"
    stages = [
        (
            sigmas[finished.index],
            finished.generations,
            finished.evaluations,
            finished.best_fitness,
            ended[finished.index],
        )
        for finished in result.stages
    ]
    return stages, result, last["point"]


exact = global_minimum()
minimum = value(exact, 0.0)
print(
    f"A tilted Rastrigin function in {len(CENTERS)} dimensions, Σ (xᵢ − aᵢ)² + {A:g} (1 − cos "
    f"2πxᵢ), from xᵢ = {START:g}"
)
print("L-BFGS-B through 6 stages of the function smoothed by a Gaussian of σ, each from the last")
stages, staged, point = run(SIGMAS, False, exact, 0)
print(" stage     σ  rounds  evaluations  value of the stage  distance to the global minimum")
for index, (sigma, rounds, evaluations, best, gap) in enumerate(stages):
    print(
        f"{index + 1:>6}  {sigma:>4.2f}  {rounds:>6}  {evaluations:>11}  {best:>18.10f}  "
        f"{scientific(gap):>29}"
    )
error = distance(point, exact)
print(
    f"converged after {staged.generations} rounds and {staged.evaluations} evaluations: f = "
    f"{value(point, 0.0):.14f}, within {scientific(error)} of the global minimum"
)
print(f"the global minimum, gene by gene by bisection: f* = {minimum:.14f}")
assert error < 1e-12

# the contrasts: σ = 0 from the same start, and the stages keeping the curvature pairs
_, cold, point = run(SIGMAS[5:], False, exact, 1)
trapped = value(point, 0.0)
print(
    f"σ = 0 from the start: {cold.generations} rounds and {cold.evaluations} evaluations, trapped "
    f"at f = {trapped:.8f}, {trapped - minimum:.8f} above f*, "
    f"{scientific(distance(point, exact))} from the global minimum"
)
_, paired, point = run(SIGMAS, True, exact, 2)
print(
    f"the stages with L-BFGS-B's curvature pairs kept: {paired.generations} rounds and "
    f"{paired.evaluations} evaluations, within {scientific(distance(point, exact))}"
)
trace.write()
