"""Adam with a learning-rate schedule: smooth 100,000 noisy points into a curve, the curve's
100,000 values the parameters, by penalized least squares with its gradient.

The data are made so that the exact answer is known, and the run ends at it: Adam's learning rate
is halved every 500 steps by ``control``, and the run stops when the gradient vanishes. Then the
same number of steps with the learning rate kept constant, which hovers around the answer
instead.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/adam/main.py
"""

import math

import numpy as np

import genoxide as gx

from trace import Trace

# the number of points, and of the curve's values
POINTS = 100_000
# the weight of the roughness against the misfit
LAMBDA = 50.0
# Adam's first learning rate, halved every HALVING steps
RATE = 0.05
HALVING = 500
# a row of the table every this many steps
EVERY = 100


def problem():
    """The exact answer x*, a smooth curve with a little noise of its own, and the data that make
    it the answer: d = x* + lambda L x*, where L is the Laplacian of the chain of points, so that
    (I + lambda L) x* = d, the normal equations of the loss."""
    noise = gx.Real((-0.01, 0.01), length=POINTS).random_genome(1)
    t = np.arange(POINTS) / (POINTS - 1)
    exact = gx.math.sin(math.tau * t) + 0.3 * gx.math.sin(5.0 * math.tau * t) + noise
    data = exact.copy()
    step = LAMBDA * np.diff(exact)
    data[1:] += step
    data[:-1] -= step
    return exact, data


EXACT, DATA = problem()


def loss(x):
    """The misfit to the data plus lambda times the roughness, and its gradient, in the order of
    the Rust example's operations, so that both take the same steps."""
    misfit = x - DATA
    gradient = 2.0 * misfit
    step = np.diff(x)
    weighted = (2.0 * LAMBDA) * step
    gradient[1:] += weighted
    gradient[:-1] -= weighted
    value = float(np.dot(misfit, misfit) + LAMBDA * np.dot(step, step))
    return value, gradient


def distance(x):
    """The largest difference between the curve and the answer."""
    return float(np.max(np.abs(x - EXACT)))


def scientific(value, digits):
    """``digits`` digits after the point, e.g. 1.2e-7."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def adam():
    return gx.FirstOrder(
        gx.Real((-10.0, 10.0), length=POINTS),
        step="adam",
        learning_rate=RATE,
        initial_genome=np.zeros(POINTS),
        gradient_tolerance=1e-9,
        objective="minimize",
    )


# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace()
rows = []


def schedule(running, progress):
    point = progress.population[0]
    rate = running.learning_rate
    trace.scheduled(progress.generation, distance(point), rate)
    rows.append(
        (progress.generation, rate, progress.scores[0], running.gradient_norm, distance(point))
    )
    # the learning rate of the next step: halved every HALVING steps
    running.learning_rate = RATE * 0.5 ** (progress.generation // HALVING)


result = adam().run(loss, gradient=True, generations=10_000, control=schedule)
steps = result.generations

print(f"Smoothing {POINTS} noisy points: the misfit plus {LAMBDA:g} times the roughness")
print(f"Adam from a flat curve, its learning rate {RATE} halved every {HALVING} steps")
print("  step  learning rate  loss         largest gradient  distance to the answer")
for index, (step, rate, value, gradient, gap) in enumerate(rows):
    if step % EVERY == 0 or index == len(rows) - 1:
        print(
            f"{step:>6}  {scientific(rate, 1):>13}  {scientific(value, 7):>11}  "
            f"{scientific(gradient, 1):>16}  {scientific(gap, 1):>22}"
        )
assert result.stop_reason == "converged"
print(
    f"converged after {steps} steps and {result.evaluations} evaluations: every value within "
    f"{scientific(rows[-1][4], 1)} of the answer"
)

# the same number of steps, the learning rate constant
constant = []


def keep(running, progress):
    gap = distance(progress.population[0])
    trace.constant(progress.generation, gap, running.learning_rate)
    constant.append(gap)


adam().run(loss, gradient=True, generations=steps, control=keep)
print(
    f"the learning rate kept at {RATE}, after the same {steps} steps: within "
    f"{scientific(constant[-1], 1)} of the answer"
)
trace.write()
