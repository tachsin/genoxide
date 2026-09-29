"""XOR neuroevolution: evolve the 9 weights of a 2-2-1 neural network until it computes XOR.

XOR isn't linearly separable, so the network needs its hidden layer: two sigmoid units, each with
a weight per input and a bias, and a sigmoid output unit with a weight per hidden unit and a bias.
The fitness is the sum of the squared errors over the four input pairs, minimized by CMA-ES with
BIPOP restarts, which escape the flat regions where the network outputs 0.5 or solves three of the
four cases. With the weights in [-10, 10], the smallest known error is 2.162140e-4, with every
weight on a bound.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/xor_neuroevolution/main.py
"""

import math

import genoxide as gx

from trace import Trace

# the inputs and the expected output
CASES = [((0.0, 0.0), 0.0), ((0.0, 1.0), 1.0), ((1.0, 0.0), 1.0), ((1.0, 1.0), 0.0)]

# the smallest known squared error with the weights in [-10, 10]: every weight on a bound, the
# biases inside
MINIMUM = 2.162140e-4


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def output(w, inputs):
    """The network's output: w[0:3] and w[3:6] are the hidden units' weights and biases, w[6:9]
    the output unit's."""
    a, b = inputs
    hidden_1 = sigmoid(w[0] * a + w[1] * b + w[2])
    hidden_2 = sigmoid(w[3] * a + w[4] * b + w[5])
    return sigmoid(w[6] * hidden_1 + w[7] * hidden_2 + w[8])


def squared_error(w):
    w = w.tolist()
    error = 0.0
    for inputs, expected in CASES:
        difference = output(w, inputs) - expected
        error += difference * difference
    return error


cmaes = gx.Cmaes(gx.Real((-10.0, 10.0), length=9), restarts="bipop", objective="minimize", seed=1)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(CASES, output, MINIMUM)
result = cmaes.run(
    squared_error,
    target=MINIMUM + 1e-6,
    evaluations=200_000,
    on_generation=trace.on_generation,
)

print(
    f"squared error {result.best_fitness:.9f} after {result.evaluations} evaluations "
    f"(the smallest known: {MINIMUM:.9f})"
)
for (a, b), expected in CASES:
    value = output(result.best_genome.tolist(), (a, b))
    print(f"{a:.0f} xor {b:.0f} = {expected:.0f}: {value:.3f}")
trace.write()
