"""XOR neuroevolution: evolve the 9 weights of a 2-2-1 neural network until it computes XOR.

XOR isn't linearly separable, so the network needs its hidden layer: two sigmoid units, each with
a weight per input and a bias, and a sigmoid output unit with a weight per hidden unit and a bias.
The fitness is the sum of the squared errors over the four input pairs, minimized by CMA-ES with
IPOP restarts, which escape the flat regions where the network outputs 0.5 or solves three of the
four cases.

With ``GENOXIDE_TRACE=<file>``, it also writes the run's trace for the plot on the example's page:
the best network's output over [0, 1]², on a grid of 21 × 21 points, in at most 64 generations.

    python examples/xor_neuroevolution/main.py
"""

import json
import math
import os

import genoxide as gx

# the inputs and the expected output
CASES = [((0.0, 0.0), 0.0), ((0.0, 1.0), 1.0), ((1.0, 0.0), 1.0), ((1.0, 1.0), 0.0)]
# the points per side of the trace's grid
GRID = 21


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


# ---- the trace of the run, for the plot on the example's page -----------------------------------


class Trace:
    """A frame per recorded generation, at most ``most``: every ``every``-th generation, with
    ``every`` doubling whenever there are ``most``, and the last generation."""

    def __init__(self, most):
        self.most, self.every, self.frames, self.last = most, 1, [], None

    def record(self, progress, state):
        """The generation's progress, the median score of its population and the plot's
        ``state``."""
        frame = {
            "generation": progress.generation,
            "evaluations": progress.evaluations,
            "best": progress.best_fitness,
            "median": median(progress.scores),
            "state": state,
        }
        self.push(frame)

    def push(self, frame):
        """Keeps ``frame`` if it's of the ``every``-th generation, or as the last one."""
        if frame["generation"] % self.every:
            self.last = frame
            return
        self.frames.append(frame)
        self.last = None
        if len(self.frames) == self.most:
            self.every *= 2
            self.frames = [frame for frame in self.frames if frame["generation"] % self.every == 0]

    def write(self, path, settings):
        """Writes the settings and the frames to ``path``, a frame per line."""
        frames = ",\n".join(to_json(frame) for frame in self.frames + [self.last] if frame)
        with open(path, "w", encoding="utf-8", newline="\n") as file:
            file.write(f'{to_json(settings)[:-1]},"frames":[\n{frames}\n]}}\n')


def median(scores):
    """The median of the valid scores, None without any."""
    scores = sorted(float(score) for score in scores if not math.isnan(score))
    middle = len(scores) // 2
    if not scores:
        return None
    return scores[middle] if len(scores) % 2 else (scores[middle - 1] + scores[middle]) / 2


def to_json(value):
    """Compact JSON with sorted keys, and numbers rounded to 6 significant digits, as the Rust
    example writes it."""
    return json.dumps(rounded(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def rounded(value):
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item) for item in value]
    if isinstance(value, float):
        return float(f"{value:.5e}") if math.isfinite(value) else None
    return value


# -------------------------------------------------------------------------------------------------


def surface(w):
    """The network's output over [0, 1]²: a row per value of the second input, from 0 to 1, and
    a column per value of the first."""
    points = [i / (GRID - 1) for i in range(GRID)]
    return [[output(w, (a, b)) for a in points] for b in points]


trace = Trace(64) if "GENOXIDE_TRACE" in os.environ else None


def record(progress):
    if trace:
        trace.record(progress, {"grid": surface(progress.best_genome.tolist())})


cmaes = gx.Cmaes(gx.Real((-10.0, 10.0), length=9), restarts="ipop", objective="minimize", seed=1)
result = cmaes.run(squared_error, target=0.01, evaluations=20_000, on_generation=record)

print(f"squared error {result.best_fitness:.6f} after {result.evaluations} evaluations")
for (a, b), expected in CASES:
    value = output(result.best_genome.tolist(), (a, b))
    print(f"{a:.0f} xor {b:.0f} = {expected:.0f}: {value:.3f}")
if trace:
    trace.write(
        os.environ["GENOXIDE_TRACE"],
        {
            "format": 1,
            "example": "xor_neuroevolution",
            "objective": "minimize",
            "x_label": "evaluations",
            "y_label": "squared error",
            "log_y": False,
            "optimum": 0.0,
            "plot": "surface",
            "problem": {
                "inputs": [inputs for inputs, _ in CASES],
                "targets": [target for _, target in CASES],
                "grid": GRID,
            },
        },
    )
