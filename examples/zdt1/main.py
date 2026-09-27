"""ZDT1: minimize two conflicting objectives over 30 variables in [0, 1].

Shows NSGA-II, a fitness function that takes a generation at a time, and the hypervolume of the
final non-dominated front.

With ``GENOXIDE_TRACE=<file>``, it also writes the run's trace for the plot on the example's page:
the front and its hypervolume, in at most 100 generations.

    python examples/zdt1/main.py
"""

import json
import math
import os

import numpy as np

import genoxide as gx


# ---- the trace of the run, for the plot on the example's page -----------------------------------


class Trace:
    """A frame per recorded generation, at most ``most``: every ``every``-th generation, with
    ``every`` doubling whenever there are ``most``, and the last generation."""

    def __init__(self, most):
        self.most, self.every, self.frames, self.last = most, 1, [], None

    def record(self, progress, state):
        """The generation's progress and the plot's ``state``, whose hypervolume is the curve."""
        frame = {
            "generation": progress.generation,
            "evaluations": progress.evaluations,
            "best": None,
            "median": None,
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

def zdt1(x):
    """x has a genome per row; the result, a row of objective values per genome."""
    f1 = x[:, 0]
    g = 1 + 9 * x[:, 1:].mean(axis=1)
    return np.column_stack([f1, g * (1 - np.sqrt(f1 / g))])


nsga2 = gx.Nsga2(
    gx.Real((0.0, 1.0), length=30),
    objectives=["minimize", "minimize"],
    population_size=100,
    crossover=gx.SimulatedBinaryCrossover(15),
    mutation=gx.PolynomialMutation(20, rate=1 / 30),
    seed=1,
)
trace = Trace(100) if "GENOXIDE_TRACE" in os.environ else None


def record(progress):
    if trace:
        front = progress.front_objectives
        volume = gx.indicators.hypervolume(front, [1.1, 1.1])
        state = {"fronts": {"NSGA-II": front.tolist()}, "hypervolume": {"NSGA-II": volume}}
        trace.record(progress, state)


result = nsga2.run(zdt1, batch=True, evaluations=25_000, on_generation=record)
front = result.front_objectives[np.argsort(result.front_objectives[:, 0])]

# the hypervolume of the front, with the reference point (1.1, 1.1): the front is sorted by f1, so
# each point adds the rectangle up to the next point's f1
widths = np.diff(np.append(front[:, 0], 1.1))
hypervolume = np.sum(widths * (1.1 - front[:, 1]))
print(f"{len(front)} solutions on the front, hypervolume {hypervolume:.4f} (the whole front: 0.8767)")
if trace:
    trace.write(
        os.environ["GENOXIDE_TRACE"],
        {
            "format": 1,
            "example": "zdt1",
            "objective": ["minimize", "minimize"],
            "x_label": "evaluations",
            "y_label": "hypervolume",
            "log_y": False,
            "optimum": 0.8767,
            "plot": "front-2d",
            "problem": {
                "objectives": ["f1", "f2"],
                "true_front": gx.problems.Zdt1(30).optimal_front(100).tolist(),
                "series": ["NSGA-II"],
            },
        },
    )
