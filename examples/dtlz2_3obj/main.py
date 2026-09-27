"""DTLZ2 with 3 objectives: minimize three conflicting objectives over 12 variables in [0, 1], whose
Pareto front is the positive eighth of the unit sphere.

NSGA-III with the 91 reference directions of Das and Dennis's method with 12 divisions, a
population of 92 and 250 generations, as in Deb and Jain (2014). Prints the size of the final
front and its hypervolume. The fitness function takes a generation at a time.

With ``GENOXIDE_TRACE=<file>``, it also writes the run's trace for the plot on the example's page:
the front and its hypervolume, in at most 100 generations.

    python examples/dtlz2_3obj/main.py
"""

import json
import math
import os

import numpy as np

import genoxide as gx

VARIABLES = 12


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
trace = Trace(100) if "GENOXIDE_TRACE" in os.environ else None


def record(progress):
    if trace:
        front = progress.front_objectives
        volume = gx.indicators.hypervolume(front, [1.1, 1.1, 1.1])
        trace.record(progress, {"front": front.tolist(), "hypervolume": volume})


result = nsga3.run(dtlz2, batch=True, generations=250, on_generation=record)

# the hypervolume of the front, with the reference point (1.1, 1.1, 1.1); the whole front's is
# 1.1³ minus the eighth of the unit ball, π/6
front = result.front_objectives
volume = hypervolume(front, np.array([1.1, 1.1, 1.1]))
print(f"{len(front)} solutions on the front, hypervolume {volume:.4f} (the whole front: 0.8074)")
if trace:
    trace.write(
        os.environ["GENOXIDE_TRACE"],
        {
            "format": 1,
            "example": "dtlz2_3obj",
            "objective": ["minimize", "minimize", "minimize"],
            "x_label": "generations",
            "y_label": "hypervolume",
            "log_y": False,
            "optimum": 0.8074,
            "plot": "front-3d",
            "problem": {"objectives": ["f1", "f2", "f3"], "true_front": "sphere"},
        },
    )
