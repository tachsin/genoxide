"""The trace of its run for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: the feasible solutions of the front, the population's infeasible
solutions and the front's hypervolume in scaled objectives, in at most 80 generations. The Rust
example writes the same file."""

import json
import math
import os

import numpy as np

import genoxide as gx


def target(w):
    """Where the reference direction w meets the optimal front, if it's feasible there: on
    the plane where the objectives sum to 1/2, all feasible."""
    return [v / 2 for v in w]


def scaled(points):
    """The objectives scaled to [0, 1] on the optimal front, by its nadir point (its ideal point is
    the origin)."""
    return np.asarray(points) / gx.problems.C1Dtlz1().nadir_point


def targets():
    """The target points: where the 91 reference directions meet the optimal front, where
    feasible."""
    points = [target(list(w)) for w in gx.das_dennis(3, 12)]
    return scaled([point for point in points if point is not None])


class Trace:
    """Records the run through ``on_generation`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self, problem):
        self.path = os.environ.get("GENOXIDE_TRACE")
        self.frames = Frames(80)
        self.problem = problem

    @property
    def on_generation(self):
        """The callback for ``run``: None without a trace to record."""
        return self.record if self.path else None

    def record(self, progress):
        """Records a generation: the front's feasible solutions and their hypervolume, and the
        population's infeasible solutions."""
        feasible = progress.violations == 0
        front = progress.front_objectives[progress.front_violations == 0]
        infeasible = progress.objectives[~feasible]
        state = {
            "front": front.tolist(),
            "infeasible": infeasible.tolist(),
            "hypervolume": gx.indicators.hypervolume(scaled(front).reshape(-1, 3), [1.1] * 3),
        }
        self.frames.push(frame(progress, state))

    def write(self):
        """Writes the trace, if there's one."""
        if self.path:
            settings = {
                "format": 1,
                "example": "c1_dtlz1_3obj",
                "objective": ["minimize", "minimize", "minimize"],
                "x_label": "generations",
                "y_label": "hypervolume (scaled objectives)",
                "log_y": False,
                # the target points' hypervolume
                "optimum": gx.indicators.hypervolume(targets(), [1.1] * 3),
                "plot": "front-3d",
                "problem": {
                    "objectives": ["f1", "f2", "f3"],
                    "true_front": "simplex",
                    "simplex": 0.5,
                },
            }
            write(self.path, settings, self.frames.to_list())


# ---- the same in every example's trace ----------------------------------------------------------


class Frames:
    """The frames of at most ``most`` generations: every ``every``-th one, with ``every`` doubling
    whenever there are ``most``, and the last one."""

    def __init__(self, most):
        self.most, self.every, self.kept, self.last = most, 1, [], None

    def push(self, frame):
        if frame["generation"] % self.every:
            self.last = frame
            return
        self.kept.append(frame)
        self.last = None
        if len(self.kept) == self.most:
            self.every *= 2
            self.kept = [kept for kept in self.kept if kept["generation"] % self.every == 0]

    def to_list(self):
        return self.kept + ([self.last] if self.last else [])


def frame(progress, state):
    """The frame of a generation: its progress and ``state``, whose hypervolume is the curve."""
    return {
        "generation": progress.generation,
        "evaluations": progress.evaluations,
        "best": None,
        "median": None,
        "state": state,
    }


def write(path, settings, frames):
    """Writes the settings and the frames to ``path``, a frame per line."""
    lines = ",\n".join(map(to_json, frames))
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write(f'{to_json(settings)[:-1]},"frames":[\n{lines}\n]}}\n')


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
